"""
dataset.py — Data Loading & Feature Extraction
===============================================
Handles:
  - Audio loading and resampling
  - Mimi tokenization via HuggingFace (kyutai/moshiko-pytorch-bf16) at 12.5 Hz
  - HuBERT-large feature extraction (50 Hz, 1024-dim)
  - Pitch (F0) and energy extraction via pyworld / librosa
  - Optional forced-alignment labels
  - Caching of pre-extracted features
  - Collation with padding masks
"""

import os
import json
import math
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (guarded so tests can import dataset.py headlessly)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False
    logger.warning("torchaudio not found – audio loading will fail at runtime.")

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    logger.warning("librosa not found – pitch extraction may fail.")

try:
    import pyworld
    PYWORLD_OK = True
except ImportError:
    PYWORLD_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extractors (wrappers around heavy models)
# ──────────────────────────────────────────────────────────────────────────────

class MimiExtractor:
    """
    Wraps the HuggingFace Mimi encoder (kyutai/moshiko-pytorch-bf16) to produce
    (T, num_codebooks) integer token tensors at 12.5 Hz.

    Install:  pip install transformers>=4.40.0
    Model:    kyutai/moshiko-pytorch-bf16  (or any repo containing a MimiModel)
    """

    # Number of codebooks exposed by the Mimi encoder
    NUM_CODEBOOKS = 8

    def __init__(self, model_name: str = "kyutai/moshiko-pytorch-bf16", device: str = "cpu"):
        self.device = device
        self._ok = False
        try:
            from transformers import MimiModel, AutoFeatureExtractor
            logger.info(f"Loading Mimi encoder from {model_name} …")
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = MimiModel.from_pretrained(model_name)
            self.model.eval().to(device)
            self._ok = True
            logger.info("Mimi encoder loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load Mimi from HuggingFace ({e}). "
                "Falling back to dummy extractor. "
                "Install with:  pip install transformers>=4.40.0"
            )

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        wav : (1, samples)  float32, at the model's expected sample rate
        Returns: (T, NUM_CODEBOOKS) int64 tensor  — T ≈ samples / 1280 ≈ 12.5 Hz
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // 1280)
            return torch.randint(0, 2048, (T, self.NUM_CODEBOOKS))

        wav_np = wav.squeeze(0).numpy()          # (samples,)
        inputs = self.processor(
            raw_audio=wav_np,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # encode() returns an EncoderOutput whose .audio_codes is (B, num_codebooks, T)
        encoder_out = self.model.encode(**inputs)
        codes = encoder_out.audio_codes          # (B, num_codebooks, T)
        codes = codes.squeeze(0)                 # (num_codebooks, T)
        codes = codes.transpose(0, 1)            # (T, num_codebooks)
        return codes.cpu().long()


class HuBERTExtractor:
    """
    Wraps a pretrained HuBERT-large model to produce (T, 1024) features at ~50 Hz.
    Default: facebook/hubert-large-ls960-ft  (1024-dim hidden states)
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        device: str = "cpu",
    ):
        self.device = device
        self._ok = False
        self._feat_dim = 1024  # HuBERT-large hidden size
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            logger.info(f"Loading HuBERT from {model_name} …")
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name)
            self.model.eval().to(device)
            # Detect actual hidden size from config
            self._feat_dim = self.model.config.hidden_size
            self._ok = True
            logger.info(f"HuBERT loaded — hidden_size={self._feat_dim}.")
        except Exception as e:
            logger.warning(f"Could not load HuBERT: {e}. Using dummy extractor.")

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        wav : (1, samples) at 16 kHz
        Returns: (T, feat_dim) float32 tensor
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // 320)   # approx 50 Hz at 16 kHz
            return torch.randn(T, self._feat_dim)

        wav_np = wav.squeeze().numpy()
        inputs = self.processor(wav_np, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state.squeeze(0).cpu().float()  # (T, feat_dim)
        return feats


def extract_f0_energy(
    wav: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalised log-F0 and log-energy using pyworld or librosa fallback.

    Returns:
        f0:      (T,) normalised in [0, 1], 0 = unvoiced
        energy:  (T,) normalised in [0, 1]
        voiced:  (T,) bool
    """
    if PYWORLD_OK:
        _wav = wav.astype(np.float64)
        f0, t = pyworld.harvest(_wav, sr, f0_floor=f0_min, f0_ceil=f0_max,
                                frame_period=hop_length / sr * 1000)
        voiced = f0 > 0
        f0_log = np.where(voiced, np.log(f0 + 1e-8), 0.0)
    elif LIBROSA_OK:
        f0_arr, voiced_flag, _ = librosa.pyin(
            wav, fmin=f0_min, fmax=f0_max, sr=sr, hop_length=hop_length
        )
        f0_arr = np.nan_to_num(f0_arr, nan=0.0)
        voiced = voiced_flag.astype(bool)
        f0_log = np.where(voiced, np.log(f0_arr + 1e-8), 0.0)
    else:
        T = math.ceil(len(wav) / hop_length)
        return np.zeros(T), np.zeros(T), np.zeros(T, dtype=bool)

    # Energy via RMS
    if LIBROSA_OK:
        rms = librosa.feature.rms(y=wav, hop_length=hop_length, frame_length=hop_length * 4)[0]
        rms = rms[:len(f0_log)]
    else:
        # Manual RMS
        frames = [wav[i:i + hop_length * 4] for i in range(0, len(wav), hop_length)]
        rms = np.array([np.sqrt(np.mean(f**2 + 1e-8)) for f in frames])
        rms = rms[:len(f0_log)]

    energy_log = np.log(rms + 1e-8)

    # Normalise to [0, 1]
    def safe_norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    return safe_norm(f0_log).astype(np.float32), safe_norm(energy_log).astype(np.float32), voiced


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MimiHuBERTDataset(Dataset):
    """
    Paired dataset that provides:
      - Mimi tokens    (T_m, 8)
      - HuBERT targets (T_h, feat_dim)   T_h = 4 * T_m
      - Prosody        (T_h,) F0 and energy
      - Optional phoneme labels
    """

    def __init__(
        self,
        manifest_path: str,
        cfg: dict,
        split: str = "train",
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.split = split
        self.sr = cfg["data"]["sample_rate"]
        self.max_len = int(cfg["data"]["max_audio_seconds"] * self.sr)
        self.cache_features = cfg["data"].get("cache_features", True)
        self.cache_dir = Path(cfg["data"].get("cache_dir", "data/cache"))
        self.hop_length = cfg["training"].get("hop_length", 160)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(manifest_path) as f:
            self.samples = [json.loads(l) for l in f]

        logger.info(f"Loaded {len(self.samples)} samples ({split})")

        # Lazy-init extractors (heavy; only when needed)
        self._mimi = None
        self._hubert = None
        self._device = device

    def _get_mimi(self):
        if self._mimi is None:
            self._mimi = MimiExtractor(self.cfg["paths"]["mimi_model"], self._device)
        return self._mimi

    def _get_hubert(self):
        if self._hubert is None:
            self._hubert = HuBERTExtractor(self.cfg["paths"]["hubert_model"], self._device)
        return self._hubert

    def _cache_path(self, audio_path: str, suffix: str) -> Path:
        h = hashlib.md5(audio_path.encode()).hexdigest()
        return self.cache_dir / f"{h}_{suffix}.pt"

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio to target SR. Returns (1, samples)."""
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        # Trim to max length
        if waveform.shape[-1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        return waveform

    def _get_or_cache(self, audio_path: str, key: str, extractor_fn):
        cp = self._cache_path(audio_path, key)
        if self.cache_features and cp.exists():
            return torch.load(cp, map_location="cpu")
        result = extractor_fn()
        if self.cache_features:
            torch.save(result, cp)
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        # --- Load audio ---
        wav = self._load_audio(audio_path)           # (1, N)
        wav_np = wav.squeeze().numpy()

        # --- Mimi tokens ---
        tokens = self._get_or_cache(
            audio_path, "mimi",
            lambda: self._get_mimi().extract(wav, self.sr)
        )  # (T_m, 8)

        # --- HuBERT features ---
        hubert = self._get_or_cache(
            audio_path, "hubert",
            lambda: self._get_hubert().extract(wav, self.sr)
        )  # (T_h, feat_dim)

        T_m = tokens.shape[0]
        T_h = hubert.shape[0]

        # Ensure 4:1 ratio (trim to min)
        T_min = min(T_m, T_h // 4)
        tokens = tokens[:T_min]          # (T_m, 8)
        hubert = hubert[:T_min * 4]      # (4*T_m, feat_dim)

        # --- Prosody ---
        f0, energy, voiced = extract_f0_energy(
            wav_np, self.sr, self.hop_length
        )
        # Resample prosody to HuBERT rate (50 Hz)
        T_h = T_min * 4
        f0     = torch.from_numpy(self._resample_array(f0, T_h))
        energy = torch.from_numpy(self._resample_array(energy, T_h))
        voiced = torch.from_numpy(self._resample_array(voiced.astype(np.float32), T_h) > 0.5)

        # --- Optional phoneme labels ---
        phone_labels = None
        if "phone_labels" in sample:
            phone_labels = torch.tensor(sample["phone_labels"][:T_h], dtype=torch.long)
            if len(phone_labels) < T_h:
                phone_labels = F.pad(phone_labels, (0, T_h - len(phone_labels)), value=-100)

        return {
            "tokens":       tokens,           # (T_m, 8) int64
            "hubert":       hubert,           # (T_h, feat_dim) float
            "f0":           f0,               # (T_h,) float
            "energy":       energy,           # (T_h,) float
            "voiced":       voiced,           # (T_h,) bool
            "phone_labels": phone_labels,     # (T_h,) or None
            "audio_path":   audio_path,
        }

    @staticmethod
    def _resample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        indices = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(indices, np.arange(len(arr)), arr).astype(arr.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Collate Function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length sequences to the longest in the batch.
    Returns a dict with batch tensors and padding masks.
    Feature dim is inferred dynamically from the first sample so this
    works for both 768-dim (HuBERT-base) and 1024-dim (HuBERT-large).
    """
    # Sort by descending token length
    batch = sorted(batch, key=lambda x: x["tokens"].shape[0], reverse=True)

    max_T_m = max(b["tokens"].shape[0] for b in batch)
    max_T_h = max_T_m * 4

    B = len(batch)
    # Infer dims dynamically — supports both 768 (base) and 1024 (large)
    feat_dim       = batch[0]["hubert"].shape[-1]
    num_codebooks  = batch[0]["tokens"].shape[-1]

    tokens_out = torch.zeros(B, max_T_m, num_codebooks, dtype=torch.long)
    hubert_out  = torch.zeros(B, max_T_h, feat_dim)
    f0_out      = torch.zeros(B, max_T_h)
    energy_out  = torch.zeros(B, max_T_h)
    voiced_out  = torch.zeros(B, max_T_h, dtype=torch.bool)
    mask_out    = torch.zeros(B, max_T_h, dtype=torch.bool)   # True = valid
    phone_out   = torch.full((B, max_T_h), -100, dtype=torch.long)

    token_lengths = []
    for i, sample in enumerate(batch):
        T_m = sample["tokens"].shape[0]
        T_h = T_m * 4
        tokens_out[i, :T_m]     = sample["tokens"]
        hubert_out[i, :T_h]     = sample["hubert"]
        f0_out[i, :T_h]         = sample["f0"]
        energy_out[i, :T_h]     = sample["energy"]
        voiced_out[i, :T_h]     = sample["voiced"]
        mask_out[i, :T_h]       = True
        if sample["phone_labels"] is not None:
            phone_out[i, :T_h]  = sample["phone_labels"]
        token_lengths.append(T_m)

    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    frame_lengths = token_lengths * 4   # HuBERT-space lengths for CTC

    return {
        "tokens":         tokens_out,           # (B, T_m, 8)
        "hubert":         hubert_out,           # (B, T_h, feat_dim)
        "f0":             f0_out,               # (B, T_h)
        "energy":         energy_out,           # (B, T_h)
        "voiced_mask":    voiced_out,           # (B, T_h)
        "mask":           mask_out,             # (B, T_h)
        "phone_labels":   phone_out,            # (B, T_h)
        "input_lengths":  frame_lengths,        # (B,) for CTC
        "ctc_targets":    None,                 # populated externally if needed
        "target_lengths": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factories
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    train_ds = MimiHuBERTDataset(cfg["data"]["train_manifest"], cfg, "train", device)
    val_ds   = MimiHuBERTDataset(cfg["data"]["val_manifest"],   cfg, "val",   device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    return train_loader, val_loader
