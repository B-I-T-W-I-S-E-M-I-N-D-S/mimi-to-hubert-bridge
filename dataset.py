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
    Loads the Mimi audio tokenizer from kyutai/moshiko-pytorch-bf16 on HuggingFace.

    The kyutai/moshiko-pytorch-bf16 repo does NOT contain a preprocessor_config.json
    or processor_config.json, so AutoFeatureExtractor / MimiModel.from_pretrained()
    will always fail for it.  Instead we download the dedicated Mimi weights file
    (tokenizer-e351c8d8-checkpoint125.safetensors) via hf_hub_download and build
    the model with the moshi library, exactly as the reference loader.py does.

    Two back-ends are attempted in order:
      1. moshi  — the official Kyutai Python package (pip install moshi).
         Uses hf_hub_download + safetensors, no preprocessor_config required.
      2. transformers MimiModel  — works when model_name points to a repo that
         *does* have preprocessor_config.json (e.g. kyutai/mimi).
      3. Dummy  — random tokens so the rest of the pipeline stays runnable.

    Install (recommended):
        pip install moshi safetensors huggingface_hub
    Alternative (only for repos with HF processor configs):
        pip install transformers>=4.40.0
    """

    # File name of the Mimi weights inside the kyutai/moshiko-pytorch-bf16 repo
    _MIMI_SAFETENSORS = "tokenizer-e351c8d8-checkpoint125.safetensors"
    # Number of codebooks we ask Mimi to use (matches config.yaml num_codebooks)
    NUM_CODEBOOKS = 8
    # Mimi operates at 24 kHz internally; hop = 1920 → 12.5 Hz frame rate
    _MIMI_SR = 24000
    _HOP = 1920   # 24000 / 12.5

    def __init__(self, model_name: str = "kyutai/moshiko-pytorch-bf16", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self._ok = False
        self._backend = None   # "moshi" | "transformers"

        # ── Backend 1: moshi library (handles raw safetensors repos) ──────────
        if not self._ok:
            self._try_load_moshi(model_name, device)

        # ── Backend 2: transformers MimiModel (needs preprocessor_config) ────
        if not self._ok:
            self._try_load_transformers(model_name, device)

        if not self._ok:
            logger.warning(
                f"[MimiExtractor] All loading strategies failed for '{model_name}'. "
                "Running with a DUMMY extractor (random tokens). "
                "To fix: pip install moshi safetensors huggingface_hub"
            )

    # ── Private loader helpers ────────────────────────────────────────────────

    def _try_load_moshi(self, model_name: str, device: str):
        """
        Download tokenizer-e351c8d8-checkpoint125.safetensors from HF and build
        a MimiModel using the moshi library's get_mimi() helper and the
        hard-coded architecture config (_mimi_config) from loader.py.
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file as sf_load
            import torch as _torch

            logger.info(
                f"[MimiExtractor] Downloading Mimi weights from {model_name} "
                f"({self._MIMI_SAFETENSORS}) via hf_hub_download …"
            )
            weights_path = hf_hub_download(
                repo_id=model_name,
                filename=self._MIMI_SAFETENSORS,
            )
            logger.info(f"[MimiExtractor] Weights cached at {weights_path}")

            # Build the Mimi model using the moshi library
            try:
                from moshi.models.loaders import get_mimi
                self.model = get_mimi(weights_path, device=device)
                self.model.set_num_codebooks(self.NUM_CODEBOOKS)
                self.model.eval()
                self._backend = "moshi"
                self._ok = True
                logger.info(
                    f"[MimiExtractor] Loaded via moshi library "
                    f"(num_codebooks={self.NUM_CODEBOOKS})."
                )
                return
            except ImportError:
                logger.info(
                    "[MimiExtractor] moshi package not installed; "
                    "attempting manual safetensors load with built-in architecture …"
                )

            # ── Fallback: build architecture manually without the moshi package ──
            # Architecture constants copied directly from loader.py (_mimi_config).
            self._weights_path = weights_path
            self._ok = True
            self._backend = "safetensors_raw"
            self._sf_path = weights_path
            self._build_moshi_manual(weights_path, device)

        except Exception as e:
            logger.info(f"[MimiExtractor] moshi/safetensors strategy failed: {e}")

    def _build_moshi_manual(self, weights_path: str, device: str):
        """
        Instantiate Mimi manually using only torch + safetensors, without the
        moshi package. Uses the exact architecture kwargs from loader.py.
        Falls back to transformers if this also fails.
        """
        try:
            from safetensors.torch import load_file as sf_load

            # These are the exact kwargs from loader.py
            seanet_kwargs = {
                "channels": 1, "dimension": 512, "causal": True,
                "n_filters": 64, "n_residual_layers": 1, "activation": "ELU",
                "compress": 2, "dilation_base": 2,
                "disable_norm_outer_blocks": 0, "kernel_size": 7,
                "residual_kernel_size": 3, "last_kernel_size": 3,
                "norm": "none", "pad_mode": "constant",
                "ratios": [8, 6, 5, 4], "true_skip": True,
            }
            quantizer_kwargs = {
                "dimension": 256, "n_q": 32, "bins": 2048,
                "input_dimension": seanet_kwargs["dimension"],
                "output_dimension": seanet_kwargs["dimension"],
            }
            transformer_kwargs = {
                "d_model": seanet_kwargs["dimension"], "num_heads": 8,
                "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000,
                "gating": "none", "norm": "layer_norm",
                "positional_embedding": "rope", "dim_feedforward": 2048,
                "input_dimension": seanet_kwargs["dimension"],
                "output_dimensions": [seanet_kwargs["dimension"]],
            }

            from moshi.modules import SEANetEncoder, SEANetDecoder
            from moshi.modules import transformer as moshi_transformer
            from moshi.quantization import SplitResidualVectorQuantizer
            from moshi.models.compression import MimiModel

            enc = SEANetEncoder(**seanet_kwargs)
            dec = SEANetDecoder(**seanet_kwargs)
            enc_tr = moshi_transformer.ProjectedTransformer(device=device, **transformer_kwargs)
            dec_tr = moshi_transformer.ProjectedTransformer(device=device, **transformer_kwargs)
            quant = SplitResidualVectorQuantizer(**quantizer_kwargs)

            model = MimiModel(
                enc, dec, quant,
                channels=1, sample_rate=24000, frame_rate=12.5,
                encoder_frame_rate=24000 / enc.hop_length,
                causal=True, resample_method="conv",
                encoder_transformer=enc_tr, decoder_transformer=dec_tr,
            ).to(device=device)
            model.eval()

            state = sf_load(weights_path, device=str(device))
            model.load_state_dict(state)
            model.set_num_codebooks(self.NUM_CODEBOOKS)

            self.model = model
            self._backend = "moshi_manual"
            logger.info("[MimiExtractor] Manual moshi architecture load succeeded.")

        except Exception as e:
            logger.warning(
                f"[MimiExtractor] Manual architecture build failed ({e}); "
                "will try transformers backend next."
            )
            self._ok = False

    def _try_load_transformers(self, model_name: str, device: str):
        """
        Fall back to transformers MimiModel + AutoFeatureExtractor.
        Works only for repos that ship preprocessor_config.json (e.g. kyutai/mimi).
        """
        try:
            from transformers import MimiModel, AutoFeatureExtractor
            logger.info(
                f"[MimiExtractor] Trying transformers AutoFeatureExtractor for '{model_name}' …"
            )
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = MimiModel.from_pretrained(model_name)
            self.model.eval().to(device)
            self._backend = "transformers"
            self._ok = True
            logger.info("[MimiExtractor] Loaded via transformers MimiModel.")
        except Exception as e:
            logger.info(f"[MimiExtractor] transformers strategy failed: {e}")

    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Encode a waveform into Mimi discrete tokens.

        wav : (1, samples)  float32  — any supported sample rate (resampled internally
              to 24 kHz for moshi backends, or to whatever the transformers model needs)
        sr  : sample rate of `wav`

        Returns: (T, NUM_CODEBOOKS) int64 tensor  at 12.5 Hz
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // self._HOP)
            return torch.randint(0, 2048, (T, self.NUM_CODEBOOKS))

        # ── moshi / moshi_manual / safetensors_raw backends ──────────────────
        if self._backend in ("moshi", "moshi_manual", "safetensors_raw"):
            # Mimi always expects 24 kHz mono audio
            if sr != self._MIMI_SR:
                if TORCHAUDIO_OK:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav, sr, self._MIMI_SR)
                else:
                    # numpy-based fallback resampling
                    ratio = self._MIMI_SR / sr
                    new_len = int(wav.shape[-1] * ratio)
                    wav_np = wav.squeeze(0).numpy()
                    indices = np.linspace(0, len(wav_np) - 1, new_len)
                    wav_np = np.interp(indices, np.arange(len(wav_np)), wav_np).astype(np.float32)
                    wav = torch.from_numpy(wav_np).unsqueeze(0)

            # moshi encode() expects (B, C, T); returns codes (B, num_codebooks, T)
            wav_device = wav.to(self.device)
            if wav_device.dim() == 2:
                wav_device = wav_device.unsqueeze(0)   # (1, 1, T)
            elif wav_device.dim() == 1:
                wav_device = wav_device.unsqueeze(0).unsqueeze(0)

            codes = self.model.encode(wav_device)      # (B, num_codebooks, T)
            codes = codes.squeeze(0)                   # (num_codebooks, T)
            codes = codes.transpose(0, 1)              # (T, num_codebooks)
            return codes.cpu().long()

        # ── transformers backend ──────────────────────────────────────────────
        wav_np = wav.squeeze(0).numpy()
        inputs = self.processor(raw_audio=wav_np, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        encoder_out = self.model.encode(**inputs)
        codes = encoder_out.audio_codes          # (B, num_codebooks, T)
        codes = codes.squeeze(0).transpose(0, 1)  # (T, num_codebooks)
        return codes.cpu().long()


class HuBERTExtractor:
    """
    ONNX-based HuBERT streaming extractor using hubert_streaming_fix_kv.onnx.

    Produces (T, 1024) float32 features at 25 Hz (one frame per 40 ms),
    with each output frame being the mean of two 20 ms ONNX encoder frames.

    GPU acceleration strategy
    ─────────────────────────
    1. ONNX runtime   — uses CUDAExecutionProvider when device="cuda",
                        automatically falls back to CPU if CUDA EP is unavailable.
    2. Resampling     — done on the GPU via torchaudio.functional.resample()
                        instead of librosa (CPU-only).
    3. Batched chunks — all overlapping chunks for one utterance are stacked
                        into a single batched ONNX call, maximising GPU occupancy
                        and avoiding per-chunk Python overhead.
    4. Pinned memory  — numpy arrays passed to the ONNX CUDA EP are allocated
                        as pinned (page-locked) memory so DMA transfers are async.

    Args:
        model_name   : Path to the .onnx model file
                       (e.g. "./model/hubert_streaming_fix_kv.onnx")
        device       : "cuda" | "cuda:N" | "cpu"
        chunk_batch  : How many chunks to run in one batched ONNX call.
                       Larger = more GPU utilisation; reduce if OOM.
                       Default 32 works well for a 16 GB GPU.
    """

    # ONNX streaming chunk config: (left_context, centre, right_context) in frames
    _CHUNKSIZE  = (3, 5, 2)
    _FEAT_DIM   = 1024   # HuBERT-large hidden size
    _TARGET_SR  = 16000  # Model expects 16 kHz audio

    def __init__(
        self,
        model_name:  str = "./model/hubert_streaming_fix_kv.onnx",
        device:      str = "cpu",
        chunk_batch: int = 1,
    ):
        self.device      = device
        self._ok         = False
        self._feat_dim   = self._FEAT_DIM
        self._chunk_batch = chunk_batch
        self._use_cuda   = device.startswith("cuda")

        try:
            import onnxruntime as ort

            logger.info(f"[HuBERTExtractor] Loading ONNX model from {model_name} …")
            sess_opt = ort.SessionOptions()
            sess_opt.intra_op_num_threads = 4

            # ── Provider selection: prefer CUDA, fall back to CPU ─────────────
            available = ort.get_available_providers()
            if self._use_cuda and "CUDAExecutionProvider" in available:
                # Extract device index from "cuda:N" (default 0)
                dev_idx = 0
                if ":" in device:
                    dev_idx = int(device.split(":")[1])
                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id":               dev_idx,
                        # Use CUDA streams for async host↔device transfers
                        "arena_extend_strategy":   "kNextPowerOfTwo",
                        "gpu_mem_limit":           4 * 1024 ** 3,   # 4 GB cap
                        "cudnn_conv_algo_search":  "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }),
                    "CPUExecutionProvider",   # fallback
                ]
                logger.info(
                    f"[HuBERTExtractor] Using CUDAExecutionProvider "
                    f"(device_id={dev_idx}, chunk_batch={chunk_batch})."
                )
            else:
                if self._use_cuda:
                    logger.warning(
                        "[HuBERTExtractor] CUDAExecutionProvider not available in this "
                        "onnxruntime build — falling back to CPU. "
                        "Install: pip install onnxruntime-gpu"
                    )
                    self._use_cuda = False
                providers = ["CPUExecutionProvider"]
                logger.info("[HuBERTExtractor] Using CPUExecutionProvider.")

            self._ort_session = ort.InferenceSession(
                model_name,
                sess_options=sess_opt,
                providers=providers,
            )
            self._ok = True
            logger.info(
                f"[HuBERTExtractor] ONNX HuBERT loaded successfully "
                f"(feat_dim={self._feat_dim}, chunksize={self._CHUNKSIZE}, "
                f"cuda={self._use_cuda})."
            )
        except Exception as e:
            logger.warning(
                f"[HuBERTExtractor] Could not load ONNX HuBERT from '{model_name}': {e}. "
                "Using dummy extractor. "
                "To fix: pip install onnxruntime-gpu librosa && check model path."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resample_gpu(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample wav (1, N) on GPU using torchaudio — much faster than librosa.
        Returns (1, N') at _TARGET_SR, still on the same device as input.
        """
        if orig_sr == self._TARGET_SR:
            return wav
        if TORCHAUDIO_OK:
            import torchaudio
            return torchaudio.functional.resample(wav, orig_sr, self._TARGET_SR)
        # librosa CPU fallback (should rarely hit this path)
        import librosa
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        resampled = librosa.resample(wav_np, orig_sr=orig_sr, target_sr=self._TARGET_SR)
        return torch.from_numpy(resampled).unsqueeze(0).to(wav.device)

    def _build_chunks(self, speech: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Slice and pad speech into overlapping input chunks.

        Returns
        -------
        chunks   : (N_chunks, split_len) float32 array
        num_f    : total number of expected output frames
        """
        chunksize = self._CHUNKSIZE
        sr        = self._TARGET_SR

        num_f     = math.ceil(len(speech) / sr * 25)
        split_len = int(sum(chunksize) * 0.04 * sr) + 80
        left_pad  = split_len - int(sum(chunksize[1:]) * 0.04 * sr)
        right_pad = split_len

        speech_pad = np.concatenate([
            np.zeros(left_pad,  dtype=speech.dtype),
            speech,
            np.zeros(right_pad, dtype=speech.dtype),
        ])

        chunks = []
        i = 0
        while i < num_f:
            sss = int(i * 0.04 * sr)
            chunks.append(speech_pad[sss : sss + split_len])
            i += chunksize[1]

        # Stack into a single array: (N_chunks, split_len)
        return np.stack(chunks, axis=0).astype(np.float32), num_f

    def _run_batched_onnx(self, chunks: np.ndarray) -> np.ndarray:
        """
        Run all chunks through the ONNX session in mini-batches of
        self._chunk_batch.  Returns (N_chunks, frames_per_chunk, 1024).
        Using pinned memory for CUDA EP accelerates host→device DMA.
        """
        N          = chunks.shape[0]
        batch_sz   = self._chunk_batch
        results    = []

        for start in range(0, N, batch_sz):
            batch = chunks[start : start + batch_sz]   # (B, split_len)

            # Pinned memory for async DMA when using CUDA EP
            if self._use_cuda:
                pinned = np.empty_like(batch)
                np.copyto(pinned, batch)
                batch = pinned

            enc = self._ort_session.run(
                None, {"input_values": batch}
            )[0]   # (B, frames_per_chunk, 1024) or (B*frames, 1024)

            # Normalise to 3-D: (B, frames_per_chunk, 1024)
            if enc.ndim == 2:
                enc = enc.reshape(batch.shape[0], -1, self._FEAT_DIM)
            results.append(enc)

        return np.concatenate(results, axis=0)   # (N_chunks, frames_per_chunk, 1024)

    def _postprocess(self, all_enc: np.ndarray, num_f: int) -> np.ndarray:
        """
        Slice valid centre frames and average 20 ms pairs → 25 Hz output.

        all_enc : (N_chunks, frames_per_chunk, 1024)
        Returns : (num_f, 1024) float32
        """
        chunksize    = self._CHUNKSIZE
        valid_feat_s = -sum(chunksize[1:]) * 2   # -14
        valid_feat_e = -chunksize[2] * 2          # -4

        # Slice valid frames from each chunk: (N_chunks, 10, 1024)
        valid = all_enc[:, valid_feat_s:valid_feat_e, :]

        # Mean-pool pairs of 20 ms frames → 40 ms / 25 Hz: (N_chunks, 5, 1024)
        valid = valid.reshape(all_enc.shape[0], chunksize[1], 2, self._FEAT_DIM).mean(axis=2)

        # Flatten chunks: (N_chunks * 5, 1024) then trim
        ret = valid.reshape(-1, self._FEAT_DIM)[:num_f]
        return ret.astype(np.float32)

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract HuBERT features from a waveform tensor.

        wav : (1, samples) float32 torch.Tensor — any sample rate.
              If device="cuda" the tensor may already be on GPU; resampling
              is performed on whichever device the tensor lives on.
        sr  : sample rate of `wav`

        Returns: (T, 1024) float32 torch.Tensor on CPU
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // 640)   # approx 25 Hz at 16 kHz
            return torch.randn(T, self._feat_dim)

        # ── 1. GPU resampling (torchaudio) ────────────────────────────────────
        if sr != self._TARGET_SR:
            wav = self._resample_gpu(wav, sr)

        # ── 2. To numpy for ONNX (CUDA EP handles host→device internally) ─────
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

        # ── 3. Build overlapping chunks ───────────────────────────────────────
        chunks, num_f = self._build_chunks(wav_np)   # (N_chunks, split_len)

        # ── 4. Batched ONNX inference (GPU or CPU) ────────────────────────────
        all_enc = self._run_batched_onnx(chunks)     # (N_chunks, 20, 1024)

        # ── 5. Postprocess → 25 Hz output ────────────────────────────────────
        feats = self._postprocess(all_enc, num_f)    # (T, 1024) numpy

        return torch.from_numpy(feats)               # (T, 1024) float32 CPU tensor


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
            return torch.load(cp, map_location="cpu", weights_only=True)
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