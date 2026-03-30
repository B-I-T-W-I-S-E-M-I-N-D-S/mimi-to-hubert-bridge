"""
preprocess.py — Data Preparation Scripts
=========================================
Prepares paired (Mimi tokens, HuBERT features, prosody) from raw audio datasets.

Supports:
  - LibriSpeech  (train-clean-100, train-clean-360, train-other-500, test-clean)
  - VoxCeleb1/2  (flat directory structure)
  - Generic      (any directory of .wav / .flac files)

Outputs:
  - data/train.jsonl
  - data/val.jsonl
  - (optionally) pre-cached feature tensors in data/cache/

Usage:
  python preprocess.py \
      --dataset librispeech \
      --root /data/LibriSpeech/train-clean-100 \
      --out_dir data \
      --split train \
      --val_frac 0.01

  python preprocess.py \
      --dataset generic \
      --root /data/my_audio \
      --out_dir data
"""

import argparse
import json
import logging
import os
import random
import math
from pathlib import Path
from typing import List, Tuple

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Audio file discovery
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def discover_audio(root: str) -> List[Path]:
    """Recursively find all audio files under root."""
    found = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(p)
    found.sort()
    return found


def discover_librispeech(root: str) -> List[Tuple[Path, str]]:
    """
    Yields (audio_path, transcript) tuples.
    LibriSpeech structure: SPEAKER/CHAPTER/SPEAKER-CHAPTER-UUUU.flac + .trans.txt
    """
    pairs = []
    root_p = Path(root)
    for trans_file in root_p.rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                utt_id = parts[0]
                text   = parts[1] if len(parts) > 1 else ""
                audio  = chapter_dir / f"{utt_id}.flac"
                if audio.exists():
                    pairs.append((audio, text))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Feature pre-extraction (optional, for faster training I/O)
# ──────────────────────────────────────────────────────────────────────────────

def preextract_features(
    audio_paths: List[Path],
    cfg: dict,
    cache_dir: Path,
    device: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 4,
):
    """
    Pre-extract and cache Mimi tokens + HuBERT features for a list of files.

    GPU acceleration
    ────────────────
    - Mimi model and waveforms are moved to `device` (CUDA when available).
    - HuBERT ONNX uses CUDAExecutionProvider when device="cuda".
    - Audio loading + resampling runs in `num_workers` background threads so
      the GPU is kept busy while the next batch of files is read from disk.
    - torchaudio resampling is done on the GPU (inside MimiExtractor /
      HuBERTExtractor) instead of on CPU.

    Cache logic
    ───────────
    A file is considered fully cached only when BOTH the Mimi and HuBERT
    cache files exist and are non-empty.  A partial cache (one file missing)
    means the previous run crashed mid-way — both files are re-extracted to
    guarantee consistency.
    """
    import hashlib
    import queue
    import threading
    import torch
    import torchaudio
    from dataset import MimiExtractor, HuBERTExtractor
    import numpy as np

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        logger.info(f"[preextract] GPU mode — device={device}, io_workers={num_workers}")
    else:
        logger.info(f"[preextract] CPU mode — io_workers={num_workers}")

    mimi_ext   = MimiExtractor(cfg["paths"]["mimi_model"], device)
    # hubert_model must point to the ONNX file, e.g. ./model/hubert_streaming_fix_kv.onnx
    hubert_ext = HuBERTExtractor(cfg["paths"]["hubert_model"], device, chunk_batch=1)
    sr         = cfg["data"]["sample_rate"]

    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(p, suffix):
        h = hashlib.md5(str(p).encode()).hexdigest()
        return cache_dir / f"{h}_{suffix}.pt"

    # ── Filter out already-cached files up front ──────────────────────────────
    todo: List[Path] = []
    n_skipped = 0
    for audio_path in audio_paths:
        cp_mimi   = cache_path(audio_path, "mimi")
        cp_hubert = cache_path(audio_path, "hubert")
        if (cp_mimi.exists() and cp_hubert.exists()
                and cp_mimi.stat().st_size > 0
                and cp_hubert.stat().st_size > 0):
            n_skipped += 1
        else:
            # Remove any stale partial files so extraction is consistent
            if cp_mimi.exists() or cp_hubert.exists():
                logger.warning(
                    f"Partial cache detected for {audio_path.name}; re-extracting."
                )
            cp_mimi.unlink(missing_ok=True)
            cp_hubert.unlink(missing_ok=True)
            todo.append(audio_path)

    n_total   = len(audio_paths)
    n_todo    = len(todo)
    logger.info(
        f"[preextract] {n_skipped}/{n_total} already cached — "
        f"extracting {n_todo} files."
    )

    if not todo:
        logger.info("Pre-extraction complete: nothing to do.")
        return

    # ── Parallel I/O: load audio in background threads ────────────────────────
    # Queue size = 2 × num_workers keeps the GPU busy without excessive RAM use.
    _SENTINEL = None
    load_q: "queue.Queue[Tuple[Path, torch.Tensor] | None]" = queue.Queue(
        maxsize=num_workers * 2
    )

    def _load_worker(paths: List[Path]):
        """Background thread: load + resample audio, push to queue."""
        for ap in paths:
            try:
                wav, file_sr = torchaudio.load(str(ap))
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if file_sr != sr:
                    wav = torchaudio.functional.resample(wav, file_sr, sr)
                # Pin memory for faster CPU→GPU transfer
                if use_cuda:
                    wav = wav.pin_memory()
                load_q.put((ap, wav))
            except Exception as e:
                logger.warning(f"[preextract] Load failed {ap}: {e}")
                load_q.put((ap, None))   # signal failure for this file
        load_q.put(_SENTINEL)

    # Split file list across workers
    chunk_size  = max(1, math.ceil(n_todo / num_workers))
    worker_chunks = [todo[i : i + chunk_size] for i in range(0, n_todo, chunk_size)]
    completed_workers = [0]
    lock = threading.Lock()
    final_sentinel_count = [0]

    def _worker_wrapper(paths):
        _load_worker(paths)

    threads = []
    for chunk in worker_chunks:
        t = threading.Thread(target=_load_worker, args=(chunk,), daemon=True)
        t.start()
        threads.append(t)

    # ── Main loop: GPU extraction ─────────────────────────────────────────────
    n_done   = 0
    n_failed = 0
    sentinels_seen = 0
    n_workers = len(threads)

    # Use tqdm if available for a progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_todo, desc="Extracting", unit="file", dynamic_ncols=True)
    except ImportError:
        pbar = None

    while sentinels_seen < n_workers:
        item = load_q.get()

        if item is _SENTINEL:
            sentinels_seen += 1
            continue

        audio_path, wav = item

        if wav is None:
            n_failed += 1
            if pbar:
                pbar.update(1)
            continue

        cp_mimi   = cache_path(audio_path, "mimi")
        cp_hubert = cache_path(audio_path, "hubert")

        try:
            # Non-blocking GPU transfer when pinned memory is available
            if use_cuda:
                wav = wav.to(device, non_blocking=True)

            with torch.no_grad():
                tokens = mimi_ext.extract(wav, sr)    # (T_m, num_codebooks)
                hubert = hubert_ext.extract(wav, sr)  # (T_h, feat_dim)

            torch.save(tokens, cp_mimi)
            torch.save(hubert, cp_hubert)
            n_done += 1

        except Exception as e:
            logger.warning(f"[preextract] Extraction failed {audio_path}: {e}")
            n_failed += 1
            cp_mimi.unlink(missing_ok=True)
            cp_hubert.unlink(missing_ok=True)

        if pbar:
            pbar.update(1)
            pbar.set_postfix(done=n_done, failed=n_failed, skipped=n_skipped)
        elif (n_done + n_failed) % 100 == 0:
            logger.info(
                f"  {n_done+n_failed}/{n_todo} — "
                f"done={n_done} skipped={n_skipped} failed={n_failed}"
            )

    if pbar:
        pbar.close()

    for t in threads:
        t.join()

    logger.info(
        f"Pre-extraction complete: {n_done} new, {n_skipped} cached, "
        f"{n_failed} failed, {n_total} total"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Manifest writers
# ──────────────────────────────────────────────────────────────────────────────

def write_manifest(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records → {path}")


def build_manifests(
    audio_pairs: List[Tuple[Path, str]],
    out_dir: Path,
    val_frac: float = 0.01,
    seed: int = 42,
):
    """Split into train/val manifests."""
    rng = random.Random(seed)
    pairs = list(audio_pairs)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_frac))
    val   = pairs[:n_val]
    train = pairs[n_val:]

    def to_record(audio_path, text):
        return {"audio_path": str(audio_path), "text": text}

    write_manifest([to_record(p, t) for p, t in train], out_dir / "train.jsonl")
    write_manifest([to_record(p, t) for p, t in val],   out_dir / "val.jsonl")
    logger.info(f"Train: {len(train)} | Val: {len(val)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data for bridge training")
    parser.add_argument("--dataset",   choices=["librispeech", "voxceleb", "generic"],
                        default="generic")
    parser.add_argument("--root",      required=True, help="Root directory of audio")
    parser.add_argument("--out_dir",   default="data", help="Output directory for manifests")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--val_frac",  type=float, default=0.01)
    parser.add_argument("--preextract", action="store_true",
                        help="Pre-extract and cache Mimi + HuBERT features")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel I/O threads for audio loading (GPU mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)

    # ── Discover files ──────────────────────────────────────────────────────
    logger.info(f"Discovering {args.dataset} audio under {args.root}")

    if args.dataset == "librispeech":
        pairs = discover_librispeech(args.root)
        logger.info(f"Found {len(pairs)} LibriSpeech utterances")
    else:
        # generic / voxceleb: no transcripts available upfront
        audio_files = discover_audio(args.root)
        pairs = [(p, "") for p in audio_files]
        logger.info(f"Found {len(pairs)} audio files")

    if not pairs:
        logger.error("No audio files found. Check --root path.")
        return

    # ── Build manifests ─────────────────────────────────────────────────────
    build_manifests(pairs, out_dir, val_frac=args.val_frac, seed=args.seed)

    # ── Optional pre-extraction ─────────────────────────────────────────────
    if args.preextract:
        logger.info("Pre-extracting features...")
        cache_dir = Path(cfg["data"]["cache_dir"])
        audio_only = [p for p, _ in pairs]
        preextract_features(audio_only, cfg, cache_dir, device=args.device,
                            num_workers=args.num_workers)

    logger.info("Preprocessing done.")


if __name__ == "__main__":
    main()