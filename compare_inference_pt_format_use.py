"""
compare_inference.py — Side-by-side comparison of real HuBERT vs Bridge-model output
======================================================================================
Pipeline
--------
  WAV  ──┬──► HuBERTExtractor (ONNX)    → hubert_gt  (T_h, 1024)  ← ground-truth
         │
         └──► MimiExtractor             → mimi_tokens (T_m, 8)
               └──► BridgeInference     → bridge_pred (4*T_m, 1024) ← model prediction

Error Metrics (after aligning lengths)
---------------------------------------
  • MSE  (mean squared error)
  • MAE  (mean absolute error)
  • RMSE (root mean squared error)
  • cosine similarity  (mean over frames)
  • SNR  (signal-to-noise ratio, dB)
  • per-dimension RMSE  (top-5 worst dims printed)

Usage
-----
  python compare_inference.py \\
      --audio path/to/audio.wav \\
      --checkpoint checkpoints/best.pt \\
      --config config.yaml

Optional flags
--------------
  --hubert-model   path to .onnx (overrides config paths.hubert_model)
  --mimi-model     HF repo or local path for Mimi (overrides config paths.mimi_model)
  --device         cuda | cpu  (default: auto)
  --save-gt        path.pt     save ground-truth HuBERT features
  --save-pred      path.pt     save bridge prediction features
  --plot           show matplotlib comparison plots (requires matplotlib)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Error metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict:
    """
    Compute comparison metrics between two feature matrices.

    Parameters
    ----------
    gt   : (T, D) float32 — ground-truth HuBERT features
    pred : (T, D) float32 — bridge model prediction (already aligned to gt length)

    Returns
    -------
    dict with float values for each metric.
    """
    assert gt.shape == pred.shape, (
        f"Shape mismatch after alignment: gt={gt.shape}, pred={pred.shape}"
    )
    diff = gt - pred

    mse  = diff.pow(2).mean().item()
    mae  = diff.abs().mean().item()
    rmse = mse ** 0.5

    # Cosine similarity per frame → mean
    cos = torch.nn.functional.cosine_similarity(gt, pred, dim=-1)  # (T,)
    mean_cos = cos.mean().item()

    # SNR: 10*log10(signal_power / noise_power)
    signal_power = gt.pow(2).mean().item()
    noise_power  = diff.pow(2).mean().item()
    snr_db = 10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))

    # Per-dimension RMSE  → (D,)
    per_dim_rmse = diff.pow(2).mean(dim=0).sqrt()  # (D,)

    return {
        "mse":          mse,
        "mae":          mae,
        "rmse":         rmse,
        "mean_cosine":  mean_cos,
        "snr_db":       snr_db,
        "per_dim_rmse": per_dim_rmse,   # kept as tensor for downstream use
    }


def print_metrics(metrics: dict, gt_shape: tuple, pred_shape: tuple):
    """Pretty-print the comparison metrics."""
    bar = "═" * 62
    print(f"\n{bar}")
    print("  HuBERT Ground-Truth  vs  Bridge Model Prediction")
    print(bar)
    print(f"  Ground-truth shape  : {gt_shape}")
    print(f"  Prediction shape    : {pred_shape}")
    print()
    print(f"  MSE             : {metrics['mse']:.6f}")
    print(f"  MAE             : {metrics['mae']:.6f}")
    print(f"  RMSE            : {metrics['rmse']:.6f}")
    print(f"  Mean cos-sim    : {metrics['mean_cosine']:.6f}  (1.0 = perfect)")
    print(f"  SNR             : {metrics['snr_db']:.2f} dB   (higher = better)")

    # Top-5 worst dimensions
    per_dim = metrics["per_dim_rmse"]
    top5_vals, top5_idx = torch.topk(per_dim, min(5, len(per_dim)))
    print()
    print("  Top-5 worst dimensions (by per-dim RMSE):")
    for rank, (idx, val) in enumerate(zip(top5_idx.tolist(), top5_vals.tolist()), 1):
        print(f"    #{rank}  dim={idx:4d}   RMSE={val:.6f}")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# Feature alignment helper
# ──────────────────────────────────────────────────────────────────────────────

def align_frames(gt: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trim both tensors to their common minimum length along dimension 0.
    This handles minor off-by-one differences between the ONNX HuBERT output
    (25 Hz) and the bridge model output (50 Hz → resampled below if needed).

    NOTE: The ONNX HuBERT extractor produces  ~25 Hz features (after avg pooling).
          The bridge model produces               50 Hz features (4× Mimi = 50 Hz).
          We upsample gt to 50 Hz by simple nearest-neighbour repeat-interleave
          so both are compared at 50 Hz.  If you prefer 25 Hz comparison, pass
          --compare-at-25hz to downsample the prediction instead.
    """
    T = min(gt.shape[0], pred.shape[0])
    return gt[:T], pred[:T]


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison function
# ──────────────────────────────────────────────────────────────────────────────

def compare(
    audio_path:       str,
    checkpoint_path:  str,
    config_path:      str,
    device:           Optional[str] = None,
    hubert_model_override: Optional[str] = None,
    mimi_model_override:   Optional[str] = None,
    save_gt:          Optional[str] = None,
    save_pred:        Optional[str] = None,
    plot:             bool = False,
    compare_at_25hz:  bool = False,
):
    """
    Full pipeline: WAV → HuBERT_gt + Bridge_pred → error metrics.
    """
    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Resolve device ────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Using device: {dev}")

    # ── Override model paths from CLI if provided ─────────────────────────────
    hubert_model_path = hubert_model_override or cfg["paths"]["hubert_model"]
    mimi_model_name   = mimi_model_override   or cfg["paths"]["mimi_model"]

    # ── Load audio ────────────────────────────────────────────────────────────
    import torchaudio
    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    # Keep original SR — each extractor resamples internally

    duration_s = waveform.shape[-1] / sr
    logger.info(f"Audio: {duration_s:.2f}s @ {sr} Hz, shape={tuple(waveform.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 1: Ground-truth HuBERT via ONNX model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Extracting ground-truth HuBERT features (ONNX)…")
    from dataset import HuBERTExtractor
    hubert_extractor = HuBERTExtractor(
        model_name=hubert_model_path,
        device=device,
    )
    hubert_gt = hubert_extractor.extract(waveform, sr)   # (T_onnx, 1024)  25 Hz
    print(f"      HuBERT GT  shape : {tuple(hubert_gt.shape)}  @ ~25 Hz")

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 2: Mimi tokens → Bridge model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Extracting Mimi tokens + running Bridge model…")
    from dataset import MimiExtractor
    from inference import BridgeInference

    mimi_extractor = MimiExtractor(mimi_model_name, device=device)
    mimi_tokens = mimi_extractor.extract(waveform, sr)   # (T_m, num_codebooks)
    print(f"      Mimi tokens shape: {tuple(mimi_tokens.shape)}  @ 12.5 Hz")

    bridge = BridgeInference(checkpoint_path, config_path, device=device)
    bridge_pred = bridge(mimi_tokens)                    # (1, 4*T_m, output_dim) → cpu
    bridge_pred = bridge_pred.squeeze(0)                 # (4*T_m, output_dim)     50 Hz
    print(f"      Bridge pred shape: {tuple(bridge_pred.shape)}  @ 50 Hz")

    # ─────────────────────────────────────────────────────────────────────────
    # Align rates: both at 50 Hz (default) or both at 25 Hz
    # ─────────────────────────────────────────────────────────────────────────
    if compare_at_25hz:
        # Downsample bridge_pred from 50→25 Hz by averaging pairs
        T_b = bridge_pred.shape[0]
        # Make even
        T_b_even = (T_b // 2) * 2
        bridge_50 = bridge_pred[:T_b_even]
        bridge_25hz = bridge_50.view(-1, 2, bridge_pred.shape[-1]).mean(dim=1)
        gt_aligned, pred_aligned = align_frames(hubert_gt, bridge_25hz)
        rate_label = "25 Hz"
    else:
        # Upsample HuBERT GT from 25→50 Hz by repeating each frame twice
        hubert_gt_50hz = hubert_gt.repeat_interleave(2, dim=0)   # (2*T_onnx, 1024)
        gt_aligned, pred_aligned = align_frames(hubert_gt_50hz, bridge_pred)
        rate_label = "50 Hz"

    print(f"\n      Aligned at {rate_label}: gt={tuple(gt_aligned.shape)}, "
          f"pred={tuple(pred_aligned.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Compute & display metrics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Computing error metrics…")
    metrics = compute_metrics(
        gt_aligned.float(),
        pred_aligned.float(),
    )
    print_metrics(metrics, gt_shape=tuple(gt_aligned.shape),
                  pred_shape=tuple(pred_aligned.shape))

    # ─────────────────────────────────────────────────────────────────────────
    # Optional: save tensors
    # ─────────────────────────────────────────────────────────────────────────
    if save_gt:
        torch.save(gt_aligned, save_gt)
        print(f"\n  Saved GT features   → {save_gt}")
    if save_pred:
        torch.save(pred_aligned, save_pred)
        print(f"  Saved Pred features → {save_pred}")

    # ─────────────────────────────────────────────────────────────────────────
    # Optional: matplotlib visualisation
    # ─────────────────────────────────────────────────────────────────────────
    if plot:
        _plot_comparison(gt_aligned, pred_aligned, metrics, rate_label)

    return metrics, gt_aligned, pred_aligned


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_comparison(
    gt: torch.Tensor,
    pred: torch.Tensor,
    metrics: dict,
    rate_label: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    gt_np   = gt.numpy()
    pred_np = pred.numpy()
    diff_np = (gt - pred).abs().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(
        f"HuBERT GT vs Bridge Prediction @ {rate_label}\n"
        f"RMSE={metrics['rmse']:.4f}  COS={metrics['mean_cosine']:.4f}  "
        f"SNR={metrics['snr_db']:.1f} dB",
        fontsize=12, fontweight="bold",
    )

    # ── Heatmap: GT features ─────────────────────────────────────────────────
    ax = axes[0]
    im0 = ax.imshow(
        gt_np.T, aspect="auto", origin="lower",
        vmin=np.percentile(gt_np, 5), vmax=np.percentile(gt_np, 95),
        cmap="magma",
    )
    ax.set_title("HuBERT Ground-Truth (ONNX)")
    ax.set_ylabel("Dimension")
    plt.colorbar(im0, ax=ax, fraction=0.015, pad=0.01)

    # ── Heatmap: Bridge prediction ────────────────────────────────────────────
    ax = axes[1]
    im1 = ax.imshow(
        pred_np.T, aspect="auto", origin="lower",
        vmin=np.percentile(gt_np, 5), vmax=np.percentile(gt_np, 95),
        cmap="magma",
    )
    ax.set_title("Bridge Model Prediction")
    ax.set_ylabel("Dimension")
    plt.colorbar(im1, ax=ax, fraction=0.015, pad=0.01)

    # ── Heatmap: Absolute error ───────────────────────────────────────────────
    ax = axes[2]
    im2 = ax.imshow(
        diff_np.T, aspect="auto", origin="lower",
        cmap="hot",
    )
    ax.set_title("Absolute Error |GT - Pred|")
    ax.set_xlabel(f"Frame (@ {rate_label})")
    ax.set_ylabel("Dimension")
    plt.colorbar(im2, ax=ax, fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare real HuBERT features vs Bridge model output for a WAV file."
    )
    parser.add_argument("--audio",       required=True,
                        help="Input WAV/FLAC audio file")
    parser.add_argument("--checkpoint",  required=True,
                        help="Trained bridge model checkpoint (.pt)")
    parser.add_argument("--config",      required=True,
                        help="config.yaml path")
    parser.add_argument("--device",      default=None,
                        help="Force device: cuda | cpu (default: auto-detect)")
    parser.add_argument("--hubert-model", default=None,
                        help="Override path to hubert .onnx model "
                             "(default: paths.hubert_model from config.yaml)")
    parser.add_argument("--mimi-model",  default=None,
                        help="Override Mimi HF repo or local path "
                             "(default: paths.mimi_model from config.yaml)")
    parser.add_argument("--save-gt",     default=None,
                        help="Save ground-truth HuBERT features to this .pt file")
    parser.add_argument("--save-pred",   default=None,
                        help="Save bridge prediction features to this .pt file")
    parser.add_argument("--compare-at-25hz", action="store_true",
                        help="Compare at 25 Hz (downsample bridge pred) "
                             "instead of 50 Hz (upsample HuBERT GT)")
    parser.add_argument("--plot",        action="store_true",
                        help="Show matplotlib heatmap comparison (requires matplotlib)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    compare(
        audio_path       = args.audio,
        checkpoint_path  = args.checkpoint,
        config_path      = args.config,
        device           = args.device,
        hubert_model_override = args.hubert_model,
        mimi_model_override   = args.mimi_model,
        save_gt          = args.save_gt,
        save_pred        = args.save_pred,
        plot             = args.plot,
        compare_at_25hz  = args.compare_at_25hz,
    )


if __name__ == "__main__":
    main()
