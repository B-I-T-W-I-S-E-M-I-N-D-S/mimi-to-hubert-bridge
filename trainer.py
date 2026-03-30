"""
trainer.py — Training Loop
===========================
Handles:
  - Multi-loss optimisation (bridge + discriminator)
  - LR scheduling (cosine with warmup)
  - Mixed precision (torch.amp — replaces deprecated torch.cuda.amp)
  - Multi-GPU via DistributedDataParallel (DDP) or DataParallel fallback
  - TensorBoard logging
  - Checkpoint save / restore
  - Validation loop with metrics
  - Early stopping
"""

import os
import json
import time
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── AMP: use the non-deprecated torch.amp namespace ───────────────────────────
from torch.amp import GradScaler, autocast   # replaces torch.cuda.amp.*

from model import MimiHuBERTBridge, FeatureDiscriminator
from losses import BridgeLoss
from dataset import build_dataloaders

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_multi_gpu(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Wrap model for multi-GPU training.

    Strategy (in priority order):
      1. CUDA_VISIBLE_DEVICES / torchrun sets up multiple GPUs → use DDP if
         LOCAL_RANK env var is set (launched via torchrun / torch.distributed.launch).
      2. Multiple GPUs visible but no DDP env → fall back to DataParallel.
      3. Single GPU or CPU → return model as-is.

    Returns the (possibly wrapped) model. Unwrap with `model.module` when
    accessing parameters directly (e.g. state_dict saving).
    """
    if device.type != "cuda":
        return model

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank >= 0:
        # ── DDP path (launched with torchrun) ─────────────────────────────────
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        model = model.to(torch.device(f"cuda:{local_rank}"))
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        logger.info(f"DDP: using GPU {local_rank} / {num_gpus}")
    elif num_gpus > 1:
        # ── DataParallel path (simple multi-GPU, no launcher needed) ──────────
        model = nn.DataParallel(model)
        logger.info(f"DataParallel: using {num_gpus} GPUs")
    else:
        logger.info(f"Single GPU: {torch.cuda.get_device_name(0)}")

    return model


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the raw module from a DataParallel / DDP wrapper, if any."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_pitch_correlation(
    pred_f0: torch.Tensor, gt_f0: torch.Tensor, voiced: torch.Tensor
) -> float:
    """Pearson correlation on voiced frames."""
    if voiced.sum() < 2:
        return 0.0
    p = pred_f0[voiced].float()
    g = gt_f0[voiced].float()
    if p.std() < 1e-8 or g.std() < 1e-8:
        return 0.0
    corr = torch.corrcoef(torch.stack([p, g]))[0, 1]
    return corr.item()


def compute_mse_db(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    if mask is not None:
        pred, target = pred[mask], target[mask]
    return torch.nn.functional.mse_loss(pred.float(), target.float()).item()


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler with Linear Warmup
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t = cfg["training"]
    warmup = t["warmup_steps"]
    total  = t["num_epochs"] * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        t_cfg = cfg["training"]

        # ── Device / AMP ──────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = t_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        # amp_device_type is needed by the new torch.amp.autocast(device_type=...) API
        self.amp_device = self.device.type   # "cuda" or "cpu"
        logger.info(f"Training on {self.device} | AMP={self.mixed_precision}")

        torch.manual_seed(t_cfg.get("seed", 42))

        # ── Build models ──────────────────────────────────────────────────────
        bridge_raw = MimiHuBERTBridge(cfg).to(self.device)
        disc_raw   = FeatureDiscriminator(
            input_dim=cfg["model"]["output_dim"],
            hidden=t_cfg["disc_hidden"],
            num_layers=t_cfg["disc_layers"],
        ).to(self.device)

        # ── Multi-GPU wrapping ────────────────────────────────────────────────
        self.bridge = _setup_multi_gpu(bridge_raw, self.device)
        self.disc   = _setup_multi_gpu(disc_raw,   self.device)

        # ── Build loss (always on primary device, never wrapped) ──────────────
        self.criterion = BridgeLoss(cfg).to(self.device)

        # ── Optimisers — always optimise raw (unwrapped) params ───────────────
        self.opt_g = AdamW(
            list(_unwrap(self.bridge).parameters()) + list(self.criterion.parameters()),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )
        self.opt_d = AdamW(
            _unwrap(self.disc).parameters(),
            lr=t_cfg["disc_lr"],
            weight_decay=t_cfg["weight_decay"],
        )

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = build_dataloaders(cfg, device="cpu")
        steps_per_epoch = len(self.train_loader)

        # ── Schedulers ────────────────────────────────────────────────────────
        self.sched_g = build_scheduler(self.opt_g, cfg, steps_per_epoch)
        self.sched_d = CosineAnnealingLR(
            self.opt_d, T_max=t_cfg["num_epochs"] * steps_per_epoch, eta_min=1e-7
        )

        # ── AMP scalers (new torch.amp.GradScaler API) ────────────────────────
        # device_type arg avoids the deprecation warning from torch.cuda.amp.GradScaler
        self.scaler_g = GradScaler(device=self.amp_device, enabled=self.mixed_precision)
        self.scaler_d = GradScaler(device=self.amp_device, enabled=self.mixed_precision)

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step  = 0
        self.epoch        = 0
        self.best_val_mse = math.inf

        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(cfg["paths"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer = None
        if cfg["paths"].get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging → {self.log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed; skipping.")

        self.disc_start_step = t_cfg.get("disc_start_step", 5000)

        p = _unwrap(self.bridge).get_param_count()
        logger.info(f"Bridge parameters: {p['trainable']:,} trainable / {p['total']:,} total")
        logger.info(f"Visible GPUs: {torch.cuda.device_count()}")

    # ──────────────────────────────────────────────────────────────────────────
    def _to_device(self, batch: dict) -> dict:
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _train_step(self, batch: dict) -> dict:
        tokens = batch["tokens"]        # (B, T_m, 8)
        target = batch["hubert"]        # (B, T_h, output_dim)

        use_adv = self.global_step >= self.disc_start_step

        # ── Discriminator step ────────────────────────────────────────────────
        d_logs = {}
        if use_adv:
            self.opt_d.zero_grad()
            # torch.amp.autocast(device_type=...) — non-deprecated API
            with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
                pred_det, _ = self.bridge(tokens)
                real_logits = self.disc(target)
                fake_logits = self.disc(pred_det.detach())
                d_loss, d_logs = self.criterion.adv.discriminator_loss(real_logits, fake_logits)

            self.scaler_d.scale(d_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                _unwrap(self.disc).parameters(), self.cfg["training"]["grad_clip"]
            )
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
            self.sched_d.step()

        # ── Generator (bridge) step ───────────────────────────────────────────
        self.opt_g.zero_grad()
        with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
            pred, _ = self.bridge(tokens)           # (B, T_h, output_dim)

            fake_disc_logits = None
            if use_adv:
                fake_disc_logits = self.disc(pred)

            g_loss, g_logs = self.criterion(pred, target, batch, fake_disc_logits)

        self.scaler_g.scale(g_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(_unwrap(self.bridge).parameters()) + list(self.criterion.parameters()),
            self.cfg["training"]["grad_clip"],
        )
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        self.sched_g.step()

        return {**g_logs, **d_logs}

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.bridge.eval()
        agg = {}
        n = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            tokens = batch["tokens"]
            target = batch["hubert"]

            with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
                pred, _ = self.bridge(tokens)

            # ── Loss (cast pred to float32 before scalar losses) ──────────────
            # StatisticsLoss and others use .mean()/.std() which are more
            # numerically stable in fp32; this is a no-op when AMP is disabled.
            pred_fp32 = pred.float()
            _, logs = self.criterion(pred_fp32, target.float(), batch)

            # ── Pitch correlation metric ──────────────────────────────────────
            if batch.get("f0") is not None:
                # Run prosody head in fp32 for stability
                f0_pred = self.criterion.prosody.f0_head(pred_fp32).squeeze(-1)
                pc = compute_pitch_correlation(
                    f0_pred.cpu().flatten(),
                    batch["f0"].cpu().flatten(),
                    batch["voiced_mask"].cpu().flatten(),
                )
                logs["pitch_corr"] = pc

            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + (v if isinstance(v, float) else float(v))
            n += 1

        self.bridge.train()
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ──────────────────────────────────────────────────────────────────────────
    def _log(self, logs: dict, prefix: str = "train"):
        if self.writer:
            for k, v in logs.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

    # ──────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, tag: str, val_logs: Optional[dict] = None):
        ckpt = {
            "step":     self.global_step,
            "epoch":    self.epoch,
            # Always save the raw (unwrapped) state_dict
            "bridge":   _unwrap(self.bridge).state_dict(),
            "disc":     _unwrap(self.disc).state_dict(),
            "opt_g":    self.opt_g.state_dict(),
            "opt_d":    self.opt_d.state_dict(),
            "sched_g":  self.sched_g.state_dict(),
            "sched_d":  self.sched_d.state_dict(),
            "best_val": self.best_val_mse,
            "val_logs": val_logs or {},
        }
        path = self.ckpt_dir / f"bridge_{tag}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        _unwrap(self.bridge).load_state_dict(ckpt["bridge"])
        _unwrap(self.disc).load_state_dict(ckpt["disc"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        self.global_step  = ckpt["step"]
        self.epoch        = ckpt["epoch"]
        self.best_val_mse = ckpt.get("best_val", math.inf)
        logger.info(f"Resumed from step {self.global_step} (epoch {self.epoch})")

    # ──────────────────────────────────────────────────────────────────────────
    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        t_cfg = self.cfg["training"]
        num_epochs = t_cfg["num_epochs"]
        logger.info(f"Starting training for {num_epochs} epochs")
        self.bridge.train()

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            t0 = time.time()
            epoch_logs: dict = {}
            n_steps = 0

            for batch in self.train_loader:
                batch = self._to_device(batch)
                step_logs = self._train_step(batch)
                self._log(step_logs, "train")

                for k, v in step_logs.items():
                    epoch_logs[k] = epoch_logs.get(k, 0.0) + v
                n_steps += 1
                self.global_step += 1

                if self.global_step % 100 == 0:
                    avg = {k: v / n_steps for k, v in epoch_logs.items()}
                    lr  = self.opt_g.param_groups[0]["lr"]
                    logger.info(
                        f"Step {self.global_step} | epoch {epoch+1}/{num_epochs} | "
                        f"loss={avg.get('total', 0):.4f} | lr={lr:.2e} | "
                        f"t={time.time()-t0:.1f}s"
                    )

            # ── Validation ────────────────────────────────────────────────────
            val_logs = self._val_epoch()
            self._log(val_logs, "val")

            val_mse = val_logs.get("recon_mse", math.inf)
            logger.info(
                f"[Epoch {epoch+1}] val_mse={val_mse:.5f} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in val_logs.items() if k != "recon_mse")
            )

            # ── Checkpointing ─────────────────────────────────────────────────
            self.save_checkpoint(f"epoch{epoch+1:03d}", val_logs)
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.save_checkpoint("best", val_logs)
                logger.info(f"  ↳ New best val MSE: {val_mse:.5f}")

        if self.writer:
            self.writer.close()
        logger.info("Training complete.")

