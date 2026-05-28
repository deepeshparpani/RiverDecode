"""
riverdecode.training
====================
Training loop, LR scheduler, checkpoint management, and validation
visualisation — shared by all OlmoEarth decoder experiments.

Functions
---------
adjust_learning_rate(optimizer, epoch, step, n_batches,
                     total_epochs, max_lr, min_lr)
    Cosine annealing with linear warm-up (per-step granularity).

train_one_lr(lr, model_cls, model_kwargs, train_loader, valid_loader,
             output_dir, pos_weight_tensor, loss_fn, metric_fn, ...)
    Single LR grid-search run with three-level checkpoint strategy.

val_sanity_viz(probe, valid_emb, valid_masks, valid_data,
               output_dir, lr, head_type, input_size, mask_key, n_show)
    Save a side-by-side qualitative validation figure.

SweepState
    Simple dataclass to track the best LR across the sweep.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    n_batches: int,
    total_epochs: int,
    max_lr: float,
    min_lr: float = 1e-5,
    warmup_frac: float = 0.10,
) -> float:
    """
    Cosine annealing with linear warm-up.

    Per-step updates give smooth LR curves at each batch rather than
    once per epoch.

    Parameters
    ----------
    epoch, step : int
        Current epoch (0-indexed) and batch index within the epoch.
    n_batches   : int
        Total number of batches per epoch.
    warmup_frac : float
        Fraction of total steps used for linear warm-up (default 10 %).

    Returns
    -------
    Current learning rate (float).
    """
    total_steps  = total_epochs * n_batches
    warmup_steps = int(warmup_frac * total_steps)
    cur_step     = epoch * n_batches + step

    if cur_step < warmup_steps:
        lr = min_lr + (max_lr - min_lr) * cur_step / max(warmup_steps, 1)
    else:
        progress = (cur_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_lr(
    lr: float,
    model_cls: type,
    model_kwargs: dict,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    output_dir: str,
    loss_fn: Callable,
    metric_fn: Callable,
    *,
    total_epochs: int = 50,
    min_lr: float = 1e-5,
    device: str | torch.device = "cuda",
    # Logging
    log_interval: int = 50,
    val_interval: int | None = None,
    # Checkpoint strategy: keep top-3
    top_k: int = 3,
    # Optional pre-trained weight init
    init_checkpoint: str | None = None,
    freeze_decoder: bool = False,
) -> tuple[nn.Module, float]:
    """
    Train one model instance with the given LR.

    Checkpoint strategy
    -------------------
    *   ``best_<lr>.pt``   — best validation metric ever seen.
    *   ``last_<lr>.pt``   — latest epoch (for continue / eval).
    *   Up to ``top_k`` epoch checkpoints kept in ``checkpoints/<lr>/``.

    Parameters
    ----------
    lr           : float  — learning rate for this run.
    model_cls    : class  — decoder class (e.g. OlmoEarthUNetDecoder448).
    model_kwargs : dict   — kwargs forwarded to model_cls(**model_kwargs).
    train_loader, valid_loader : DataLoader
        Expected to yield (embeddings, masks) tuples.
    loss_fn  : Callable(logits, masks) → scalar tensor.
    metric_fn: Callable(logits, masks) → float (higher is better, e.g. F1).

    Returns
    -------
    (best_model, best_val_metric)
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **kwargs): return it  # type: ignore

    val_interval = val_interval or max(1, total_epochs // 10)
    lr_str       = f"{lr:.0e}"
    ckpt_dir     = os.path.join(output_dir, "checkpoints", lr_str)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Build model ──────────────────────────────────────────────────────────
    model = model_cls(**model_kwargs).to(device)

    if init_checkpoint and os.path.exists(init_checkpoint):
        ckpt = torch.load(init_checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded init checkpoint — missing: {len(missing)}, "
              f"unexpected: {len(unexpected)}")

    if freeze_decoder:
        for name, p in model.named_parameters():
            if "lora" not in name.lower():
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Decoder frozen — {trainable} trainable params (LoRA only).")

    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    n_batches    = len(train_loader)
    best_metric  = -float("inf")
    best_model   = None
    epoch_scores: list[tuple[float, str]] = []  # (metric, ckpt_path)

    # ── Training loop ────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(total_epochs), desc=f"LR={lr_str}", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss, train_metric = 0.0, 0.0

        for step, (emb, masks) in enumerate(train_loader):
            emb   = emb.to(device)
            masks = masks.to(device)

            cur_lr = adjust_learning_rate(
                optimizer, epoch, step, n_batches, total_epochs, lr, min_lr
            )
            optimizer.zero_grad()
            logits = model(emb)
            loss   = loss_fn(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss   += loss.item()
            train_metric += metric_fn(logits.detach(), masks)

            if (step + 1) % log_interval == 0:
                avg_l = train_loss / (step + 1)
                avg_m = train_metric / (step + 1)
                epoch_bar.set_postfix(
                    loss=f"{avg_l:.4f}", f1=f"{avg_m:.4f}", lr=f"{cur_lr:.2e}"
                )

        # ── Validation ───────────────────────────────────────────────────────
        if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
            model.eval()
            val_loss, val_metric, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for emb_v, masks_v in valid_loader:
                    emb_v   = emb_v.to(device)
                    masks_v = masks_v.to(device)
                    logits_v = model(emb_v)
                    val_loss   += loss_fn(logits_v, masks_v).item()
                    val_metric += metric_fn(logits_v, masks_v)
                    n_val += 1
            val_metric /= max(n_val, 1)
            val_loss   /= max(n_val, 1)

            ep_ckpt = os.path.join(ckpt_dir, f"ep{epoch + 1:04d}.pt")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "val_f1": val_metric},
                ep_ckpt,
            )
            epoch_scores.append((val_metric, ep_ckpt))

            # Keep only top-k epoch checkpoints
            if len(epoch_scores) > top_k:
                epoch_scores.sort(reverse=True)
                old_path = epoch_scores.pop()[1]
                if os.path.exists(old_path):
                    os.remove(old_path)

            # Update best
            if val_metric > best_metric:
                best_metric = val_metric
                best_model  = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
                torch.save(
                    {"epoch": epoch + 1, "model_state_dict": best_model,
                     "val_f1": best_metric},
                    os.path.join(output_dir, "checkpoints", f"best_{lr_str}.pt"),
                )

        # ── Last epoch checkpoint ────────────────────────────────────────────
        torch.save(
            {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
            os.path.join(output_dir, "checkpoints", f"last_{lr_str}.pt"),
        )

    # Load best weights back into model before returning
    if best_model is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model.items()})

    print(
        f"  LR={lr_str} done — best val F1 = {best_metric:.4f}"
    )
    return model, best_metric


# ─────────────────────────────────────────────────────────────────────────────
# Sweep state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepState:
    """Track the best result across an LR sweep."""
    best_lr:     float = 0.0
    best_metric: float = -float("inf")
    best_model:  nn.Module | None = None
    history:     list[dict] = field(default_factory=list)

    def update(
        self,
        lr: float,
        metric: float,
        model: nn.Module,
    ) -> None:
        self.history.append({"lr": lr, "metric": metric})
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_lr     = lr
            self.best_model  = model
            print(f"  ★ New best: LR={lr:.1e}, F1={metric:.4f}")
        else:
            print(f"    LR={lr:.1e}: F1={metric:.4f}  (best so far: {self.best_metric:.4f})")

    def print_summary(self) -> None:
        print("\n" + "═" * 55)
        print("LR SWEEP SUMMARY")
        print("═" * 55)
        for h in sorted(self.history, key=lambda x: -x["metric"]):
            marker = " ★" if h["lr"] == self.best_lr else ""
            print(f"  LR={h['lr']:.1e}  F1={h['metric']:.4f}{marker}")
        print(f"\n  Best: LR={self.best_lr:.1e}, F1={self.best_metric:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation visualisation
# ─────────────────────────────────────────────────────────────────────────────

def val_sanity_viz(
    model: nn.Module,
    valid_emb: dict,
    output_dir: str,
    *,
    head_type: str = "sigmoid",   # "sigmoid" or "argmax"
    mask_key: str = "masks_448",
    n_show: int = 4,
    lr: float | None = None,
    threshold: float = 0.5,
    device: str | torch.device = "cuda",
) -> None:
    """
    Save a side-by-side qualitative validation figure showing predicted
    probability maps alongside ground-truth masks.

    Parameters
    ----------
    model      : decoder in eval mode (moved to *device*).
    valid_emb  : dict from ``torch.load('valid_embeddings.pt', ...)``.
    output_dir : experiment root (saved to ``visualizations/``).
    head_type  : "sigmoid" (BCE head) or "argmax" (CE head).
    mask_key   : key for the mask tensor in valid_emb.
    n_show     : number of samples to display.
    lr         : label suffix for the saved figure filename.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping visualisation.")
        return

    model.eval()
    embs  = valid_emb["embeddings"][:n_show].to(device)
    masks = valid_emb[mask_key][:n_show].to(device)

    with torch.no_grad():
        logits = model(embs)

    if head_type == "sigmoid":
        preds = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
    else:
        preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    masks_np = masks.cpu().numpy()

    fig, axes = plt.subplots(n_show, 2, figsize=(8, 3 * n_show))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i][0].imshow(preds[i],    cmap="Blues",  vmin=0, vmax=1)
        axes[i][0].set_title("Predicted probability")
        axes[i][0].axis("off")
        axes[i][1].imshow(masks_np[i], cmap="Greys_r", vmin=0, vmax=1)
        axes[i][1].set_title("Ground truth mask")
        axes[i][1].axis("off")

    plt.tight_layout()
    lr_suffix = f"_lr{lr:.0e}" if lr is not None else ""
    save_path = os.path.join(output_dir, "visualizations",
                             f"val_sanity{lr_suffix}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved validation figure → {save_path}")
