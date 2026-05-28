"""
riverdecode.losses
==================
Loss functions and evaluation metrics used across all experiments.

Loss heads
----------
Two decoder head variants are used across the notebooks:

* **BCE head** (binary, ``[B,1,H,W]`` logit + sigmoid):
  Used by ``olmoearth_zero_shot_unet_448``.
  → ``bce_loss``, ``dice_loss_binary``, ``combined_loss_bce``

* **CE head** (2-class, ``[B,2,H,W]`` logit + argmax):
  Used by ``olmoearth_zero_shot_linearprobe_224/448``, ``unet_496``, etc.
  → ``ce_criterion``, ``dice_loss_ce``, ``combined_loss_ce``

Training-time metrics
---------------------
``compute_f1_sigmoid`` — for BCE head (threshold=0.5)
``compute_f1_argmax``  — for CE head

Evaluation metrics (numpy, tile-level)
---------------------------------------
``compute_iou_np``
``compute_f1_np``

Utilities
---------
``compute_pos_weight(embeddings_path, mask_key, clamp_max)``
    Derive BCE pos_weight from training masks.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BCE head (binary, [B,1,H,W] logit)
# ─────────────────────────────────────────────────────────────────────────────

def bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy with pos_weight for class imbalance.

    Parameters
    ----------
    logits  : [B, 1, H, W] float32
    targets : [B, H, W]    int64 {0, 1}
    pos_weight_tensor : scalar tensor on same device as logits
    """
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(1).float(),
        targets.float(),
        pos_weight=pos_weight_tensor,
    )


def dice_loss_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss on sigmoid probability — directly optimises recall.

    logits  : [B, 1, H, W] float32
    targets : [B, H, W]    int64 {0, 1}
    """
    prob  = torch.sigmoid(logits.squeeze(1).float())
    gt    = targets.float()
    inter = (prob * gt).sum()
    denom = prob.sum() + gt.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def combined_loss_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight_tensor: torch.Tensor,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """BCE + dice_weight×Dice.  No boundary loss."""
    return (
        bce_loss(logits, targets, pos_weight_tensor)
        + dice_weight * dice_loss_binary(logits, targets)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CE head (2-class, [B,2,H,W] logit)
# ─────────────────────────────────────────────────────────────────────────────

def build_ce_criterion(
    pos_weight: float,
    device: str | torch.device,
) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss with class weights [1.0, pos_weight]."""
    return nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device)
    )


def dice_loss_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice on the water class (class 1) from 2-class logits.

    logits  : [B, 2, H, W] float32
    targets : [B, H, W]    int64 {0, 1}
    """
    prob  = torch.softmax(logits.float(), dim=1)[:, 1]
    gt    = (targets == 1).float()
    inter = (prob * gt).sum()
    denom = prob.sum() + gt.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def combined_loss_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """CrossEntropy + dice_weight×Dice.  No boundary loss."""
    return criterion(logits, targets) + dice_weight * dice_loss_ce(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# pos_weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(
    embeddings_path: str,
    mask_key: str = "masks_448",
    clamp_min: float = 1.0,
    clamp_max: float = 20.0,
) -> float:
    """
    Compute BCE / CE pos_weight from training mask water fraction.

    pos_weight = (1 - water_frac) / water_frac, clamped to [clamp_min, clamp_max].

    Parameters
    ----------
    embeddings_path : str
        Path to ``train_embeddings.pt``.
    mask_key : str
        Key for the mask tensor (e.g. "masks_448" or "masks_224").
    """
    print("Computing pos_weight from training masks...")
    data    = torch.load(embeddings_path, map_location="cpu")
    masks   = data[mask_key].float()
    frac    = masks.mean().item()
    raw_pw  = (1.0 - frac) / max(frac, 1e-6)
    pw      = float(np.clip(raw_pw, clamp_min, clamp_max))
    del data, masks
    print(f"  Mean water fraction : {frac:.4f}  ({frac * 100:.1f}% water)")
    print(f"  Raw pos_weight      : {raw_pw:.2f}  →  clamped: {pw:.2f}")
    return pw


# ─────────────────────────────────────────────────────────────────────────────
# Training-time F1 metrics (torch)
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1_sigmoid(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    F1 via sigmoid + threshold for training-time validation (BCE head).

    logits : [B, 1, H, W]   masks : [B, H, W] int64
    """
    preds = (torch.sigmoid(logits.squeeze(1).float()) >= threshold).long()
    gt    = masks.long()
    TP = ((preds == 1) & (gt == 1)).sum().float()
    FP = ((preds == 1) & (gt == 0)).sum().float()
    FN = ((preds == 0) & (gt == 1)).sum().float()
    prec = TP / (TP + FP + 1e-7)
    rec  = TP / (TP + FN + 1e-7)
    return (2 * prec * rec / (prec + rec + 1e-7)).item()


def compute_f1_argmax(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    """
    F1 via argmax — no threshold needed (CE head).

    logits : [B, 2, H, W]   masks : [B, H, W] int64
    """
    preds = torch.argmax(logits, dim=1).float()
    gt    = masks.float()
    TP = ((preds == 1) & (gt == 1)).sum().float()
    FP = ((preds == 1) & (gt == 0)).sum().float()
    FN = ((preds == 0) & (gt == 1)).sum().float()
    prec = TP / (TP + FP + 1e-7)
    rec  = TP / (TP + FN + 1e-7)
    return (2 * prec * rec / (prec + rec + 1e-7)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (numpy, tile-level)
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_np(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """IoU between binary *gt* and thresholded *pred* (numpy arrays)."""
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin   = (gt   >= threshold).astype(np.uint8)
    TP = np.sum((pred_bin == 1) & (gt_bin == 1))
    FP = np.sum((pred_bin == 1) & (gt_bin == 0))
    FN = np.sum((pred_bin == 0) & (gt_bin == 1))
    return float(TP / (TP + FP + FN + 1e-7))


def compute_f1_np(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """F1 between binary *gt* and thresholded *pred* (numpy arrays)."""
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin   = (gt   >= threshold).astype(np.uint8)
    TP = np.sum((pred_bin == 1) & (gt_bin == 1))
    FP = np.sum((pred_bin == 1) & (gt_bin == 0))
    FN = np.sum((pred_bin == 0) & (gt_bin == 1))
    prec = TP / (TP + FP + 1e-7)
    rec  = TP / (TP + FN + 1e-7)
    return float(2 * prec * rec / (prec + rec + 1e-7))
