"""
riverdecode.io
==============
GeoTIFF reading helpers for PlanetScope imagery and segmentation masks.

All functions mirror the behaviour of ``dataset_planet.py`` exactly so that
training-time and evaluation-time preprocessing are identical.

Functions
---------
read_image_for_model(path)
    Global min-max normalisation.  Used for model input — matches
    ``dataset_planet.py`` exactly.

read_image_for_display(path)
    Per-band 2nd/98th-percentile stretch.  For visualisation only —
    NOT used as model input.

read_mask_raw(path)
    Binary water mask: label != 0 → 1 (treats both river and other water
    as positive).  Matches ``dataset_planet.py``.

verify_sample_tile(dataset_dir, df_test)
    Assert that a sample image + label pair resolves and print their shape.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio


# ─────────────────────────────────────────────────────────────────────────────
# Image readers
# ─────────────────────────────────────────────────────────────────────────────

def read_image_for_model(path: str) -> np.ndarray:
    """
    Read a PlanetScope GeoTIFF and apply global min-max normalisation.

    Matches ``dataset_planet.py`` exactly — do NOT change this normalisation
    or the frozen embeddings will be inconsistent.

    Returns
    -------
    np.ndarray of shape [C, H, W] float32 in [0, 1].
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    gmin, gmax = float(img.min()), float(img.max())
    if gmax - gmin > 0:
        img = (img - gmin) / max(gmax, 1.0)
    else:
        img = np.zeros_like(img)
    return img


def read_image_for_display(path: str) -> np.ndarray:
    """
    Read a PlanetScope GeoTIFF with per-band 2nd/98th percentile stretch.

    **VISUALISATION ONLY** — never use as model input.

    Returns
    -------
    np.ndarray of shape [C, H, W] float32 in [0, 1].
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    for i in range(img.shape[0]):
        valid = img[i][img[i] > 0]
        if len(valid) == 0:
            continue
        p2, p98 = np.percentile(valid, (2, 98))
        img[i] = np.clip((img[i] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Mask reader
# ─────────────────────────────────────────────────────────────────────────────

def read_mask_raw(path: str) -> np.ndarray:
    """
    Read a single-band label GeoTIFF and binarise to water / land.

    label != 0 → 1.0 (water).  Treats both label 1 (river) and label 2
    (other water) as positive.  Matches ``dataset_planet.py``.

    Returns
    -------
    np.ndarray of shape [H, W] float32 in {0.0, 1.0}.
    """
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)
    return np.where(mask != 0, 1.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_sample_tile(dataset_dir: str, df_test) -> None:
    """
    Assert that the first test tile and its label both resolve on disk,
    then print the band count and spatial size.
    """
    import pandas as pd  # local import to keep module lightweight

    sample_img   = os.path.join(dataset_dir, df_test["normalized_planetscope_path"].iloc[0])
    sample_label = os.path.join(dataset_dir, df_test["label_path"].iloc[0])

    assert os.path.exists(sample_img),   f"Image not found: {sample_img}"
    assert os.path.exists(sample_label), f"Label not found: {sample_label}"

    with rasterio.open(sample_img) as src:
        print(f"\nSample tile — bands: {src.count}, shape: {src.height}×{src.width}")
