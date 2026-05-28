"""
riverdecode.data
================
Dataset loading, date parsing, and random seed helpers.

Functions
---------
set_seeds(seed)
    Set random / numpy / torch seeds for reproducibility.

load_splits(dataset_dir)
    Read train.csv, valid.csv, test.csv and return (df_train, df_valid, df_test).

parse_dates(df, split_name)
    Parse acquisition dates from planetscope_id or filename, with fallback.

load_resolution_ceiling(json_path)
    Load the resolution loss summary JSON and return (ceiling_iou, ceiling_std,
    ceiling_ssim, ceiling_psnr).

print_date_stats(df_train, df_valid, df_test)
    Print date distribution across all splits.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Set random, numpy, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CSV splits
# ─────────────────────────────────────────────────────────────────────────────

def _load_split(dataset_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"{name}.csv — {len(df)} tiles")
    return df


def load_splits(
    dataset_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read train / valid / test CSV splits from *dataset_dir*.

    Returns
    -------
    (df_train, df_valid, df_test)
    """
    print("Loading CSV splits...")
    df_train = _load_split(dataset_dir, "train")
    df_valid = _load_split(dataset_dir, "valid")
    df_test  = _load_split(dataset_dir, "test")
    return df_train, df_valid, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Date parsing
# ─────────────────────────────────────────────────────────────────────────────

_DUMMY_DATE = (15, 6, 2023)  # (day, month, year)


def _parse_yyyymmdd(token: str) -> tuple[int, int, int] | None:
    try:
        if len(token) != 8 or not token.isdigit():
            return None
        year, month, day = int(token[:4]), int(token[4:6]), int(token[6:8])
        if not (2014 <= year <= 2026 and 1 <= month <= 12 and 1 <= day <= 31):
            return None
        return day, month, year
    except Exception:
        return None


def _parse_date_from_planetscope_id(ps_id) -> tuple[int, int, int] | None:
    try:
        return _parse_yyyymmdd(str(ps_id).split("_")[0])
    except Exception:
        return None


def _parse_date_from_path(path) -> tuple[int, int, int] | None:
    try:
        return _parse_yyyymmdd(os.path.basename(str(path)).split("_")[0])
    except Exception:
        return None


def parse_dates(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Parse acquisition dates from ``planetscope_id`` or path, adding
    ``_day``, ``_month``, ``_year`` columns.  Falls back to a dummy date
    (15 Jun 2023) when parsing fails.

    Parameters
    ----------
    df : pd.DataFrame
        Split dataframe (train / valid / test).
    split_name : str
        Label used in warning messages (e.g. "train").

    Returns
    -------
    pd.DataFrame with three new date columns.
    """
    df = df.copy()
    days, months, years, fallback_count = [], [], [], 0

    for _, row in df.iterrows():
        result = None
        if "planetscope_id" in df.columns:
            result = _parse_date_from_planetscope_id(row["planetscope_id"])
        if result is None and "normalized_planetscope_path" in df.columns:
            result = _parse_date_from_path(row["normalized_planetscope_path"])
        if result is None:
            result = _DUMMY_DATE
            fallback_count += 1
        day, month, year = result
        days.append(day)
        months.append(month)
        years.append(year)

    df["_day"], df["_month"], df["_year"] = days, months, years

    if fallback_count > 0:
        print(
            f"  WARNING [{split_name}]: {fallback_count}/{len(df)} tiles "
            f"used dummy date."
        )
    else:
        print(f"  [{split_name}]: All {len(df)} dates parsed successfully.")
    return df


def parse_all_splits(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: parse dates for all three splits."""
    print("\nParsing dates...")
    df_train = parse_dates(df_train, "train")
    df_valid = parse_dates(df_valid, "valid")
    df_test  = parse_dates(df_test,  "test")
    return df_train, df_valid, df_test


def print_date_stats(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Print year / month distribution across all splits."""
    all_years  = pd.concat([df_train["_year"],  df_valid["_year"],  df_test["_year"]])
    all_months = pd.concat([df_train["_month"], df_valid["_month"], df_test["_month"]])
    print(f"\nDate distribution across {len(all_years)} tiles:")
    print(
        f"  Year  — min: {int(all_years.min())}, "
        f"max: {int(all_years.max())}, "
        f"median: {all_years.median():.0f}"
    )
    print(f"  Month — {dict(all_months.value_counts().sort_index())}")


# ─────────────────────────────────────────────────────────────────────────────
# Resolution ceiling
# ─────────────────────────────────────────────────────────────────────────────

def load_resolution_ceiling(
    json_path: str,
) -> tuple[float, float, float, float]:
    """
    Load resolution loss summary JSON.

    Returns
    -------
    (ceiling_iou, ceiling_std, ceiling_ssim, ceiling_psnr)
    """
    with open(json_path) as fh:
        summary = json.load(fh)
    ceiling_iou  = summary["channel1_mask_iou_mean"]
    ceiling_std  = summary["channel1_mask_iou_std"]
    ceiling_ssim = summary["channel2_ssim_mean"]
    ceiling_psnr = summary["channel2_psnr_mean"]

    print("═" * 55)
    print("RESOLUTION CEILING (from resolution_loss_analysis)")
    print("═" * 55)
    print(f"  Channel 1 — Max achievable IoU : {ceiling_iou:.4f} ± {ceiling_std:.4f}")
    print(f"  Channel 2 — Input SSIM         : {ceiling_ssim:.4f}")
    print(f"  Channel 2 — Input PSNR         : {ceiling_psnr:.2f} dB")
    return ceiling_iou, ceiling_std, ceiling_ssim, ceiling_psnr
