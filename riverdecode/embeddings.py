"""
riverdecode.embeddings
======================
Three-tier embedding cache used by all training notebooks.

Tier 1 — Already computed for this experiment (skip).
Tier 2 — Copy from a previously computed experiment (validated).
Tier 3 — Compute fresh from scratch using the frozen backbone.

Functions
---------
load_or_compute_embeddings(df, split_name, output_dir,
                            existing_embeddings_dir, mask_key,
                            h_patches, input_size,
                            dataset_dir, olmo_model,
                            patch_size, device)
    Cache-aware embedding pipeline for one data split.

prepare_all_embeddings(df_train, df_valid, df_test, ...)
    Convenience wrapper for all three splits.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F


def load_or_compute_embeddings(
    df,
    split_name: str,
    output_dir: str,
    *,
    existing_embeddings_dir: str | None = None,
    mask_key: str = "masks_448",
    h_patches: int | None = None,
    input_size: int | None = None,
    dataset_dir: str | None = None,
    olmo_model=None,
    patch_size: int = 8,
    device: str | torch.device = "cuda",
) -> None:
    """
    Populate ``<output_dir>/embeddings/<split_name>_embeddings.pt``.

    Parameters
    ----------
    df : pd.DataFrame
        Split dataframe with columns ``normalized_planetscope_path``,
        ``label_path``, ``planetscope_id``, ``_day``, ``_month``, ``_year``.
    split_name : str
        One of "train", "valid", "test".
    output_dir : str
        Experiment output root (subdirectory ``embeddings/`` is used).
    existing_embeddings_dir : str | None
        Path to embeddings from a prior experiment to copy from (Tier 2).
    mask_key : str
        Key for the resized mask tensor in the saved .pt file.
    h_patches : int | None
        Expected patch grid height — used for assertion in Tier 2.
    input_size : int | None
        Expected mask spatial size — used for assertion in Tier 2.
    dataset_dir : str | None
        Root of the RiverScope dataset — required for Tier 3.
    olmo_model :
        Frozen OlmoEarth model — required for Tier 3.
    patch_size : int
        ViT patch size (default 8).
    device : str | torch.device
    """
    from riverdecode.io import read_image_for_model, read_mask_raw
    from riverdecode.backbone import extract_spatial_tokens

    out_path = os.path.join(output_dir, "embeddings", f"{split_name}_embeddings.pt")

    # ── Tier 1: Already computed ──────────────────────────────────────────────
    if os.path.exists(out_path):
        print(f"  {split_name}: already exists — skipping.")
        return

    # ── Tier 2: Copy from prior experiment ───────────────────────────────────
    if existing_embeddings_dir is not None:
        src = os.path.join(existing_embeddings_dir, f"{split_name}_embeddings.pt")
        if os.path.exists(src):
            check = torch.load(src, map_location="cpu")
            assert mask_key in check, \
                f"Prior embeddings missing key '{mask_key}' — re-run that experiment first."
            if h_patches is not None:
                assert check["embeddings"].shape[1] == h_patches, \
                    (f"Patch grid mismatch: expected {h_patches}, "
                     f"got {check['embeddings'].shape[1]}")
            if input_size is not None:
                assert check[mask_key].shape[1] == input_size, \
                    (f"Mask size mismatch: expected {input_size}, "
                     f"got {check[mask_key].shape[1]}")
            shutil.copy(src, out_path)
            print(
                f"  {split_name}: copied from prior embeddings — "
                f"emb={check['embeddings'].shape}, masks={check[mask_key].shape}"
            )
            del check
            return

    # ── Tier 3: Compute fresh ─────────────────────────────────────────────────
    assert dataset_dir is not None,  "dataset_dir required for fresh computation."
    assert olmo_model  is not None,  "olmo_model required for fresh computation."
    assert input_size  is not None,  "input_size required for fresh computation."

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **kwargs): return it  # type: ignore

    print(f"  Computing {split_name} fresh ({len(df)} tiles)...")
    embeddings, masks_list = [], []
    native_shapes, gt_meta_paths, tile_ids = [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        img_path   = os.path.join(dataset_dir, row["normalized_planetscope_path"])
        label_path = os.path.join(dataset_dir, row["label_path"])

        img_np  = read_image_for_model(img_path)
        mask_np = read_mask_raw(label_path)
        H, W    = mask_np.shape

        tokens = extract_spatial_tokens(
            img_np,
            int(row["_day"]), int(row["_month"]), int(row["_year"]),
            olmo_model, patch_size, input_size, device,
        )

        mask_t    = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        mask_resz = F.interpolate(
            mask_t, size=(input_size, input_size), mode="nearest"
        ).squeeze().long().numpy()

        embeddings.append(tokens)
        masks_list.append(mask_resz)
        native_shapes.append((H, W))
        gt_meta_paths.append(label_path)
        tile_ids.append(row.get("planetscope_id", idx))

    data = {
        "tile_ids":      tile_ids,
        "embeddings":    torch.tensor(np.stack(embeddings), dtype=torch.float32),
        mask_key:        torch.tensor(np.stack(masks_list),  dtype=torch.long),
        "native_shapes": native_shapes,
        "gt_meta_paths": gt_meta_paths,
    }
    torch.save(data, out_path)
    print(
        f"  Saved: emb={data['embeddings'].shape}  "
        f"masks={data[mask_key].shape}"
    )


def prepare_all_embeddings(
    df_train,
    df_valid,
    df_test,
    **kwargs,
) -> None:
    """Run ``load_or_compute_embeddings`` for train, valid, and test."""
    print("Preparing embeddings...")
    load_or_compute_embeddings(df_train, "train", **kwargs)
    load_or_compute_embeddings(df_valid, "valid", **kwargs)
    load_or_compute_embeddings(df_test,  "test",  **kwargs)
    print("\nAll embeddings ready.")
