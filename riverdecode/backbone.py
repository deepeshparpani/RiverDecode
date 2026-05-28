"""
riverdecode.backbone
====================
OlmoEarth backbone loading and spatial token extraction.

Functions
---------
load_olmoearth(model_size, device)
    Load, freeze and return the OlmoEarth encoder.
    Returns (model, embed_dim).

preprocess_image_for_olmo(img_np, input_size, device)
    Map PlanetScope bands to Sentinel-2 layout, resize to input_size,
    and reshape to OlmoEarth's expected [1, H, W, 1, C] tensor.

extract_spatial_tokens(img_np, day, month, year, model,
                        patch_size, input_size, device)
    Run the frozen encoder and return [H_p, W_p, D] token array.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

S2_TOTAL_BANDS: int = 12          # Sentinel-2 L2A: B1–B9, B8A, B11, B12
BAND_MAP: dict[int, int] = {      # PlanetScope (B,G,R,NIR) → S2 indices
    0: 1,   # Blue  → B2
    1: 2,   # Green → B3
    2: 3,   # Red   → B4
    3: 7,   # NIR   → B8
}

_EMBED_DIM_MAP: dict[str, int] = {
    "OLMOEARTH_V1_NANO":  128,
    "OLMOEARTH_V1_TINY":  192,
    "OLMOEARTH_V1_BASE":  768,
    "OLMOEARTH_V1_LARGE": 1024,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_olmoearth(
    model_size: str = "OLMOEARTH_V1_BASE",
    device: str | torch.device = "cuda",
) -> tuple:
    """
    Load the OlmoEarth pretrained encoder, freeze all parameters, and put it
    in eval mode.

    Parameters
    ----------
    model_size : str
        One of OLMOEARTH_V1_{NANO,TINY,BASE,LARGE}.
    device : str | torch.device

    Returns
    -------
    (model, embed_dim) where embed_dim is the token dimensionality.
    """
    from olmoearth_pretrain.model_loader import load_model_from_id, ModelID  # type: ignore

    model_id_map = {k: getattr(ModelID, k) for k in _EMBED_DIM_MAP}
    if model_size not in model_id_map:
        raise ValueError(f"Unknown model_size {model_size!r}. "
                         f"Choose from {list(model_id_map)}")

    print(f"Loading OlmoEarth {model_size}...")
    model = load_model_from_id(model_id_map[model_size])
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    assert trainable == 0, "Backbone has trainable parameters — check freeze!"
    print(
        f"OlmoEarth {model_size} loaded and frozen.  "
        f"Total params: {total / 1e6:.1f}M"
    )

    embed_dim = _EMBED_DIM_MAP[model_size]
    return model, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_for_olmo(
    img_np: np.ndarray,
    input_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Convert a [C, H, W] PlanetScope image to OlmoEarth's expected format.

    Steps
    -----
    1. Map PS bands → S2 band slots (BAND_MAP).
    2. Bilinear resize to (input_size, input_size).
    3. Reshape to [1, input_size, input_size, 1, S2_TOTAL_BANDS].

    Returns
    -------
    torch.Tensor on *device*.
    """
    _, H, W = img_np.shape
    s2_img  = np.zeros((S2_TOTAL_BANDS, H, W), dtype=np.float32)
    for ps_idx, s2_idx in BAND_MAP.items():
        if ps_idx < img_np.shape[0] and s2_idx < S2_TOTAL_BANDS:
            s2_img[s2_idx] = img_np[ps_idx]

    t = torch.from_numpy(s2_img).unsqueeze(0)                   # [1, C, H, W]
    t = F.interpolate(t, size=(input_size, input_size),
                      mode="bilinear", align_corners=False)
    # → [1, input_size, input_size, 1, C]
    t = t.squeeze(0).permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
    return t.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_spatial_tokens(
    img_np: np.ndarray,
    day: int,
    month: int,
    year: int,
    model,
    patch_size: int,
    input_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """
    Run the frozen OlmoEarth encoder on a single PlanetScope tile.

    Parameters
    ----------
    img_np : np.ndarray
        [C, H, W] float32 image from ``read_image_for_model``.
    day, month, year : int
        Acquisition date.
    model :
        Frozen OlmoEarth model from ``load_olmoearth``.
    patch_size : int
        ViT patch size (typically 8).
    input_size : int
        Spatial resolution fed to the encoder (e.g. 224, 448, 496).
    device : str | torch.device

    Returns
    -------
    np.ndarray of shape [H_patches, W_patches, embed_dim] float32.
    """
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample       # type: ignore
    from olmoearth_pretrain.nn.latent_mim import unpack_encoder_output   # type: ignore

    x         = preprocess_image_for_olmo(img_np, input_size, device)
    timestamp = torch.tensor([[[day, month, year]]], dtype=torch.long, device=device)
    sample    = MaskedOlmoEarthSample(
        sentinel2_l2a=x,
        sentinel2_l2a_mask=torch.zeros_like(x),
        timestamps=timestamp,
    )

    model.eval()
    output_dict  = model.encoder(sample, patch_size=patch_size)
    latent, _, _ = unpack_encoder_output(output_dict)
    s2_tokens    = latent.sentinel2_l2a

    if s2_tokens.dim() == 6:
        spatial_tokens = s2_tokens.mean(dim=(3, 4))
    elif s2_tokens.dim() == 5:
        spatial_tokens = s2_tokens.mean(dim=3)
    else:
        spatial_tokens = s2_tokens

    return spatial_tokens.squeeze(0).cpu().numpy()   # [H_p, W_p, D]
