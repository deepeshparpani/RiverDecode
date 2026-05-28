"""
riverdecode.models.linear_probe
================================
LinearProbePixelShuffle — the exact LP protocol from OlmoEarth evals.

Architecture
------------
Input:  [B, H_p, W_p, embed_dim]   — frozen OlmoEarth tokens (channel-last)
Output: [B, num_classes, H_px, W_px]  — raw logits (CE head)

  Linear(embed_dim, hidden * pixels_per_patch²)  — patch-level
  → PixelShuffle(pixels_per_patch)               — → pixel-level
  → [B, num_classes, H_px, W_px]

For OLMOEARTH_V1_BASE @ input 224:
  embed_dim=768, hidden=128, pixels_per_patch=8
  → 28×28 patches → 224×224 pixels

Used by: olmoearth_zero_shot_linearprobe_224.ipynb
         olmoearth_zero_shot_linearprobe_448.ipynb  (pixels_per_patch=16)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbePixelShuffle(nn.Module):
    """
    Linear probe + pixel-shuffle for per-pixel segmentation.

    Parameters
    ----------
    embed_dim       : int — OlmoEarth token dimensionality (e.g. 768).
    num_classes     : int — output classes (e.g. 2 for water / land).
    pixels_per_patch: int — spatial upscale factor (e.g. 8 for 28→224).
    hidden_ch       : int — hidden channel count before pixel shuffle (default 128).
    """

    def __init__(
        self,
        embed_dim: int       = 768,
        num_classes: int     = 2,
        pixels_per_patch: int = 8,
        hidden_ch: int       = 128,
    ) -> None:
        super().__init__()
        self.pixels_per_patch = pixels_per_patch
        self.num_classes      = num_classes

        # One linear layer maps each token to (hidden × r²) values
        out_features = hidden_ch * pixels_per_patch * pixels_per_patch
        self.linear  = nn.Linear(embed_dim, out_features)

        # Pixel shuffle upsamples patch grid → pixel grid
        self.pixel_shuffle = nn.PixelShuffle(pixels_per_patch)

        # Final 1×1 conv to get num_classes channels
        self.head = nn.Conv2d(hidden_ch, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : [B, H_p, W_p, embed_dim]  — channel-last OlmoEarth tokens.

        Returns
        -------
        logits : [B, num_classes, H_px, W_px]  — raw logits.
        """
        B, H_p, W_p, D = tokens.shape

        # [B, H_p, W_p, hidden * r²]
        x = self.linear(tokens)

        # → [B, hidden * r², H_p, W_p]  (NCHW for pixel shuffle)
        x = x.permute(0, 3, 1, 2).contiguous()

        # → [B, hidden, H_p * r, W_p * r]
        x = self.pixel_shuffle(x)

        # → [B, num_classes, H_px, W_px]
        return self.head(x)


if __name__ == "__main__":
    # Smoke test — 224 px
    probe = LinearProbePixelShuffle(768, 2, pixels_per_patch=8)
    x = torch.zeros(2, 28, 28, 768)
    out = probe(x)
    assert out.shape == (2, 2, 224, 224), f"Bad shape: {out.shape}"
    n = sum(p.numel() for p in probe.parameters())
    print(f"LinearProbePixelShuffle (224): {list(x.shape)} → {list(out.shape)}, params={n:,}")

    # 448 px variant
    probe2 = LinearProbePixelShuffle(768, 2, pixels_per_patch=16)
    x2 = torch.zeros(2, 28, 28, 768)
    out2 = probe2(x2)
    assert out2.shape == (2, 2, 448, 448), f"Bad shape: {out2.shape}"
    print(f"LinearProbePixelShuffle (448): {list(x2.shape)} → {list(out2.shape)}, "
          f"params={sum(p.numel() for p in probe2.parameters()):,}")
