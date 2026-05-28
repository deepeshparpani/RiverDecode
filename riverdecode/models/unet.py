"""
riverdecode.models.unet
=======================
Simple 3-level UNet decoder for OlmoEarth tokens at 448×448.

Architecture
------------
Input:  [B, H_p, W_p, embed_dim]  — frozen OlmoEarth tokens (channel-last)
Output: [B, 1, 448, 448]          — raw sigmoid logit (BCE head)

  [B, 768, 28, 28]  → bottleneck → [B, 256, 28, 28]
                      up1 ×2     → [B, 128, 56, 56]
                      up2 ×2     → [B,  64, 112, 112]
                      up3 ×2     → [B,  32, 224, 224]
                      up4 ×2     → [B,  16, 448, 448]
                      head       → [B,   1, 448, 448]

Used by: olmoearth_zero_shot_unet_448.ipynb
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Bilinear ×2 → Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _ConvBnRelu(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class OlmoEarthUNetDecoder448(nn.Module):
    """
    Lightweight 4-level UNet decoder for 448×448 pixel segmentation.

    Uses a **binary (BCE)** output head — call ``torch.sigmoid`` on logits.

    Parameters
    ----------
    embed_dim  : int   — OlmoEarth token dimensionality (default 768).
    num_classes: int   — must be 1 for the BCE head.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 1) -> None:
        super().__init__()
        assert num_classes == 1, (
            "OlmoEarthUNetDecoder448 uses a binary BCE head (num_classes=1). "
            "For 2-class CE use OlmoEarthUNetDecoderExtended."
        )
        self.bottleneck = nn.Sequential(
            _ConvBnRelu(embed_dim, 256),
            _ConvBnRelu(256, 256),
        )
        self.up1    = _UpBlock(256, 128)   # 28  → 56
        self.up2    = _UpBlock(128,  64)   # 56  → 112
        self.up3    = _UpBlock( 64,  32)   # 112 → 224
        self.up4    = _UpBlock( 32,  16)   # 224 → 448
        self.head   = nn.Conv2d(16, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : [B, H_p, W_p, embed_dim] — OlmoEarth tokens, channel-last.

        Returns
        -------
        logits : [B, 1, 448, 448] — raw logit (apply sigmoid for probability).
        """
        x = tokens.permute(0, 3, 1, 2).contiguous()  # [B, D, H_p, W_p]
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


if __name__ == "__main__":
    model = OlmoEarthUNetDecoder448()
    dummy = torch.zeros(2, 56, 56, 768)  # 496 input → 62 patches
    out   = model(dummy)
    print(f"OlmoEarthUNetDecoder448: {list(dummy.shape)} → {list(out.shape)}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
