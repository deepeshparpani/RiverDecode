"""
riverdecode.models.cnn_head
============================
ShallowCNNHead — Shallow CNN adapter on top of frozen OlmoEarth tokens.

Architecture
------------
Input:  [B, H_p, W_p, embed_dim]  — frozen OlmoEarth tokens (channel-last)
Output: [B, num_classes, H_p*8, W_p*8]  — raw logits (CE head, argmax)

  28×28 patch grid → 224×224 pixel output (8× upscale, 3 bilinear ×2 stages)

  permute   → [B, 768, 28,  28]
  proj      → Conv(768→256, 3×3)+BN+ReLU  → [B, 256, 28,  28]
  up1       → Upsample×2+Conv(256→128)+BN+ReLU → [B, 128, 56,  56]
  up2       → Upsample×2+Conv(128→64) +BN+ReLU → [B,  64, 112, 112]
  up3       → Upsample×2+Conv(64→32)  +BN+ReLU → [B,  32, 224, 224]
  head      → Conv(32→num_classes, 1×1)          → [B,   2, 224, 224]

Trainable params ≈ 2.16M  (vs 98K for the linear probe, vs 11M for UNet-Extended).

Used by: olmoearth_zero_shot_cnn_448.ipynb
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _UpBlock(nn.Module):
    """Bilinear upsample ×2 → Conv2d(3×3) → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.up(x))))


class ShallowCNNHead(nn.Module):
    """
    Shallow 3-level CNN decoder for token-to-pixel upscaling.

    Preferred over the linear probe when you want spatial inductive bias
    at low parameter count, and over the UNet when you want fast training.

    Parameters
    ----------
    embed_dim   : int — OlmoEarth token dimensionality (default 768).
    num_classes : int — output classes (default 2 for water / land).
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 2) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1  = _UpBlock(256, 128)   # 28  → 56
        self.up2  = _UpBlock(128,  64)   # 56  → 112
        self.up3  = _UpBlock( 64,  32)   # 112 → 224
        # Raw logits — no BN/activation before CrossEntropyLoss
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
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
        tokens : [B, H_p, W_p, embed_dim]  — channel-last OlmoEarth tokens.

        Returns
        -------
        logits : [B, num_classes, H_p*8, W_p*8]  — raw logits.
        """
        x = tokens.permute(0, 3, 1, 2).contiguous()   # [B, D, H_p, W_p]
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)


if __name__ == "__main__":
    model = ShallowCNNHead(768, 2)
    dummy = torch.zeros(2, 28, 28, 768)
    out   = model(dummy)
    assert out.shape == (2, 2, 224, 224), f"Bad shape: {out.shape}"
    n = sum(p.numel() for p in model.parameters())
    print(f"ShallowCNNHead: {list(dummy.shape)} → {list(out.shape)}, params={n:,}")
