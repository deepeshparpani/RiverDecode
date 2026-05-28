"""
riverdecode.models.unet_extended
=================================
OlmoEarthUNetDecoderExtended — 4-level UNet decoder with lateral skip
connections from the frozen token grid.

This is the canonical extended decoder, extracted verbatim from
``olmoearth_unet_decoder_extended.py`` at the repository root.

Architecture
------------
Input:  [B, H_p, W_p, embed_dim]  — frozen OlmoEarth tokens (channel-last)
Output: [B, num_classes, 448, 448]  — raw logits (CE head, argmax)

  embed_dim=768, H_p=W_p=28 (for 224px input), or H_p=W_p=62 (for 496px)

  bottleneck  : 768 → 512  @ 28×28
  up1         : 512 → 256  @ 56×56   (+ lat1 skip @ 56)
  up2         : 256 → 256  @ 112×112 (+ lat2 skip @ 112)
  up3         : 256 → 256  @ 224×224 (+ lat3 skip @ 224)
  up4         : 256 → 128  @ 448×448 (+ lat4 skip @ 448)
  refine      : 128 →  64  @ 448×448
  head        :  64 →   2  @ 448×448

Total trainable params ≈ 11–12 M.

Used by: olmoearth_zero_shot_unet_496.ipynb  (62×62 patch grid)
         olmoearth_lora_unet_496.ipynb
         (also as a drop-in for 448-input variant)
"""

# ── Imported verbatim from repository root olmoearth_unet_decoder_extended.py ──
# Keeping the original class in this module avoids import-path issues and
# makes the notebook self-contained via `from riverdecode.models.unet_extended import ...`

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Bilinear upsample ×2 → concat skip → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LateralProject(nn.Module):
    """Project the token grid to a target spatial size and channel width."""

    def __init__(
        self, embed_dim: int, out_ch: int, target_size: tuple[int, int]
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.conv        = ConvBnRelu(embed_dim, out_ch, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(tokens, size=self.target_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class OlmoEarthUNetDecoderExtended(nn.Module):
    """
    4-level UNet decoder with lateral skips for 448×448 river segmentation.

    Parameters
    ----------
    embed_dim     : int
    num_classes   : int
    bottleneck_ch : int
    decoder_chs   : tuple[int, ...]  — channels for up1…up4 + refine.
    """

    SKIP_CH: int = 64

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        bottleneck_ch: int = 512,
        decoder_chs: tuple[int, ...] = (256, 256, 256, 128, 64),
    ) -> None:
        super().__init__()

        d1, d2, d3, d4, d5 = decoder_chs
        s = self.SKIP_CH

        self.bottleneck = nn.Sequential(
            ConvBnRelu(embed_dim, bottleneck_ch),
            ConvBnRelu(bottleneck_ch, bottleneck_ch),
        )

        self.lat1 = LateralProject(embed_dim, s, target_size=(56,  56))
        self.lat2 = LateralProject(embed_dim, s, target_size=(112, 112))
        self.lat3 = LateralProject(embed_dim, s, target_size=(224, 224))
        self.lat4 = LateralProject(embed_dim, s, target_size=(448, 448))

        self.up1    = UNetUpBlock(bottleneck_ch, s, d1)
        self.up2    = UNetUpBlock(d1,            s, d2)
        self.up3    = UNetUpBlock(d2,            s, d3)
        self.up4    = UNetUpBlock(d3,            s, d4)

        self.refine = nn.Sequential(
            ConvBnRelu(d4, d5),
            ConvBnRelu(d5, d5),
        )
        self.head = nn.Conv2d(d5, num_classes, kernel_size=1)
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
        tokens : [B, H_p, W_p, embed_dim]

        Returns
        -------
        logits : [B, num_classes, 448, 448]
        """
        x = tokens.permute(0, 3, 1, 2).contiguous()   # [B, D, H_p, W_p]

        skip1 = self.lat1(x)
        skip2 = self.lat2(x)
        skip3 = self.lat3(x)
        skip4 = self.lat4(x)

        x = self.bottleneck(x)
        x = self.up1(x, skip1)
        x = self.up2(x, skip2)
        x = self.up3(x, skip3)
        x = self.up4(x, skip4)
        x = self.refine(x)
        return self.head(x)


if __name__ == "__main__":
    model = OlmoEarthUNetDecoderExtended()
    dummy = torch.zeros(2, 28, 28, 768)
    out   = model(dummy)
    assert out.shape == (2, 2, 448, 448), f"Bad shape: {out.shape}"
    n = sum(p.numel() for p in model.parameters())
    print(f"OlmoEarthUNetDecoderExtended: {list(dummy.shape)} → {list(out.shape)}, "
          f"params={n / 1e6:.1f}M")
