"""
riverdecode.models.lora_unet
=============================
LoRA (Low-Rank Adaptation) for the OlmoEarth ViT backbone.

Classes
-------
LoRALinear
    Wraps an ``nn.Linear`` layer with a low-rank adapter A·B.
    Only A and B are trainable; the original weight is frozen.

Functions
---------
inject_lora(model, rank, alpha, target_modules)
    Replace Q and V projection layers in all ViT attention blocks with
    LoRALinear wrappers.  Returns the number of adapters injected.

Usage
-----
    from riverdecode.models.lora_unet import inject_lora

    olmo_model, embed_dim = load_olmoearth(MODEL_SIZE, DEVICE)
    n_adapters = inject_lora(olmo_model, rank=8, alpha=16)
    # encoder now has trainable LoRA params; decoder is still fresh init

Notes
-----
- Phase 1 (e.g. 5 epochs): freeze LoRA, train decoder only.
- Phase 2 (remaining):     unfreeze LoRA + train decoder jointly.
- Freeze LoRA with: `for n, p in model.named_parameters():
                        if 'lora' in n: p.requires_grad_(False)`
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping a frozen ``nn.Linear``.

    W_new = W_frozen + (alpha / rank) * B @ A

    Parameters
    ----------
    original : nn.Linear
        The layer to wrap.  Its ``weight`` and optional ``bias`` are frozen.
    rank     : int  — LoRA rank (r).
    alpha    : float — scaling factor (alpha / r multiplied at forward time).
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int   = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features  = original.in_features
        self.out_features = original.out_features
        self.rank         = rank
        self.scale        = alpha / rank

        # Freeze original weight
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B initialised to zero → no change at step 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        result = result + self.scale * nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return result

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.2f}"
        )


def inject_lora(
    model: nn.Module,
    rank:  int   = 8,
    alpha: float = 16.0,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
) -> int:
    """
    Replace Q/V projection ``nn.Linear`` layers in *model* with
    ``LoRALinear`` wrappers.

    Parameters
    ----------
    model          : nn.Module — the OlmoEarth encoder (or full model).
    rank           : int — LoRA rank.
    alpha          : float — LoRA scaling.
    target_modules : sequence of attribute name substrings to match.

    Returns
    -------
    int — total number of LoRA adapters injected.
    """
    n_injected = 0
    for name, module in list(model.named_modules()):
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                orig = getattr(module, attr_name)
                if isinstance(orig, nn.Linear):
                    setattr(module, attr_name, LoRALinear(orig, rank=rank, alpha=alpha))
                    n_injected += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected {n_injected} adapters  "
        f"(rank={rank}, alpha={alpha}).  "
        f"Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )
    return n_injected


if __name__ == "__main__":
    # Sanity check on a toy transformer
    layer = nn.Linear(768, 64)
    lora  = LoRALinear(layer, rank=8, alpha=16)
    x     = torch.randn(2, 10, 768)
    out   = lora(x)
    assert out.shape == (2, 10, 64), f"Bad shape: {out.shape}"
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"LoRALinear output: {list(out.shape)}, trainable params: {trainable:,}")
