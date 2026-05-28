"""
riverdecode.models
==================
Decoder architectures for OlmoEarth river segmentation.

Classes
-------
OlmoEarthUNetDecoder448      — Simple 3-level UNet (448×448, BCE head)
OlmoEarthUNetDecoderExtended — 4-level UNet + lateral skips (any size)
LinearProbePixelShuffle      — LP paper approach (CE head)
ShallowCNNHead               — Shallow 3-level CNN adapter (CE head)
LoRALinear                   — Low-rank adapter
inject_lora()                — Insert LoRA into OlmoEarth ViT Q/V layers
"""

from riverdecode.models.unet          import OlmoEarthUNetDecoder448
from riverdecode.models.unet_extended import OlmoEarthUNetDecoderExtended
from riverdecode.models.linear_probe  import LinearProbePixelShuffle
from riverdecode.models.cnn_head      import ShallowCNNHead
from riverdecode.models.lora_unet     import LoRALinear, inject_lora

__all__ = [
    "OlmoEarthUNetDecoder448",
    "OlmoEarthUNetDecoderExtended",
    "LinearProbePixelShuffle",
    "ShallowCNNHead",
    "LoRALinear",
    "inject_lora",
]
