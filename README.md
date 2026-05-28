# RiverDecode

**River segmentation experiments using the [OlmoEarth](https://github.com/cvl-umass/riverscope-models) frozen ViT backbone.**

This repository contains:
- A modular Python package (`riverdecode/`) with all shared training utilities
- Six clean experiment notebooks, each refactored to import from the package rather than defining code inline

Part of the [RiverScope](https://github.com/deepeshparpani/riverscope-notebooks) project for automated river width estimation from PlanetScope satellite imagery.

---

## Repository Structure

```
RiverDecode/
├── riverdecode/                    # Shared Python package
│   ├── __init__.py
│   ├── setup.py                    # Drive mount, dynamo patch, GPU info
│   ├── data.py                     # CSV splits, date parsing, seeds
│   ├── io.py                       # GeoTIFF image/mask readers
│   ├── backbone.py                 # OlmoEarth loader & token extraction
│   ├── embeddings.py               # 3-tier embedding cache
│   ├── losses.py                   # BCE/Dice/CE loss heads, F1, IoU
│   ├── training.py                 # Training loop, LR schedule, sweep state
│   └── models/
│       ├── unet.py                 # OlmoEarthUNetDecoder448   (BCE head)
│       ├── unet_extended.py        # OlmoEarthUNetDecoderExtended (CE head)
│       ├── linear_probe.py         # LinearProbePixelShuffle  (CE head)
│       ├── cnn_head.py             # ShallowCNNHead            (CE head)
│       └── lora_unet.py            # LoRALinear + inject_lora
│
├── olmoearth_zero_shot_unet_448_refactored.ipynb
├── olmoearth_zero_shot_unet_496_refactored.ipynb
├── olmoearth_zero_shot_linearprobe_224_refactored.ipynb
├── olmoearth_zero_shot_linearprobe_448_refactored.ipynb
├── olmoearth_zero_shot_cnn_448_refactored.ipynb
├── olmoearth_lora_unet_496_refactored.ipynb
│
├── resolution_loss_analysis.ipynb  # Standalone: resolution ceiling analysis
├── riverscope_dpt_baseline.ipynb   # Standalone: DPT baseline
├── width_est.ipynb                 # Standalone: river width estimation
│
└── rebuild_notebooks.py            # Regenerate all *_refactored.ipynb files
```

---

## Experiments

| Notebook | Backbone Input | Decoder | Head | Output |
|----------|---------------|---------|------|--------|
| `zero_shot_unet_448` | 448 px → 56×56 patches | 4-level UNet (simple) | BCE + Dice | 448×448 |
| `zero_shot_unet_496` | 496 px → 62×62 patches | UNetExtended (4-level + laterals) | CE + Dice | 448×448 |
| `zero_shot_linearprobe_224` | 224 px → 28×28 patches | LinearProbe + PixelShuffle | CE + Dice | 224×224 |
| `zero_shot_linearprobe_448` | 448 px → 56×56 patches | LinearProbe + PixelShuffle | CE + Dice | 448×448 |
| `zero_shot_cnn_448` | 224 px → 28×28 patches | ShallowCNNHead (3 upsamples) | CE + Dice | 224×224 |
| `lora_unet_496` | 496 px (live) | UNetExtended | CE + Dice | 448×448 |

All zero-shot experiments use a **frozen** OlmoEarth encoder — only the decoder is trained.  
The LoRA experiment fine-tunes Q/V projections in the ViT jointly with the decoder.

---

## Package Overview

### `riverdecode.setup`
```python
from riverdecode.setup import mount_drive, patch_dynamo, setup_output_dirs, print_gpu_info
```
Colab environment helpers: Drive mounting, the `torch._dynamo.disable` keyword-argument patch, output directory creation, and GPU info.

### `riverdecode.data`
```python
from riverdecode.data import set_seeds, load_splits, parse_all_splits, load_resolution_ceiling
```
Reproducibility seeds, CSV split loading, acquisition date parsing from PlanetScope IDs, and resolution ceiling JSON loader.

### `riverdecode.io`
```python
from riverdecode.io import read_image_for_model, read_image_for_display, read_mask_raw
```
GeoTIFF readers with normalisation matching `dataset_planet.py` exactly.  
`read_image_for_model` uses global min-max normalisation (for model input).  
`read_image_for_display` uses per-band 2/98 percentile stretch (visualisation only).

### `riverdecode.backbone`
```python
from riverdecode.backbone import load_olmoearth, extract_spatial_tokens
```
Loads and freezes the OlmoEarth encoder, maps PlanetScope bands to Sentinel-2 slots, and returns `[H_p, W_p, D]` spatial token arrays.

### `riverdecode.embeddings`
```python
from riverdecode.embeddings import load_or_compute_embeddings
```
Three-tier cache:
1. **Already exists** → skip
2. **Prior experiment has compatible embeddings** → copy (fast)
3. **Neither** → compute fresh with the frozen backbone

### `riverdecode.losses`
```python
from riverdecode.losses import combined_loss_bce, combined_loss_ce, compute_pos_weight
from riverdecode.losses import compute_f1_sigmoid, compute_f1_argmax, compute_iou_np
```
BCE+Dice (binary head) and CE+Dice (2-class head) with `pos_weight` for class imbalance. Includes both training-time (torch) and evaluation-time (numpy) metrics.

### `riverdecode.training`
```python
from riverdecode.training import train_one_lr, SweepState, val_sanity_viz, adjust_learning_rate
```
Generic training loop with cosine LR annealing + warm-up, top-k checkpoint strategy, and `SweepState` to track the best LR across a sweep.

### `riverdecode.models`
```python
from riverdecode.models import (
    OlmoEarthUNetDecoder448,
    OlmoEarthUNetDecoderExtended,
    LinearProbePixelShuffle,
    ShallowCNNHead,
    LoRALinear, inject_lora,
)
```

| Class | Params | Input | Output |
|-------|--------|-------|--------|
| `OlmoEarthUNetDecoder448` | ~1.5M | `[B, H_p, W_p, 768]` | `[B, 1, 448, 448]` |
| `OlmoEarthUNetDecoderExtended` | ~11M | `[B, H_p, W_p, 768]` | `[B, 2, 448, 448]` |
| `LinearProbePixelShuffle` | ~98K–400K | `[B, H_p, W_p, 768]` | `[B, 2, H, W]` |
| `ShallowCNNHead` | ~2.16M | `[B, H_p, W_p, 768]` | `[B, 2, H_p×8, W_p×8]` |

---

## Quick Start (Google Colab)

1. **Upload** this repo to Google Drive, e.g. at  
   `MyDrive/CS682/project/riverscope-models/RiverDecode/`

2. **Open** any `*_refactored.ipynb` notebook in Colab.

3. **Edit** the `RIVERDECODE_DIR` variable in Cell 1 to match your Drive path:
   ```python
   RIVERDECODE_DIR = '/content/drive/MyDrive/CS682/project/riverscope-models/RiverDecode'
   ```

4. **Edit** the `## 0. Config` cell paths (`DATASET_DIR`, `OUTPUT_DIR`, etc.) to point at your dataset and results folders.

5. **Run all cells.** The setup cell handles Drive mounting, dependency installation, and repo cloning automatically.

---

## Regenerating Notebooks

If you modify the package and want to regenerate the notebook stubs:

```bash
python3 rebuild_notebooks.py
```

This rewrites all six `*_refactored.ipynb` files. The originals (without `_refactored` suffix) are kept unchanged.

---

## Dependencies

Installed automatically by `install_dependencies()` in the setup cell:

| Package | Purpose |
|---------|---------|
| `rasterio` | GeoTIFF reading/writing |
| `scikit-image` | Prediction upscaling (`skimage.transform.resize`) |
| `olmoearth-pretrain` | OlmoEarth ViT backbone |
| `torch` | Pre-installed on Colab A100 |
| `einops` | Token reshaping (used internally by OlmoEarth) |

---

## Notes

- **Embeddings are not stored in this repo** — they are large (several GB) and are saved to Google Drive under each experiment's `OUTPUT_DIR/embeddings/`.
- **Checkpoints are not stored in this repo** — saved to `OUTPUT_DIR/checkpoints/` on Drive.
- The `lora_unet_496` experiment does **not** use pre-computed embeddings; it runs a live forward pass through the encoder at each batch (encoder must stay on GPU).
