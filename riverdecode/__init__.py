"""
riverdecode
===========
Shared utilities for all RiverDecode experiment notebooks.

Quick-start
-----------
    import sys; sys.path.insert(0, '/content/drive/MyDrive/<your-path>/RiverDecode')
    from riverdecode.setup   import mount_drive, patch_dynamo, setup_output_dirs, print_gpu_info
    from riverdecode.data    import set_seeds, load_splits, load_resolution_ceiling
    from riverdecode.io      import read_image_for_model, read_image_for_display, read_mask_raw
    from riverdecode.backbone import load_olmoearth, extract_spatial_tokens
    from riverdecode.embeddings import load_or_compute_embeddings
    from riverdecode.losses  import combined_loss_bce, combined_loss_ce, compute_pos_weight
    from riverdecode.training import adjust_learning_rate, train_one_lr
    from riverdecode.models.unet          import OlmoEarthUNetDecoder448
    from riverdecode.models.unet_extended import OlmoEarthUNetDecoderExtended
    from riverdecode.models.linear_probe  import LinearProbePixelShuffle
    from riverdecode.models.lora_unet     import inject_lora, LoRALinear
"""

__version__ = "1.0.0"
