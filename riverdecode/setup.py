"""
riverdecode.setup
=================
Google Colab environment helpers shared by all experiment notebooks.

Functions
---------
mount_drive()
    Mount Google Drive with retry fallback.

patch_dynamo()
    Patch torch._dynamo.disable to accept a `wrapping` kwarg (Colab bug fix).
    Must be called before any torch.optim usage.

setup_output_dirs(output_dir)
    Create standard subdirectory layout under output_dir.

print_gpu_info()
    Print GPU name and memory via nvidia-smi.

install_dependencies(extra_packages=None)
    pip-install rasterio, scikit-image, olmoearth-pretrain, and any extras.

clone_or_update_repo(repo_dir, repo_url)
    Clone riverscope-models repo if not already present.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────────

def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive at *mount_point*, skipping if already mounted."""
    if os.path.exists(os.path.join(mount_point, "MyDrive")):
        print("Drive already mounted.")
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_point, force_remount=True)
    except Exception as exc:
        print(f"Drive mount failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dynamo patch (Colab-specific bug)
# ─────────────────────────────────────────────────────────────────────────────

def patch_dynamo() -> None:
    """
    Patch torch._dynamo.decorators.disable to accept a ``wrapping`` kwarg.

    Some Colab runtimes ship a PyTorch version whose optimizer wrappers call
    ``torch._dynamo.disable(fn, wrapping=True)`` but the installed dynamo
    doesn't yet accept that argument.  This one-time file patch + module
    reload fixes it.  Safe to call multiple times.
    """
    dynamo_file = "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/decorators.py"
    if not os.path.exists(dynamo_file):
        # Not a standard Colab path — skip silently
        return

    with open(dynamo_file) as fh:
        src = fh.read()

    if "wrapping" not in src:
        src = re.sub(
            r"def disable\(fn=None,\s*recursive=True\)",
            "def disable(fn=None, recursive=True, wrapping=None)",
            src,
        )
        with open(dynamo_file, "w") as fh:
            fh.write(src)
        print("PyTorch Dynamo patched: added wrapping=None to disable().")
    else:
        print("PyTorch Dynamo already patched — skipping.")

    # Reload affected modules so the new signature is live
    for mod in [
        "torch._dynamo.decorators",
        "torch._dynamo",
        "torch.optim.optimizer",
        "torch.optim.adam",
        "torch.optim.adamw",
        "torch.optim",
    ]:
        if mod in sys.modules:
            try:
                importlib.reload(sys.modules[mod])
            except Exception as exc:
                print(f"  reload {mod}: {exc}")

    # Smoke-test
    import torch
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    del opt
    print("AdamW smoke test OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SUBDIRS = ("embeddings", "checkpoints", "predictions", "visualizations")


def setup_output_dirs(
    output_dir: str,
    subdirs: tuple[str, ...] = _DEFAULT_SUBDIRS,
) -> None:
    """Create *output_dir* and its standard subdirectories."""
    for sub in (output_dir, *[os.path.join(output_dir, s) for s in subdirs]):
        os.makedirs(sub, exist_ok=True)
    print(f"Output dir: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU info
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_info() -> None:
    """Print GPU name + memory via nvidia-smi (no-op if not available)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        print(f"GPU: {result.stdout.strip()}")
    except FileNotFoundError:
        print("nvidia-smi not found — GPU info unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Dependency installation
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies(extra_packages: list[str] | None = None) -> None:
    """
    Install rasterio, scikit-image, olmoearth-pretrain, and any *extra_packages*.

    Uses ``pip install -q`` to suppress verbose output.
    """
    base = ["rasterio", "scikit-image", "olmoearth-pretrain"]
    pkgs = base + (extra_packages or [])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    print("Dependencies ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Repo clone / update
# ─────────────────────────────────────────────────────────────────────────────

def clone_or_update_repo(
    repo_dir: str,
    repo_url: str = "https://github.com/cvl-umass/riverscope-models",
) -> None:
    """Clone *repo_url* to *repo_dir* if not present, then add to sys.path."""
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        print(f"Cloned repo → {repo_dir}")
    else:
        print(f"Repo already present at {repo_dir}")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
