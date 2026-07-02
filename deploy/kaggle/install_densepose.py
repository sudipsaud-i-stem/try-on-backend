#!/usr/bin/env python3
"""
Install Detectron2 + DensePose on Linux (Kaggle / Colab).

CatVTON AutoMasker uses real DensePose for pose-aware masks (hands/face protected).
Without it, SCHP-only fallback is weaker on crossed arms and complex poses.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _can_import_densepose() -> bool:
    try:
        import detectron2  # noqa: F401
        from densepose import add_densepose_config  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_wheel_tag() -> tuple[str, str]:
    """Return (cuda_tag, torch_major_minor) for detectron2 wheel index."""
    try:
        import torch

        version = torch.__version__.split("+")[0]
        major_minor = ".".join(version.split(".")[:2])
        if "+cu128" in torch.__version__ or "cu128" in torch.__version__:
            cuda = "cu128"
        elif "+cu124" in torch.__version__:
            cuda = "cu124"
        elif "+cu121" in torch.__version__:
            cuda = "cu121"
        elif "+cu118" in torch.__version__:
            cuda = "cu118"
        else:
            cuda = "cu128"
        return cuda, major_minor
    except ImportError:
        return "cu128", "2.5"


def install_detectron2_densepose(verbose: bool = True) -> bool:
    """Best-effort install; returns True if DensePose imports succeed."""
    if _can_import_densepose():
        print("Detectron2 + DensePose already installed.")
        return True

    print("\n=== Installing Detectron2 + DensePose (Linux GPU) ===")
    print("This improves pose/hand/face masks. Build may take 5–15 minutes on Kaggle.\n")

    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    # T4=7.5, P100=6.0, V100=7.0, A100=8.0
    env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"
    env["MAX_JOBS"] = "2"

    cuda, torch_mm = _torch_wheel_tag()
    wheel_url = f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch{torch_mm}/index.html"

    attempts = [
        [sys.executable, "-m", "pip", "install", "-q", "detectron2", "-f", wheel_url],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/facebookresearch/detectron2.git@v0.6",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/facebookresearch/detectron2.git",
        ],
    ]

    for cmd in attempts:
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=not verbose, text=True)
        if result.returncode == 0 and _can_import_densepose():
            print("Detectron2 installed via pip.")
            break
        if result.returncode != 0 and verbose and result.stderr:
            print(result.stderr[-800:])

    if not _can_import_densepose():
        print("WARNING: Detectron2 install failed — continuing with SCHP-only masks.")
        return False

    # DensePose project (separate editable install)
    build_dir = Path("/kaggle/working/_detectron2_build")
    if not build_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/detectron2.git", str(build_dir)],
            check=False,
        )

    densepose_dir = build_dir / "projects" / "DensePose"
    if densepose_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(densepose_dir)],
            env=env,
            check=False,
        )

    ok = _can_import_densepose()
    if ok:
        print("DensePose ready — AutoMasker will use pose-aware segmentation.")
    else:
        print("DensePose project install incomplete — SCHP fallback still active.")
    return ok


if __name__ == "__main__":
    ok = install_detectron2_densepose(verbose=True)
    sys.exit(0 if ok else 1)
