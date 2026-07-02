#!/usr/bin/env python3
"""
Install pose libraries on Linux (Kaggle / Colab).

1. Detectron2 + DensePose — best masks (often fails on Py3.12 + torch 2.11).
2. MediaPipe — reliable fallback for face + hand protection on Kaggle.
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


def install_mediapipe() -> bool:
    print("Installing MediaPipe (face + hand fallback)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mediapipe>=0.10.9"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("MediaPipe install failed:", (result.stderr or result.stdout)[-400:])
        return False
    try:
        import mediapipe  # noqa: F401

        print("MediaPipe ready — face/hand protection enabled.")
        return True
    except ImportError:
        print("MediaPipe installed but import failed.")
        return False


def _torch_wheel_tag() -> tuple[str, str]:
    try:
        import torch

        version = torch.__version__.split("+")[0]
        major_minor = ".".join(version.split(".")[:2])
        if "+cu128" in torch.__version__:
            cuda = "cu128"
        elif "+cu124" in torch.__version__:
            cuda = "cu124"
        elif "+cu121" in torch.__version__:
            cuda = "cu121"
        else:
            cuda = "cu118"
        return cuda, major_minor
    except ImportError:
        return "cu118", "2.5"


def install_detectron2_densepose(verbose: bool = True) -> bool:
    """Best-effort DensePose install. Always attempts MediaPipe fallback."""
    mediapipe_ok = install_mediapipe()

    if _can_import_densepose():
        print("Detectron2 + DensePose already installed.")
        return True

    print("\n=== Installing Detectron2 + DensePose (optional, Linux GPU) ===")
    print("Py3.12 + torch 2.11 often has no prebuilt wheel — build may fail.\n")

    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "7.5"
    env["MAX_JOBS"] = "2"

    try:
        import torch

        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            env["CUDA_HOME"] = CUDA_HOME
    except Exception:
        pass

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ninja", "pybind11"],
        env=env,
    )

    cuda, torch_mm = _torch_wheel_tag()
    wheel_urls = [
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch{torch_mm}/index.html",
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch2.5/index.html",
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch2.4/index.html",
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.5/index.html",
    ]

    for url in wheel_urls:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "detectron2", "-f", url]
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0 and _can_import_densepose():
            print("Detectron2 installed from wheel index.")
            break

    if not _can_import_densepose():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-v",
            "--no-build-isolation",
            "git+https://github.com/facebookresearch/detectron2.git",
        ]
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=False)
        if result.returncode != 0 and verbose:
            print("Detectron2 source build failed (expected on Kaggle Py3.12).")

    if _can_import_densepose():
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

    dense_ok = _can_import_densepose()
    if dense_ok:
        print("DensePose ready — AutoMasker will use pose-aware segmentation.")
    else:
        print("DensePose unavailable — using SCHP + MediaPipe face/hand protection.")
        if mediapipe_ok:
            print("MediaPipe fallback is active (recommended for Kaggle).")

    return dense_ok


if __name__ == "__main__":
    ok = install_detectron2_densepose(verbose=True)
    sys.exit(0 if ok else 0)
