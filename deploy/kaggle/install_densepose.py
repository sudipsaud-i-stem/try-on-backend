#!/usr/bin/env python3
"""
Pose helpers for Kaggle / Linux.

Detectron2+DensePose does NOT build on Kaggle (Python 3.12 + torch 2.11).
This script only pins NumPy 1.26 (required by CatVTON/OpenCV) and optionally
tries MediaPipe after that pin.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def pin_numpy_stack() -> None:
    """CatVTON + OpenCV require NumPy 1.x — Kaggle defaults to NumPy 2.x."""
    print("Pinning numpy 1.26.4 + scipy 1.13.0 (required for CatVTON/OpenCV)...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "numpy==1.26.4",
            "scipy==1.13.0",
            "opencv-python==4.9.0.80",
        ],
        check=False,
    )


def _can_import_densepose() -> bool:
    try:
        import detectron2  # noqa: F401
        from densepose import add_densepose_config  # noqa: F401

        return True
    except ImportError:
        return False


def _can_import_mediapipe() -> bool:
    try:
        import mediapipe as mp  # noqa: F401

        _ = mp.solutions.hands
        return True
    except Exception:
        return False


def install_mediapipe() -> bool:
    """Install MediaPipe only after NumPy 1.x pin."""
    if _can_import_mediapipe():
        print("MediaPipe already works.")
        return True

    print("Installing MediaPipe (face + hand fallback)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.14"],
        check=False,
    )

    if _can_import_mediapipe():
        print("MediaPipe ready — face/hand protection enabled.")
        return True

    print("MediaPipe unavailable — SCHP + OpenCV face detection will be used.")
    return False


def install_detectron2_densepose(verbose: bool = True) -> bool:
    """
    Best-effort pose libs. On Kaggle: pin NumPy + MediaPipe only (no Detectron2).
    """
    pin_numpy_stack()

    on_kaggle = Path("/kaggle").exists()
    if on_kaggle:
        print("\n=== Kaggle detected ===")
        print("Skipping Detectron2/DensePose (no Py3.12 wheel; source build breaks NumPy).")
        print("Using SCHP + MediaPipe/OpenCV for face and hand protection.\n")
        install_mediapipe()
        return _can_import_densepose()

    if _can_import_densepose():
        print("Detectron2 + DensePose already installed.")
        install_mediapipe()
        return True

    print("\n=== Detectron2 + DensePose (optional, non-Kaggle) ===")
    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"
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
        check=False,
    )

    import torch

    ver = torch.__version__.split("+")[0]
    mm = ".".join(ver.split(".")[:2])
    cuda = "cu128" if "cu128" in torch.__version__ else "cu118"
    wheel = f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch{mm}/index.html"

    cmd = [sys.executable, "-m", "pip", "install", "-q", "detectron2", "-f", wheel]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=False)

    dense_ok = _can_import_densepose()
    if dense_ok:
        print("Detectron2 + DensePose installed.")
    else:
        print("Detectron2 not available — SCHP + MediaPipe/OpenCV fallback active.")

    install_mediapipe()
    return dense_ok


if __name__ == "__main__":
    install_detectron2_densepose(verbose=True)
    sys.exit(0)
