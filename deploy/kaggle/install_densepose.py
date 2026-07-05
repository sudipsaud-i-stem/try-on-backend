#!/usr/bin/env python3
"""
Pose helpers for Kaggle / Linux.

On Kaggle: only pins NumPy 1.26 (MediaPipe breaks NumPy on Py3.12).
Detectron2+DensePose is skipped — use SCHP + OpenCV for face/hand protection.
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


def install_detectron2_densepose(verbose: bool = True) -> bool:
    """Best-effort pose libs. On Kaggle: NumPy pin only (no Detectron2 / MediaPipe)."""
    on_kaggle = Path("/kaggle").exists()
    if on_kaggle:
        print("\n=== Kaggle detected ===")
        print("Skipping Detectron2/DensePose and MediaPipe (break NumPy 2.x on Py3.12).")
        print("Using SCHP + OpenCV for segmentation and face protection.\n")
        pin_script = Path(__file__).resolve().parent / "pin_numpy.py"
        subprocess.run([sys.executable, str(pin_script)], check=True)
        return False

    pin_script = Path(__file__).resolve().parent / "pin_numpy.py"
    subprocess.run([sys.executable, str(pin_script)], check=False)

    if _can_import_densepose():
        print("Detectron2 + DensePose already installed.")
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

    subprocess.run([sys.executable, str(pin_script)], check=False)
    return _can_import_densepose()


if __name__ == "__main__":
    install_detectron2_densepose(verbose=True)
    sys.exit(0)
