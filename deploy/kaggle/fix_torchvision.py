#!/usr/bin/env python3
"""One-shot fix for Kaggle torch/torchvision mismatch (torchvision::nms error)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
INDEX = "https://download.pytorch.org/whl/cu128"


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=str(BACKEND))


def main() -> None:
    print("Fixing Kaggle torch + torchvision CUDA pair...")
    run(f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio")
    run(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {INDEX}")
    run(
        f"{sys.executable} -m pip install --no-deps --force-reinstall "
        "peft==0.11.1 transformers==4.40.2 diffusers==0.27.2 "
        "accelerate==0.30.0 huggingface-hub==0.23.0 safetensors==0.4.3"
    )
    run(f"{sys.executable} -m pip install 'tokenizers>=0.19,<0.20'")
    run(f"{sys.executable} -m pip install 'numpy<2.0.0'")
    run(f"{sys.executable} -m pip install --force-reinstall 'numpy==1.26.4' 'scipy==1.13.0'")
    run(f"{sys.executable} -m pip install --force-reinstall 'Pillow==10.3.0'")
    run(
        f"{sys.executable} -c \"from worker.compat import verify_torchvision_cuda_ops; "
        "verify_torchvision_cuda_ops()\""
    )
    print("\nDone. Restart the backend: python deploy/kaggle/kaggle_backend_runner.py")


if __name__ == "__main__":
    main()
