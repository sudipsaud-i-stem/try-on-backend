#!/usr/bin/env python3
"""Pin NumPy 1.26 + OpenCV 4.9 for CatVTON on Kaggle (must run before uvicorn)."""

from __future__ import annotations

import subprocess
import sys


def pin_numpy_stack(*, verify: bool = True) -> None:
    """
    CatVTON + OpenCV require NumPy 1.x.

    MediaPipe / Kaggle pre-installs can upgrade NumPy to 2.x and break cv2 at runtime.
    Always call this as the LAST pip step before starting the API server.
    """
    print("\n=== Final NumPy 1.26.4 + OpenCV pin (required before server start) ===")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "opencv-python-headless",
            "opencv-contrib-python",
        ],
        check=False,
    )
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
        check=True,
    )
    if verify:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                (
                    "import numpy as np; "
                    "assert np.__version__.startswith('1.26'), np.__version__; "
                    "import cv2; "
                    "print('NumPy', np.__version__, '+ OpenCV', cv2.__version__, 'OK')"
                ),
            ]
        )


if __name__ == "__main__":
    pin_numpy_stack(verify=True)
