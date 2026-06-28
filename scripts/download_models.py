from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import huggingface_hub
from loguru import logger

from app.config import settings

ATTN_SUBFOLDERS = {
    "mix": "mix-48k-1024",
    "vitonhd": "vitonhd-16k-512",
    "dresscode": "dresscode-16k-512",
}


def _dir_size_gb(path: Path) -> float:
    """Calculate total size of a directory in gigabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def _verify_model(model_dir: Path, version: str, label: str) -> None:
    """Verify CatVTON attention weights exist after download."""
    subfolder = ATTN_SUBFOLDERS[version]
    required = [
        model_dir / subfolder / "attention" / "model.safetensors",
        model_dir / "SCHP" / "exp-schp-201908301523-atr.pth",
        model_dir / "DensePose" / "model_final_162be9.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label} verification failed — missing:\n" + "\n".join(missing))
    logger.info("{} verification passed ({})", label, subfolder)


def download_models() -> None:
    """Download and cache all required model weights from HuggingFace."""
    settings.ensure_directories()
    total_gb = 0.0

    catvton_dir = settings.catvton_model_path
    weights_file = (
        catvton_dir
        / ATTN_SUBFOLDERS[settings.CATVTON_ATTN_VERSION]
        / "attention"
        / "model.safetensors"
    )

    if weights_file.exists():
        logger.info("CatVTON weights already present at {}", weights_file)
    else:
        logger.info("Downloading {} -> {}", settings.CATVTON_MODEL_ID, catvton_dir)
        huggingface_hub.snapshot_download(
            settings.CATVTON_MODEL_ID,
            local_dir=str(catvton_dir),
            resume_download=True,
        )

    _verify_model(catvton_dir, settings.CATVTON_ATTN_VERSION, "CatVTON")
    catvton_gb = _dir_size_gb(catvton_dir)
    logger.info("CatVTON size: {:.2f} GB", catvton_gb)
    total_gb += catvton_gb

    print(f"\nAll models ready. Total size: {total_gb:.2f} GB")
    print(f"Attention weights: {weights_file}")
    print(
        "\nNote: Base SD inpainting model (runwayml/stable-diffusion-inpainting) "
        "downloads automatically on first inference."
    )


if __name__ == "__main__":
    download_models()
