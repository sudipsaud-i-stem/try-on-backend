from __future__ import annotations

from pathlib import Path

from diffusers.image_processor import VaeImageProcessor
from loguru import logger
from PIL import Image

from app.config import settings
from worker.catvton.model.cloth_masker import AutoMasker

_automasker: AutoMasker | None = None
_mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)


def _load_automasker() -> AutoMasker:
    global _automasker
    if _automasker is not None:
        return _automasker

    model_root = settings.catvton_model_path
    densepose_ckpt = str(model_root / "DensePose")
    schp_ckpt = str(model_root / "SCHP")
    device = str(settings.device)

    logger.info("Loading CatVTON AutoMasker (DensePose={}, SCHP={})", densepose_ckpt, schp_ckpt)
    _automasker = AutoMasker(
        densepose_ckpt=densepose_ckpt,
        schp_ckpt=schp_ckpt,
        device=device,
    )
    if _automasker.use_densepose:
        logger.info("AutoMasker using official DensePose + SCHP")
    else:
        logger.warning(
            "Detectron2/DensePose not installed — using SCHP + MediaPipe mask (install detectron2 for full quality)"
        )
    return _automasker


def generate_clothing_mask(
    person_image: Image.Image,
    cloth_type: str | None = None,
) -> Image.Image:
    """Generate an agnostic mask using the official CatVTON AutoMasker pipeline."""
    result = generate_clothing_mask_full(person_image, cloth_type=cloth_type)
    return result["mask"]


def generate_clothing_mask_full(
    person_image: Image.Image,
    cloth_type: str | None = None,
) -> dict:
    """Return mask plus SCHP parse maps for identity protection."""
    mask_type = cloth_type or settings.CLOTH_TYPE
    automasker = _load_automasker()
    raw = automasker(person_image, mask_type=mask_type)
    mask = _mask_processor.blur(raw["mask"], blur_factor=settings.MASK_BLUR_FACTOR)
    return {
        "mask": mask,
        "schp_atr": raw["schp_atr"],
        "schp_lip": raw["schp_lip"],
        "densepose": raw["densepose"],
    }
