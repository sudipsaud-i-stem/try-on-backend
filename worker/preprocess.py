from __future__ import annotations

from typing import TypedDict

from PIL import Image

from app.config import settings
from worker.catvton.image_utils import preprocess_garment_image, resize_and_crop
from worker.catvton.mask_service import generate_clothing_mask


class PreprocessInputs(TypedDict):
    """Dictionary of PIL images prepared for CatVTON inference."""

    person: Image.Image
    garment: Image.Image
    mask: Image.Image


def prepare_inputs(
    person_image_path: str,
    garment_image_path: str,
    cloth_type: str | None = None,
) -> PreprocessInputs:
    """
    Load and preprocess images using the same resize rules as official CatVTON.

    Person images are center-cropped; garment images are letterboxed.
    Masks are generated with the official AutoMasker (SCHP + DensePose when available).
    """
    person_image = Image.open(person_image_path).convert("RGB")
    garment_image = Image.open(garment_image_path).convert("RGB")

    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
    person_image = resize_and_crop(person_image, target_size)
    garment_image = preprocess_garment_image(garment_image, target_size)

    mask_image = generate_clothing_mask(person_image, cloth_type=cloth_type)

    from worker.postprocess import tighten_mask

    mask_image = tighten_mask(mask_image)

    return {
        "person": person_image,
        "garment": garment_image,
        "mask": mask_image,
    }
