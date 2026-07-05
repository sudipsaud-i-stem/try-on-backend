from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of
from worker.postprocess import grabcut_person_mask


def build_schp_person_mask(schp_atr: Image.Image, schp_lip: Image.Image) -> np.ndarray:
    """Full-body person silhouette from SCHP parse maps (non-background labels)."""
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    bg = (
        part_mask_of("Background", atr, ATR_MAPPING)
        | part_mask_of("Background", lip, LIP_MAPPING)
    )
    person = ((atr > 0) | (lip > 0)) & ~(bg > 0)
    return (person.astype(np.uint8) * 255)


def build_person_matte(
    person: Image.Image,
    schp_atr: Image.Image | None = None,
    schp_lip: Image.Image | None = None,
    feather: int = 5,
) -> Image.Image:
    """
    Person alpha matte: SCHP body parse + GrabCut silhouette, feathered edges.

    Works on messy / noisy backgrounds without BiRefNet (Kaggle-safe).
    """
    h, w = person.height, person.width
    schp_mask = np.zeros((h, w), dtype=np.uint8)
    if schp_atr is not None and schp_lip is not None:
        schp_mask = build_schp_person_mask(schp_atr, schp_lip)

    gc_mask = grabcut_person_mask(person)

    if schp_mask.max() > 0 and gc_mask.max() > 0:
        schp_f = schp_mask.astype(np.float32) / 255.0
        gc_f = gc_mask.astype(np.float32) / 255.0
        combined = np.clip(0.55 * schp_f + 0.45 * gc_f, 0, 1)
        combined = np.where(schp_f > 0.5, np.maximum(combined, schp_f * 0.85), combined)
    elif schp_mask.max() > 0:
        combined = schp_mask.astype(np.float32) / 255.0
    else:
        combined = gc_mask.astype(np.float32) / 255.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    matte_u8 = (combined * 255).astype(np.uint8)
    matte_u8 = cv2.morphologyEx(matte_u8, cv2.MORPH_CLOSE, kernel)
    matte_u8 = cv2.morphologyEx(matte_u8, cv2.MORPH_OPEN, kernel)

    matte = Image.fromarray(matte_u8, mode="L")
    if feather > 0:
        matte = matte.filter(ImageFilter.GaussianBlur(radius=feather))
    return matte


def composite_on_white(
    person: Image.Image,
    alpha: Image.Image,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Place extracted person on a clean white studio background for CatVTON."""
    rgb = np.array(person.convert("RGB"), dtype=np.float32)
    a = np.array(alpha.convert("L").resize(person.size, Image.Resampling.LANCZOS), dtype=np.float32)
    a = np.clip(a / 255.0, 0.0, 1.0)[..., np.newaxis]
    bg = np.full_like(rgb, background, dtype=np.float32)
    out = rgb * a + bg * (1.0 - a)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def recomposite_on_original_background(
    original: Image.Image,
    vton_person: Image.Image,
    person_alpha: Image.Image,
) -> Image.Image:
    """
    Paste try-on person (from white-bg inference) back onto the original photo.

    Background pixels outside the person matte are preserved exactly.
    """
    if vton_person.size != original.size:
        vton_person = vton_person.resize(original.size, Image.Resampling.LANCZOS)

    orig = np.array(original.convert("RGB"), dtype=np.float32)
    vton = np.array(vton_person.convert("RGB"), dtype=np.float32)
    alpha = np.array(
        person_alpha.convert("L").resize(original.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    alpha = np.clip(alpha / 255.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha_3 = alpha[..., np.newaxis]

    out = orig * (1.0 - alpha_3) + vton * alpha_3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
