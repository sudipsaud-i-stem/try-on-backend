from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of


def _face_hair_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return (
        part_mask_of("Face", schp_lip, LIP_MAPPING)
        | part_mask_of("Hair", schp_lip, LIP_MAPPING)
        | part_mask_of("Hat", schp_lip, LIP_MAPPING)
        | part_mask_of("Sunglasses", schp_lip, LIP_MAPPING)
        | part_mask_of("Face", schp_atr, ATR_MAPPING)
        | part_mask_of("Hair", schp_atr, ATR_MAPPING)
        | part_mask_of("Hat", schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def _arm_distal_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    """Protect forearms and hands (lower ~45% of each SCHP arm segment)."""
    protect = np.zeros_like(schp_atr, dtype=np.uint8)
    for arm_label in ("Left-arm", "Right-arm"):
        arm = (
            part_mask_of(arm_label, schp_lip, LIP_MAPPING)
            | part_mask_of(arm_label, schp_atr, ATR_MAPPING)
        )
        if not arm.any():
            continue
        ys, _ = np.where(arm > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        y_cut = int(y0 + (y1 - y0) * 0.50)
        protect[(arm > 0) & (np.arange(arm.shape[0])[:, None] >= y_cut)] = 255
    return protect


def _full_arm_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    """Protect entire arms when compositing back onto the original photo."""
    return (
        part_mask_of("Left-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Right-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Left-arm", schp_atr, ATR_MAPPING)
        | part_mask_of("Right-arm", schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def _schp_upper_garment(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return (
        part_mask_of(["Upper-clothes", "Dress", "Coat", "Jumpsuits"], schp_lip, LIP_MAPPING)
        | part_mask_of(["Upper-clothes", "Dress", "Coat"], schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def build_identity_protect_mask(
    schp_atr: Image.Image,
    schp_lip: Image.Image,
) -> np.ndarray:
    """Pixels pasted back from the original after try-on (face, hair, full arms)."""
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    protect = _face_hair_protect(atr, lip) | _full_arm_protect(atr, lip)
    return (protect > 0).astype(np.float32)


def _inpaint_exclude_mask(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    """Remove from inpaint: face, hair, hands — but keep shirt body and sleeves."""
    return _face_hair_protect(schp_atr, schp_lip) | _arm_distal_protect(schp_atr, schp_lip)


def ensure_minimum_garment_coverage(
    mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    min_coverage: float = 0.14,
) -> np.ndarray:
    """Guarantee enough torso area for CatVTON when SCHP/refine shrinks the mask too far."""
    if float((mask > 127).mean()) >= min_coverage:
        return mask

    upper = _schp_upper_garment(schp_atr, schp_lip).copy()
    exclude = _inpaint_exclude_mask(schp_atr, schp_lip)
    upper[exclude > 0] = 0

    if float((upper > 127).mean()) >= min_coverage * 0.8:
        merged = np.maximum(mask, upper * 255)
    else:
        h, w = mask.shape
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[int(h * 0.18) : int(h * 0.78), int(w * 0.18) : int(w * 0.82)] = 255
        exclude = _inpaint_exclude_mask(schp_atr, schp_lip)
        fallback[exclude > 0] = 0
        merged = np.maximum(mask, fallback)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    return merged


def refine_inpaint_mask(
    mask: Image.Image,
    schp_atr: Image.Image,
    schp_lip: Image.Image,
    cloth_type: str,
) -> Image.Image:
    """
    Trim inpaint mask to shirt region while keeping sleeves.

    Face/hands are excluded from inpaint but pasted back after generation.
    """
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    arr = np.array(mask.convert("L"))

    exclude = _inpaint_exclude_mask(atr, lip)
    arr[exclude > 0] = 0

    if cloth_type in {"sleeveless", "tank", "tank_top"}:
        h, w = arr.shape
        arr[:, : int(w * 0.16)] = 0
        arr[:, int(w * 0.84) :] = 0

    arr = ensure_minimum_garment_coverage(arr, atr, lip)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(arr, mode="L")
