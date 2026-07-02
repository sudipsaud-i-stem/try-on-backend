from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of


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
        ys, xs = np.where(arm > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        y_cut = int(y0 + (y1 - y0) * 0.52)
        protect[(arm > 0) & (np.arange(arm.shape[0])[:, None] >= y_cut)] = 255
    return protect


def build_identity_protect_mask(
    schp_atr: Image.Image,
    schp_lip: Image.Image,
) -> np.ndarray:
    """Pixels that must stay identical to the original photo (face, hair, hands)."""
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    protect = (
        part_mask_of("Face", lip, LIP_MAPPING)
        | part_mask_of("Hair", lip, LIP_MAPPING)
        | part_mask_of("Hat", lip, LIP_MAPPING)
        | part_mask_of("Sunglasses", lip, LIP_MAPPING)
        | part_mask_of("Face", atr, ATR_MAPPING)
        | part_mask_of("Hair", atr, ATR_MAPPING)
        | part_mask_of("Hat", atr, ATR_MAPPING)
        | _arm_distal_protect(atr, lip)
    )
    return (protect > 0).astype(np.float32)


def refine_inpaint_mask(
    mask: Image.Image,
    schp_atr: Image.Image,
    schp_lip: Image.Image,
    cloth_type: str,
) -> Image.Image:
    """
    Remove face, hair, and hands from the inpaint mask.

    SCHP-only mode does not protect hands in CatVTON AutoMasker — this fixes that.
    """
    arr = np.array(mask.convert("L"))
    protect = build_identity_protect_mask(schp_atr, schp_lip)
    arr[protect > 0.5] = 0

    if cloth_type in {"upper", "inner", "outer"}:
        h, w = arr.shape
        # Drop outer side columns below the shoulders to avoid repainting full arms.
        shoulder = int(h * 0.24)
        for y in range(shoulder, h):
            t = (y - shoulder) / max(h - shoulder, 1)
            margin = int(w * (0.06 + 0.14 * t))
            arr[y, :margin] = 0
            arr[y, w - margin :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(arr, mode="L")
