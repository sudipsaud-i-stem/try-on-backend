from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of


def _full_arm_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    """Protect entire arms (critical for crossed-arm poses)."""
    return (
        part_mask_of("Left-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Right-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Left-arm", schp_atr, ATR_MAPPING)
        | part_mask_of("Right-arm", schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def _arm_distal_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    """Protect forearms and hands when full-arm mask is unavailable."""
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
        | _full_arm_protect(atr, lip)
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
        # Remove entire arm columns from inpaint (crossed-arm safe).
        arms = _full_arm_protect(np.array(schp_atr), np.array(schp_lip))
        arr[arms > 0] = 0

        shoulder = int(h * 0.24)
        for y in range(shoulder, h):
            t = (y - shoulder) / max(h - shoulder, 1)
            margin = int(w * (0.04 + 0.10 * t))
            arr[y, :margin] = 0
            arr[y, w - margin :] = 0

    # Close small neck holes without expanding into arms.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return Image.fromarray(arr, mode="L")
