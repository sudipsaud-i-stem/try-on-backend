from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def sam2_available() -> bool:
    try:
        from sam2.build_sam import build_sam2  # noqa: F401

        return True
    except ImportError:
        return False


def _grabcut_envelope_mask(image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Visual segmentation fallback when SAM2 is unavailable."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    rect = (x0, y0, x1 - x0, y1 - y0)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return out
    except cv2.error:
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[y0:y1, x0:x1] = 255
        return fallback


def segment_with_sam2_or_fallback(
    person_image: Image.Image,
    bbox: tuple[int, int, int, int],
    center_point: tuple[int, int] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Try SAM2 point/box prompt; fall back to GrabCut on the body envelope bbox.
    Returns (mask, source_tag).
    """
    if sam2_available():
        try:
            return _segment_sam2(person_image, bbox, center_point), "sam2"
        except Exception:
            pass
    return _grabcut_envelope_mask(person_image, bbox), "grabcut"
