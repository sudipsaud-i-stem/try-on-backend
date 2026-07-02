from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.config import settings
from worker.catvton.image_utils import center_crop_box, resize_and_crop, resize_and_padding
from worker.catvton.mask_service import generate_clothing_mask
from worker.pipeline.types import ParseReport, PipelineContext

# Maps HUBA garment categories to CatVTON mask types.
CLOTH_TYPE_MAP = {
    "upper": "upper",
    "shirt": "upper",
    "jacket": "upper",
    "coat": "outer",
    "outer": "outer",
    "lower": "lower",
    "pants": "lower",
    "overall": "overall",
    "dress": "overall",
    "sleeveless": "upper",
    "tank": "upper",
    "tank_top": "upper",
}


def _mask_coverage(mask: np.ndarray) -> float:
    return float((mask > 127).mean())


def _mask_confidence(mask: np.ndarray, cloth_type: str) -> float:
    """Heuristic confidence from mask shape and coverage."""
    binary = (mask > 127).astype(np.uint8)
    coverage = binary.mean()
    if coverage < 0.02 or coverage > 0.65:
        return 0.25

    ys, xs = np.where(binary > 0)
    if len(xs) < 50:
        return 0.2

    h, w = mask.shape
    y_span = (ys.max() - ys.min()) / max(h, 1)
    x_span = (xs.max() - xs.min()) / max(w, 1)

    score = 0.55
    if 0.12 <= coverage <= 0.42:
        score += 0.2
    if 0.25 <= y_span <= 0.75:
        score += 0.15
    if 0.2 <= x_span <= 0.85:
        score += 0.1

    if cloth_type in {"sleeveless", "tank", "tank_top"}:
        # Tank tops should not cover too much horizontal area (no sleeves).
        if x_span < 0.62:
            score += 0.1
        else:
            score -= 0.15

    return float(np.clip(score, 0.0, 1.0))


def _fallback_torso_mask(size: tuple[int, int]) -> Image.Image:
    """Simple torso heuristic when SCHP confidence is low."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, x1 = int(w * 0.22), int(w * 0.78)
    y0, y1 = int(h * 0.18), int(h * 0.72)
    mask[y0:y1, x0:x1] = 255
    return Image.fromarray(mask, mode="L")


def _refine_sleeveless_mask(mask: Image.Image) -> Image.Image:
    """Remove arm regions for tank tops / sleeveless tops."""
    arr = np.array(mask.convert("L"))
    h, w = arr.shape
    # Zero out side columns where arms typically appear.
    left = int(w * 0.18)
    right = int(w * 0.82)
    arr[:, :left] = 0
    arr[:, right:] = 0
    # Keep central torso band only.
    return Image.fromarray(arr, mode="L")


def _person_segment_fallback(person: Image.Image) -> np.ndarray:
    """GrabCut-based coarse person mask as SCHP sanity check."""
    from worker.postprocess import grabcut_person_mask

    return grabcut_person_mask(person)


def _blend_masks(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if fallback.max() == 0:
        return primary
    p = (primary > 127).astype(np.float32)
    f = (fallback > 127).astype(np.float32)
    blended = np.clip(0.65 * p + 0.35 * f, 0, 1)
    return (blended * 255).astype(np.uint8)


def run_stage1_parsing(ctx: PipelineContext) -> Image.Image:
    """SCHP mask with garment-type logic and low-confidence fallback."""
    if ctx.person is None:
        raise RuntimeError("stage1 requires stage0 person image")

    cloth_type = CLOTH_TYPE_MAP.get(ctx.cloth_type.lower(), ctx.cloth_type.lower())
    if cloth_type not in {"upper", "lower", "overall", "inner", "outer"}:
        cloth_type = "upper"

    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
    blend_base = ctx.person
    src_w, src_h = blend_base.size
    target_w, target_h = target_size

    # Wide/landscape photos: letterbox so arms/pose are not cut off by center crop.
    if src_w / src_h > target_w / target_h * 1.05:
        person = resize_and_padding(blend_base, target_size)
        ctx.crop_box = None
        ctx.normalize_mode = "letterbox"
        ctx.log(f"stage1: letterbox ({src_w}x{src_h} -> {target_w}x{target_h})")
    else:
        crop_box = center_crop_box(blend_base.size, target_size)
        person = resize_and_crop(blend_base, target_size)
        ctx.crop_box = crop_box
        ctx.normalize_mode = "center_crop"
        ctx.log(f"stage1: center crop {crop_box}")

    ctx.blend_base = blend_base

    primary_mask = generate_clothing_mask(person, cloth_type=cloth_type)
    primary_arr = np.array(primary_mask.convert("L"))

    if ctx.cloth_type.lower() in {"sleeveless", "tank", "tank_top"}:
        primary_mask = _refine_sleeveless_mask(primary_mask)
        primary_arr = np.array(primary_mask.convert("L"))

    confidence = _mask_confidence(primary_arr, ctx.cloth_type)
    used_fallback = False

    if confidence < settings.PIPELINE_PARSE_CONFIDENCE:
        used_fallback = True
        ctx.log(f"stage1: low SCHP confidence ({confidence:.2f}) — blending GrabCut fallback")
        fallback_arr = _person_segment_fallback(person)
        if ctx.cloth_type.lower() in {"sleeveless", "tank", "tank_top"}:
            fallback_arr = np.array(_refine_sleeveless_mask(Image.fromarray(fallback_arr, mode="L")))
        blended = _blend_masks(primary_arr, fallback_arr)
        mask = Image.fromarray(blended, mode="L").filter(ImageFilter.GaussianBlur(radius=2))
        confidence = max(confidence, _mask_confidence(blended, ctx.cloth_type) * 0.9)
    else:
        mask = primary_mask
        ctx.log(f"stage1: SCHP mask ok (confidence={confidence:.2f}, type={cloth_type})")

    ctx.parse = ParseReport(
        confidence=confidence,
        used_fallback=used_fallback,
        cloth_type=cloth_type,
        mask_coverage=_mask_coverage(np.array(mask.convert("L"))),
    )
    ctx.person = person
    ctx.inpaint_mask = mask
    return mask
