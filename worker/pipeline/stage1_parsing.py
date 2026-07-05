from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.catvton.image_utils import resize_and_padding
from worker.catvton.mask_service import generate_clothing_mask_full
from worker.pipeline.types import ParseReport, PipelineContext

# Maps HUBA garment categories to CatVTON mask types.
CLOTH_TYPE_MAP = {
    "upper": "upper",
    "shirt": "upper",
    "tshirt": "upper",
    "tee": "upper",
    "jacket": "outer",
    "bomber": "outer",
    "blazer": "outer",
    "hoodie": "outer",
    "coat": "outer",
    "outer": "outer",
    "inner": "inner",
    "undershirt": "inner",
    "lower": "lower",
    "pants": "lower",
    "trousers": "lower",
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
    if 0.08 <= coverage <= 0.22:
        score += 0.25
    elif 0.22 < coverage <= 0.30:
        score += 0.1
    elif coverage > 0.30:
        score -= 0.35
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

    # Always letterbox — center crop cuts off raised arms and yoga poses.
    person = resize_and_padding(blend_base, target_size)
    ctx.crop_box = None
    ctx.normalize_mode = "letterbox"
    ctx.log(f"stage1: letterbox ({src_w}x{src_h} -> {target_w}x{target_h})")

    ctx.blend_base = blend_base

    primary = generate_clothing_mask_full(person, cloth_type=cloth_type)
    ctx.schp_atr = primary["schp_atr"]
    ctx.schp_lip = primary["schp_lip"]
    primary_mask = primary["mask"]

    from worker.mask_refine import refine_inpaint_mask

    primary_mask = refine_inpaint_mask(primary_mask, ctx.schp_atr, ctx.schp_lip, cloth_type)
    primary_arr = np.array(primary_mask.convert("L"))
    coverage = _mask_coverage(primary_arr)

    if coverage > 0.30:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        for _ in range(6):
            if coverage <= 0.28:
                break
            primary_arr = cv2.erode(primary_arr, kernel, iterations=1)
            coverage = _mask_coverage(primary_arr)
        primary_mask = Image.fromarray(primary_arr, mode="L")
        ctx.log(f"stage1: trimmed oversized mask to {coverage:.2%} coverage")

    ctx.log(f"stage1: garment mask coverage={coverage:.2%} after refine")

    if coverage < 0.14:
        from worker.mask_refine import ensure_minimum_garment_coverage

        expanded = ensure_minimum_garment_coverage(
            primary_arr,
            np.array(ctx.schp_atr),
            np.array(ctx.schp_lip),
            min_coverage=0.14,
        )
        primary_mask = Image.fromarray(expanded, mode="L")
        coverage = _mask_coverage(expanded)
        ctx.log(f"stage1: expanded undersized mask to {coverage:.2%} coverage")

    if ctx.cloth_type.lower() in {"sleeveless", "tank", "tank_top"}:
        primary_mask = _refine_sleeveless_mask(primary_mask)
        primary_arr = np.array(primary_mask.convert("L"))

    confidence = _mask_confidence(primary_arr, ctx.cloth_type)
    used_fallback = False

    if confidence < settings.PIPELINE_PARSE_CONFIDENCE:
        used_fallback = True
        ctx.log(f"stage1: low SCHP confidence ({confidence:.2f}) — expanding with torso heuristic")
        fallback_arr = np.array(_fallback_torso_mask(person.size))
        if ctx.schp_atr is not None and ctx.schp_lip is not None:
            from worker.mask_refine import ensure_minimum_garment_coverage

            fallback_arr = ensure_minimum_garment_coverage(
                fallback_arr,
                np.array(ctx.schp_atr),
                np.array(ctx.schp_lip),
                min_coverage=0.14,
            )
        merged = np.maximum(primary_arr, fallback_arr)
        if ctx.cloth_type.lower() in {"sleeveless", "tank", "tank_top"}:
            merged = np.array(_refine_sleeveless_mask(Image.fromarray(merged, mode="L")))
        mask = Image.fromarray(merged, mode="L")
        confidence = max(confidence, _mask_confidence(merged, ctx.cloth_type) * 0.85)
    else:
        mask = primary_mask
        ctx.log(f"stage1: SCHP conservative mask (confidence={confidence:.2f}, type={cloth_type}, identity-protected)")

    ctx.parse = ParseReport(
        confidence=confidence,
        used_fallback=used_fallback,
        cloth_type=cloth_type,
        mask_coverage=_mask_coverage(np.array(mask.convert("L"))),
    )
    ctx.person = person
    ctx.inpaint_mask = mask
    return mask
