from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.catvton.image_utils import resize_and_padding
from worker.catvton.mask_service import generate_clothing_mask_full
from worker.exceptions import MaskValidationError
from worker.mask_pipeline import build_garment_mask
from worker.pipeline.types import ParseReport, PipelineContext

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


def run_stage1_parsing(ctx: PipelineContext) -> Image.Image:
    """Garment-aware mask pipeline: classify target garment, estimate body, synthesize mask."""
    if ctx.person is None:
        raise RuntimeError("stage1 requires stage0 person image")
    if ctx.garment is None:
        raise RuntimeError("stage1 requires target garment image")

    cloth_type = CLOTH_TYPE_MAP.get(ctx.cloth_type.lower(), ctx.cloth_type.lower())
    if cloth_type not in {"upper", "lower", "overall", "inner", "outer"}:
        cloth_type = "upper"

    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
    blend_base = ctx.person
    src_w, src_h = blend_base.size

    person = resize_and_padding(blend_base, target_size)
    ctx.crop_box = None
    ctx.normalize_mode = "letterbox"
    ctx.log(f"stage1: letterbox ({src_w}x{src_h} -> {target_size[0]}x{target_size[1]})")
    ctx.blend_base = blend_base

    primary = generate_clothing_mask_full(person, cloth_type=cloth_type)
    ctx.schp_atr = primary["schp_atr"]
    ctx.schp_lip = primary["schp_lip"]

    schp_atr_arr = np.array(ctx.schp_atr)
    schp_lip_arr = np.array(ctx.schp_lip)

    try:
        result = build_garment_mask(
            person=person,
            garment=ctx.garment,
            schp_mask=primary["mask"],
            schp_atr=schp_atr_arr,
            schp_lip=schp_lip_arr,
            cloth_type=ctx.cloth_type,
            log_fn=ctx.log,
        )
    except MaskValidationError as exc:
        ctx.log(f"stage1: mask rejected ({exc.code}): {exc}")
        raise

    mask_arr = result.mask
    diagnostics = result.diagnostics

    if float((mask_arr > 127).mean()) > 0.30:
        mask_arr = cv2.erode(
            mask_arr,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        ctx.log(f"stage1: trimmed oversized mask to {(mask_arr > 127).mean():.2%} coverage")

    ctx.parse = ParseReport(
        confidence=result.confidence,
        used_fallback=result.used_fallback != "schp",
        cloth_type=cloth_type,
        mask_coverage=float((mask_arr > 127).mean()),
        mask_coverage_person_bbox=diagnostics.get("mask_coverage_person_bbox"),
        connectivity_component_count=diagnostics.get("connectivity_component_count"),
        neckline_offset_from_chin_keypoint=diagnostics.get("neckline_offset_from_chin_keypoint"),
        symmetry_ratio=diagnostics.get("symmetry_ratio"),
        used_fallback_source=result.used_fallback,
        garment_neckline_class=diagnostics.get("garment_neckline_class"),
        garment_sleeve_class=diagnostics.get("garment_sleeve_class"),
    )
    ctx.log(
        f"stage1: mask ready (confidence={result.confidence:.2f}, "
        f"fallback={result.used_fallback}, type={cloth_type})"
    )

    ctx.person = person
    ctx.inpaint_mask = Image.fromarray(mask_arr, mode="L")
    ctx.mask_diagnostics = diagnostics
    return ctx.inpaint_mask
