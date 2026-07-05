from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.config import settings
from worker.exceptions import MaskValidationError
from worker.garment_classifier import GarmentProfile, classify_garment
from worker.mask_refine import (
    clamp_mask_above_collar,
    clip_mask_to_person,
    compute_mask_confidence,
    fill_mask_holes,
    keep_torso_component,
    mask_is_boxy,
    mask_shape_is_valid,
    rebuild_garment_mask_from_schp,
    refine_inpaint_mask,
    rejection_message,
)
from worker.mask_synthesis import synthesize_garment_mask
from worker.pose_body import BodyEnvelope, estimate_body_envelope
from worker.sam_fallback import segment_with_sam2_or_fallback


@dataclass
class MaskBuildResult:
    mask: np.ndarray
    garment_profile: GarmentProfile
    body: BodyEnvelope
    confidence: float
    used_fallback: str
    diagnostics: dict


def build_garment_mask(
    person: Image.Image,
    garment: Image.Image,
    schp_mask: Image.Image,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    cloth_type: str,
    log_fn=None,
) -> MaskBuildResult:
    """
    End-to-end mask pipeline (spec steps 1–6).
    Raises MaskValidationError when no valid mask can be produced.
    """
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    garment_profile = classify_garment(garment, cloth_type_hint=cloth_type)
    log(
        f"mask: garment class neckline={garment_profile.neckline} "
        f"sleeve={garment_profile.sleeve_length} fit={garment_profile.fit}"
    )

    body = estimate_body_envelope(
        person,
        schp_atr,
        schp_lip,
        sleeve_length=garment_profile.sleeve_length,
        fit=garment_profile.fit,
    )
    log(f"mask: body envelope source={body.keypoints.source} pose_conf={body.keypoints.confidence:.2f}")

    neckline_key = garment_profile.neckline
    if neckline_key in {"none", "off_shoulder"}:
        neckline_key = "scoop"
    elif neckline_key == "hoodie":
        neckline_key = "crew"

    primary = refine_inpaint_mask(
        schp_mask,
        Image.fromarray(schp_atr),
        Image.fromarray(schp_lip),
        cloth_type,
        neckline_type=neckline_key,
        keypoints=body.keypoints,
        sleeve_length=garment_profile.sleeve_length,
    )
    mask_arr = np.array(primary.convert("L"))
    mask_arr = clip_mask_to_person(mask_arr, person, schp_atr, schp_lip)
    mask_arr = keep_torso_component(mask_arr, schp_atr, schp_lip)
    mask_arr = clamp_mask_above_collar(mask_arr, body.keypoints, schp_atr, schp_lip)

    used_fallback = "schp"
    valid, diagnostics = mask_shape_is_valid(
        mask_arr,
        schp_atr,
        schp_lip,
        person_bbox=body.person_bbox,
        keypoints=body.keypoints,
        sleeve_length=garment_profile.sleeve_length,
        min_coverage=settings.MASK_MIN_COVERAGE,
    )

    if not valid or mask_is_boxy(mask_arr):
        log("mask: SCHP mask invalid or boxy — synthesizing from body envelope + SCHP")
        mask_arr = synthesize_garment_mask(
            schp_atr, schp_lip, body, garment_profile, person_image=person
        )
        mask_arr = clamp_mask_above_collar(mask_arr, body.keypoints, schp_atr, schp_lip)
        used_fallback = "synthesis"
        valid, diagnostics = mask_shape_is_valid(
            mask_arr,
            schp_atr,
            schp_lip,
            person_bbox=body.person_bbox,
            keypoints=body.keypoints,
            sleeve_length=garment_profile.sleeve_length,
            min_coverage=settings.MASK_MIN_COVERAGE,
        )

    if not valid:
        log("mask: synthesis failed — trying SAM2/GrabCut fallback")
        cx = (body.person_bbox[0] + body.person_bbox[2]) // 2
        cy = (body.person_bbox[1] + body.person_bbox[3]) // 2
        seg, seg_source = segment_with_sam2_or_fallback(person, body.person_bbox, (cx, cy))
        seg = seg & body.mask
        seg = clip_mask_to_person(seg, person, schp_atr, schp_lip)
        rebuilt = rebuild_garment_mask_from_schp(
            schp_atr,
            schp_lip,
            person,
            neckline_type=neckline_key,
            keypoints=body.keypoints,
            sleeve_length=garment_profile.sleeve_length,
        )
        mask_arr = np.maximum(rebuilt, seg)
        mask_arr = keep_torso_component(mask_arr, schp_atr, schp_lip)
        mask_arr = fill_mask_holes(mask_arr)
        mask_arr = clamp_mask_above_collar(mask_arr, body.keypoints, schp_atr, schp_lip)
        used_fallback = seg_source
        valid, diagnostics = mask_shape_is_valid(
            mask_arr,
            schp_atr,
            schp_lip,
            person_bbox=body.person_bbox,
            keypoints=body.keypoints,
            sleeve_length=garment_profile.sleeve_length,
            min_coverage=settings.MASK_MIN_COVERAGE,
        )

    confidence = compute_mask_confidence(diagnostics, min_coverage=settings.MASK_MIN_COVERAGE)

    # Last resort: good coverage but neckline drift — clamp to collar and re-check.
    if (not valid or confidence < settings.PIPELINE_PARSE_CONFIDENCE) and diagnostics.get(
        "mask_coverage_person_bbox", 0
    ) >= 0.35:
        mask_arr = clamp_mask_above_collar(mask_arr, body.keypoints, schp_atr, schp_lip)
        valid, diagnostics = mask_shape_is_valid(
            mask_arr,
            schp_atr,
            schp_lip,
            person_bbox=body.person_bbox,
            keypoints=body.keypoints,
            sleeve_length=garment_profile.sleeve_length,
            min_coverage=settings.MASK_MIN_COVERAGE,
        )
        confidence = compute_mask_confidence(diagnostics, min_coverage=settings.MASK_MIN_COVERAGE)
        if valid:
            log("mask: collar clamp recovered valid mask")

    diagnostics["garment_neckline_class"] = garment_profile.neckline
    diagnostics["garment_sleeve_class"] = garment_profile.sleeve_length
    diagnostics["used_fallback"] = used_fallback
    diagnostics["confidence"] = confidence

    log(
        f"mask: coverage_bbox={diagnostics.get('mask_coverage_person_bbox', 0):.2%} "
        f"components={diagnostics.get('connectivity_component_count')} "
        f"symmetry={diagnostics.get('symmetry_ratio', 0):.2f} "
        f"fallback={used_fallback} confidence={confidence:.2f}"
    )

    if not valid or confidence < settings.PIPELINE_PARSE_CONFIDENCE:
        code, message = rejection_message(diagnostics)
        raise MaskValidationError(message, code=code, diagnostics=diagnostics)

    return MaskBuildResult(
        mask=mask_arr,
        garment_profile=garment_profile,
        body=body,
        confidence=confidence,
        used_fallback=used_fallback,
        diagnostics=diagnostics,
    )
