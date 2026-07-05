from __future__ import annotations

import cv2
import numpy as np

from worker.garment_classifier import GarmentProfile
from worker.mask_refine import (
    _schp_upper_garment,
    bridge_disconnected_sleeves,
    clip_mask_to_person,
    fill_mask_holes,
    keep_torso_component,
    normalize_neckline,
    trim_sleeves_to_length,
)
from worker.pose_body import BodyEnvelope


def _dilate_small(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.dilate((mask > 127).astype(np.uint8) * 255, k, iterations=1)


def synthesize_garment_mask(
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    body: BodyEnvelope,
    garment_profile: GarmentProfile,
    person_image=None,
) -> np.ndarray:
    """
    Intersect SCHP labels with pose body envelope, bridge sleeves, reshape to target garment.
    """
    raw = _schp_upper_garment(schp_atr, schp_lip).astype(np.uint8) * 255
    envelope = body.mask

    if raw.max() > 0:
        dilated = _dilate_small(raw, radius=max(11, min(raw.shape) // 40))
        recovery = (envelope > 127) & (dilated > 127)
        candidate = np.maximum(raw, recovery.astype(np.uint8) * 255)
    else:
        candidate = (envelope > 127).astype(np.uint8) * 255

    if person_image is not None:
        candidate = clip_mask_to_person(candidate, person_image, schp_atr, schp_lip)

    candidate = bridge_disconnected_sleeves(candidate, schp_atr, schp_lip)
    candidate = keep_torso_component(candidate, schp_atr, schp_lip)

    neckline = garment_profile.neckline
    if neckline in {"none", "off_shoulder"}:
        neckline_key = "scoop"
    elif neckline == "hoodie":
        neckline_key = "crew"
    else:
        neckline_key = neckline

    candidate = normalize_neckline(candidate, schp_atr, schp_lip, neckline_key)
    candidate = trim_sleeves_to_length(
        candidate,
        body.keypoints,
        garment_profile.sleeve_length,
        schp_atr,
        schp_lip,
    )

    if garment_profile.sleeve_length == "sleeveless":
        h, w = candidate.shape[:2]
        candidate[:, : int(w * 0.16)] = 0
        candidate[:, int(w * 0.84) :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    return fill_mask_holes(candidate)
