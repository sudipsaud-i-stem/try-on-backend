from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of
from worker.pose_body import BodyKeypoints


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
    """Protect forearms/hands using outer lateral bands (works when arms are crossed)."""
    h, w = schp_atr.shape[:2]
    protect = np.zeros((h, w), dtype=np.uint8)
    for arm_label, outer_side in (("Left-arm", "left"), ("Right-arm", "right")):
        arm = (
            part_mask_of(arm_label, schp_lip, LIP_MAPPING)
            | part_mask_of(arm_label, schp_atr, ATR_MAPPING)
        )
        if not arm.any():
            continue
        ys, xs = np.where(arm > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        y_cut = int(y0 + (y1 - y0) * 0.55)
        if outer_side == "left":
            x_cut = int(x0 + (x1 - x0) * 0.45)
            lateral = (arm > 0) & (np.arange(w)[None, :] <= x_cut)
        else:
            x_cut = int(x0 + (x1 - x0) * 0.55)
            lateral = (arm > 0) & (np.arange(w)[None, :] >= x_cut)
        distal = (arm > 0) & (np.arange(h)[:, None] >= y_cut)
        protect[(lateral | distal) & (arm > 0)] = 255
    return protect


def _arm_union(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return (
        part_mask_of("Left-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Right-arm", schp_lip, LIP_MAPPING)
        | part_mask_of("Left-arm", schp_atr, ATR_MAPPING)
        | part_mask_of("Right-arm", schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def bridge_disconnected_sleeves(
    garment_mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    max_gap_px: int = 45,
) -> np.ndarray:
    """Reconnect sleeve islands disconnected from torso by raised/bent arms."""
    binary = (garment_mask > 127).astype(np.uint8)
    if binary.max() == 0:
        return garment_mask

    arm_mask = _arm_union(schp_atr, schp_lip)
    if not arm_mask.any():
        return garment_mask

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return garment_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = int(np.argmax(areas)) + 1
    main_component = (labels == main_label).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap_px, max_gap_px))
    main_dilated = cv2.dilate(main_component * 255, kernel, iterations=1)

    bridged = main_component.copy() * 255
    for label in range(1, n):
        if label == main_label:
            continue
        component = (labels == label).astype(np.uint8)
        overlaps_arm = bool((component & arm_mask).any())
        near_main = bool((component.astype(bool) & (main_dilated > 0)).any())
        if overlaps_arm and near_main:
            bridged = np.maximum(bridged, component * 255)
            gap_fill = cv2.dilate(component * 255, kernel, iterations=1)
            gap_fill = gap_fill & main_dilated & (arm_mask * 255)
            bridged = np.maximum(bridged, gap_fill)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, close_kernel)
    return bridged


def fill_mask_holes(mask: np.ndarray, max_hole_ratio: float = 0.10) -> np.ndarray:
    """Fill small enclosed holes only — avoid flooding letterbox/desk regions."""
    binary = (mask > 127).astype(np.uint8)
    if binary.max() == 0:
        return mask
    filled = binary.copy()
    flood = filled.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0) & (binary == 0)
    if not holes.any():
        return mask
    hole_limit = int(h * w * max_hole_ratio)
    n, labels = cv2.connectedComponents(holes.astype(np.uint8))
    for label in range(1, n):
        component = labels == label
        if int(component.sum()) <= hole_limit:
            filled[component] = 1
    return (filled * 255).astype(np.uint8)


def _schp_person_silhouette(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    parts = [
        "Face", "Hair", "Hat", "Upper-clothes", "Dress", "Coat",
        "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Pants", "Skirt", "Jumpsuits",
    ]
    person = (
        part_mask_of(parts, schp_lip, LIP_MAPPING)
        | part_mask_of(parts, schp_atr, ATR_MAPPING)
    ).astype(np.uint8)
    if person.max() == 0:
        return person
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    person = cv2.morphologyEx(person * 255, cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(person, kernel, iterations=1)


def clip_mask_to_person(
    mask: np.ndarray,
    person_image: Image.Image,
    schp_atr: np.ndarray | None = None,
    schp_lip: np.ndarray | None = None,
) -> np.ndarray:
    from worker.postprocess import grabcut_person_mask

    if schp_atr is not None and schp_lip is not None:
        person = _schp_person_silhouette(schp_atr, schp_lip)
    else:
        person = np.zeros(mask.shape, dtype=np.uint8)
    if person.max() == 0:
        person = grabcut_person_mask(person_image)
    if person.max() == 0:
        return mask
    clipped = np.array(mask, dtype=np.uint8)
    clipped[person <= 127] = 0
    return clipped


def keep_torso_component(
    mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
) -> np.ndarray:
    mask = bridge_disconnected_sleeves(mask, schp_atr, schp_lip)

    binary = (mask > 127).astype(np.uint8)
    if binary.max() == 0:
        return mask
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return mask

    face = _face_hair_protect(schp_atr, schp_lip)
    if face.any():
        fy, fx = np.where(face > 0)
        anchor = (float(fx.mean()), float(fy.mean()))
    else:
        anchor = (binary.shape[1] / 2.0, binary.shape[0] / 3.0)

    best_label = 1
    best_score = -1.0
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        dist = ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
        score = area / (1.0 + dist * 0.35)
        if score > best_score:
            best_score = score
            best_label = label

    return (labels == best_label).astype(np.uint8) * 255


def mask_is_boxy(mask: np.ndarray, coverage_threshold: float = 0.20) -> bool:
    binary = (mask > 127).astype(np.uint8)
    coverage = float(binary.mean())
    if coverage < coverage_threshold:
        return False
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
    rect = cv2.minAreaRect(contour)
    box_area = max(rect[1][0] * rect[1][1], 1.0)
    return (area / box_area) > 0.88 and coverage > 0.22


def normalize_neckline(
    garment_mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    neckline_type: str = "auto",
    keypoints: BodyKeypoints | None = None,
) -> np.ndarray:
    if neckline_type in {"auto", "none"}:
        return garment_mask

    face = _face_hair_protect(schp_atr, schp_lip)
    if not face.any():
        return garment_mask

    binary = (garment_mask > 127).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if ys.size == 0:
        return garment_mask

    h, w = garment_mask.shape[:2]
    fy, fx = np.where(face > 0)
    chin_y = int(fy.max())
    face_cx = float(fx.mean())
    face_width = max(float(fx.max() - fx.min()), 1.0)
    torso_x0, torso_x1 = int(xs.min()), int(xs.max())
    shoulder_width = max(torso_x1 - torso_x0, int(face_width * 2.0))

    result = binary.copy() * 255
    result[face > 0] = 0

    if neckline_type in {"crew", "collar", "hoodie"}:
        collar_y = chin_y + int(0.035 * h)
        if keypoints and keypoints.left_shoulder and keypoints.right_shoulder:
            collar_y = int(
                (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) * 0.5 * h
            )
        neck_half = max(int(shoulder_width * 0.09), int(face_width * 0.32))
        depth = max(int(shoulder_width * 0.05), 3)
        cutout = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            cutout,
            (int(face_cx), int(collar_y)),
            (neck_half, depth),
            0, 0, 180, 255, -1,
        )
        result[cutout > 0] = 0
        return result

    shoulder_y = int(np.percentile(ys, 8))
    neck_half_width = max(int(shoulder_width * 0.14), int(face_width * 0.45))
    neck_top = min(chin_y + int(0.04 * h), shoulder_y + int(shoulder_width * 0.04))
    depth_ratio = {"scoop": 0.14, "vneck": 0.24}.get(neckline_type, 0.12)
    neck_depth = max(int(shoulder_width * depth_ratio), 4)

    cutout = np.zeros((h, w), dtype=np.uint8)
    if neckline_type == "vneck":
        pts = np.array(
            [
                [face_cx - neck_half_width, neck_top],
                [face_cx + neck_half_width, neck_top],
                [face_cx, neck_top + neck_depth],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(cutout, [pts], 255)
    else:
        cv2.ellipse(
            cutout,
            (int(face_cx), int(neck_top)),
            (neck_half_width, neck_depth),
            0, 0, 180, 255, -1,
        )

    result[cutout > 0] = 0
    result[face > 0] = 0
    return result


def expand_mask_to_arms(
    garment_mask: np.ndarray,
    keypoints: BodyKeypoints,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    sleeve_length: str = "long",
) -> np.ndarray:
    """Add YOLO/SCHP arm corridors so raised/bent sleeves stay in the inpaint region."""
    if sleeve_length == "sleeveless":
        return garment_mask

    h, w = garment_mask.shape[:2]
    result = np.array(garment_mask, dtype=np.uint8)
    radius = max(10, int(min(h, w) * 0.032))
    corridor = np.zeros((h, w), dtype=np.uint8)

    limbs = (
        (keypoints.left_shoulder, keypoints.left_elbow, keypoints.left_wrist),
        (keypoints.right_shoulder, keypoints.right_elbow, keypoints.right_wrist),
    )
    for shoulder, elbow, wrist in limbs:
        if shoulder is None:
            continue
        end = wrist or elbow or shoulder
        if sleeve_length == "short" and elbow is not None:
            end = elbow
        elif sleeve_length == "three_quarter" and elbow is not None and wrist is not None:
            end = (
                elbow[0] * 0.35 + wrist[0] * 0.65,
                elbow[1] * 0.35 + wrist[1] * 0.65,
            )
        x0, y0 = int(shoulder[0] * w), int(shoulder[1] * h)
        x1, y1 = int(end[0] * w), int(end[1] * h)
        cv2.line(corridor, (x0, y0), (x1, y1), 255, thickness=max(radius * 2, 14))

    result = np.maximum(result, corridor)
    arm_schp = (_arm_union(schp_atr, schp_lip).astype(np.uint8) * 255)
    result = np.maximum(result, cv2.bitwise_and(arm_schp, corridor))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)

    face = _face_hair_protect(schp_atr, schp_lip)
    result[face > 0] = 0
    return result


def trim_sleeves_to_length(
    garment_mask: np.ndarray,
    keypoints: BodyKeypoints,
    sleeve_length: str,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
) -> np.ndarray:
    """Remove forearm regions from mask when target garment has shorter sleeves."""
    if sleeve_length in {"long", "sleeveless"}:
        if sleeve_length == "sleeveless":
            h, w = garment_mask.shape[:2]
            result = np.array(garment_mask, dtype=np.uint8)
            arm = _arm_union(schp_atr, schp_lip)
            result[arm > 0] = 0
            face = _face_hair_protect(schp_atr, schp_lip)
            result[face > 0] = 0
            return result
        return garment_mask

    h, w = garment_mask.shape[:2]
    result = np.array(garment_mask, dtype=np.uint8)
    arm = _arm_union(schp_atr, schp_lip)

    for shoulder, elbow, wrist in (
        (keypoints.left_shoulder, keypoints.left_elbow, keypoints.left_wrist),
        (keypoints.right_shoulder, keypoints.right_elbow, keypoints.right_wrist),
    ):
        if shoulder is None:
            continue
        cut_y = h
        if sleeve_length == "short" and elbow is not None:
            cut_y = int(elbow[1] * h)
        elif sleeve_length == "three_quarter" and elbow is not None and wrist is not None:
            cut_y = int((elbow[1] * 0.35 + wrist[1] * 0.65) * h)
        if cut_y >= h:
            continue
        forearm = np.zeros((h, w), dtype=np.uint8)
        forearm[cut_y:, :] = 255
        trim_region = forearm & (arm * 255)
        result[trim_region > 0] = 0

    face = _face_hair_protect(schp_atr, schp_lip)
    result[face > 0] = 0
    return result


def rebuild_garment_mask_from_schp(
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    person_image: Image.Image,
    neckline_type: str = "auto",
    keypoints: BodyKeypoints | None = None,
    sleeve_length: str = "long",
) -> np.ndarray:
    upper = _schp_upper_garment(schp_atr, schp_lip).astype(np.uint8) * 255
    upper[_inpaint_exclude_mask(schp_atr, schp_lip) > 0] = 0
    upper = clip_mask_to_person(upper, person_image, schp_atr, schp_lip)
    upper = keep_torso_component(upper, schp_atr, schp_lip)
    upper = normalize_neckline(upper, schp_atr, schp_lip, neckline_type, keypoints)
    if keypoints is not None:
        upper = trim_sleeves_to_length(upper, keypoints, sleeve_length, schp_atr, schp_lip)
        if sleeve_length != "sleeveless":
            upper = expand_mask_to_arms(upper, keypoints, schp_atr, schp_lip, sleeve_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    upper = cv2.morphologyEx(upper, cv2.MORPH_CLOSE, kernel)
    return fill_mask_holes(upper)


def _full_arm_protect(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return _arm_union(schp_atr, schp_lip)


def _schp_upper_garment(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return (
        part_mask_of(["Upper-clothes", "Dress", "Coat", "Jumpsuits"], schp_lip, LIP_MAPPING)
        | part_mask_of(["Upper-clothes", "Dress", "Coat"], schp_atr, ATR_MAPPING)
    ).astype(np.uint8)


def build_identity_protect_mask(
    schp_atr: Image.Image,
    schp_lip: Image.Image,
    garment_mask: np.ndarray | None = None,
) -> np.ndarray:
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    protect = _face_hair_protect(atr, lip) | _arm_distal_protect(atr, lip)
    protect = (protect > 0).astype(np.float32)
    if garment_mask is not None:
        g = np.array(garment_mask.convert("L") if isinstance(garment_mask, Image.Image) else garment_mask)
        if g.shape[:2] == protect.shape[:2]:
            g = (g > 127).astype(np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            g = cv2.dilate(g, k, iterations=1)
            protect = protect * (1.0 - g.astype(np.float32))
    return np.clip(protect, 0.0, 1.0)


def _inpaint_exclude_mask(schp_atr: np.ndarray, schp_lip: np.ndarray) -> np.ndarray:
    return _face_hair_protect(schp_atr, schp_lip)


def ensure_minimum_garment_coverage(
    mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    min_coverage: float = 0.14,
) -> np.ndarray:
    if float((mask > 127).mean()) >= min_coverage:
        return mask

    upper = _schp_upper_garment(schp_atr, schp_lip).copy()
    exclude = _inpaint_exclude_mask(schp_atr, schp_lip)
    upper[exclude > 0] = 0
    merged = np.maximum(mask, upper * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    return merged


def clamp_mask_above_collar(
    mask: np.ndarray,
    keypoints: BodyKeypoints | None,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
) -> np.ndarray:
    """Strip mask pixels above the collarbone — fixes GrabCut/segmentation including the head."""
    h, w = mask.shape[:2]
    collar_y: int | None = None

    if keypoints and keypoints.left_shoulder and keypoints.right_shoulder:
        collar_y = int(
            (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) * 0.5 * h + 0.025 * h
        )
    else:
        face = _face_hair_protect(schp_atr, schp_lip)
        if face.any():
            collar_y = int(np.where(face > 0)[0].max()) + int(0.045 * h)

    if collar_y is None:
        return mask

    result = np.array(mask, dtype=np.uint8)
    result[: max(0, collar_y), :] = 0
    face = _face_hair_protect(schp_atr, schp_lip)
    result[face > 0] = 0
    return result


def _collar_reference_y(
    h: int,
    keypoints: BodyKeypoints | None,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
) -> int | None:
    if keypoints and keypoints.left_shoulder and keypoints.right_shoulder:
        return int(
            (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) * 0.5 * h + 0.025 * h
        )
    face = _face_hair_protect(schp_atr, schp_lip)
    if face.any():
        return int(np.where(face > 0)[0].max()) + int(0.04 * h)
    return None


def mask_coverage_in_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float((crop > 127).mean())


def mask_shape_is_valid(
    mask: np.ndarray,
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    person_bbox: tuple[int, int, int, int] | None = None,
    keypoints=None,
    min_coverage: float = 0.08,
    sleeve_length: str = "long",
) -> tuple[bool, dict]:
    binary = (mask > 127).astype(np.uint8)
    if person_bbox is not None:
        x0, y0, x1, y1 = person_bbox
        bbox_area = max((x1 - x0) * (y1 - y0), 1)
        coverage = float(binary[y0:y1, x0:x1].sum()) / float(bbox_area)
    else:
        bbox_area = binary.size
        coverage = float(binary.sum()) / float(bbox_area)

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    significant = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.01 * binary.size)
    max_components = 1 if sleeve_length == "sleeveless" else 3

    face = _face_hair_protect(schp_atr, schp_lip)
    neckline_ok = True
    neckline_offset = None
    if binary.any():
        top_y = int(np.where(binary > 0)[0].min())
        h = binary.shape[0]
        collar_y = _collar_reference_y(h, keypoints, schp_atr, schp_lip)

        if collar_y is not None:
            neckline_offset = top_y - collar_y
            margin_above = int(0.06 * h) if (keypoints and getattr(keypoints, "side_pose", False)) else int(0.04 * h)
            margin_below = int(0.18 * h) if (keypoints and getattr(keypoints, "side_pose", False)) else int(0.14 * h)
            neckline_ok = (collar_y - margin_above) <= top_y <= (collar_y + margin_below)
        elif face.any():
            fy = np.where(face > 0)[0]
            chin_y = int(fy.max())
            neckline_offset = top_y - chin_y
            neckline_ok = chin_y - int(0.02 * h) <= top_y <= chin_y + int(0.14 * h)
        else:
            neckline_ok = top_y >= int(0.10 * h)

    symmetry_ok = True
    symmetry_ratio = 1.0
    if binary.any() and not (keypoints and getattr(keypoints, "side_pose", False)):
        ys, xs = np.where(binary > 0)
        cx = float(xs.mean())
        left_w = float((xs[xs < cx]).max() - cx) if np.any(xs < cx) else 0.0
        right_w = float(cx - (xs[xs > cx]).min()) if np.any(xs > cx) else 0.0
        if left_w > 0 and right_w > 0:
            symmetry_ratio = max(left_w, right_w) / max(min(left_w, right_w), 1.0)
            symmetry_ok = symmetry_ratio <= 3.0

    diagnostics = {
        "mask_coverage_person_bbox": coverage,
        "connectivity_component_count": significant,
        "neckline_offset_from_chin_keypoint": neckline_offset,
        "symmetry_ratio": symmetry_ratio,
        "neckline_ok": neckline_ok,
        "symmetry_ok": symmetry_ok,
    }
    is_valid = (
        coverage >= min_coverage
        and significant <= max_components
        and neckline_ok
        and symmetry_ok
    )
    return is_valid, diagnostics


def compute_mask_confidence(diagnostics: dict, min_coverage: float = 0.08) -> float:
    coverage = diagnostics.get("mask_coverage_person_bbox", diagnostics.get("coverage", 0.0))
    if coverage < min_coverage or not diagnostics.get("neckline_ok", False):
        return 0.0
    if diagnostics.get("connectivity_component_count", 99) > 3:
        return 0.0
    base = min(1.0, max(0.0, (coverage - min_coverage) / max(0.20 - min_coverage, 1e-6)))
    penalty = 0.5 if not diagnostics.get("symmetry_ok", True) else 1.0
    return float(base * penalty)


def rejection_message(diagnostics: dict) -> tuple[str, str]:
    coverage = diagnostics.get("mask_coverage_person_bbox", 0.0)
    components = diagnostics.get("connectivity_component_count", 0)
    if coverage < 0.08:
        return (
            "mask_low_coverage",
            "We couldn't find enough of the garment area in this photo. "
            "Try a photo with the torso clearly visible.",
        )
    if components > 3:
        return (
            "mask_disconnected",
            "The clothing region we detected was split into disconnected areas. "
            "Try a clearer, more evenly lit photo.",
        )
    if not diagnostics.get("neckline_ok", True):
        return (
            "mask_neckline",
            "We couldn't reliably locate the neckline in this photo. "
            "Try a front-facing photo with shoulders visible.",
        )
    if not diagnostics.get("symmetry_ok", True):
        return (
            "mask_asymmetric",
            "We couldn't fully detect a raised/bent arm. "
            "Try a pose with arms closer to the body, or crop to show the torso only.",
        )
    return ("mask_invalid", "Unable to build a valid garment mask for this photo.")


def refine_inpaint_mask(
    mask: Image.Image,
    schp_atr: Image.Image,
    schp_lip: Image.Image,
    cloth_type: str,
    neckline_type: str = "auto",
    keypoints: BodyKeypoints | None = None,
    sleeve_length: str = "long",
) -> Image.Image:
    atr = np.array(schp_atr)
    lip = np.array(schp_lip)
    arr = np.array(mask.convert("L"))

    exclude = _inpaint_exclude_mask(atr, lip)
    arr[exclude > 0] = 0

    if cloth_type in {"sleeveless", "tank", "tank_top"}:
        h, w = arr.shape
        arr[:, : int(w * 0.16)] = 0
        arr[:, int(w * 0.84) :] = 0
    else:
        arr = bridge_disconnected_sleeves(arr, atr, lip)

    arr = ensure_minimum_garment_coverage(arr, atr, lip)
    arr = normalize_neckline(arr, atr, lip, neckline_type, keypoints)
    if keypoints is not None:
        arr = trim_sleeves_to_length(arr, keypoints, sleeve_length, atr, lip)
        if sleeve_length != "sleeveless":
            arr = expand_mask_to_arms(arr, keypoints, atr, lip, sleeve_length)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    arr = fill_mask_holes(arr)
    return Image.fromarray(arr, mode="L")
