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
    """Union of SCHP body-part labels — excludes desk/laptop background noise."""
    parts = [
        "Face",
        "Hair",
        "Hat",
        "Upper-clothes",
        "Dress",
        "Coat",
        "Left-arm",
        "Right-arm",
        "Left-leg",
        "Right-leg",
        "Pants",
        "Skirt",
        "Jumpsuits",
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
    """Remove laptop/desk/background blobs outside the person silhouette."""
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
    """Keep the garment blob anchored on the face/torso, drop stray SCHP noise."""
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

    kept = (labels == best_label).astype(np.uint8) * 255
    return kept


def mask_is_boxy(mask: np.ndarray, coverage_threshold: float = 0.20) -> bool:
    """Detect rectangular fallback / over-eroded masks that break try-on."""
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
    rectangularity = area / box_area
    return rectangularity > 0.88 and coverage > 0.22


def rebuild_garment_mask_from_schp(
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    person_image: Image.Image,
) -> np.ndarray:
    """Rebuild a person-shaped shirt mask when AutoMasker output is a box."""
    upper = _schp_upper_garment(schp_atr, schp_lip).astype(np.uint8) * 255
    upper[_inpaint_exclude_mask(schp_atr, schp_lip) > 0] = 0
    upper = clip_mask_to_person(upper, person_image, schp_atr, schp_lip)
    upper = keep_torso_component(upper, schp_atr, schp_lip)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    upper = cv2.morphologyEx(upper, cv2.MORPH_CLOSE, kernel)
    return fill_mask_holes(upper)


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
    garment_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paste back face, hair, and visible forearms — skip regions where garment was swapped."""
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
    """Exclude face/hair only — crossed arms over the chest stay in the shirt mask."""
    return _face_hair_protect(schp_atr, schp_lip)


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
        merged = np.maximum(mask, upper * 255)

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
    arr = fill_mask_holes(arr)
    return Image.fromarray(arr, mode="L")
