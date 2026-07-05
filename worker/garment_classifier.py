from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class GarmentProfile:
    category: str = "top"  # top | bottom | dress | outerwear
    neckline: str = "crew"  # crew | vneck | scoop | collar | hoodie | off_shoulder | none
    sleeve_length: str = "long"  # sleeveless | short | three_quarter | long
    fit: str = "regular"  # fitted | regular | oversized

    def to_dict(self) -> dict:
        return asdict(self)


def _upper_region(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: int(h * 0.42), :]


def _collar_openness(upper: np.ndarray) -> float:
    """Higher score => more V/open collar in the flat-lay collar band."""
    h, w = upper.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    band = upper[int(h * 0.08) : int(h * 0.35), int(w * 0.35) : int(w * 0.65)]
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center_col = binary.shape[1] // 2
    top_third = binary[: max(1, binary.shape[0] // 3), :]
    center_gap = float((top_third[:, center_col - 2 : center_col + 3] < 128).mean())
    return center_gap


def _silhouette_width_ratio(arr: np.ndarray) -> float:
    """Fraction of image width covered by garment at shoulder height (flat-lay)."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float(fg.mean()) > 200:
        fg = cv2.bitwise_not(fg)
    h, w = fg.shape[:2]
    band = fg[int(h * 0.18) : int(h * 0.42), :]
    if band.size == 0:
        return 0.5
    return float((band.mean(axis=0) > 127).sum()) / max(w, 1)


def _sleeve_extent(arr: np.ndarray) -> tuple[float, float]:
    """Return left/right sleeve width ratios relative to image width."""
    h, w = arr.shape[:2]
    upper = arr[: int(h * 0.55), :]
    gray = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    col_energy = edges.sum(axis=0).astype(np.float32)
    if col_energy.max() <= 0:
        return 0.0, 0.0
    col_energy /= col_energy.max()
    left = float(col_energy[: int(w * 0.22)].mean())
    right = float(col_energy[int(w * 0.78) :].mean())
    return left, right


def classify_garment(
    garment: Image.Image,
    cloth_type_hint: str = "upper",
) -> GarmentProfile:
    """
    Lightweight rule-based classifier on the target (flat-lay) garment image.
    Drives neckline template + sleeve cutoff in mask synthesis.
    """
    arr = np.array(garment.convert("RGB"))
    h, w = arr.shape[:2]
    aspect = w / max(h, 1)

    hint = cloth_type_hint.lower()
    category = "top"
    if hint in {"lower", "pants", "trousers", "skirt"}:
        category = "bottom"
    elif hint in {"dress", "overall", "jumpsuit"}:
        category = "dress"
    elif hint in {"outer", "jacket", "hoodie", "coat", "bomber", "blazer"}:
        category = "outerwear"
    elif hint in {"sleeveless", "tank", "tank_top"}:
        category = "top"

    upper = _upper_region(arr)
    openness = _collar_openness(upper)
    left_sleeve, right_sleeve = _sleeve_extent(arr)
    sleeve_signal = max(left_sleeve, right_sleeve)
    shoulder_width_ratio = _silhouette_width_ratio(arr)

    neckline = "crew"
    if category == "outerwear" or hint in {"hoodie", "coat", "jacket"}:
        neckline = "hoodie"
    elif openness > 0.55:
        neckline = "vneck"
    elif openness > 0.38:
        neckline = "scoop"
    elif openness > 0.22:
        neckline = "collar"
    elif hint in {"sleeveless", "tank", "tank_top"}:
        neckline = "none"

    # Never classify as sleeveless from weak edge signal alone — dark flat-lays
    # (e.g. black sweatshirt on gray) have low Canny energy but clear wide silhouette.
    if hint in {"sleeveless", "tank", "tank_top"}:
        sleeve_length = "sleeveless"
    elif shoulder_width_ratio >= 0.58 or category in {"outerwear"} or hint in {"hoodie", "coat", "jacket"}:
        sleeve_length = "long"
    elif shoulder_width_ratio >= 0.46:
        sleeve_length = "short"
    elif shoulder_width_ratio >= 0.38:
        sleeve_length = "three_quarter"
    elif sleeve_signal >= 0.12:
        sleeve_length = "long"
    else:
        sleeve_length = "long" if hint in {"upper", "shirt", "tshirt", "tee", "outer"} else "short"

    fit = "regular"
    torso_w_ratio = 0.55
    if category == "top":
        gray = cv2.cvtColor(_upper_region(arr), cv2.COLOR_RGB2GRAY)
        ys, xs = np.where(gray < 245)
        if len(xs) > 50:
            torso_w_ratio = (xs.max() - xs.min()) / max(w, 1)
    if torso_w_ratio > 0.72:
        fit = "oversized"
    elif torso_w_ratio < 0.48:
        fit = "fitted"

    return GarmentProfile(
        category=category,
        neckline=neckline,
        sleeve_length=sleeve_length,
        fit=fit,
    )
