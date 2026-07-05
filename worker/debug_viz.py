from __future__ import annotations

import numpy as np
from PIL import Image

# Human-readable labels for debug artifacts (order matches save sequence).
DEBUG_STEP_LABELS: dict[str, str] = {
    "00_original": "Original upload",
    "01_person_normalized": "Normalized person (768×1024 workspace)",
    "02_inpaint_mask": "Garment inpaint mask",
    "02b_inference_mask": "Tightened inference mask (CatVTON)",
    "02c_mask_overlay": "Mask overlay on person (red = swap region)",
    "02d_schp_parsing": "SCHP body parsing (colored labels)",
    "03_garment": "Garment reference",
    "04_vton": "CatVTON raw output",
    "05_blended": "After stage 4 composite",
    "06_final": "Final result",
    "pipeline_summary": "Pipeline metrics (JSON)",
    "pipeline_logs": "Stage log lines (text)",
}

_SCHP_ATR_COLORS = {
    0: (30, 30, 30),
    4: (220, 20, 60),
    5: (255, 105, 180),
    6: (65, 105, 225),
    7: (148, 0, 211),
    11: (255, 218, 185),
    12: (100, 149, 237),
    13: (100, 149, 237),
    14: (255, 165, 0),
    15: (255, 165, 0),
}


def mask_overlay_rgb(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.48,
) -> Image.Image:
    """Visualize swap mask as a red tint on the person image."""
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    m = np.array(mask.convert("L").resize(image.size, Image.Resampling.LANCZOS), dtype=np.float32)
    m = np.clip(m / 255.0, 0.0, 1.0)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    out = rgb * (1.0 - m[..., None] * alpha) + tint * (m[..., None] * alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def colorize_schp_atr(schp_atr: Image.Image) -> Image.Image:
    """Quick SCHP ATR label visualization for debugging bad masks."""
    labels = np.array(schp_atr.convert("L"))
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in _SCHP_ATR_COLORS.items():
        rgb[labels == label_id] = color
    unmapped = ~np.isin(labels, list(_SCHP_ATR_COLORS.keys()))
    rgb[unmapped] = (180, 180, 180)
    return Image.fromarray(rgb)
