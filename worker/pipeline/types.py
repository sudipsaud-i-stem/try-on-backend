from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    is_blurry: bool
    is_low_res: bool
    upscaled: bool
    white_balanced: bool


@dataclass
class ParseReport:
    confidence: float
    used_fallback: bool
    cloth_type: str
    mask_coverage: float
    mask_coverage_person_bbox: float | None = None
    connectivity_component_count: int | None = None
    neckline_offset_from_chin_keypoint: int | None = None
    symmetry_ratio: float | None = None
    used_fallback_source: str | None = None
    garment_neckline_class: str | None = None
    garment_sleeve_class: str | None = None


@dataclass
class PipelineContext:
    """Holds images and metrics across all pipeline stages."""

    original_person: Image.Image
    garment: Image.Image
    cloth_type: str

    person: Image.Image | None = None
    person_white: Image.Image | None = None
    person_segment: Image.Image | None = None
    inpaint_mask: Image.Image | None = None
    alpha_matte: Image.Image | None = None
    vton_result: Image.Image | None = None
    blended: Image.Image | None = None
    final: Image.Image | None = None

    quality: QualityReport | None = None
    parse: ParseReport | None = None
    stage_logs: list[str] = field(default_factory=list)

    blend_base: Image.Image | None = None
    crop_box: tuple[int, int, int, int] | None = None
    inference_mask: Image.Image | None = None
    normalize_mode: str = "center_crop"  # center_crop | letterbox
    schp_atr: Image.Image | None = None
    schp_lip: Image.Image | None = None
    mask_diagnostics: dict | None = None

    def log(self, message: str) -> None:
        self.stage_logs.append(message)
        logger.info("pipeline | {}", message)

    def summary(self) -> dict[str, Any]:
        out = {
            "cloth_type": self.cloth_type,
            "quality": self.quality.__dict__ if self.quality else None,
            "parse": self.parse.__dict__ if self.parse else None,
            "stages": self.stage_logs,
        }
        if self.mask_diagnostics:
            out["mask_diagnostics"] = self.mask_diagnostics
        return out
