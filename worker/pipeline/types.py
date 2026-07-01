from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
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


@dataclass
class PipelineContext:
    """Holds images and metrics across all pipeline stages."""

    original_person: Image.Image
    garment: Image.Image
    cloth_type: str

    person: Image.Image | None = None
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

    def log(self, message: str) -> None:
        self.stage_logs.append(message)

    def summary(self) -> dict[str, Any]:
        return {
            "cloth_type": self.cloth_type,
            "quality": self.quality.__dict__ if self.quality else None,
            "parse": self.parse.__dict__ if self.parse else None,
            "stages": self.stage_logs,
        }
