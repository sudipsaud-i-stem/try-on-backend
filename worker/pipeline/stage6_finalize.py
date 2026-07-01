from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.pipeline.types import PipelineContext


def _deblock(rgb: np.ndarray) -> np.ndarray:
    if not settings.PIPELINE_DEBLOCK:
        return rgb
    return cv2.bilateralFilter(rgb, d=5, sigmaColor=18, sigmaSpace=18)


def _upscale_final(image: Image.Image) -> Image.Image:
    factor = settings.PIPELINE_UPSCALE_FACTOR
    if factor <= 1.01:
        return image

    try:
        from worker.pipeline.optional_models import upscale_realesrgan

        upscaled = upscale_realesrgan(image, outscale=factor)
        if upscaled is not None:
            return upscaled
    except Exception:
        pass

    w, h = image.size
    return image.resize((int(w * factor), int(h * factor)), Image.Resampling.LANCZOS)


def run_stage6_finalize(ctx: PipelineContext) -> Image.Image:
    """Deblock JPEG artifacts and optional 2x upscale."""
    source = ctx.final or ctx.blended or ctx.vton_result
    if source is None:
        raise RuntimeError("stage6 requires a generated image")

    rgb = _deblock(np.array(source.convert("RGB")))
    image = Image.fromarray(rgb)
    image = _upscale_final(image)

    ctx.final = image
    ctx.log(f"stage6: finalize (upscale={settings.PIPELINE_UPSCALE_FACTOR}x)")
    return image
