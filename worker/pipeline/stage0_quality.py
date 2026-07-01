from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.pipeline.types import PipelineContext, QualityReport


def _laplacian_blur_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _auto_white_balance(rgb: np.ndarray) -> np.ndarray:
    """Gray-world white balance for phone photos under mixed lighting."""
    result = rgb.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    scale = gray_mean / np.clip(channel_means, 1.0, None)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def _upscale_image(image: Image.Image, scale: float) -> Image.Image:
    if scale <= 1.01:
        return image
    w, h = image.size
    new_size = (int(w * scale), int(h * scale))
    try:
        from worker.pipeline.optional_models import upscale_realesrgan

        upscaled = upscale_realesrgan(image, outscale=scale)
        if upscaled is not None:
            return upscaled
    except Exception:
        pass
    return image.resize(new_size, Image.Resampling.LANCZOS)


def run_stage0_quality(ctx: PipelineContext) -> Image.Image:
    """Normalize noisy phone photos before parsing and VTON."""
    rgb = np.array(ctx.original_person.convert("RGB"))
    h, w = rgb.shape[:2]
    short_edge = min(h, w)
    blur_score = _laplacian_blur_score(rgb)

    is_blurry = blur_score < settings.PIPELINE_BLUR_THRESHOLD
    is_low_res = short_edge < settings.PIPELINE_MIN_SHORT_EDGE
    upscaled = False
    white_balanced = False

    if settings.PIPELINE_AUTO_WHITE_BALANCE:
        rgb = _auto_white_balance(rgb)
        white_balanced = True

    image = Image.fromarray(rgb)

    if is_low_res or (is_blurry and settings.PIPELINE_PRE_UPSCALE):
        target = settings.PIPELINE_MIN_SHORT_EDGE
        scale = max(1.0, target / short_edge)
        if scale > 1.05:
            image = _upscale_image(image, scale)
            upscaled = True
            ctx.log(f"stage0: upscaled {scale:.2f}x (blur={blur_score:.1f}, short_edge={short_edge})")
        else:
            ctx.log(f"stage0: skip upscale (blur={blur_score:.1f}, short_edge={short_edge})")
    else:
        ctx.log(f"stage0: quality ok (blur={blur_score:.1f}, {w}x{h})")

    ctx.quality = QualityReport(
        width=image.width,
        height=image.height,
        blur_score=blur_score,
        is_blurry=is_blurry,
        is_low_res=is_low_res,
        upscaled=upscaled,
        white_balanced=white_balanced,
    )
    ctx.person = image
    return image
