from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.config import settings
from worker.pipeline.types import PipelineContext


def _matte_from_mask(mask: Image.Image) -> Image.Image:
    """Soft alpha matte from inpaint mask (fallback when BiRefNet unavailable)."""
    arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    # Feather edges for hair-like softness approximation.
    matte = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    return matte.filter(ImageFilter.GaussianBlur(radius=settings.PIPELINE_MATTING_BLUR))


def run_stage2_matting(ctx: PipelineContext) -> Image.Image:
    """BiRefNet alpha matte with mask-based fallback."""
    if ctx.person is None or ctx.inpaint_mask is None:
        raise RuntimeError("stage2 requires person + inpaint mask")

    alpha: Image.Image | None = None
    if settings.ENABLE_BIREFNET:
        try:
            from worker.pipeline.optional_models import generate_birefnet_matte

            alpha = generate_birefnet_matte(ctx.person)
            if alpha is not None:
                ctx.log("stage2: BiRefNet matte")
        except Exception as exc:
            ctx.log(f"stage2: BiRefNet unavailable ({exc}) — using mask matte")

    if alpha is None:
        alpha = _matte_from_mask(ctx.inpaint_mask)
        ctx.log("stage2: soft mask matte (fallback)")

    ctx.alpha_matte = alpha
    return alpha
