from __future__ import annotations

from PIL import Image, ImageFilter

from app.config import settings
from worker.person_segment import build_person_matte, composite_on_white
from worker.pipeline.types import PipelineContext


def _matte_from_mask(mask: Image.Image) -> Image.Image:
    """Soft alpha from garment inpaint mask (last-resort fallback)."""
    matte = mask.convert("L")
    return matte.filter(ImageFilter.GaussianBlur(radius=settings.PIPELINE_MATTING_BLUR))


def run_stage2_matting(ctx: PipelineContext) -> Image.Image:
    """Person segmentation + optional white-background prep for CatVTON."""
    if ctx.person is None or ctx.inpaint_mask is None:
        raise RuntimeError("stage2 requires person + inpaint mask")

    alpha: Image.Image | None = None

    # Full-body person matte (SCHP + GrabCut) — works on messy backgrounds without BiRefNet.
    person_matte = build_person_matte(
        ctx.person,
        ctx.schp_atr,
        ctx.schp_lip,
        feather=settings.PIPELINE_PERSON_MATTING_FEATHER,
    )
    ctx.person_segment = person_matte
    ctx.log("stage2: SCHP+GrabCut person matte")

    if settings.ENABLE_BIREFNET:
        try:
            from worker.pipeline.optional_models import generate_birefnet_matte

            biref = generate_birefnet_matte(ctx.person)
            if biref is not None:
                alpha = biref
                ctx.log("stage2: BiRefNet matte merged with person segment")
        except Exception as exc:
            ctx.log(f"stage2: BiRefNet unavailable ({exc})")

    if alpha is None:
        alpha = person_matte
    else:
        from PIL import Image as PILImage
        import numpy as np

        a = np.array(alpha.convert("L"), dtype=np.float32) / 255.0
        p = np.array(person_matte.convert("L"), dtype=np.float32) / 255.0
        merged = np.clip(np.maximum(a, p * 0.85), 0, 1)
        alpha = PILImage.fromarray((merged * 255).astype("uint8"), mode="L")

    ctx.alpha_matte = alpha

    if settings.PIPELINE_WHITE_BG_INFERENCE:
        ctx.person_white = composite_on_white(ctx.person, alpha)
        ctx.log("stage2: person isolated on white background for CatVTON")

    return alpha
