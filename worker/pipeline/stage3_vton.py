from __future__ import annotations

from PIL import Image

from app.config import settings
from worker.catvton.image_utils import preprocess_garment_image
from worker.pipeline.types import PipelineContext


def run_stage3_vton(ctx: PipelineContext, infer_fn) -> Image.Image:
    """CatVTON generation using pipeline-produced person + mask."""
    if ctx.person is None or ctx.inpaint_mask is None:
        raise RuntimeError("stage3 requires person + inpaint mask")

    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
    garment = preprocess_garment_image(ctx.garment, target_size)

    from worker.postprocess import (
        apply_garment_color_preserve,
        composite_garment_only,
        tighten_mask,
    )
    from worker.preprocess import PreprocessInputs

    inference_mask = tighten_mask(ctx.inpaint_mask)
    ctx.inference_mask = inference_mask

    inputs: PreprocessInputs = {
        "person": ctx.person,
        "garment": garment,
        "mask": inference_mask,
    }
    output = infer_fn(inputs)

    # Lock original skin/body outside the shirt — CatVTON only changes the mask core.
    output = composite_garment_only(output, ctx.person, inference_mask)

    if settings.COLOR_PRESERVE_STRENGTH > 0:
        color_mask = tighten_mask(inference_mask, erode_px=settings.MASK_ERODE_PIXELS + 4)
        output = apply_garment_color_preserve(
            output,
            garment,
            color_mask,
            settings.COLOR_PRESERVE_STRENGTH,
        )

    ctx.vton_result = output
    ctx.log(
        f"stage3: CatVTON ({settings.OUTPUT_WIDTH}x{settings.OUTPUT_HEIGHT}, "
        f"steps={settings.INFERENCE_STEPS}, mask_erode={settings.MASK_ERODE_PIXELS}px)"
    )
    return output
