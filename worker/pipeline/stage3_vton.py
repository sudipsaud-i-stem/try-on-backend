from __future__ import annotations

import torch
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

    from worker.preprocess import PreprocessInputs

    inputs: PreprocessInputs = {
        "person": ctx.person,
        "garment": garment,
        "mask": ctx.inpaint_mask,
    }
    output = infer_fn(inputs)
    ctx.vton_result = output
    ctx.log(f"stage3: CatVTON ({settings.OUTPUT_WIDTH}x{settings.OUTPUT_HEIGHT}, steps={settings.INFERENCE_STEPS})")
    return output
