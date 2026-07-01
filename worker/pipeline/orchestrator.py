from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from app.config import settings
from worker.pipeline import (
    stage0_quality,
    stage1_parsing,
    stage2_matting,
    stage3_vton,
    stage4_blend,
    stage5_face,
    stage6_finalize,
)
from worker.pipeline.types import PipelineContext


class TryOnOrchestrator:
    """
    7-stage HUBA streetwear pipeline for noisy real-world phone photos.

    Stages:
      0 quality triage, 1 SCHP+fallback parsing, 2 matting, 3 CatVTON,
      4 Poisson blend, 5 GFPGAN, 6 deblock+upscale.
    """

    def __init__(self, infer_fn) -> None:
        self._infer_fn = infer_fn
        self._models_warmed = False

    def preload_models(self) -> None:
        """Load heavy models once at startup (CatVTON + optional extras)."""
        if self._models_warmed:
            return

        from worker.catvton.mask_service import _load_automasker
        from worker.inference import _load_pipeline

        _load_pipeline()
        _load_automasker()

        if settings.ENABLE_BIREFNET:
            try:
                from worker.pipeline.optional_models import preload_birefnet

                preload_birefnet()
            except Exception as exc:
                logger.warning("BiRefNet preload skipped: {}", exc)

        if settings.ENABLE_GFPGAN:
            try:
                from worker.pipeline.optional_models import preload_gfpgan

                preload_gfpgan()
            except Exception as exc:
                logger.warning("GFPGAN preload skipped: {}", exc)

        if settings.ENABLE_REALESRGAN:
            try:
                from worker.pipeline.optional_models import preload_realesrgan

                preload_realesrgan()
            except Exception as exc:
                logger.warning("Real-ESRGAN preload skipped: {}", exc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._models_warmed = True
        logger.info("TryOnOrchestrator models preloaded")

    def generate_tryon(
        self,
        person_image_path: str,
        garment_image_path: str,
        garment_type: str = "upper",
        debug_dir: Path | None = None,
    ) -> tuple[Image.Image, PipelineContext]:
        person = Image.open(person_image_path).convert("RGB")
        garment = Image.open(garment_image_path).convert("RGB")

        ctx = PipelineContext(
            original_person=person,
            garment=garment,
            cloth_type=garment_type or settings.CLOTH_TYPE,
        )

        if settings.ENABLE_PIPELINE_STAGE0:
            stage0_quality.run_stage0_quality(ctx)
        else:
            ctx.person = person
            ctx.log("stage0: skipped")

        stage1_parsing.run_stage1_parsing(ctx)

        if ctx.parse and ctx.parse.confidence < settings.PIPELINE_PARSE_CONFIDENCE:
            logger.warning(
                "Low parse confidence ({:.2f}) — using fallback mask (coverage={:.2f})",
                ctx.parse.confidence,
                ctx.parse.mask_coverage,
            )

        if settings.ENABLE_PIPELINE_STAGE2:
            stage2_matting.run_stage2_matting(ctx)
        else:
            ctx.log("stage2: skipped")

        stage3_vton.run_stage3_vton(ctx, self._infer_fn)

        if settings.ENABLE_PIPELINE_STAGE4:
            if settings.PIPELINE_BLEND_MODE == "garment_only":
                from worker import postprocess

                if ctx.vton_result and ctx.person and ctx.inpaint_mask:
                    swap_mask = ctx.inference_mask or ctx.inpaint_mask
                    blended_crop = postprocess.composite_garment_only(
                        ctx.vton_result,
                        ctx.person,
                        swap_mask,
                    )
                    base = ctx.blend_base or ctx.original_person
                    if ctx.crop_box is not None:
                        blended = postprocess.embed_crop_on_base(base, blended_crop, ctx.crop_box)
                    else:
                        blended = blended_crop
                    if blended.size != ctx.original_person.size:
                        blended = blended.resize(
                            ctx.original_person.size,
                            Image.Resampling.LANCZOS,
                        )
                    ctx.blended = blended
                    ctx.log("stage4: garment-only composite (preserves catalog colors)")
                else:
                    ctx.blended = ctx.vton_result
            else:
                stage4_blend.run_stage4_blend(ctx)
        else:
            from worker import postprocess

            if ctx.vton_result and ctx.person and ctx.inpaint_mask:
                ctx.blended = postprocess.composite_garment_only(
                    ctx.vton_result,
                    ctx.person,
                    ctx.inpaint_mask,
                )
                ctx.log("stage4: garment-only composite (legacy)")
            else:
                ctx.blended = ctx.vton_result

        if settings.ENABLE_PIPELINE_STAGE5:
            stage5_face.run_stage5_face(ctx)
        else:
            ctx.final = ctx.blended
            ctx.log("stage5: skipped")

        if settings.ENABLE_PIPELINE_STAGE6:
            stage6_finalize.run_stage6_finalize(ctx)
        else:
            ctx.final = ctx.blended or ctx.vton_result
            ctx.log("stage6: skipped")

        if debug_dir is not None:
            self._save_debug(ctx, debug_dir)

        final = ctx.final or ctx.blended or ctx.vton_result
        if final is None:
            raise RuntimeError("Pipeline produced no output image")
        return final, ctx

    def _save_debug(self, ctx: PipelineContext, debug_dir: Path) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)

        def _save(name: str, image: Image.Image | None) -> None:
            if image is None:
                return
            image.save(debug_dir / f"{name}.jpg", quality=92)

        _save("00_original", ctx.original_person)
        _save("01_person_normalized", ctx.person)
        _save("02_inpaint_mask", ctx.inpaint_mask)
        _save("03_alpha_matte", ctx.alpha_matte)
        _save("04_vton", ctx.vton_result)
        _save("05_blended", ctx.blended)
        _save("06_final", ctx.final)

        (debug_dir / "pipeline_summary.json").write_text(
            json.dumps(ctx.summary(), indent=2),
            encoding="utf-8",
        )
        ctx.log(f"debug: saved intermediates to {debug_dir}")
