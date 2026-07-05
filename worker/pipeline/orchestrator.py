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

    @staticmethod
    def _composite_base(ctx: PipelineContext) -> Image.Image:
        """Full-resolution photo used for compositing (may be upscaled in stage0)."""
        return ctx.blend_base or ctx.original_person

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
                from worker.person_segment import recomposite_on_original_background

                if ctx.vton_result and ctx.person and ctx.inpaint_mask:
                    swap_mask = ctx.inference_mask or ctx.inpaint_mask
                    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
                    composite_base = self._composite_base(ctx)
                    inference_base = (
                        ctx.person_white
                        if settings.PIPELINE_WHITE_BG_INFERENCE and ctx.person_white
                        else ctx.person
                    )
                    blended_crop = postprocess.composite_garment_only(
                        ctx.vton_result,
                        inference_base,
                        swap_mask,
                    )
                    base = composite_base
                    use_white_recomposite = (
                        settings.PIPELINE_WHITE_BG_INFERENCE
                        and ctx.person_segment is not None
                    )

                    if ctx.normalize_mode == "letterbox":
                        restored = postprocess.restore_from_letterbox(
                            blended_crop,
                            composite_base.size,
                            target_size,
                        )
                        if use_white_recomposite:
                            alpha_full = postprocess.restore_mask_from_letterbox(
                                ctx.person_segment,
                                composite_base.size,
                                target_size,
                            )
                            restored = recomposite_on_original_background(
                                composite_base,
                                restored,
                                alpha_full,
                            )
                            ctx.log("stage4: white-bg VTON → original background restore")
                        blended = postprocess.finalize_on_original(
                            restored,
                            composite_base,
                            swap_mask,
                            ctx.normalize_mode,
                            ctx.crop_box,
                            target_size,
                            ctx.schp_atr,
                            ctx.schp_lip,
                        )
                        ctx.log("stage4: letterbox restore + original composite + identity lock")
                    elif ctx.crop_box is not None:
                        orig_crop = base.crop(ctx.crop_box)
                        embed_mask = postprocess.build_embed_mask(
                            orig_crop,
                            ctx.inpaint_mask,
                            ctx.person_segment or ctx.alpha_matte,
                        )
                        embedded = postprocess.embed_crop_on_base(
                            base,
                            blended_crop,
                            ctx.crop_box,
                            embed_mask=embed_mask,
                        )
                        if use_white_recomposite:
                            alpha_full = postprocess.map_mask_to_full(
                                ctx.person_segment,
                                ctx.crop_box,
                                composite_base.size,
                            )
                            embedded = recomposite_on_original_background(
                                composite_base,
                                embedded,
                                alpha_full,
                            )
                            ctx.log("stage4: white-bg VTON → original background restore")
                        blended = postprocess.finalize_on_original(
                            embedded,
                            composite_base,
                            swap_mask,
                            ctx.normalize_mode,
                            ctx.crop_box,
                            target_size,
                            ctx.schp_atr,
                            ctx.schp_lip,
                        )
                        ctx.log("stage4: masked crop embed + original composite + identity lock")
                    else:
                        restored = blended_crop
                        if use_white_recomposite:
                            alpha_full = postprocess.map_mask_to_full(
                                ctx.person_segment,
                                ctx.crop_box,
                                composite_base.size,
                            ) if ctx.crop_box else ctx.person_segment.resize(
                                composite_base.size, Image.Resampling.LANCZOS
                            )
                            restored = recomposite_on_original_background(
                                composite_base,
                                restored.resize(
                                    composite_base.size, Image.Resampling.LANCZOS
                                ),
                                alpha_full,
                            )
                            ctx.log("stage4: white-bg VTON → original background restore")
                        blended = postprocess.finalize_on_original(
                            restored,
                            composite_base,
                            swap_mask,
                            ctx.normalize_mode,
                            ctx.crop_box,
                            target_size,
                            ctx.schp_atr,
                            ctx.schp_lip,
                        )
                        ctx.log("stage4: garment-only composite + identity lock")
                    ctx.blended = blended
                else:
                    ctx.blended = ctx.vton_result
            else:
                stage4_blend.run_stage4_blend(ctx)
                from worker import postprocess

                if ctx.blended and ctx.original_person:
                    composite_base = self._composite_base(ctx)
                    schp_atr_full = schp_lip_full = None
                    target_size = (settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT)
                    if ctx.schp_atr is not None and ctx.schp_lip is not None:
                        schp_atr_full = postprocess.map_parse_to_original(
                            ctx.schp_atr,
                            ctx.normalize_mode,
                            ctx.crop_box,
                            composite_base.size,
                            target_size,
                        )
                        schp_lip_full = postprocess.map_parse_to_original(
                            ctx.schp_lip,
                            ctx.normalize_mode,
                            ctx.crop_box,
                            composite_base.size,
                            target_size,
                        )
                    ctx.blended = postprocess.preserve_identity_regions(
                        ctx.blended,
                        composite_base,
                        schp_atr_full,
                        schp_lip_full,
                    )
                    ctx.log("stage4: Poisson blend + identity lock")
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

        if final.size != ctx.original_person.size:
            final = final.resize(ctx.original_person.size, Image.Resampling.LANCZOS)
            ctx.final = final
            ctx.log(
                f"output: resized to original {ctx.original_person.size[0]}x{ctx.original_person.size[1]}"
            )

        return final, ctx

    def _save_debug(self, ctx: PipelineContext, debug_dir: Path) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)

        def _save(name: str, image: Image.Image | None) -> None:
            if image is None:
                return
            image.save(debug_dir / f"{name}.jpg", quality=92)

        _save("00_original", ctx.original_person)
        _save("01_person_normalized", ctx.person)
        _save("01b_person_white", ctx.person_white)
        _save("01c_person_segment", ctx.person_segment)
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
