from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from app.config import settings
from worker import postprocess, preprocess
from worker.catvton.pipeline import CatVTONPipeline
from worker.pipeline.orchestrator import TryOnOrchestrator

_pipeline: CatVTONPipeline | None = None
_orchestrator: TryOnOrchestrator | None = None


def _is_catvton_ready(model_path: Path) -> bool:
    """Return True if CatVTON attention weights are present locally."""
    subfolder = {
        "mix": "mix-48k-1024",
        "vitonhd": "vitonhd-16k-512",
        "dresscode": "dresscode-16k-512",
    }[settings.CATVTON_ATTN_VERSION]
    return (model_path / subfolder / "attention" / "model.safetensors").exists()


def _load_pipeline() -> CatVTONPipeline:
    """Lazy-load the CatVTON pipeline into GPU memory (official settings, no slicing)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    model_path = settings.catvton_model_path
    if not _is_catvton_ready(model_path):
        raise RuntimeError(
            f"CatVTON weights not found at {model_path}. "
            "Run: python scripts\\download_models.py"
        )

    try:
        logger.info(
            "Loading CatVTON pipeline (base={}, attn={}, steps={})",
            settings.CATVTON_BASE_MODEL_ID,
            model_path,
            settings.INFERENCE_STEPS,
        )
        _pipeline = CatVTONPipeline(
            base_ckpt=settings.CATVTON_BASE_MODEL_ID,
            attn_ckpt=str(model_path),
            attn_ckpt_version=settings.CATVTON_ATTN_VERSION,
            weight_dtype=settings.torch_dtype,
            device=settings.device,
            use_tf32=True,
        )
        logger.info("CatVTON pipeline loaded on {} (full quality, no attention slicing)", settings.device)
        return _pipeline
    except Exception as exc:
        logger.exception("Failed to load CatVTON pipeline")
        raise RuntimeError(f"Model load failed: {exc}") from exc


def _get_orchestrator() -> TryOnOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TryOnOrchestrator(infer_fn=_run_pipeline_inference)
    return _orchestrator


def preload_inference_models() -> None:
    """Warm up CatVTON and optional pipeline models once at startup."""
    from worker.compat import ensure_torchvision_functional_tensor, verify_ml_dependency_stack

    ensure_torchvision_functional_tensor()
    verify_ml_dependency_stack()
    _load_pipeline()
    if settings.ENABLE_HUBA_PIPELINE:
        _get_orchestrator().preload_models()


def _make_generator() -> torch.Generator | None:
    """Create a CUDA generator matching the official CatVTON demo."""
    if settings.INFERENCE_SEED < 0:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(settings.INFERENCE_SEED)


def _validate_output(image: Image.Image) -> None:
    """Raise if the model returned an empty or corrupt image."""
    import numpy as np

    pixels = np.array(image.convert("RGB"))
    if pixels.size == 0:
        raise RuntimeError("Generated image is empty.")
    if float(pixels.mean()) < 4.0:
        raise RuntimeError(
            "Generated image appears blank. Use a clear front-facing person photo."
        )


def _run_pipeline_inference(inputs: preprocess.PreprocessInputs) -> Image.Image:
    """Execute CatVTON inference on preprocessed PIL images."""
    pipeline = _load_pipeline()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = pipeline(
        image=inputs["person"],
        condition_image=inputs["garment"],
        mask=inputs["mask"],
        num_inference_steps=settings.INFERENCE_STEPS,
        guidance_scale=settings.GUIDANCE_SCALE,
        height=settings.OUTPUT_HEIGHT,
        width=settings.OUTPUT_WIDTH,
        generator=_make_generator(),
    )
    output = results[0]
    _validate_output(output)
    return output


def run_inference_direct(
    person_image_path: str,
    garment_image_path: str,
    output_path: str | Path,
    cloth_type: str | None = None,
) -> Path:
    """Run inference directly without database or queue interaction."""
    start_time = time.time()
    garment_type = cloth_type or settings.CLOTH_TYPE
    output_path = Path(output_path)
    debug_dir = None
    if settings.PIPELINE_DEBUG:
        debug_dir = output_path.parent / "debug"

    if settings.ENABLE_HUBA_PIPELINE:
        orchestrator = _get_orchestrator()
        output_image, ctx = orchestrator.generate_tryon(
            person_image_path,
            garment_image_path,
            garment_type=garment_type,
            debug_dir=debug_dir,
        )
        for line in ctx.stage_logs:
            logger.info("pipeline | {}", line)
        if ctx.parse:
            logger.info(
                "pipeline summary | confidence={:.2f} fallback={} type={}",
                ctx.parse.confidence,
                ctx.parse.used_fallback,
                ctx.parse.cloth_type,
            )
    else:
        inputs = preprocess.prepare_inputs(person_image_path, garment_image_path, cloth_type=garment_type)
        output_image = _run_pipeline_inference(inputs)
        output_image = postprocess.composite_garment_only(
            output_image,
            inputs["person"],
            inputs["mask"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_image.mode != "RGB":
        output_image = output_image.convert("RGB")
    output_image.save(output_path, format="JPEG", quality=97)

    elapsed = time.time() - start_time
    logger.info("Direct inference completed in {:.2f}s -> {}", elapsed, output_path)
    return output_path
