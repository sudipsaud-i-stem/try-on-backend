from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.pipeline.types import PipelineContext


def _extract_noise_patch(original: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Sample background noise near the subject for grain matching."""
    bg_mask = alpha < 0.08
    if int(bg_mask.sum()) < 64:
        return np.zeros(3, dtype=np.float32)
    patch = original[bg_mask]
    if patch.size < 300:
        return np.zeros(3, dtype=np.float32)
    blurred = cv2.GaussianBlur(patch.reshape(-1, 1, 3), (0, 0), 1.2).reshape(-1, 3)
    residual = patch.astype(np.float32) - blurred.astype(np.float32)
    return residual.std(axis=0)


def _inject_noise(image: np.ndarray, alpha: np.ndarray, noise_std: np.ndarray) -> np.ndarray:
    if float(noise_std.max()) < 0.5:
        return image
    h, w = alpha.shape
    rng = np.random.default_rng(settings.INFERENCE_SEED if settings.INFERENCE_SEED >= 0 else None)
    noise = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
    noise *= noise_std.reshape(1, 1, 3)
    strength = np.clip(alpha, 0, 1)[..., None] * settings.PIPELINE_NOISE_MATCH_STRENGTH
    out = image.astype(np.float32) + noise * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_center_and_size(alpha: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    ys, xs = np.where(alpha > 0.15)
    if len(xs) == 0:
        h, w = alpha.shape
        return (w // 2, h // 2), (w // 3, h // 2)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    size = (max(8, x1 - x0), max(8, y1 - y0))
    return center, size


def _embed_crop(
    base: np.ndarray,
    crop: np.ndarray,
    alpha_crop: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Place VTON crop back into the full-resolution frame."""
    left, top, right, bottom = box
    crop_w, crop_h = right - left, bottom - top
    gen_full = base.copy()
    alpha_full = np.zeros(base.shape[:2], dtype=np.float32)

    crop_resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_resized = cv2.resize(alpha_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    gen_full[top:bottom, left:right] = crop_resized
    alpha_full[top:bottom, left:right] = alpha_resized
    return gen_full, alpha_full


def _poisson_blend(original: Image.Image, generated: Image.Image, alpha: Image.Image) -> Image.Image:
    orig = np.array(original.convert("RGB"))
    gen = np.array(generated.convert("RGB"))
    if orig.shape != gen.shape:
        gen = np.array(generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS))

    alpha_arr = np.array(alpha.convert("L").resize(original.size, Image.Resampling.LANCZOS), dtype=np.float32)
    alpha_arr = alpha_arr / 255.0

    noise_std = _extract_noise_patch(orig, alpha_arr)
    gen = _inject_noise(gen, alpha_arr, noise_std)

    mask_u8 = (np.clip(alpha_arr, 0, 1) * 255).astype(np.uint8)
    center, _ = _alpha_center_and_size(alpha_arr)

    try:
        blended = cv2.seamlessClone(gen, orig, mask_u8, center, cv2.NORMAL_CLONE)
        return Image.fromarray(blended)
    except cv2.error:
        a3 = alpha_arr[..., None]
        out = orig.astype(np.float32) * (1 - a3) + gen.astype(np.float32) * a3
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def run_stage4_blend(ctx: PipelineContext) -> Image.Image:
    """Poisson re-composition into original background with noise matching."""
    if ctx.vton_result is None:
        raise RuntimeError("stage4 requires vton result")

    base_image = ctx.blend_base or ctx.person or ctx.original_person
    alpha = ctx.alpha_matte or ctx.inpaint_mask
    if alpha is None:
        raise RuntimeError("stage4 requires alpha matte or inpaint mask")

    vton = np.array(ctx.vton_result.convert("RGB"))
    alpha_crop = np.array(alpha.convert("L"), dtype=np.float32) / 255.0
    base = np.array(base_image.convert("RGB"))

    if ctx.crop_box is not None:
        gen_full, alpha_full = _embed_crop(base, vton, alpha_crop, ctx.crop_box)
        blended = _poisson_blend(
            Image.fromarray(base),
            Image.fromarray(gen_full),
            Image.fromarray((alpha_full * 255).astype(np.uint8), mode="L"),
        )
    else:
        blended = _poisson_blend(base_image, ctx.vton_result, alpha)

    if blended.size != ctx.original_person.size:
        blended = blended.resize(ctx.original_person.size, Image.Resampling.LANCZOS)

    ctx.blended = blended
    ctx.log("stage4: Poisson/noise-matched background recomposition")
    return blended
