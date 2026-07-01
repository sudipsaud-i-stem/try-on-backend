from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.config import settings


def tighten_mask(mask: Image.Image, erode_px: int | None = None) -> Image.Image:
    """
    Shrink the inpaint mask so CatVTON does not repaint skin, arms, or neck.

    A mask that is too large is the main cause of body-texture drift.
    """
    px = settings.MASK_ERODE_PIXELS if erode_px is None else erode_px
    arr = np.array(mask.convert("L"))
    if px <= 0:
        return Image.fromarray(arr, mode="L")

    k = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = cv2.erode(arr, kernel, iterations=1)
    return Image.fromarray(arr, mode="L")


def composite_garment_only(
    result: Image.Image,
    person: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Keep the original photo everywhere except the core garment mask.

    Only pixels inside the garment mask are taken from the model output;
    skin, hair, background, and other clothing stay untouched.
    """
    person_arr = np.array(person.convert("RGB"), dtype=np.float32)
    result_arr = np.array(result.convert("RGB"), dtype=np.float32)
    mask_arr = np.array(
        mask.convert("L").resize(result.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    mask_arr = mask_arr / 255.0

    # Strict core: only confident mask pixels are replaced (preserves skin at edges).
    alpha = np.clip((mask_arr - 0.45) / 0.35, 0.0, 1.0)
    alpha = alpha ** 1.6

    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    alpha = np.array(alpha_img, dtype=np.float32) / 255.0

    alpha_3 = alpha[..., np.newaxis]
    out = person_arr * (1.0 - alpha_3) + result_arr * alpha_3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_garment_color_preserve(
    result: Image.Image,
    garment: Image.Image,
    mask: Image.Image,
    strength: float,
) -> Image.Image:
    """
    Nudge generated garment pixels toward the reference garment colors (LAB stats).

    Reduces random hue/texture drift from diffusion + background blending.
    """
    if strength <= 0:
        return result

    mask_arr = np.array(
        mask.convert("L").resize(result.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    mask_arr = np.clip(mask_arr / 255.0, 0.0, 1.0)
    if float(mask_arr.max()) < 0.05:
        return result

    alpha = np.clip((mask_arr - 0.25) / 0.45, 0.0, 1.0) ** 1.3
    alpha_3 = (alpha * strength)[..., np.newaxis]

    result_rgb = np.array(result.convert("RGB"), dtype=np.float32)
    garment_rgb = np.array(
        garment.convert("RGB").resize(result.size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )

    garment_mask = alpha > 0.35
    if int(garment_mask.sum()) < 32:
        return result

    out = result_rgb.copy()
    for c in range(3):
        g_pixels = garment_rgb[:, :, c][garment_mask]
        r_pixels = result_rgb[:, :, c][garment_mask]
        g_mean, g_std = float(g_pixels.mean()), float(g_pixels.std()) + 1e-6
        r_mean, r_std = float(r_pixels.mean()), float(r_pixels.std()) + 1e-6
        corrected = (result_rgb[:, :, c] - r_mean) * (g_std / r_std) + g_mean
        out[:, :, c] = result_rgb[:, :, c] * (1.0 - alpha_3[..., 0]) + corrected * alpha_3[..., 0]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def embed_crop_on_base(
    base: Image.Image,
    crop_result: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> Image.Image:
    """Paste a VTON crop back into the full-resolution person frame."""
    left, top, right, bottom = crop_box
    cw, ch = right - left, bottom - top
    canvas = base.copy()
    patch = crop_result.resize((cw, ch), Image.Resampling.LANCZOS)
    canvas.paste(patch, (left, top))
    return canvas
