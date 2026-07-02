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


def grabcut_person_mask(image: Image.Image) -> np.ndarray:
    """Coarse person silhouette for embedding / fallback matting."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8, dtype=np.uint8)
    rect = (int(w * 0.08), int(h * 0.03), int(w * 0.84), int(h * 0.94))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)


def build_embed_mask(
    original_crop: Image.Image,
    inpaint_mask: Image.Image | None = None,
    alpha_matte: Image.Image | None = None,
) -> Image.Image:
    """
    Person-shaped alpha for pasting VTON crop back onto the full photo.

    Without this, embed_crop_on_base pastes a hard rectangle (the 'shirt box' bug).
    """
    if alpha_matte is not None:
        return alpha_matte.convert("L")

    person = grabcut_person_mask(original_crop)

    if inpaint_mask is not None:
        garment = np.array(inpaint_mask.convert("L"))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        torso = cv2.dilate(garment, kernel, iterations=2)
        person = np.maximum(person, torso)

    return Image.fromarray(person, mode="L")


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
    """Nudge generated garment pixels toward the reference garment colors."""
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
    embed_mask: Image.Image | None = None,
) -> Image.Image:
    """Blend VTON crop back into the full-resolution frame using a person mask."""
    left, top, right, bottom = crop_box
    cw, ch = right - left, bottom - top
    canvas = base.copy()
    original_crop = canvas.crop(crop_box)
    patch = crop_result.resize((cw, ch), Image.Resampling.LANCZOS)

    if embed_mask is not None:
        mask = embed_mask.convert("L").resize((cw, ch), Image.Resampling.LANCZOS)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        blended = Image.composite(patch, original_crop, mask)
    else:
        blended = patch

    canvas.paste(blended, (left, top))
    return canvas


def restore_from_letterbox(
    letterboxed: Image.Image,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """Remove letterbox padding and resize to the original photo dimensions."""
    tw, th = target_size
    ow, oh = original_size
    w, h = ow, oh

    if w / h < tw / th:
        content_w = w * th // h
        content_h = th
    else:
        content_w = tw
        content_h = h * tw // w

    left = (tw - content_w) // 2
    top = (th - content_h) // 2
    content = letterboxed.crop((left, top, left + content_w, top + content_h))
    return content.resize(original_size, Image.Resampling.LANCZOS)
