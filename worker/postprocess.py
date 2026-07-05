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


def grabcut_person_mask(image: Image.Image, max_side: int = 512) -> np.ndarray:
    """Coarse person silhouette for embedding / fallback matting."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask = _grabcut_mask_array(small)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return _grabcut_mask_array(rgb)


def _grabcut_mask_array(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = (int(w * 0.08), int(h * 0.03), int(w * 0.84), int(h * 0.94))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)


def build_garment_embed_mask(
    original_crop: Image.Image,
    inpaint_mask: Image.Image,
) -> Image.Image:
    """Soft garment-region mask for embedding VTON crop (avoids full-body ghost paste)."""
    garment = np.array(
        inpaint_mask.convert("L").resize(original_crop.size, Image.Resampling.LANCZOS)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    expanded = cv2.dilate(garment, kernel, iterations=2)
    matte = Image.fromarray(expanded, mode="L")
    return matte.filter(ImageFilter.GaussianBlur(radius=4))


def build_embed_mask(
    original_crop: Image.Image,
    inpaint_mask: Image.Image | None = None,
    alpha_matte: Image.Image | None = None,
) -> Image.Image:
    """
    Person-shaped alpha for pasting VTON crop back onto the full photo.

    Prefer garment-only embed mask to avoid ghost overlays on arms/props/background.
    """
    if inpaint_mask is not None:
        return build_garment_embed_mask(original_crop, inpaint_mask)

    if alpha_matte is not None:
        return alpha_matte.convert("L")

    return Image.fromarray(grabcut_person_mask(original_crop), mode="L")


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

    alpha = np.clip((mask_arr - 0.22) / 0.35, 0.0, 1.0)
    alpha = alpha ** 1.15

    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.8))
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


def restore_labels_from_letterbox(
    label_image: Image.Image,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """Restore a parse label map from letterboxed coordinates (nearest-neighbor)."""
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
    content = label_image.crop((left, top, left + content_w, top + content_h))
    return content.resize(original_size, Image.Resampling.NEAREST)


def map_parse_to_original(
    parse_image: Image.Image,
    normalize_mode: str,
    crop_box: tuple[int, int, int, int] | None,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Image.Image:
    """Map SCHP label maps from normalized inference space to the original photo."""
    if normalize_mode == "letterbox":
        return restore_labels_from_letterbox(parse_image, original_size, target_size)
    if crop_box is not None:
        left, top, right, bottom = crop_box
        cw, ch = right - left, bottom - top
        full = Image.new("L", original_size, 0)
        full.paste(parse_image.convert("L").resize((cw, ch), Image.Resampling.NEAREST), (left, top))
        return full
    return parse_image.convert("L").resize(original_size, Image.Resampling.NEAREST)


def restore_mask_from_letterbox(
    mask: Image.Image,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """Map a letterboxed mask back to the original photo size."""
    return restore_labels_from_letterbox(mask, original_size, target_size)


def map_mask_to_full(
    mask: Image.Image,
    crop_box: tuple[int, int, int, int],
    full_size: tuple[int, int],
) -> Image.Image:
    """Place a normalized mask into full-resolution image coordinates."""
    left, top, right, bottom = crop_box
    cw, ch = right - left, bottom - top
    full = Image.new("L", full_size, 0)
    full.paste(mask.convert("L").resize((cw, ch), Image.Resampling.NEAREST), (left, top))
    return full


def _detect_face_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    if w * h < 60 * 60:
        return None
    return int(x), int(y), int(w), int(h)


def preserve_identity_regions(
    result: Image.Image,
    original: Image.Image,
    schp_atr: Image.Image | None = None,
    schp_lip: Image.Image | None = None,
) -> Image.Image:
    """
    Paste face, hair, and hands from the original photo.

    Prevents CatVTON from altering identity, pose, or missing extremities.
    """
    if not settings.MASK_PROTECT_IDENTITY:
        return result

    if result.size != original.size:
        result = result.resize(original.size, Image.Resampling.LANCZOS)

    h, w = original.height, original.width
    protect = np.zeros((h, w), dtype=np.float32)

    if schp_atr is not None and schp_lip is not None:
        from worker.mask_refine import build_identity_protect_mask

        schp_protect = build_identity_protect_mask(schp_atr, schp_lip)
        if schp_protect.shape[:2] != (h, w):
            schp_protect = cv2.resize(schp_protect, (w, h), interpolation=cv2.INTER_LINEAR)
        protect = np.maximum(protect, schp_protect)

    from worker.pose_mediapipe import build_mediapipe_protect_mask

    mp_protect = build_mediapipe_protect_mask(original)
    if mp_protect is not None:
        if mp_protect.shape[:2] != (h, w):
            mp_protect = cv2.resize(mp_protect, (w, h), interpolation=cv2.INTER_LINEAR)
        protect = np.maximum(protect, mp_protect)

    rgb = np.array(original.convert("RGB"))
    face = _detect_face_bbox(rgb)
    if face is not None:
        x, y, fw, fh = face
        pad_x = int(fw * 0.45)
        pad_y = int(fh * 0.55)
        y0 = max(0, y - pad_y)
        y1 = min(h, y + fh + int(fh * 0.35))
        x0 = max(0, x - pad_x)
        x1 = min(w, x + fw + pad_x)
        protect[y0:y1, x0:x1] = 1.0

    if float(protect.max()) < 0.05:
        return result

    protect = cv2.GaussianBlur(protect, (15, 15), 0)
    protect = np.clip(protect, 0.0, 1.0)

    orig_arr = np.array(original.convert("RGB"), dtype=np.float32)
    result_arr = np.array(result.convert("RGB"), dtype=np.float32)
    alpha = protect[..., np.newaxis]
    out = result_arr * (1.0 - alpha) + orig_arr * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def finalize_on_original(
    generated: Image.Image,
    original: Image.Image,
    swap_mask: Image.Image,
    ctx_normalize_mode: str,
    crop_box: tuple[int, int, int, int] | None,
    target_size: tuple[int, int],
    schp_atr: Image.Image | None = None,
    schp_lip: Image.Image | None = None,
) -> Image.Image:
    """
    Composite try-on onto the untouched original photo and lock identity regions.
    """
    if ctx_normalize_mode == "letterbox":
        full_mask = restore_mask_from_letterbox(swap_mask, original.size, target_size)
    elif crop_box is not None:
        full_mask = map_mask_to_full(swap_mask, crop_box, original.size)
    else:
        full_mask = swap_mask.resize(original.size, Image.Resampling.LANCZOS)

    tight_mask = tighten_mask(full_mask, erode_px=max(4, settings.MASK_ERODE_PIXELS - 2))
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    schp_atr_full = schp_lip_full = None
    if schp_atr is not None and schp_lip is not None:
        schp_atr_full = map_parse_to_original(
            schp_atr, ctx_normalize_mode, crop_box, original.size, target_size
        )
        schp_lip_full = map_parse_to_original(
            schp_lip, ctx_normalize_mode, crop_box, original.size, target_size
        )

    out = composite_garment_only(generated, original, tight_mask)
    return preserve_identity_regions(out, original, schp_atr_full, schp_lip_full)


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
