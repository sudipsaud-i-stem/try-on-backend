from __future__ import annotations

from typing import Union

import numpy as np
import PIL
import torch
from PIL import Image


def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """Encode an image tensor through the VAE."""
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        latent_dist = vae.encode(pixel_values).latent_dist
        model_input = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
    return model_input


def prepare_image(image: Union[PIL.Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert a PIL/numpy image to a normalized batch tensor."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(dtype=torch.float32)

    if isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = [image]
    if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)

    image = image.transpose(0, 3, 1, 2)
    return torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0


def prepare_mask_image(mask_image: Union[PIL.Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert a mask to a binarized batch tensor."""
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image = mask_image.clone()
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        return mask_image

    if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
        mask_image = [mask_image]

    if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
        mask_image = np.concatenate(
            [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
        )
        mask_image = mask_image.astype(np.float32) / 255.0
    elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
        mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

    mask_image = mask_image.copy()
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1
    return torch.from_numpy(mask_image)


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """Convert a numpy batch to PIL images."""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]


def resize_and_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop and resize an image to the target size."""
    box = center_crop_box(image.size, size)
    return image.crop(box).resize(size, Image.Resampling.LANCZOS)


def center_crop_box(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) for the center crop before resize."""
    w, h = image_size
    target_w, target_h = target_size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return left, top, left + new_w, top + new_h


def crop_to_content(image: Image.Image, bg_threshold: int = 235) -> Image.Image:
    """Tight-crop flat-lay garments (light or dark backgrounds, PNG alpha)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        alpha = np.array(rgba.split()[-1])
        mask = alpha > 12
        if not mask.any():
            return image.convert("RGB")
        ys, xs = np.where(mask)
        margin = max(4, int(min(rgba.size[::-1]) * 0.02))
        box = (
            max(0, int(xs.min()) - margin),
            max(0, int(ys.min()) - margin),
            min(rgba.width, int(xs.max()) + margin + 1),
            min(rgba.height, int(ys.max()) + margin + 1),
        )
        return rgba.crop(box).convert("RGB")

    rgb = np.array(image.convert("RGB"))
    corners = np.array(
        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=np.float32
    )
    dark_bg = float(corners.mean()) < 80.0

    if dark_bg:
        # Black/dark studio backgrounds (e.g. leather jacket PNG on black).
        mask = np.max(rgb, axis=2) > 28
    else:
        mask = np.min(rgb, axis=2) < bg_threshold

    if not mask.any():
        return image

    ys, xs = np.where(mask)
    margin = max(4, int(min(rgb.shape[:2]) * 0.02))
    box = (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(rgb.shape[1], int(xs.max()) + margin + 1),
        min(rgb.shape[0], int(ys.max()) + margin + 1),
    )
    return image.crop(box)


def preprocess_garment_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop empty margins then letterbox — keeps catalog colors/textures sharp."""
    w, h = image.size
    cropped = crop_to_content(image)
    cw, ch = cropped.size
    if cw * ch >= 0.35 * w * h:
        image = cropped
    # Scale garment to ~88% of canvas so CatVTON sees more pixel detail.
    target_w, target_h = size
    inner = (int(target_w * 0.88), int(target_h * 0.88))
    fitted = resize_and_padding(image, inner)
    canvas = Image.new("RGB", size, (255, 255, 255))
    ox = (target_w - inner[0]) // 2
    oy = (target_h - inner[1]) // 2
    canvas.paste(fitted, (ox, oy))
    return canvas


def resize_and_padding(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize with letterboxing on a white background."""
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding
