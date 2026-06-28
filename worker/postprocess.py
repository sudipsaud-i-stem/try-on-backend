from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def composite_garment_only(
    result: Image.Image,
    person: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Keep the original photo everywhere except the inpaint mask.

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

    # White mask regions = garment swap target from CatVTON AutoMasker.
    alpha = np.clip((mask_arr - 0.2) / 0.5, 0.0, 1.0)
    alpha = alpha ** 1.35

    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    alpha = np.array(alpha_img, dtype=np.float32) / 255.0

    alpha_3 = alpha[..., np.newaxis]
    out = person_arr * (1.0 - alpha_3) + result_arr * alpha_3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
