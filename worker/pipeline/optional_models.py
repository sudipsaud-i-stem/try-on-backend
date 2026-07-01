from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from app.config import settings

if TYPE_CHECKING:
    from transformers import PreTrainedModel

_birefnet_model: PreTrainedModel | None = None
_birefnet_device: torch.device | None = None
_gfpgan_restorer = None
_realesrgan_upsampler = None


def preload_birefnet() -> None:
    global _birefnet_model, _birefnet_device
    if _birefnet_model is not None:
        return
    from transformers import AutoModelForImageSegmentation

    _birefnet_device = settings.device
    _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        settings.BIREFNET_MODEL_ID,
        trust_remote_code=True,
    )
    _birefnet_model.to(_birefnet_device)
    _birefnet_model.eval()


def generate_birefnet_matte(person: Image.Image) -> Image.Image | None:
    global _birefnet_model, _birefnet_device
    if _birefnet_model is None:
        preload_birefnet()

    from torchvision import transforms

    assert _birefnet_model is not None and _birefnet_device is not None

    rgb = person.convert("RGB")
    size = (1024, 1024)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(rgb).unsqueeze(0).to(_birefnet_device)

    with torch.no_grad():
        preds = _birefnet_model(tensor)[-1].sigmoid().cpu()

    matte = (preds[0, 0].numpy() * 255).astype(np.uint8)
    matte_img = Image.fromarray(matte, mode="L").resize(rgb.size, Image.Resampling.LANCZOS)
    return matte_img


def preload_gfpgan() -> None:
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return
    from gfpgan import GFPGANer

    model_path = settings.MODEL_CACHE_DIR / "gfpgan" / "GFPGANv1.4.pth"
    _gfpgan_restorer = GFPGANer(
        model_path=str(model_path),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=str(settings.device),
    )


def restore_face_gfpgan(image: Image.Image) -> Image.Image | None:
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        preload_gfpgan()
    assert _gfpgan_restorer is not None

    bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    _, _, restored_bgr = _gfpgan_restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
    )
    if restored_bgr is None:
        return None
    rgb = restored_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def preload_realesrgan() -> None:
    global _realesrgan_upsampler
    if _realesrgan_upsampler is not None:
        return
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    model_path = settings.MODEL_CACHE_DIR / "realesrgan" / "RealESRGAN_x2plus.pth"
    _realesrgan_upsampler = RealESRGANer(
        scale=2,
        model_path=str(model_path),
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=settings.TORCH_DTYPE == "float16",
        device=str(settings.device),
    )


def upscale_realesrgan(image: Image.Image, outscale: float = 2.0) -> Image.Image | None:
    global _realesrgan_upsampler
    if _realesrgan_upsampler is None:
        preload_realesrgan()
    assert _realesrgan_upsampler is not None

    bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    output, _ = _realesrgan_upsampler.enhance(bgr, outscale=outscale)
    rgb = output[:, :, ::-1]
    return Image.fromarray(rgb)
