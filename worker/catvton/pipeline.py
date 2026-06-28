from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Union

import PIL
import numpy as np
import torch
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download

from worker.catvton.attn_processor import SkipAttnProcessor
from worker.catvton.image_utils import (
    compute_vae_encodings,
    numpy_to_pil,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)
from worker.catvton.model_utils import get_trainable_module, init_adapter

ATTN_SUBFOLDERS = {
    "mix": "mix-48k-1024",
    "vitonhd": "vitonhd-16k-512",
    "dresscode": "dresscode-16k-512",
}


class CatVTONPipeline:
    """CatVTON virtual try-on pipeline (adapted from Zheng-Chong/CatVTON)."""

    def __init__(
        self,
        base_ckpt: str,
        attn_ckpt: str | Path,
        attn_ckpt_version: str = "mix",
        weight_dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        use_tf32: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
            self.device, dtype=weight_dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(
            self.device, dtype=weight_dtype
        )
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self._load_attention_weights(attn_ckpt, attn_ckpt_version)

        if use_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def _load_attention_weights(self, attn_ckpt: str | Path, version: str) -> None:
        """Load fine-tuned attention weights from a local path or HuggingFace repo."""
        sub_folder = ATTN_SUBFOLDERS[version]
        attn_path = Path(attn_ckpt)
        if attn_path.exists():
            weights_dir = attn_path / sub_folder / "attention"
        else:
            repo_path = Path(snapshot_download(repo_id=str(attn_ckpt)))
            weights_dir = repo_path / sub_folder / "attention"
        load_checkpoint_in_model(self.attn_modules, str(weights_dir))

    def check_inputs(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        width: int,
        height: int,
    ):
        """Resize inputs to the target resolution."""
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta: float) -> dict:
        """Prepare optional kwargs for the scheduler step."""
        extra_step_kwargs = {}
        if "eta" in inspect.signature(self.noise_scheduler.step).parameters:
            extra_step_kwargs["eta"] = eta
        if "generator" in inspect.signature(self.noise_scheduler.step).parameters:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta: float = 1.0,
        **kwargs,
    ) -> list[PIL.Image.Image]:
        """Run CatVTON inference and return PIL result images."""
        concat_dim = -2
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        masked_image = image * (mask < 0.5)
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        latents = torch.randn(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order

        from tqdm import tqdm
        for i, t in enumerate(tqdm(timesteps, desc="Inference")):
            non_inpainting_latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(
                non_inpainting_latent_model_input, t
            )
            inpainting_latent_model_input = torch.cat(
                [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1
            )
            noise_pred = self.unet(
                inpainting_latent_model_input,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        decoded = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(decoded)
