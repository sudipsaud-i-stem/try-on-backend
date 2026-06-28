from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Literal

from fastapi import Request, UploadFile
from loguru import logger
from PIL import Image

from app.config import settings


def save_upload(file: UploadFile, job_id: str, kind: Literal["person", "garment"]) -> Path:
    """
    Save an uploaded image to disk after validation and resizing.

    Validates the file is a real image, resizes to max 1024px on the longest side,
    converts to RGB JPEG at quality 95.
    """
    job_dir = settings.UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest_path = job_dir / f"{kind}.jpg"

    contents = file.file.read()
    if not contents:
        raise ValueError(f"Empty file uploaded for {kind} image")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Invalid {kind} image: not a valid image file") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    max_dim = 1024
    width, height = image.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    image.save(dest_path, format="JPEG", quality=95)
    logger.debug("Saved {} upload for job {} to {}", kind, job_id, dest_path)
    return dest_path


def get_result_url(job_id: str, request: Request) -> str:
    """Return the full absolute URL to fetch the result image for a job."""
    host = request.headers.get("host", f"{settings.API_HOST}:{settings.API_PORT}")
    scheme = request.url.scheme
    return f"{scheme}://{host}/result/{job_id}"


def result_image_path(job_id: str) -> Path:
    """Return the filesystem path to the result image for a job."""
    return settings.OUTPUT_DIR / job_id / "result.jpg"


def cleanup_job_files(job_id: str) -> None:
    """Delete uploaded images for a job. Output images are preserved."""
    upload_dir = settings.UPLOAD_DIR / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
        logger.info("Cleaned up upload files for job {}", job_id)
