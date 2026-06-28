from __future__ import annotations

import subprocess
import time
from typing import Any

import torch
from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Product
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])

_process_start_time = time.time()


def _get_gpu_info() -> dict[str, Any]:
    """Collect GPU availability and VRAM statistics."""
    info: dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_total_mb": None,
        "gpu_vram_free_mb": None,
    }
    if torch.cuda.is_available():
        info["gpu_available"] = True
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(0)
        info["gpu_vram_total_mb"] = total // (1024 * 1024)
        info["gpu_vram_free_mb"] = (total - reserved) // (1024 * 1024)
    return info


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Return aggregated health status for the API, GPU, and database."""
    logger.debug("GET /health")
    gpu = _get_gpu_info()
    uptime = time.time() - _process_start_time

    db_status: str = "ok"
    product_count = 0
    try:
        product_count = db.query(Product).count()
    except Exception:
        db_status = "error"

    status_value: str = "ok"
    if not gpu["gpu_available"] or db_status == "error":
        status_value = "degraded"

    response = HealthResponse(
        status=status_value,  # type: ignore[arg-type]
        gpu_available=gpu["gpu_available"],
        gpu_name=gpu["gpu_name"],
        gpu_vram_total_mb=gpu["gpu_vram_total_mb"],
        gpu_vram_free_mb=gpu["gpu_vram_free_mb"],
        uptime_seconds=round(uptime, 2),
        database=db_status,  # type: ignore[arg-type]
        product_count=product_count,
    )
    logger.info("GET /health -> 200 status={} products={}", status_value, product_count)
    return response


@router.get("/health/gpu")
async def health_gpu() -> dict[str, Any]:
    """Return detailed GPU diagnostic information."""
    logger.debug("GET /health/gpu")
    gpu_available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "gpu_available": gpu_available,
        "gpu_name": None,
        "cuda_version": torch.version.cuda if gpu_available else None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "vram_used_mb": None,
        "vram_used_percent": None,
        "current_temperature": None,
    }

    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(0)
        total_mb = total // (1024 * 1024)
        used_mb = reserved // (1024 * 1024)
        free_mb = (total - reserved) // (1024 * 1024)
        result.update(
            {
                "gpu_name": props.name,
                "vram_total_mb": total_mb,
                "vram_free_mb": free_mb,
                "vram_used_mb": used_mb,
                "vram_used_percent": round((used_mb / total_mb) * 100, 1) if total_mb else 0,
            }
        )

        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                result["current_temperature"] = int(smi.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

    logger.info("GET /health/gpu -> 200")
    return result
