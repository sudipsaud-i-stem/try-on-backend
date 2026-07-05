from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.services import storage
from app.services.rate_limit import (
    check_rate_limit,
    get_client_id,
    get_client_ip,
    record_tryon_request,
    update_tryon_status,
)

router = APIRouter(tags=["tryon"])

MIN_PERSON_SHORT_EDGE = 400


def _job_dir(job_id: str) -> Path:
    return settings.OUTPUT_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_status(job_id: str, payload: dict) -> None:
    path = _status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_status(job_id: str) -> dict | None:
    path = _status_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_person_image(path: Path) -> None:
    """Warn-quality gate: tiny photos produce poor try-on and slow upscaling."""
    with Image.open(path) as img:
        w, h = img.size
    short = min(w, h)
    if short < MIN_PERSON_SHORT_EDGE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Person photo too small ({w}x{h}). Use an original photo with "
                f"short edge >= {MIN_PERSON_SHORT_EDGE}px — not a prior try-on thumbnail."
            ),
        )


async def _run_tryon_job(
    job_id: str,
    request_id: str,
    person_path: Path,
    garment_path: Path,
    result_path: Path,
    cloth_type: str,
) -> None:
    from worker.inference import run_inference_direct

    _write_status(job_id, {"job_id": job_id, "status": "processing", "progress": "inference"})
    try:
        await asyncio.to_thread(
            run_inference_direct,
            str(person_path),
            str(garment_path),
            result_path,
            cloth_type,
        )
        if not result_path.exists():
            raise RuntimeError("Result image was not created")

        from app.db.database import SessionLocal

        with SessionLocal() as db:
            update_tryon_status(db, request_id, "success")

        _write_status(
            job_id,
            {
                "job_id": job_id,
                "status": "completed",
                "result_url": f"/tryon/result/{job_id}",
            },
        )
        logger.info("Async try-on completed job_id={}", job_id)
    except Exception as exc:
        from app.db.database import SessionLocal

        with SessionLocal() as db:
            update_tryon_status(db, request_id, "failed", str(exc))
        _write_status(job_id, {"job_id": job_id, "status": "failed", "error": str(exc)})
        logger.exception("Async try-on failed job_id={}", job_id)


@router.post(
    "/tryon/async",
    summary="Virtual try-on (async — avoids Cloudflare 120s timeout)",
)
async def virtual_tryon_async(
    request: Request,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    cloth_type: str = "upper",
    db: Session = Depends(get_db),
) -> JSONResponse:
    ip_address = get_client_ip(request)
    client_id = get_client_id(request)
    check_rate_limit(db, ip_address)

    job_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    record_tryon_request(
        db,
        request_id=request_id,
        job_id=job_id,
        ip_address=ip_address,
        client_id=client_id,
        status_value="processing",
    )

    person_path = storage.save_upload(person_image, job_id, "person")
    garment_path = storage.save_upload(garment_image, job_id, "garment")
    _validate_person_image(person_path)

    result_path = _job_dir(job_id) / "result.jpg"
    _write_status(job_id, {"job_id": job_id, "status": "queued"})

    asyncio.create_task(
        _run_tryon_job(job_id, request_id, person_path, garment_path, result_path, cloth_type)
    )

    logger.info("POST /tryon/async job_id={} person={}", job_id, person_image.filename)
    return JSONResponse(
        {
            "job_id": job_id,
            "status": "processing",
            "status_url": f"/tryon/status/{job_id}",
            "result_url": f"/tryon/result/{job_id}",
            "message": "Poll status_url every 5s until status=completed (typical 90-130s on Kaggle).",
        }
    )


@router.get("/tryon/status/{job_id}", summary="Poll async try-on status")
async def tryon_status(job_id: str) -> JSONResponse:
    payload = _read_status(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(payload)


@router.get(
    "/tryon/result/{job_id}",
    responses={200: {"content": {"image/jpeg": {}}}},
    summary="Download async try-on result",
)
async def tryon_result(job_id: str) -> FileResponse:
    result_path = _job_dir(job_id) / "result.jpg"
    payload = _read_status(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if payload.get("status") != "completed" or not result_path.exists():
        raise HTTPException(status_code=202, detail="Result not ready — poll /tryon/status/{job_id}")
    return FileResponse(
        path=str(result_path),
        media_type="image/jpeg",
        filename="tryon_result.jpg",
    )


@router.post(
    "/tryon",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Try-on result image"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Virtual try-on (sync — may timeout via Cloudflare after 120s)",
)
async def virtual_tryon(
    request: Request,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    cloth_type: str = "upper",
    db: Session = Depends(get_db),
) -> FileResponse:
    ip_address = get_client_ip(request)
    client_id = get_client_id(request)
    check_rate_limit(db, ip_address)

    job_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    record_tryon_request(
        db,
        request_id=request_id,
        job_id=job_id,
        ip_address=ip_address,
        client_id=client_id,
        status_value="processing",
    )

    logger.info(
        "POST /tryon — ip={}, client={}, person={}, garment={}",
        ip_address,
        client_id,
        person_image.filename,
        garment_image.filename,
    )

    try:
        person_path = storage.save_upload(person_image, job_id, "person")
        garment_path = storage.save_upload(garment_image, job_id, "garment")
        _validate_person_image(person_path)
        result_path = settings.OUTPUT_DIR / job_id / "result.jpg"

        from worker.inference import run_inference_direct

        await asyncio.to_thread(
            run_inference_direct,
            str(person_path),
            str(garment_path),
            result_path,
            cloth_type,
        )

        if not result_path.exists():
            update_tryon_status(db, request_id, "failed", "Result image was not created")
            raise HTTPException(status_code=500, detail="Result image was not created")

        update_tryon_status(db, request_id, "success")
        logger.info("POST /tryon -> 200 saved to {}", result_path)
        return FileResponse(
            path=str(result_path),
            media_type="image/jpeg",
            filename="tryon_result.jpg",
        )
    except HTTPException:
        raise
    except Exception as exc:
        update_tryon_status(db, request_id, "failed", str(exc))
        logger.exception("Try-on failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
