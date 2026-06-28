from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from loguru import logger
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


@router.post(
    "/tryon",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Try-on result image"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Virtual try-on",
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
