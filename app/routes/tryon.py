from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
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
from worker.debug_viz import DEBUG_STEP_LABELS

router = APIRouter(tags=["tryon"])

MIN_PERSON_SHORT_EDGE = 400

_DEBUG_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_DEBUG_TEXT_FILES = {"pipeline_summary.json", "pipeline_logs.txt"}


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


def _debug_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "debug"


def _list_debug_steps(job_id: str) -> list[dict[str, str]]:
    debug_path = _debug_dir(job_id)
    if not debug_path.is_dir():
        return []

    steps: list[dict[str, str]] = []
    for path in sorted(debug_path.iterdir()):
        if path.name == "pipeline_summary.json":
            steps.append(
                {
                    "id": "pipeline_summary",
                    "label": DEBUG_STEP_LABELS["pipeline_summary"],
                    "url": f"/tryon/debug/{job_id}/pipeline_summary.json",
                    "kind": "json",
                }
            )
            continue
        if path.name == "pipeline_logs.txt":
            steps.append(
                {
                    "id": "pipeline_logs",
                    "label": DEBUG_STEP_LABELS["pipeline_logs"],
                    "url": f"/tryon/debug/{job_id}/pipeline_logs.txt",
                    "kind": "text",
                }
            )
            continue
        if path.suffix.lower() not in _DEBUG_IMAGE_SUFFIXES:
            continue
        step_id = path.stem
        steps.append(
            {
                "id": step_id,
                "label": DEBUG_STEP_LABELS.get(step_id, step_id.replace("_", " ")),
                "url": f"/tryon/debug/{job_id}/{path.name}",
                "kind": "image",
            }
        )
    return steps


def _resolve_debug_file(job_id: str, filename: str) -> Path:
    if ".." in filename or filename.startswith("/") or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if filename not in _DEBUG_TEXT_FILES and Path(filename).suffix.lower() not in _DEBUG_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported debug file type")
    path = _debug_dir(job_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Debug file not found")
    return path


async def _run_tryon_job(
    job_id: str,
    request_id: str,
    person_path: Path,
    garment_path: Path,
    result_path: Path,
    cloth_type: str,
    debug: bool = False,
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
            debug,
        )
        if not result_path.exists():
            raise RuntimeError("Result image was not created")

        from app.db.database import SessionLocal

        with SessionLocal() as db:
            update_tryon_status(db, request_id, "success")

        completed: dict = {
            "job_id": job_id,
            "status": "completed",
            "result_url": f"/tryon/result/{job_id}",
        }
        if debug and _debug_dir(job_id).is_dir():
            completed["debug_url"] = f"/tryon/debug/{job_id}"
            completed["debug_steps"] = _list_debug_steps(job_id)
        _write_status(job_id, completed)
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
    debug: bool = Query(False, description="Save and expose per-stage pipeline images for analysis"),
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
        _run_tryon_job(job_id, request_id, person_path, garment_path, result_path, cloth_type, debug)
    )

    logger.info("POST /tryon/async job_id={} person={} debug={}", job_id, person_image.filename, debug)
    payload: dict = {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/tryon/status/{job_id}",
        "result_url": f"/tryon/result/{job_id}",
        "message": "Poll status_url every 5s until status=completed (typical 90-130s on Kaggle).",
    }
    if debug:
        payload["debug_url"] = f"/tryon/debug/{job_id}"
        payload["message"] += " Debug step images available at debug_url when complete."
    return JSONResponse(payload)


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


@router.get("/tryon/debug/{job_id}", summary="List pipeline debug step outputs for a job")
async def tryon_debug_manifest(job_id: str) -> JSONResponse:
    if not _debug_dir(job_id).is_dir():
        raise HTTPException(
            status_code=404,
            detail="No debug output for this job. Re-run with ?debug=true on POST /tryon/async",
        )
    summary_path = _debug_dir(job_id) / "pipeline_summary.json"
    summary = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "job_id": job_id,
            "summary": summary,
            "steps": _list_debug_steps(job_id),
        }
    )


@router.get(
    "/tryon/debug/{job_id}/{filename}",
    summary="Download one pipeline debug artifact",
)
async def tryon_debug_file(job_id: str, filename: str) -> FileResponse:
    path = _resolve_debug_file(job_id, filename)
    media = "application/json" if filename.endswith(".json") else "text/plain"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media = "image/jpeg"
    elif path.suffix.lower() == ".png":
        media = "image/png"
    return FileResponse(path=str(path), media_type=media, filename=filename)


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
    debug: bool = Query(False, description="Save per-stage pipeline images under outputs/{job_id}/debug/"),
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
            debug,
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
