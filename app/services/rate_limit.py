from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import TryOnRequest


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def get_client_id(request: Request) -> str | None:
    return request.headers.get("X-Client-Id") or request.headers.get("x-client-id")


def check_rate_limit(db: Session, ip_address: str) -> None:
    """Raise HTTP 429 if IP exceeded TRYON_RATE_LIMIT in the configured window."""
    window_start = datetime.now(timezone.utc) - timedelta(hours=settings.TRYON_RATE_WINDOW_HOURS)
    count = (
        db.query(TryOnRequest)
        .filter(
            TryOnRequest.ip_address == ip_address,
            TryOnRequest.created_at >= window_start,
        )
        .count()
    )
    if count >= settings.TRYON_RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Try-on limit reached ({settings.TRYON_RATE_LIMIT} per hour). "
                "Please wait and try again later."
            ),
        )


def record_tryon_request(
    db: Session,
    *,
    request_id: str,
    job_id: str,
    ip_address: str,
    client_id: str | None,
    status_value: str = "pending",
    error_message: str | None = None,
) -> TryOnRequest:
    row = TryOnRequest(
        id=request_id,
        job_id=job_id,
        ip_address=ip_address,
        client_id=client_id,
        status=status_value,
        error_message=error_message,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_tryon_status(
    db: Session,
    request_id: str,
    status_value: str,
    error_message: str | None = None,
) -> None:
    row = db.query(TryOnRequest).filter(TryOnRequest.id == request_id).one_or_none()
    if row is None:
        return
    row.status = status_value
    row.error_message = error_message
    db.commit()
