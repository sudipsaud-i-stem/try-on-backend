from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.db import init_db, seed_products
from app.db.database import SessionLocal
from app.routes import cart, health, products, tryon, wishlist


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings.ensure_directories()

    init_db()
    with SessionLocal() as db:
        seed_products(db)

    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info("=" * 60)
    logger.info("Virtual Try-On API starting (production mode)")
    logger.info("Host: {}:{}", settings.API_HOST, settings.API_PORT)
    logger.info("GPU: {} (CUDA available: {})", gpu_name, torch.cuda.is_available())
    logger.info("Database: {}", settings.DATABASE_PATH)
    logger.info("Upload dir: {}", settings.UPLOAD_DIR)
    logger.info("Output dir: {}", settings.OUTPUT_DIR)
    logger.info("Model cache: {}", settings.MODEL_CACHE_DIR)
    logger.info("Rate limit: {} try-ons / {} hour(s) per IP", settings.TRYON_RATE_LIMIT, settings.TRYON_RATE_WINDOW_HOURS)
    logger.info("HUBA pipeline: {}", "enabled" if settings.ENABLE_HUBA_PIPELINE else "legacy")
    logger.info("=" * 60)

    if settings.ENABLE_HUBA_PIPELINE or torch.cuda.is_available():
        try:
            from worker.inference import preload_inference_models

            preload_inference_models()
        except Exception as exc:
            logger.warning("Model preload deferred (will load on first request): {}", exc)

    yield

    logger.info("Virtual Try-On API shutting down cleanly")


app = FastAPI(
    title="TrialOn Virtual Try-On API",
    version="2.0.0",
    description="Production API: products, cart, wishlist, try-on (SQLite + CatVTON GPU).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tryon.router)
app.include_router(health.router)
app.include_router(products.router)
app.include_router(cart.router)
app.include_router(wishlist.router)

if settings.ENABLE_PROMETHEUS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

if settings.SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[FastApiIntegration()],
    )


@app.get("/")
async def root() -> dict[str, str | int]:
    return {
        "service": "trialon-api",
        "version": "2.0.0",
        "docs": "/docs",
        "tryon": "POST /tryon",
        "products": "GET /products",
        "cart": "GET /cart",
        "wishlist": "GET /wishlist",
        "rate_limit_per_hour": settings.TRYON_RATE_LIMIT,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on {} {}", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
