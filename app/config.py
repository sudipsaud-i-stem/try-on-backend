from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: virtual-tryon-backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"
    API_KEY: str = "dev-secret-key-change-in-production"
    ADMIN_API_KEY: str = "dev-admin-key-change-in-production"

    # Database
    DATABASE_PATH: Path = BACKEND_ROOT / "data" / "db" / "trialon.db"

    # Rate limiting (no login — per IP)
    TRYON_RATE_LIMIT: int = 10
    TRYON_RATE_WINDOW_HOURS: int = 1

    # Storage paths (relative to backend root by default)
    UPLOAD_DIR: Path = BACKEND_ROOT / "data" / "uploads"
    OUTPUT_DIR: Path = BACKEND_ROOT / "data" / "outputs"
    MODEL_CACHE_DIR: Path = BACKEND_ROOT / "models"

    # ML settings
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "float16"
    INFERENCE_STEPS: int = 50
    GUIDANCE_SCALE: float = 3.0
    OUTPUT_WIDTH: int = 768
    OUTPUT_HEIGHT: int = 1024
    MASK_BLUR_FACTOR: int = 5
    CLOTH_TYPE: str = "upper"
    INFERENCE_SEED: int = 42
    COLOR_PRESERVE_STRENGTH: float = 0.0
    ENABLE_XFORMERS: bool = False
    ENABLE_CPU_OFFLOAD: bool = False
    ENABLE_ATTENTION_SLICING: bool = False

    # Model identifiers
    CATVTON_MODEL_ID: str = "zhengchong/CatVTON"
    CATVTON_BASE_MODEL_ID: str = "runwayml/stable-diffusion-inpainting"
    CATVTON_ATTN_VERSION: str = "mix"

    # Monitoring
    SENTRY_DSN: str = ""
    ENABLE_PROMETHEUS: bool = True
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def device(self) -> torch.device:
        if self.DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.TORCH_DTYPE == "float16":
            return torch.float16
        return torch.float32

    @property
    def catvton_model_path(self) -> Path:
        return self.MODEL_CACHE_DIR / "catvton"

    def ensure_directories(self) -> None:
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> Settings:
        for name in ("UPLOAD_DIR", "OUTPUT_DIR", "MODEL_CACHE_DIR", "DATABASE_PATH"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, (BACKEND_ROOT / value).resolve())
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
