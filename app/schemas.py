from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_total_mb: Optional[int] = None
    gpu_vram_free_mb: Optional[int] = None
    uptime_seconds: float
    database: Literal["ok", "error"] = "ok"
    product_count: int = 0


class ProductResponse(BaseModel):
    id: str
    name: str
    gender: Literal["men", "women"]
    image: str
    priceNpr: int
    description: str
    sizes: list[str]
    tag: Optional[str] = None


class CartItemResponse(BaseModel):
    key: str
    productId: str
    name: str
    gender: Literal["men", "women"]
    image: str
    priceNpr: int
    size: str
    quantity: int


class AddCartItemRequest(BaseModel):
    productId: str
    size: str
    quantity: int = Field(default=1, ge=1, le=99)


class UpdateCartItemRequest(BaseModel):
    quantity: int = Field(ge=0, le=99)


class WishlistItemResponse(BaseModel):
    productId: str
    name: str
    gender: Literal["men", "women"]
    image: str
    priceNpr: int


class ToggleWishlistRequest(BaseModel):
    productId: str
