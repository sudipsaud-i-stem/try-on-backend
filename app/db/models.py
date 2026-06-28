from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Product(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    gender: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(255))
    image: Mapped[str] = mapped_column(String(512))
    price_npr: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text)
    sizes_json: Mapped[str] = mapped_column(Text)
    tag: Mapped[str | None] = mapped_column(String(32), nullable=True)

    cart_items: Mapped[list["CartItem"]] = relationship(back_populates="product")
    wishlist_items: Mapped[list["WishlistItem"]] = relationship(back_populates="product")


class CartItem(Base):
    __tablename__ = "cart_items"
    __table_args__ = (UniqueConstraint("client_id", "product_id", "size", name="uq_cart_line"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    product_id: Mapped[str] = mapped_column(String(64), ForeignKey("products.id"))
    size: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    product: Mapped[Product] = relationship(back_populates="cart_items")


class WishlistItem(Base):
    __tablename__ = "wishlist_items"
    __table_args__ = (UniqueConstraint("client_id", "product_id", name="uq_wishlist_line"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(64), ForeignKey("products.id"))
    client_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    product: Mapped[Product] = relationship(back_populates="wishlist_items")


class TryOnRequest(Base):
    __tablename__ = "tryon_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ip_address: Mapped[str] = mapped_column(String(64), index=True)
    client_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    job_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
