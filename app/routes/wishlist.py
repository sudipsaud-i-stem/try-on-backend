from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Product, WishlistItem
from app.schemas import ToggleWishlistRequest, WishlistItemResponse
from app.services.products import product_to_schema

router = APIRouter(prefix="/wishlist", tags=["wishlist"])


def _require_client_id(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id or len(x_client_id) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Client-Id header",
        )
    return x_client_id


@router.get("", response_model=list[WishlistItemResponse])
def get_wishlist(
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> list[WishlistItemResponse]:
    rows = db.query(WishlistItem).filter(WishlistItem.client_id == client_id).all()
    items: list[WishlistItemResponse] = []
    for row in rows:
        product = db.query(Product).filter(Product.id == row.product_id).one_or_none()
        if not product:
            continue
        schema = product_to_schema(product)
        items.append(
            WishlistItemResponse(
                productId=schema.id,
                name=schema.name,
                gender=schema.gender,
                image=schema.image,
                priceNpr=schema.priceNpr,
            )
        )
    return items


@router.post("/toggle", response_model=dict)
def toggle_wishlist(
    body: ToggleWishlistRequest,
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> dict:
    product = db.query(Product).filter(Product.id == body.productId).one_or_none()
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    row = (
        db.query(WishlistItem)
        .filter(WishlistItem.client_id == client_id, WishlistItem.product_id == body.productId)
        .one_or_none()
    )
    if row:
        db.delete(row)
        db.commit()
        return {"productId": body.productId, "wishlisted": False}

    db.add(WishlistItem(client_id=client_id, product_id=body.productId))
    db.commit()
    return {"productId": body.productId, "wishlisted": True}


@router.delete("", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def clear_wishlist(
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> Response:
    db.query(WishlistItem).filter(WishlistItem.client_id == client_id).delete()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
