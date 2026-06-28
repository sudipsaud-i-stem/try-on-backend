from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import CartItem, Product
from app.schemas import AddCartItemRequest, CartItemResponse, UpdateCartItemRequest
from app.services.products import product_to_schema

router = APIRouter(prefix="/cart", tags=["cart"])


def _require_client_id(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id or len(x_client_id) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Client-Id header",
        )
    return x_client_id


def _cart_line_key(product_id: str, size: str) -> str:
    return f"{product_id}:{size}"


def _to_response(item: CartItem, product: Product) -> CartItemResponse:
    schema = product_to_schema(product)
    return CartItemResponse(
        key=_cart_line_key(product.id, item.size),
        productId=product.id,
        name=schema.name,
        gender=schema.gender,
        image=schema.image,
        priceNpr=schema.priceNpr,
        size=item.size,
        quantity=item.quantity,
    )


@router.get("", response_model=list[CartItemResponse])
def get_cart(
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> list[CartItemResponse]:
    rows = db.query(CartItem).filter(CartItem.client_id == client_id).all()
    responses: list[CartItemResponse] = []
    for row in rows:
        product = db.query(Product).filter(Product.id == row.product_id).one_or_none()
        if product:
            responses.append(_to_response(row, product))
    return responses


@router.post("", response_model=CartItemResponse, status_code=status.HTTP_201_CREATED)
def add_to_cart(
    body: AddCartItemRequest,
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> CartItemResponse:
    product = db.query(Product).filter(Product.id == body.productId).one_or_none()
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    sizes = product_to_schema(product).sizes
    if body.size not in sizes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid size")

    row = (
        db.query(CartItem)
        .filter(
            CartItem.client_id == client_id,
            CartItem.product_id == body.productId,
            CartItem.size == body.size,
        )
        .one_or_none()
    )
    if row:
        row.quantity += body.quantity
    else:
        row = CartItem(
            client_id=client_id,
            product_id=body.productId,
            size=body.size,
            quantity=body.quantity,
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return _to_response(row, product)


@router.patch("/{product_id}/{size}", response_model=CartItemResponse | None)
def update_cart_item(
    product_id: str,
    size: str,
    body: UpdateCartItemRequest,
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> CartItemResponse | None:
    row = (
        db.query(CartItem)
        .filter(
            CartItem.client_id == client_id,
            CartItem.product_id == product_id,
            CartItem.size == size,
        )
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart item not found")

    if body.quantity <= 0:
        db.delete(row)
        db.commit()
        return None

    row.quantity = body.quantity
    db.commit()
    db.refresh(row)
    product = db.query(Product).filter(Product.id == product_id).one()
    return _to_response(row, product)


@router.delete("/{product_id}/{size}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def remove_cart_item(
    product_id: str,
    size: str,
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> Response:
    row = (
        db.query(CartItem)
        .filter(
            CartItem.client_id == client_id,
            CartItem.product_id == product_id,
            CartItem.size == size,
        )
        .one_or_none()
    )
    if row:
        db.delete(row)
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def clear_cart(
    client_id: str = Depends(_require_client_id),
    db: Session = Depends(get_db),
) -> Response:
    db.query(CartItem).filter(CartItem.client_id == client_id).delete()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
