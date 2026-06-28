from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Product
from app.schemas import ProductResponse
from app.services.products import product_to_schema

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=list[ProductResponse])
def list_products(
    gender: str | None = Query(default=None, pattern="^(men|women)$"),
    db: Session = Depends(get_db),
) -> list[ProductResponse]:
    query = db.query(Product)
    if gender:
        query = query.filter(Product.gender == gender)
    products = query.order_by(Product.gender, Product.name).all()
    return [product_to_schema(p) for p in products]


@router.get("/{gender}/{product_id}", response_model=ProductResponse)
def get_product(
    gender: str,
    product_id: str,
    db: Session = Depends(get_db),
) -> ProductResponse:
    product = (
        db.query(Product)
        .filter(Product.gender == gender, Product.id == product_id)
        .one_or_none()
    )
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return product_to_schema(product)
