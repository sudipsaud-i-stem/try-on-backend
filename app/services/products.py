from __future__ import annotations

import json

from app.db.models import Product
from app.schemas import ProductResponse


def product_to_schema(product: Product) -> ProductResponse:
    sizes = json.loads(product.sizes_json)
    return ProductResponse(
        id=product.id,
        name=product.name,
        gender=product.gender,
        image=product.image,
        priceNpr=product.price_npr,
        description=product.description,
        sizes=sizes,
        tag=product.tag,
    )
