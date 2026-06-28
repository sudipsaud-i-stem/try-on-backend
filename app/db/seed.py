from __future__ import annotations

from loguru import logger
from sqlalchemy.orm import Session

from app.db.models import Product
from app.db.seed_data import all_seed_products, sizes_to_json


def seed_products(db: Session, force: bool = False) -> int:
    """Insert catalog products if the table is empty (or force re-seed)."""
    existing = db.query(Product).count()
    if existing > 0 and not force:
        logger.info("Database already has {} products — skipping seed", existing)
        return 0

    if force and existing > 0:
        logger.warning("Force re-seeding: removing {} existing products", existing)
        db.query(Product).delete()
        db.commit()

    count = 0
    for item in all_seed_products():
        db.add(
            Product(
                id=item.id,
                gender=item.gender,
                name=item.name,
                image=item.image,
                price_npr=item.price_npr,
                description=item.description,
                sizes_json=sizes_to_json(item.sizes),
                tag=item.tag,
            )
        )
        count += 1

    db.commit()
    logger.info("Seeded {} products into SQLite", count)
    return count
