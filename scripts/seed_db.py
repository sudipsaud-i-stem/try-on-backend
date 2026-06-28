#!/usr/bin/env python3
"""Seed or re-seed the SQLite product catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow: python scripts/seed_db.py (from virtual-tryon-backend/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import init_db, seed_products
from app.db.database import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed TrialOn SQLite database")
    parser.add_argument("--force", action="store_true", help="Replace existing products")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        count = seed_products(db, force=args.force)
    print(f"Done. Seeded {count} products.")


if __name__ == "__main__":
    main()
