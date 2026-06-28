"""Catalog seed data — mirrors frontend/lib/catalog.ts."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedProduct:
    id: str
    gender: str
    name: str
    image: str
    price_npr: int
    description: str
    sizes: list[str]
    tag: str | None = None


MEN_NAMES = [
    "Oxford Classic Shirt",
    "Urban Denim Jacket",
    "Slim Fit Polo",
    "Heritage Wool Blazer",
    "Essential Crew Tee",
    "Performance Track Jacket",
    "Linen Summer Shirt",
    "Merino Knit Sweater",
    "Tailored Chino Look",
    "Streetwear Hoodie",
    "Premium Bomber",
    "Casual Button-Down",
    "Modern Fit Henley",
    "City Commuter Coat",
    "Weekend Flannel",
    "Studio Overshirt",
]

WOMEN_NAMES = [
    "Silk Wrap Blouse",
    "Tailored Midi Dress",
    "Cashmere Crew Knit",
    "High-Rise Wide Leg",
    "Satin Evening Top",
    "Structured Blazer",
    "Floral Day Dress",
    "Ribbed Knit Set",
    "Linen Resort Shirt",
    "Cropped Denim Jacket",
    "Soft Drape Cami",
    "Belted Trench Coat",
    "Minimal Slip Dress",
    "Cable Knit Cardigan",
    "City Trousers",
    "Statement Sleeve Top",
]

MEN_DESCRIPTIONS = [
    "Crisp cotton oxford with a relaxed drape — ideal for office wear or weekend outings in Kathmandu.",
    "Classic denim jacket with structured shoulders and a versatile indigo wash for all seasons.",
    "Breathable slim-fit polo crafted for comfort during warm days and casual evenings.",
    "Tailored wool-blend blazer with refined lapels, perfect for formal events and business meetings.",
    "Soft everyday crew tee with a premium hand-feel and minimal branding.",
    "Lightweight track jacket with zip closure — great for travel and active city days.",
    "Airy linen shirt designed for summer heat with a clean, modern silhouette.",
    "Fine merino knit sweater offering warmth without bulk — layers easily under coats.",
    "Smart chino-inspired look with a tapered leg and neutral tone for daily wear.",
    "Cozy streetwear hoodie with a relaxed fit and durable fleece interior.",
    "Premium bomber jacket with ribbed cuffs and a sleek, contemporary finish.",
    "Easy casual button-down in a versatile shade — pairs with jeans or trousers.",
    "Modern henley with a flattering fit and soft jersey fabric.",
    "City-ready commuter coat with clean lines and practical pocket detailing.",
    "Weekend flannel with a brushed texture and timeless check pattern.",
    "Studio overshirt that works as a light layer or standalone statement piece.",
]

WOMEN_DESCRIPTIONS = [
    "Elegant silk-blend wrap blouse with a flattering neckline for day-to-evening wear.",
    "Tailored midi dress with a structured waist — polished enough for work or celebrations.",
    "Luxurious cashmere-feel crew knit with a soft drape and refined finish.",
    "High-rise wide-leg trousers with a flowing silhouette and comfortable waistband.",
    "Satin evening top with a subtle sheen — perfect for dinners and special occasions.",
    "Structured blazer with sharp tailoring to elevate any outfit instantly.",
    "Floral day dress with a breezy fit, ideal for brunches and outdoor events.",
    "Ribbed knit co-ord set offering stretch, comfort, and a coordinated look.",
    "Resort-style linen shirt with a relaxed cut for warm weather and travel.",
    "Cropped denim jacket with a modern length and classic wash.",
    "Soft drape cami with delicate straps — layers beautifully under blazers.",
    "Belted trench coat with timeless details and a confident, structured shape.",
    "Minimal slip dress with clean lines for effortless elegance.",
    "Cable knit cardigan with cozy texture and button-front styling.",
    "City trousers with a tailored fit and all-day comfort.",
    "Statement sleeve top with volume details and a contemporary edge.",
]

SIZE_POOLS = [
    ["S", "M", "L", "XL"],
    ["XS", "S", "M", "L"],
    ["M", "L", "XL", "XXL"],
    ["S", "M", "L"],
    ["XS", "S", "M", "L", "XL"],
]

MEN_FILES = [
    "789.jpg", "806.jpg", "823.jpg", "869.jpg", "871.jpg", "912.jpg", "934.jpg",
    "948.jpg", "963.jpg", "965.jpg", "972.jpg", "973.jpg", "982.jpg", "1905.jpg", "1998.jpg",
]

WOMEN_FILES = [
    "10448_00.jpg", "10548_00.jpg", "10723_00.jpg", "10947_00.jpg", "12130_00.jpg",
    "12219_00.jpg", "13079_00.jpg", "13204_00.jpg", "13562_00.jpg", "14059_00.jpg",
    "14107_00.jpg", "14112_00.jpg", "14173_00.jpg", "14212_00.jpg", "14458_00.jpg",
    "14533_00.jpg", "14627_00.jpg",
]


def _build_products(
    gender: str,
    files: list[str],
    names: list[str],
    descriptions: list[str],
    base_price: int,
) -> list[SeedProduct]:
    products: list[SeedProduct] = []
    for index, file in enumerate(files):
        product_id = file.rsplit(".", 1)[0]
        tag = None
        if index % 4 == 0:
            tag = "New"
        elif index % 5 == 0:
            tag = "Bestseller"
        products.append(
            SeedProduct(
                id=product_id,
                gender=gender,
                name=names[index % len(names)],
                image=f"/clothes/{gender}/{file}",
                price_npr=base_price + (index % 5) * 450 + (index % 3) * 250,
                description=descriptions[index % len(descriptions)],
                sizes=SIZE_POOLS[index % len(SIZE_POOLS)],
                tag=tag,
            )
        )
    return products


def all_seed_products() -> list[SeedProduct]:
    return _build_products("men", MEN_FILES, MEN_NAMES, MEN_DESCRIPTIONS, 2499) + _build_products(
        "women", WOMEN_FILES, WOMEN_NAMES, WOMEN_DESCRIPTIONS, 2999
    )


def sizes_to_json(sizes: list[str]) -> str:
    return json.dumps(sizes)
