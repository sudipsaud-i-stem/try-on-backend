#!/usr/bin/env python3
"""
Mask pipeline regression test — prints pass/fail table with logged diagnostics.

Usage:
  python scripts/test_mask_matrix.py --manifest tests/mask_matrix.json
  python scripts/test_mask_matrix.py --person path/to/person.jpg --garment path/to/garment.jpg

Manifest JSON format:
[
  {
    "name": "crouch-crew-long",
    "person": "tests/persons/crouch.jpg",
    "garment": "tests/garments/crew_long.jpg",
    "cloth_type": "upper",
    "expect": "pass"
  }
]

expect: "pass" | "reject" — pass means valid mask; reject means MaskValidationError expected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
from PIL import Image

from worker.catvton.image_utils import resize_and_padding
from worker.catvton.mask_service import generate_clothing_mask_full
from worker.exceptions import MaskValidationError
from worker.mask_pipeline import build_garment_mask
from worker.mask_refine import mask_shape_is_valid


def _run_case(
    name: str,
    person_path: Path,
    garment_path: Path,
    cloth_type: str,
    expect: str,
) -> dict:
    person_raw = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    person = resize_and_padding(person_raw, (768, 1024))

    primary = generate_clothing_mask_full(person, cloth_type=cloth_type)
    schp_atr = np.array(primary["schp_atr"])
    schp_lip = np.array(primary["schp_lip"])

    row = {
        "name": name,
        "expect": expect,
        "status": "FAIL",
        "error": None,
    }

    try:
        result = build_garment_mask(
            person=person,
            garment=garment,
            schp_mask=primary["mask"],
            schp_atr=schp_atr,
            schp_lip=schp_lip,
            cloth_type=cloth_type,
        )
        row.update(result.diagnostics)
        if expect == "reject":
            row["status"] = "FAIL"
            row["error"] = "Expected rejection but mask passed"
        else:
            valid, diag = mask_shape_is_valid(
                result.mask,
                schp_atr,
                schp_lip,
                person_bbox=result.body.person_bbox,
                keypoints=result.body.keypoints,
                sleeve_length=result.garment_profile.sleeve_length,
            )
            row.update(diag)
            row["status"] = "PASS" if valid else "FAIL"
            if not valid:
                row["error"] = "Mask failed validity checks"
    except MaskValidationError as exc:
        row["error"] = str(exc)
        row["code"] = exc.code
        row.update(exc.diagnostics or {})
        row["status"] = "PASS" if expect == "reject" else "FAIL"

    return row


def _print_table(rows: list[dict]) -> None:
    headers = [
        "name",
        "status",
        "expect",
        "coverage_bbox",
        "components",
        "symmetry",
        "fallback",
        "confidence",
        "neckline_class",
        "sleeve_class",
        "error",
    ]
    print("\n" + " | ".join(headers))
    print("-" * 120)
    for row in rows:
        print(
            " | ".join(
                [
                    str(row.get("name", "")),
                    str(row.get("status", "")),
                    str(row.get("expect", "")),
                    f"{row.get('mask_coverage_person_bbox', 0):.3f}",
                    str(row.get("connectivity_component_count", "")),
                    f"{row.get('symmetry_ratio', 0):.2f}",
                    str(row.get("used_fallback", "")),
                    f"{row.get('confidence', 0):.2f}",
                    str(row.get("garment_neckline_class", "")),
                    str(row.get("garment_sleeve_class", "")),
                    str(row.get("error", "") or "")[:40],
                ]
            )
        )

    passed = sum(1 for r in rows if r["status"] == "PASS")
    print(f"\n{passed}/{len(rows)} passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask pipeline regression matrix")
    parser.add_argument("--manifest", type=Path, help="JSON manifest of test cases")
    parser.add_argument("--person", type=Path, help="Single person image")
    parser.add_argument("--garment", type=Path, help="Single garment image")
    parser.add_argument("--cloth-type", default="upper")
    parser.add_argument("--expect", default="pass", choices=["pass", "reject"])
    args = parser.parse_args()

    rows: list[dict] = []
    if args.manifest:
        cases = json.loads(args.manifest.read_text(encoding="utf-8"))
        for case in cases:
            rows.append(
                _run_case(
                    case["name"],
                    BACKEND_ROOT / case["person"],
                    BACKEND_ROOT / case["garment"],
                    case.get("cloth_type", "upper"),
                    case.get("expect", "pass"),
                )
            )
    elif args.person and args.garment:
        rows.append(
            _run_case(
                args.person.stem,
                args.person,
                args.garment,
                args.cloth_type,
                args.expect,
            )
        )
    else:
        parser.error("Provide --manifest or both --person and --garment")

    _print_table(rows)
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
