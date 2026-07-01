"""
Local smoke tests — run without production GPU.

Usage:
  python scripts/smoke_local.py              # fast checks + pipeline stages 0-2
  python scripts/smoke_local.py --full       # includes one CatVTON try-on (~3-5 min)
  python scripts/smoke_local.py --api        # hit running server at localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = PROJECT_ROOT / "data" / "fixtures"
PERSON_URL = "https://picsum.photos/seed/trialon-person/768/1024"


def _create_garment_fixture(dest: Path) -> None:
    """Simple flat-lay shirt for pipeline testing when no CDN garment is available."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (768, 1024), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 120, 588, 900), fill=(30, 80, 180))
    draw.rectangle((120, 120, 180, 420), fill=(30, 80, 180))
    draw.rectangle((588, 120, 648, 420), fill=(30, 80, 180))
    draw.ellipse((330, 100, 438, 190), fill=(250, 250, 250))
    img.save(dest, quality=95)


def download_fixtures() -> tuple[Path, Path]:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    person = FIXTURES / "person.jpg"
    garment = FIXTURES / "garment.jpg"
    if not person.exists() or person.stat().st_size < 10_000:
        print(f"Downloading {person.name} ...")
        urllib.request.urlretrieve(PERSON_URL, person)
    if not garment.exists() or garment.stat().st_size < 5_000:
        print(f"Creating {garment.name} (synthetic flat-lay) ...")
        _create_garment_fixture(garment)
    return person, garment


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def check_models() -> bool:
    from app.config import settings

    print("\n== Model weights ==")
    model_path = settings.catvton_model_path
    required = [
        model_path / "mix-48k-1024" / "attention" / "model.safetensors",
        model_path / "SCHP" / "exp-schp-201908301523-atr.pth",
    ]
    ok = True
    for path in required:
        if path.exists() and path.stat().st_size > 1_000_000:
            _ok(f"{path.name} ({path.stat().st_size // (1024 * 1024)} MB)")
        else:
            _fail(f"Missing or incomplete: {path}")
            ok = False
    return ok


def check_gpu() -> bool:
    import torch

    print("\n== GPU ==")
    if not torch.cuda.is_available():
        _warn("CUDA not available — full try-on will fail or be very slow on CPU")
        return False
    name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    _ok(f"{name} ({total_mb} MB VRAM)")
    if total_mb < 8000:
        _warn("Less than 8 GB VRAM — keep ENABLE_BIREFNET/GFPGAN/REALESRGAN=false")
    return True


def check_imports() -> bool:
    print("\n== Pipeline imports ==")
    try:
        from worker.pipeline.orchestrator import TryOnOrchestrator  # noqa: F401

        _ok("TryOnOrchestrator")
        from worker.pipeline import stage0_quality, stage1_parsing, stage2_matting  # noqa: F401

        _ok("stages 0-2")
        return True
    except Exception as exc:
        _fail(f"Import error: {exc}")
        return False


def check_pipeline_stages(person: Path, garment: Path) -> bool:
    from PIL import Image

    from worker.pipeline.orchestrator import TryOnOrchestrator
    from worker.pipeline.stage0_quality import run_stage0_quality
    from worker.pipeline.stage1_parsing import run_stage1_parsing
    from worker.pipeline.stage2_matting import run_stage2_matting
    from worker.pipeline.types import PipelineContext

    print("\n== Pipeline stages 0-2 (no CatVTON) ==")
    person_img = Image.open(person).convert("RGB")
    garment_img = Image.open(garment).convert("RGB")
    ctx = PipelineContext(original_person=person_img, garment=garment_img, cloth_type="upper")

    run_stage0_quality(ctx)
    assert ctx.person is not None
    _ok(f"stage0 blur={ctx.quality.blur_score:.1f} upscaled={ctx.quality.upscaled}")

    run_stage1_parsing(ctx)
    assert ctx.inpaint_mask is not None
    _ok(
        f"stage1 confidence={ctx.parse.confidence:.2f} "
        f"fallback={ctx.parse.used_fallback} coverage={ctx.parse.mask_coverage:.2f}"
    )

    run_stage2_matting(ctx)
    assert ctx.alpha_matte is not None
    _ok(f"stage2 matte size={ctx.alpha_matte.size}")

    debug_dir = PROJECT_ROOT / "data" / "outputs" / "smoke_test" / "debug"
    TryOnOrchestrator(infer_fn=lambda x: None)._save_debug(ctx, debug_dir)
    _ok(f"debug images -> {debug_dir}")
    return True


def check_full_tryon(person: Path, garment: Path) -> bool:
    import torch

    from app.config import settings

    print("\n== Full try-on (CatVTON) ==")
    if not torch.cuda.is_available():
        _warn("Skipping — no CUDA")
        return False

    out = PROJECT_ROOT / "data" / "outputs" / "smoke_test" / "result.jpg"
    settings.PIPELINE_DEBUG = True

    from worker.inference import run_inference_direct

    t0 = time.time()
    try:
        run_inference_direct(str(person), str(garment), out, cloth_type="upper")
    except Exception as exc:
        _fail(f"Try-on failed: {exc}")
        return False

    elapsed = time.time() - t0
    if out.exists() and out.stat().st_size > 5000:
        _ok(f"result saved ({out.stat().st_size // 1024} KB, {elapsed:.0f}s)")
        return True
    _fail("Result image missing or too small")
    return False


def check_api(base: str) -> bool:
    print(f"\n== API ({base}) ==")
    ok = True
    for path in ("/health", "/health/gpu", "/products?gender=men"):
        url = f"{base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = resp.read().decode()
                data = json.loads(body) if body.startswith("{") or body.startswith("[") else body
                _ok(f"GET {path} -> {resp.status}")
                if path == "/health" and isinstance(data, dict):
                    print(f"         gpu={data.get('gpu_available')} products={data.get('product_count')}")
        except Exception as exc:
            _fail(f"GET {path}: {exc}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Local smoke tests for TrialOn backend")
    parser.add_argument("--full", action="store_true", help="Run full CatVTON try-on")
    parser.add_argument("--api", action="store_true", help="Test HTTP API (server must be running)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    print("TrialOn local smoke test")
    print("=" * 40)

    results: list[bool] = []
    results.append(check_imports())
    results.append(check_models())
    results.append(check_gpu())

    person, garment = download_fixtures()
    _ok(f"fixtures: {person.name}, {garment.name}")

    try:
        results.append(check_pipeline_stages(person, garment))
    except Exception as exc:
        _fail(f"Pipeline stages: {exc}")
        results.append(False)

    if args.full:
        results.append(check_full_tryon(person, garment))

    if args.api:
        results.append(check_api(args.api_url))

    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    if not args.full:
        print("Tip: run with --full for end-to-end CatVTON (needs GPU, ~3-5 min)")
    if not args.api:
        print("Tip: start uvicorn then run with --api to test HTTP endpoints")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
