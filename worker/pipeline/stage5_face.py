from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from worker.pipeline.types import PipelineContext


def _detect_face_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    if w * h < 80 * 80:
        return None
    return int(x), int(y), int(w), int(h)


def run_stage5_face(ctx: PipelineContext) -> Image.Image:
    """GFPGAN face restoration gated by face-detection confidence."""
    source = ctx.blended or ctx.vton_result
    if source is None:
        raise RuntimeError("stage5 requires blended or vton image")

    if not settings.ENABLE_GFPGAN:
        ctx.final = source
        ctx.log("stage5: skipped (GFPGAN disabled)")
        return source

    rgb = np.array(source.convert("RGB"))
    bbox = _detect_face_bbox(rgb)
    if bbox is None:
        ctx.final = source
        ctx.log("stage5: no confident face detected — skipped")
        return source

    try:
        from worker.pipeline.optional_models import restore_face_gfpgan

        restored = restore_face_gfpgan(source)
        if restored is not None:
            ctx.final = restored
            ctx.log("stage5: GFPGAN face restoration")
            return restored
    except Exception as exc:
        ctx.log(f"stage5: GFPGAN unavailable ({exc})")

    ctx.final = source
    return source
