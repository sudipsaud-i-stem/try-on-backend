from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def mediapipe_available() -> bool:
    try:
        import mediapipe as mp  # noqa: F401

        _ = mp.solutions.hands
        return True
    except Exception:
        return False


def build_mediapipe_protect_mask(image: Image.Image) -> np.ndarray | None:
    """
    Face + hand regions from MediaPipe (Kaggle fallback when DensePose missing).
    """
    if not mediapipe_available():
        return None

    import mediapipe as mp

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    protect = np.zeros((h, w), dtype=np.float32)

    mp_face = mp.solutions.face_detection
    mp_hands = mp.solutions.hands

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45) as face_det:
        result = face_det.process(rgb)
        if result.detections:
            for det in result.detections:
                box = det.location_data.relative_bounding_box
                x0 = max(0, int(box.xmin * w) - int(box.width * w * 0.35))
                y0 = max(0, int(box.ymin * h) - int(box.height * h * 0.45))
                x1 = min(w, int((box.xmin + box.width) * w) + int(box.width * w * 0.35))
                y1 = min(h, int((box.ymin + box.height) * h) + int(box.height * h * 0.25))
                protect[y0:y1, x0:x1] = 1.0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.35,
    ) as hands:
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                xs = [int(lm.x * w) for lm in hand.landmark]
                ys = [int(lm.y * h) for lm in hand.landmark]
                pad = max(12, int(max(h, w) * 0.025))
                x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
                y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
                protect[y0:y1, x0:x1] = 1.0

    if float(protect.max()) < 0.05:
        return None
    return protect
