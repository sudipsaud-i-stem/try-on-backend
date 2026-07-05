from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from worker.catvton.model.cloth_masker import ATR_MAPPING, LIP_MAPPING, part_mask_of

_YOLO_MODEL = None


@dataclass
class BodyKeypoints:
    """Normalized [0,1] keypoints; None when not visible/confident."""

    neck: tuple[float, float] | None = None
    left_shoulder: tuple[float, float] | None = None
    right_shoulder: tuple[float, float] | None = None
    left_elbow: tuple[float, float] | None = None
    right_elbow: tuple[float, float] | None = None
    left_wrist: tuple[float, float] | None = None
    right_wrist: tuple[float, float] | None = None
    left_hip: tuple[float, float] | None = None
    right_hip: tuple[float, float] | None = None
    chin: tuple[float, float] | None = None
    source: str = "none"
    confidence: float = 0.0
    side_pose: bool = False


@dataclass
class BodyEnvelope:
    mask: np.ndarray
    person_bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    keypoints: BodyKeypoints = field(default_factory=BodyKeypoints)


def _bbox_from_mask(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x0, x1 = max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad)
    return x0, y0, x1, y1


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()) / mask.shape[1], float(ys.mean()) / mask.shape[0]


def _arm_keypoints_from_schp(
    schp_atr: np.ndarray,
    schp_lip: np.ndarray,
    side: str,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None]:
    label = f"{side}-arm"
    arm = (
        part_mask_of(label, schp_lip, LIP_MAPPING)
        | part_mask_of(label, schp_atr, ATR_MAPPING)
    )
    if not arm.any():
        return None, None, None
    h, w = arm.shape[:2]
    ys, xs = np.where(arm > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    shoulder = (float((x0 + x1) / 2) / w, float(y0) / h)
    wrist = (float((x0 + x1) / 2) / w, float(y1) / h)
    elbow_y = int(y0 + (y1 - y0) * 0.45)
    elbow_xs = xs[ys == elbow_y] if np.any(ys == elbow_y) else xs
    elbow = (float(elbow_xs.mean()) / w, float(elbow_y) / h)
    return shoulder, elbow, wrist


def _keypoints_from_schp(schp_atr: np.ndarray, schp_lip: np.ndarray) -> BodyKeypoints:
    h, w = schp_atr.shape[:2]
    face = (
        part_mask_of("Face", schp_lip, LIP_MAPPING)
        | part_mask_of("Face", schp_atr, ATR_MAPPING)
    )
    kp = BodyKeypoints(source="schp", confidence=0.45)
    if face.any():
        fy, fx = np.where(face > 0)
        kp.chin = (float(fx.mean()) / w, float(fy.max()) / h)
        kp.neck = (float(fx.mean()) / w, min(1.0, float(fy.max()) / h + 0.04))

    ls, le, lw = _arm_keypoints_from_schp(schp_atr, schp_lip, "Left")
    rs, re, rw = _arm_keypoints_from_schp(schp_atr, schp_lip, "Right")
    kp.left_shoulder, kp.left_elbow, kp.left_wrist = ls, le, lw
    kp.right_shoulder, kp.right_elbow, kp.right_wrist = rs, re, rw

    for leg_side, attr in (("Left", "left_hip"), ("Right", "right_hip")):
        leg = (
            part_mask_of(f"{leg_side}-leg", schp_lip, LIP_MAPPING)
            | part_mask_of(f"{leg_side}-leg", schp_atr, ATR_MAPPING)
        )
        if leg.any():
            ys, xs = np.where(leg > 0)
            setattr(kp, attr, (float(xs.mean()) / w, float(ys.min()) / h))

    if ls and rs:
        shoulder_dx = abs(ls[0] - rs[0])
        kp.side_pose = shoulder_dx < 0.12
        kp.confidence = 0.55 + min(0.25, shoulder_dx * 2.0)
    return kp


def _keypoints_from_yolo(image: Image.Image) -> BodyKeypoints | None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return None

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8n-pose.pt")
    results = _YOLO_MODEL.predict(rgb, verbose=False, conf=0.25)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return None
    xy = kps.xy[0].cpu().numpy()
    conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

    # COCO order: 0 nose, 5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow,
    # 9 L wrist, 10 R wrist, 11 L hip, 12 R hip
    def pt(i: int, min_conf: float = 0.25) -> tuple[float, float] | None:
        if i >= len(xy) or conf[i] < min_conf:
            return None
        return float(xy[i][0]) / w, float(xy[i][1]) / h

    kp = BodyKeypoints(
        neck=pt(0, 0.2),
        chin=pt(0, 0.2),
        left_shoulder=pt(5),
        right_shoulder=pt(6),
        left_elbow=pt(7),
        right_elbow=pt(8),
        left_wrist=pt(9),
        right_wrist=pt(10),
        left_hip=pt(11),
        right_hip=pt(12),
        source="yolo",
        confidence=float(conf[5:11].mean()) if len(conf) > 10 else 0.5,
    )
    if kp.left_shoulder and kp.right_shoulder:
        kp.side_pose = abs(kp.left_shoulder[0] - kp.right_shoulder[0]) < 0.10
    return kp


def estimate_body_keypoints(
    person_image: Image.Image,
    schp_atr: np.ndarray | None = None,
    schp_lip: np.ndarray | None = None,
) -> BodyKeypoints:
    yolo_kp = _keypoints_from_yolo(person_image)
    if yolo_kp is not None and yolo_kp.confidence >= 0.35:
        return yolo_kp
    if schp_atr is not None and schp_lip is not None:
        return _keypoints_from_schp(schp_atr, schp_lip)
    return BodyKeypoints(source="none", confidence=0.0)


def _draw_limb(
    canvas: np.ndarray,
    p0: tuple[float, float] | None,
    p1: tuple[float, float] | None,
    radius: int,
) -> None:
    if p0 is None or p1 is None:
        return
    h, w = canvas.shape[:2]
    x0, y0 = int(p0[0] * w), int(p0[1] * h)
    x1, y1 = int(p1[0] * w), int(p1[1] * h)
    cv2.line(canvas, (x0, y0), (x1, y1), 255, thickness=max(radius * 2, 6))


def build_body_envelope(
    shape: tuple[int, int],
    keypoints: BodyKeypoints,
    sleeve_length: str = "long",
    fit: str = "regular",
) -> BodyEnvelope:
    """Torso + visible arm polygons independent of clothing labels."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    ls, rs = keypoints.left_shoulder, keypoints.right_shoulder
    lh, rh = keypoints.left_hip, keypoints.right_hip
    neck = keypoints.neck or keypoints.chin

    if ls and rs and lh and rh:
        neck_pt = neck or ((ls[0] + rs[0]) / 2, min(ls[1], rs[1]) - 0.03)
        pts = np.array(
            [
                [int(neck_pt[0] * w), int(max(0, neck_pt[1] * h))],
                [int(ls[0] * w), int(ls[1] * h)],
                [int(lh[0] * w), int(lh[1] * h)],
                [int(rh[0] * w), int(rh[1] * h)],
                [int(rs[0] * w), int(rs[1] * h)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], 255)

    fit_scale = {"fitted": 0.92, "regular": 1.0, "oversized": 1.12}.get(fit, 1.0)
    arm_radius = max(6, int(min(h, w) * 0.035 * fit_scale))

    if sleeve_length != "sleeveless":
        for shoulder, elbow, wrist in (
            (keypoints.left_shoulder, keypoints.left_elbow, keypoints.left_wrist),
            (keypoints.right_shoulder, keypoints.right_elbow, keypoints.right_wrist),
        ):
            if shoulder is None:
                continue
            end = wrist
            if sleeve_length == "short" and elbow is not None:
                end = elbow
            elif sleeve_length == "three_quarter" and elbow is not None and wrist is not None:
                end = (
                    elbow[0] * 0.35 + wrist[0] * 0.65,
                    elbow[1] * 0.35 + wrist[1] * 0.65,
                )
            if end is None and elbow is not None:
                end = elbow
            _draw_limb(canvas, shoulder, end, arm_radius)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    bbox = _bbox_from_mask(canvas)
    return BodyEnvelope(mask=canvas, person_bbox=bbox, keypoints=keypoints)


def estimate_body_envelope(
    person_image: Image.Image,
    schp_atr: np.ndarray | None = None,
    schp_lip: np.ndarray | None = None,
    sleeve_length: str = "long",
    fit: str = "regular",
) -> BodyEnvelope:
    arr = np.array(person_image.convert("RGB"))
    h, w = arr.shape[:2]
    keypoints = estimate_body_keypoints(person_image, schp_atr, schp_lip)
    envelope = build_body_envelope((h, w), keypoints, sleeve_length=sleeve_length, fit=fit)

    if envelope.mask.max() == 0 and schp_atr is not None and schp_lip is not None:
        parts = ["Face", "Hair", "Upper-clothes", "Left-arm", "Right-arm", "Pants", "Dress"]
        person = (
            part_mask_of(parts, schp_lip, LIP_MAPPING)
            | part_mask_of(parts, schp_atr, ATR_MAPPING)
        ).astype(np.uint8) * 255
        if person.max() > 0:
            envelope = BodyEnvelope(
                mask=person,
                person_bbox=_bbox_from_mask(person),
                keypoints=keypoints,
            )
    if envelope.mask.max() == 0:
        from worker.postprocess import grabcut_person_mask

        person = grabcut_person_mask(person_image)
        envelope = BodyEnvelope(
            mask=person,
            person_bbox=_bbox_from_mask(person),
            keypoints=keypoints,
        )
    return envelope
