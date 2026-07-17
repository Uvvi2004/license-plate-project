"""YOLO plate detection for Pi deployment - onnxruntime instead of ultralytics/torch.

Same public interface as license_plate_pipeline.detection (get_model/detect_boxes),
so license_plate_pipeline.pi.pipeline can mirror the dev-machine pipeline.py exactly.

Validated against the dev-machine best.pt on test_images/demo2.jpg before this was
written: ONNX gave box=(251.5,114.5,406.2,205.4) conf=0.876 vs. best.pt's
box=(252.2,115.0,405.9,205.5) conf=0.874 - sub-pixel difference, expected
floating-point/letterbox-rounding variance, not a regression.
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort

from license_plate_pipeline.config import PAD_RATIO, PROJECT_ROOT

logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "best.onnx"
IMGSZ = 640
CONF_THRESHOLD = 0.25

_session = None


def get_model():
    global _session
    if _session is None:
        logger.info("Loading ONNX detection model from %s", ONNX_MODEL_PATH)
        _session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def pad_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=PAD_RATIO):
    # Duplicated from license_plate_pipeline.detection rather than imported, so this
    # module never pulls in ultralytics/torch - the whole point of the ONNX path.
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def _letterbox(img, new_shape=(IMGSZ, IMGSZ), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)


def detect_boxes(image):
    """Run detection on an already-loaded BGR image array.

    Returns a list of (x1, y1, x2, y2) padded boxes, in the original image's
    coordinate space. Returns an empty list - rather than raising - if inference
    fails, so a single bad frame doesn't crash a video/live loop.
    """
    h, w = image.shape[:2]

    try:
        padded, scale, (pad_x, pad_y) = _letterbox(image)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]

        output = get_model().run(None, {"images": blob})[0]  # [1, 300, 6]
    except Exception:
        logger.exception("Detection failed on this frame - skipping it")
        return []

    boxes = []
    for x1, y1, x2, y2, conf, _cls in output[0]:
        if conf < CONF_THRESHOLD:
            continue
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        boxes.append(pad_box(int(x1), int(y1), int(x2), int(y2), w, h))
    return boxes
