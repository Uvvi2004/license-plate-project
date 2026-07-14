"""YOLO plate detection: model loading and bounding-box helpers."""

import logging

from ultralytics import YOLO  # import before paddleocr - see PROJECT_CONTEXT.md "Import order" note

from license_plate_pipeline.config import MODEL_PATH, PAD_RATIO

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading detection model from %s", MODEL_PATH)
        _model = YOLO(str(MODEL_PATH))
    return _model


def pad_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=PAD_RATIO):
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def detect_boxes(image):
    """Run detection on an image (path or already-loaded array).

    Returns a list of (x1, y1, x2, y2) padded boxes. Returns an empty list -
    rather than raising - if inference fails on this particular frame/image,
    so a single bad frame in a video loop doesn't crash the whole run.
    """
    h, w = image.shape[:2] if hasattr(image, "shape") else (None, None)

    try:
        results = get_model().predict(source=image, save=False, verbose=False)
    except Exception:
        logger.exception("Detection failed on this frame - skipping it")
        return []

    boxes = []
    for r in results:
        if h is None:
            h, w = r.orig_shape
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append(pad_box(x1, y1, x2, y2, w, h))
    return boxes
