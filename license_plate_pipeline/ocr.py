"""PaddleOCR reading: engine loading, crop preprocessing, and plate-text selection."""

import logging

import cv2
from paddleocr import PaddleOCR

from license_plate_pipeline.config import UPSCALE_FACTOR
from license_plate_pipeline.validation import display_plate, is_valid_plate

logger = logging.getLogger(__name__)

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        logger.info("Loading PaddleOCR engine")
        # enable_mkldnn=False: these pretrained PP-OCR models were exported with an older
        # PaddlePaddle version; the newer runtime's oneDNN fused-conv path errors on them.
        _reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, enable_mkldnn=False)
    return _reader


def preprocess_for_ocr(crop):
    upscaled = cv2.resize(crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def select_plate_text(ocr_lines):
    """Pick the highest-confidence fragment that is a valid plate string.

    Returns None if no fragment looks like a real plate - this deliberately does
    NOT fall back to "highest confidence anything", because that let sticker
    text, state names, and OCR junk through as events. See
    license_plate_pipeline.validation.
    """
    valid = [(display_plate(t), c) for t, c in ocr_lines if is_valid_plate(t)]
    if valid:
        return max(valid, key=lambda tc: tc[1])
    return None


def read_crop(crop):
    """Run OCR on a plate crop. Returns a list of (text, confidence) fragments.

    Returns an empty list - rather than raising - if OCR fails on this crop,
    so one bad crop doesn't crash a whole video/frame loop.
    """
    if crop.size == 0:
        return []

    processed = preprocess_for_ocr(crop)
    try:
        ocr_result = get_reader().ocr(processed, cls=True)
    except Exception:
        logger.exception("OCR failed on this crop - skipping it")
        return []

    return [(line[1][0], line[1][1]) for line in (ocr_result[0] or [])]
