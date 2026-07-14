"""PaddleOCR reading: engine loading, crop preprocessing, and plate-text selection."""

import logging
import re

import cv2
from paddleocr import PaddleOCR

from license_plate_pipeline.config import PLATE_TEXT_PATTERN, UPSCALE_FACTOR

logger = logging.getLogger(__name__)

_reader = None

_PLATE_TEXT_RE = re.compile(PLATE_TEXT_PATTERN)


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
    """Pick the fragment most likely to be the actual plate number, not nearby sticker/dealer text."""
    plate_like = [(t, c) for t, c in ocr_lines if _PLATE_TEXT_RE.fullmatch(t)]
    if plate_like:
        return max(plate_like, key=lambda tc: tc[1])
    if ocr_lines:
        return max(ocr_lines, key=lambda tc: tc[1])
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
