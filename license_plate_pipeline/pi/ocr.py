"""OCR reading for Pi deployment - RapidOCR (onnxruntime) instead of PaddleOCR.

Same public interface as license_plate_pipeline.ocr (get_reader/read_crop/
select_plate_text), so license_plate_pipeline.pi.pipeline can mirror the
dev-machine pipeline.py exactly.

Preprocessing is intentionally NOT the same as license_plate_pipeline.ocr's
preprocess_for_ocr: tested against the demo2.jpg plate crop (ground truth
6FVZ747) before writing this -

  raw crop, no preprocessing:            '6FVZ747' @ 0.985 (correct)
  3x upscale, color, no grayscale:       '6FVZ747' @ 0.975 (correct)
  3x upscale + grayscale + equalize
  (the dev-machine PaddleOCR tuning):    plate text NOT detected at all

RapidOCR's bundled model reacts differently to the grayscale/contrast-equalize
step than PaddleOCR's did - reusing that tuning here would have been a silent
regression. Using upscale-only: nearly ties the raw-crop score on this test
and should still help on smaller/more-distant real-world crops than this one
well-framed test image represents.
"""

import logging
import re

import cv2
from rapidocr_onnxruntime import RapidOCR

from license_plate_pipeline.config import PLATE_TEXT_PATTERN, UPSCALE_FACTOR

logger = logging.getLogger(__name__)

_reader = None

_PLATE_TEXT_RE = re.compile(PLATE_TEXT_PATTERN)


def get_reader():
    global _reader
    if _reader is None:
        logger.info("Loading RapidOCR engine")
        _reader = RapidOCR()
    return _reader


def preprocess_for_ocr(crop):
    return cv2.resize(crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)


def select_plate_text(ocr_lines):
    # Duplicated from license_plate_pipeline.ocr rather than imported, so this
    # module never pulls in paddleocr/paddlepaddle - the whole point of the
    # RapidOCR path.
    plate_like = [(t, c) for t, c in ocr_lines if _PLATE_TEXT_RE.fullmatch(t)]
    if plate_like:
        return max(plate_like, key=lambda tc: tc[1])
    if ocr_lines:
        return max(ocr_lines, key=lambda tc: tc[1])
    return None


def read_crop(crop):
    """Run OCR on a plate crop. Returns a list of (text, confidence) fragments.

    Returns an empty list - rather than raising - if OCR fails on this crop,
    so one bad crop doesn't crash a whole video/live loop.
    """
    if crop.size == 0:
        return []

    processed = preprocess_for_ocr(crop)
    try:
        result, _elapse = get_reader()(processed)
    except Exception:
        logger.exception("OCR failed on this crop - skipping it")
        return []

    if not result:
        return []
    return [(text, confidence) for _box, text, confidence in result]
