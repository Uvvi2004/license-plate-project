"""OCR reading for Pi deployment - RapidOCR (onnxruntime) instead of PaddleOCR.

Same public interface as license_plate_pipeline.ocr (get_reader/read_crop/
select_plate_text), so license_plate_pipeline.pi.pipeline can mirror the
dev-machine pipeline.py exactly.

Two speed decisions here, both measured on the demo2.jpg plate crop (see
IMPLEMENTATION_PLAN.md "Root-cause diagnosis") rather than guessed:

- preprocess_for_ocr returns the crop UNCHANGED. Upscaling 1x/2x/3x gave
  RapidOCR no accuracy gain (all read 6FVZ747 correctly) but cost a cubic
  resize per call - so we don't. Contrast the dev-machine ocr.py, which DOES
  upscale+equalize for PaddleOCR; RapidOCR's model reacts differently (grayscale
  +equalize actually made it miss the plate entirely). The function is kept as
  an identity pass so the interface/tests stay stable and a future tight-crop
  optimization has an obvious hook.
- read_crop calls RapidOCR with use_cls=False. The angle classifier added
  ~400 ms/call (1363 -> 966 ms) and does nothing for a fixed, upright gate
  camera that never produces rotated plates.

The remaining ~950 ms/call is RapidOCR's internal text-detection stage. That is
NOT trimmed here - it's addressed at the pipeline level by only OCR-ing a few
frames per tracked plate (license_plate_pipeline.tracking), not every frame.
"""

import logging
import re

from rapidocr_onnxruntime import RapidOCR

from license_plate_pipeline.config import PLATE_TEXT_PATTERN

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
    # Identity: upscaling gave RapidOCR no accuracy gain but cost a resize/call.
    # See module docstring. Kept as a hook for a future tight-crop fast path.
    return crop


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
        result, _elapse = get_reader()(processed, use_cls=False)
    except Exception:
        logger.exception("OCR failed on this crop - skipping it")
        return []

    if not result:
        return []
    return [(text, confidence) for _box, text, confidence in result]
