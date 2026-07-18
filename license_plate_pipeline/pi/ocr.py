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

from rapidocr_onnxruntime import RapidOCR

from license_plate_pipeline.validation import display_plate, is_valid_plate

logger = logging.getLogger(__name__)

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        # RapidOCR's bundled PP-OCRv4 recognition model is kept deliberately.
        # The English "mobile" alternatives (PP-OCRv4/v5) were tested head-to-head
        # and read Latin plates LESS accurately (4/4 vs 1-2/4 on real plate crops -
        # see IMPLEMENTATION_PLAN.md Step 3). Its only downside, occasional Chinese
        # glyphs, is fully removed downstream by license_plate_pipeline.validation
        # (non-ASCII is stripped before a plate is ever emitted).
        logger.info("Loading RapidOCR engine")
        _reader = RapidOCR()
    return _reader


def preprocess_for_ocr(crop):
    # Identity: upscaling gave RapidOCR no accuracy gain but cost a resize/call.
    # See module docstring. Kept as a hook for a future tight-crop fast path.
    return crop


def select_plate_text(ocr_lines):
    """Highest-confidence fragment that is a valid plate string, else None.

    Same logic as license_plate_pipeline.ocr.select_plate_text (shares the
    validation module); does not fall back to "highest confidence anything".
    """
    valid = [(display_plate(t), c) for t, c in ocr_lines if is_valid_plate(t)]
    if valid:
        return max(valid, key=lambda tc: tc[1])
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

    # RapidOCR returns [box, text, score] per line. `score` is a float in some
    # versions (1.4.x) but a STRING in others (1.2.3, which is what installs on
    # the Pi's Python 3.13) - coerce to float so every downstream numeric compare
    # (confidence floor, best-of-N selection) works regardless of version.
    readings = []
    for _box, text, score in result:
        try:
            readings.append((str(text), float(score)))
        except (TypeError, ValueError):
            continue
    return readings
