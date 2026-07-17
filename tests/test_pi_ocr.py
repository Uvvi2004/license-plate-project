import pytest

pytest.importorskip("rapidocr_onnxruntime")

from license_plate_pipeline.pi.ocr import select_plate_text  # noqa: E402


# Same behavior as license_plate_pipeline.ocr.select_plate_text, duplicated on
# purpose (see pi/ocr.py module docstring - avoids pulling in paddleocr/paddle).
# These tests exist so the two copies can't silently drift apart.


def test_prefers_plate_like_fragment_over_higher_confidence_sticker_text():
    lines = [("SEP", 0.99), ("6FVZ747", 0.97), ("STALCHRYSLER.CO", 0.95)]
    assert select_plate_text(lines) == ("6FVZ747", 0.97)


def test_returns_none_when_nothing_looks_like_a_valid_plate():
    lines = [("STALCHRYSLER.CO", 0.95), ("SEP", 0.99)]
    assert select_plate_text(lines) is None


def test_strips_non_ascii_and_keeps_valid_plate():
    # RapidOCR's Chinese default models sometimes prepend a province glyph.
    lines = [("皖EKH9211", 0.96)]
    assert select_plate_text(lines) == ("EKH9211", 0.96)


def test_returns_none_for_no_readings():
    assert select_plate_text([]) is None
