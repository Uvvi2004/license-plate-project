import pytest

pytest.importorskip("paddleocr")

from license_plate_pipeline.ocr import select_plate_text  # noqa: E402


def test_prefers_plate_like_fragment_over_higher_confidence_sticker_text():
    # Real example from PROJECT_CONTEXT.md: the plate crop also picked up
    # a registration sticker ("SEP") and dealer frame text, both scoring
    # higher confidence than the actual plate number.
    lines = [("SEP", 0.99), ("6FVZ747", 0.97), ("STALCHRYSLER.CO", 0.95)]
    assert select_plate_text(lines) == ("6FVZ747", 0.97)


def test_returns_none_when_nothing_looks_like_a_valid_plate():
    # No fragment is a valid plate (letter+digit, 5-8 chars) - do NOT fall back
    # to "highest confidence anything"; that let junk through as events.
    lines = [("STALCHRYSLER.CO", 0.95), ("SEP", 0.99)]
    assert select_plate_text(lines) is None


def test_returns_none_for_no_readings():
    assert select_plate_text([]) is None
