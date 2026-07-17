"""Plate-string validation - reject OCR output that isn't a plausible plate.

Pure Python, no ML deps. Shared by the OCR modules (to pick the plate-like
fragment) and the video pipeline (to drop junk events). Motivated by real
failures on 4K real-world footage (see PROJECT_CONTEXT.md "Real-world video
testing"): RapidOCR's default Chinese models emit numeric fragments
("619879"), state-name misreads ("IDAHO" -> "HDAHO"), and even Chinese
characters ("皖EKH9211"). A plate must be 5-8 alphanumeric characters
(separators ignored) containing BOTH a letter and a digit - which every plate
in the test footage satisfies and essentially all this junk fails.
"""

import re

from license_plate_pipeline.config import MAX_PLATE_CHARS, MIN_PLATE_CHARS

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")
_NON_DISPLAY = re.compile(r"[^A-Za-z0-9\- ]")


def canonical_plate(text):
    """Uppercase, alphanumeric only. Drops separators AND non-ASCII (e.g. 皖)."""
    return _NON_ALNUM.sub("", text).upper()


def display_plate(text):
    """Human-readable form: keep ASCII alphanumerics plus dash/space, strip the rest.

    Preserves "R-183-JF" as-is but turns "皖EKH9211" into "EKH9211".
    """
    return _NON_DISPLAY.sub("", text).strip()


def is_valid_plate(text):
    """True if `text` is a plausible plate: 5-8 alnum chars with a letter AND a digit."""
    c = canonical_plate(text)
    if not (MIN_PLATE_CHARS <= len(c) <= MAX_PLATE_CHARS):
        return False
    return any(ch.isalpha() for ch in c) and any(ch.isdigit() for ch in c)
