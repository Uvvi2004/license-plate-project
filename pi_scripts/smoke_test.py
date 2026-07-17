"""One-time check after setting up the Pi: confirm detection + OCR give the
same validated result as the dev machine before doing anything else.

Run from the repo root: python pi_scripts/smoke_test.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from license_plate_pipeline.pi.pipeline import read_plates_from_image  # noqa: E402

EXPECTED_TEXT = "6FVZ747"
TEST_IMAGE = PROJECT_ROOT / "test_images" / "demo2.jpg"


def main():
    print(f"Running detection + OCR on {TEST_IMAGE} ...")
    readings = read_plates_from_image(TEST_IMAGE)

    if not readings:
        print("FAILED: no plate readings at all - check the model files and dependencies.")
        sys.exit(1)

    for text, confidence in readings:
        print(f"  '{text}' (confidence: {confidence:.3f})")

    matched = [t for t, c in readings if t == EXPECTED_TEXT]
    if matched:
        print(f"\nPASSED: found expected plate text '{EXPECTED_TEXT}'.")
    else:
        print(f"\nWARNING: expected '{EXPECTED_TEXT}' but didn't see an exact match above.")
        print("This doesn't necessarily mean something's broken - but stop and compare")
        print("against the dev-machine result before moving on to the webcam test.")


if __name__ == "__main__":
    main()
