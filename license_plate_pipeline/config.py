"""Tunable constants for the detection + OCR + dedup pipeline.

Every value here was calibrated against real test data (see PROJECT_CONTEXT.md
"OCR Accuracy" and "Dedup Clustering" sections for the numbers) - don't change
these without re-running that calibration against actual footage.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Detection crop preprocessing (see PROJECT_CONTEXT.md "OCR Accuracy")
PAD_RATIO = 0.1
UPSCALE_FACTOR = 3

# OCR fragment selection - plate-like text vs. nearby sticker/dealer text
PLATE_TEXT_PATTERN = r"[A-Za-z0-9]{5,8}"

# Video/live dedup: group same-text sightings into one event
GAP_SECONDS = 1.5
MIN_FRAMES = 3

# Cross-text clustering (pass 1): merge different OCR readings of the same plate
CLUSTER_GAP_SECONDS = GAP_SECONDS
SIMILARITY_THRESHOLD = 70  # rapidfuzz ratio (0-100)

# Finished-cluster merge (pass 2): wider gap allowed only for near-exact matches
EXTENDED_GAP_SECONDS = 3.0
HIGH_SIMILARITY_THRESHOLD = 85
