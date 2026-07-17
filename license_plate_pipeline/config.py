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

# Fragment absorption (pass 3): a short read that is a substring of a longer,
# time-overlapping plate is a partial detection of that same plate (e.g. YOLO
# emitting a second box of just the right half -> "HZ" for "L-605-HZ"). Absorb
# it. After that, any surviving event whose alphanumeric length is below
# MIN_PLATE_LEN is an unresolvable stub, not a plate - drop it.
MIN_PLATE_LEN = 4

# Box tracking + selective OCR (see IMPLEMENTATION_PLAN.md Step 1).
# The measured fix for OCR speed on both engines: track each physical plate
# across frames by box overlap (IOU) and OCR it only a few times, instead of
# running the ~1s-per-call OCR on every frame it's visible.
IOU_THRESHOLD = 0.2         # min box overlap to count as the same plate frame-to-frame
                            # (0.2 not 0.3: a large plate crossing the frame moves enough
                            #  per frame that 0.3 split single plates into two tracks)
MAX_MISSED_FRAMES = 20      # keep a track alive through this many plate-less frames (~0.7s @30fps)
MIN_TRACK_FRAMES = 3        # don't OCR (or emit) a track seen fewer times - filters flicker/noise
OCR_FRAME_INTERVAL = 8      # once eligible, OCR a track at most once per this many frames
MAX_OCR_PER_TRACK = 8       # hard cap on OCR calls per physical plate (more samples = better
                            #  chance of catching the sharp mid-transit frame)
EARLY_STOP_CONFIDENCE = 0.97  # a read this good ends OCR for that track early
# When collapsing a track's reads to one plate string, prefer the LONGEST read
# among those within this confidence margin of the best - a fragment ("H-6" @0.97)
# shouldn't outrank the full plate ("H-644-LX" @0.96) just by a hair of confidence.
READING_CONFIDENCE_MARGIN = 0.06
