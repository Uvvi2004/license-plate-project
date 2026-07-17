"""High-level pipeline: detect + read plates in a single image, or across a video."""

import logging

import cv2

from license_plate_pipeline.config import GAP_SECONDS, MIN_FRAMES
from license_plate_pipeline.detection import detect_boxes
from license_plate_pipeline.ocr import read_crop, select_plate_text
from license_plate_pipeline.video import process_video_tracked

logger = logging.getLogger(__name__)


def read_plates_from_frame(frame):
    """Detect + read every plate in a single already-loaded frame.

    Returns a list of (text, confidence) - one entry per detected plate that
    produced any OCR text. Detection/OCR failures on individual boxes are
    already handled (logged, skipped) inside detect_boxes/read_crop.
    """
    readings = []
    for x1, y1, x2, y2 in detect_boxes(frame):
        crop = frame[y1:y2, x1:x2]
        picked = select_plate_text(read_crop(crop))
        if picked:
            readings.append(picked)
    return readings


def read_plates_from_image(image_path):
    """Detect + read every plate in an image file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return read_plates_from_frame(image)


def process_video(video_path, gap_seconds=GAP_SECONDS, min_frames=MIN_FRAMES):
    """Run the full pipeline over a video file and return deduped events.

    Each event is a dict: plate_text, first_seen, last_seen, best_confidence,
    frame_count, readings (all raw OCR fragments merged into this event).

    Detection runs every frame; OCR runs only a few times per tracked plate
    (see license_plate_pipeline.video / .tracking for why - it's the fix for
    the per-frame OCR cost). Raises FileNotFoundError if the video can't be
    opened at all.
    """
    return process_video_tracked(
        video_path, detect_boxes, read_crop, select_plate_text,
        gap_seconds=gap_seconds, min_frames=min_frames,
    )
