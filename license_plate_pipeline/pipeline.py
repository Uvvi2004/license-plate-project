"""High-level pipeline: detect + read plates in a single image, or across a video."""

import logging

import cv2

from license_plate_pipeline.config import GAP_SECONDS, MIN_FRAMES
from license_plate_pipeline.dedup import dedup_events
from license_plate_pipeline.detection import detect_boxes
from license_plate_pipeline.ocr import read_crop, select_plate_text

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

    Raises FileNotFoundError if the video can't be opened at all. A frame that
    fails to decode ends the read loop (correct for a finite video file - see
    PROJECT_CONTEXT.md for why this isn't yet built out for a live/reconnecting
    camera feed, which needs different handling in Phase 8).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        cap.release()
        raise RuntimeError(f"Video reports 0 fps, can't compute timestamps: {video_path}")

    active_events = {}
    finished_events = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps

            for text, confidence in read_plates_from_frame(frame):
                event = active_events.get(text)
                if event is not None and timestamp - event["last_seen"] <= gap_seconds:
                    event["last_seen"] = timestamp
                    event["frame_count"] += 1
                    event["best_confidence"] = max(event["best_confidence"], confidence)
                else:
                    if event is not None:
                        finished_events.append(event)
                    active_events[text] = {
                        "plate_text": text,
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "best_confidence": confidence,
                        "frame_count": 1,
                    }

            frame_idx += 1
    finally:
        cap.release()

    finished_events.extend(active_events.values())
    logger.info("Raw sub-events before dedup: %d", len(finished_events))

    clusters = dedup_events(finished_events)
    return [c for c in clusters if c["frame_count"] >= min_frames]
