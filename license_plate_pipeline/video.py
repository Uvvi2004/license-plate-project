"""Shared tracked-video driver, used by both the dev-machine and Pi pipelines.

Imports cv2, tracking, and dedup - but NOT any detection/OCR engine. The
engine-specific functions are injected as callables, so this one implementation
serves both license_plate_pipeline.pipeline (ultralytics/paddleocr) and
license_plate_pipeline.pi.pipeline (onnxruntime/rapidocr) without pulling either
engine's heavy dependencies into the other's environment.
"""

import logging

import cv2

from license_plate_pipeline.config import GAP_SECONDS, MIN_FRAMES
from license_plate_pipeline.dedup import dedup_events
from license_plate_pipeline.tracking import PlateTracker

logger = logging.getLogger(__name__)


def process_video_tracked(
    video_path,
    detect_boxes,
    read_crop,
    select_plate_text,
    gap_seconds=GAP_SECONDS,
    min_frames=MIN_FRAMES,
):
    """Run detection every frame but OCR each plate only a few times, via tracking.

    detect_boxes(frame) -> [(x1,y1,x2,y2), ...]
    read_crop(crop)     -> [(text, confidence), ...]
    select_plate_text(readings) -> (text, confidence) | None

    Returns deduped event dicts (plate_text/first_seen/last_seen/best_confidence/
    frame_count/readings), same shape as the old per-frame pipeline, filtered to
    events seen at least `min_frames` times.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        cap.release()
        raise RuntimeError(f"Video reports 0 fps, can't compute timestamps: {video_path}")

    tracker = PlateTracker()
    ocr_calls = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps

            boxes = detect_boxes(frame)
            for track_id, (x1, y1, x2, y2) in tracker.update(boxes, timestamp, frame_idx):
                crop = frame[y1:y2, x1:x2]
                ocr_calls += 1
                picked = select_plate_text(read_crop(crop))
                if picked:
                    tracker.add_reading(track_id, picked[0], picked[1])

            frame_idx += 1
    finally:
        cap.release()

    raw_events = tracker.finish()
    logger.info("Tracked %d frames with %d OCR calls -> %d raw plate events", frame_idx, ocr_calls, len(raw_events))

    clusters = dedup_events(raw_events)
    return [c for c in clusters if c["frame_count"] >= min_frames]
