"""Shared tracked-video driver, used by both the dev-machine and Pi pipelines.

Imports cv2, tracking, and dedup - but NOT any detection/OCR engine. The
engine-specific functions are injected as callables, so this one implementation
serves both license_plate_pipeline.pipeline (ultralytics/paddleocr) and
license_plate_pipeline.pi.pipeline (onnxruntime/rapidocr) without pulling either
engine's heavy dependencies into the other's environment.
"""

import logging
import time

import cv2

from license_plate_pipeline.config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    GAP_SECONDS,
    LIVE_DEDUP_WINDOW_SECONDS,
    MIN_FRAMES,
    MIN_PLATE_CONFIDENCE,
)
from license_plate_pipeline.dedup import dedup_events
from license_plate_pipeline.tracking import PlateTracker
from license_plate_pipeline.validation import canonical_plate, is_valid_plate

logger = logging.getLogger(__name__)


def _recently_logged(canon, recent, now, window):
    """True if the same plate (exact or as a substring) was logged within `window`.

    Streaming backstop for a track that split mid-vehicle: exact match catches
    OCR-flicker of the same string, substring catches a partial read of the same
    plate ("6HH07" vs "66HH07"). Deliberately NOT fuzzy similarity - at a gate,
    missing a genuinely different truck with a look-alike plate is worse than an
    occasional double-log, and substring is a strong same-plate signal that a
    distinct plate won't trip. `recent` is pruned of entries older than `window`.
    """
    for prev in list(recent):
        if now - recent[prev] >= window:
            del recent[prev]
            continue
        if prev == canon or canon in prev or prev in canon:
            return True
    return False


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
    return [
        c
        for c in clusters
        if c["frame_count"] >= min_frames
        and c["best_confidence"] >= MIN_PLATE_CONFIDENCE
        and is_valid_plate(c["plate_text"])
    ]


def _passes_output_filters(event, min_frames):
    return (
        event["frame_count"] >= min_frames
        and event["best_confidence"] >= MIN_PLATE_CONFIDENCE
        and is_valid_plate(event["plate_text"])
    )


def process_camera(
    detect_boxes,
    read_crop,
    select_plate_text,
    camera_index=0,
    width=CAPTURE_WIDTH,
    height=CAPTURE_HEIGHT,
    min_frames=MIN_FRAMES,
    dedup_window=LIVE_DEDUP_WINDOW_SECONDS,
    on_frame=None,
    stop=None,
):
    """Run the tracked pipeline on a LIVE camera, yielding one event per vehicle.

    Unlike process_video (finite file), this is the deployment entry point: it
    opens a camera, tracks plates, and yields a validated event dict the moment a
    vehicle leaves frame (its track ages out) - so callers can log/insert it
    immediately with a wall-clock timestamp. Runs until `stop()` returns True
    (if given) or the camera fails.

    detect_boxes/read_crop/select_plate_text are injected (engine-agnostic, same
    as process_video). `on_frame(frame, boxes)` is an optional hook for drawing a
    live preview. Each yielded event has the usual keys plus "timestamp" (epoch
    seconds when the vehicle was first seen).

    Capture is forced to `width`x`height` (default 720p) and the driver buffer to
    1 frame, so on a slow device we process the freshest frame instead of falling
    behind on a backlog.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep near-real-time on slow hardware
    logger.info(
        "Live capture opened on camera %d at %dx%d",
        camera_index,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    tracker = PlateTracker()
    recent = {}  # canonical plate -> last-emitted epoch time (live double-log suppression)
    frame_idx = 0

    try:
        while True:
            if stop is not None and stop():
                break
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera read failed - stopping live capture")
                break
            now = time.time()

            boxes = detect_boxes(frame)
            for track_id, (x1, y1, x2, y2) in tracker.update(boxes, now, frame_idx):
                crop = frame[y1:y2, x1:x2]
                picked = select_plate_text(read_crop(crop))
                if picked:
                    tracker.add_reading(track_id, picked[0], picked[1])

            if on_frame is not None:
                on_frame(frame, boxes)

            for event in tracker.drain_finished():
                if not _passes_output_filters(event, min_frames):
                    continue
                canon = canonical_plate(event["plate_text"])
                if _recently_logged(canon, recent, now, dedup_window):
                    continue  # one vehicle logged moments ago - don't double-log
                recent[canon] = now
                # first_seen/last_seen are epoch seconds here (we pass `now` as the
                # tracker timestamp), so first_seen IS the vehicle's arrival time.
                event["timestamp"] = event["first_seen"]
                yield event

            frame_idx += 1
    finally:
        cap.release()
