"""Box tracking + selective OCR scheduling - the measured fix for OCR speed.

Root cause (IMPLEMENTATION_PLAN.md "Root-cause diagnosis"): ~95% of every OCR
call is the engine's internal text-detection stage, and the old pipeline ran
OCR on *every* frame a plate was visible (~600+ calls for a 21s clip). This
module makes each physical plate get OCR'd only a handful of times by tracking
its bounding box across frames via IOU overlap.

Deliberately pure Python - no cv2, no ML imports - so it's fully unit-testable
with synthetic boxes and can be shared by both the dev-machine and Pi pipelines
(license_plate_pipeline.video drives it and injects the real detect/OCR calls).

Flow (see license_plate_pipeline.video.process_video_tracked):

    tracker = PlateTracker(...)
    for each frame (index, timestamp):
        boxes = detect_boxes(frame)                       # cheap, run every frame
        for track_id, box in tracker.update(boxes, timestamp, frame_idx):
            text, conf = ocr(crop_of(box))                # expensive, run rarely
            tracker.add_reading(track_id, text, conf)
    events = tracker.finish()                              # one event per plate

Each returned event is the same dict shape the dedup passes expect
(plate_text/first_seen/last_seen/best_confidence/frame_count), so
license_plate_pipeline.dedup runs downstream unchanged as a cross-track safety
net.
"""

import logging
from dataclasses import dataclass, field

from license_plate_pipeline.config import (
    EARLY_STOP_CONFIDENCE,
    IOU_THRESHOLD,
    MAX_MISSED_FRAMES,
    MAX_OCR_PER_TRACK,
    MIN_TRACK_FRAMES,
    OCR_FRAME_INTERVAL,
    READING_CONFIDENCE_MARGIN,
)

logger = logging.getLogger(__name__)


def iou(box_a, box_b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes. 0.0 if disjoint."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    box: tuple
    first_seen: float
    last_seen: float
    frames_seen: int = 1
    missed_frames: int = 0
    ocr_attempts: int = 0
    last_ocr_frame: int = -(10**9)  # "never" - guarantees first eligible frame OCRs
    done: bool = False              # early-stopped on a high-confidence read
    readings: list = field(default_factory=list)  # [(text, confidence), ...]

    def needs_ocr(self, frame_idx):
        return (
            not self.done
            and self.frames_seen >= MIN_TRACK_FRAMES
            and self.ocr_attempts < MAX_OCR_PER_TRACK
            and (frame_idx - self.last_ocr_frame) >= OCR_FRAME_INTERVAL
        )

    def to_event(self):
        """Collapse this track's readings into one dedup-compatible event dict.

        Returns None if the track never produced a usable read (noise / a plate
        that was detected but never resolved to text) - those are dropped.
        """
        if not self.readings:
            return None
        best_conf = max(c for _t, c in self.readings)
        # Among near-best reads, prefer the longest (fuller plate over a fragment).
        near_best = [(t, c) for t, c in self.readings if c >= best_conf - READING_CONFIDENCE_MARGIN]
        text, conf = max(near_best, key=lambda tc: (len(tc[0]), tc[1]))
        return {
            "plate_text": text,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "best_confidence": conf,
            "frame_count": self.frames_seen,
        }


class PlateTracker:
    """Tracks plate boxes across frames and schedules sparse OCR.

    All tuning comes from config.py by default; overridable per-instance for
    tests. Timestamps are whatever the caller passes (seconds into video, or
    wall-clock) - the tracker only compares them, never interprets units.
    """

    def __init__(
        self,
        iou_threshold=IOU_THRESHOLD,
        max_missed_frames=MAX_MISSED_FRAMES,
        early_stop_confidence=EARLY_STOP_CONFIDENCE,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.early_stop_confidence = early_stop_confidence
        self._open = []       # list[_Track]
        self._finished = []   # list[_Track] - aged-out tracks awaiting finish()
        self._next_id = 0

    def update(self, boxes, timestamp, frame_idx):
        """Advance one frame. Returns [(track_id, box), ...] that need OCR now.

        `boxes` is this frame's detections (list of (x1,y1,x2,y2)). The caller
        is expected to OCR each returned box and report back via add_reading().
        An OCR attempt is counted here whether or not the caller finds text, so
        a track that never resolves still can't consume unbounded OCR calls.
        """
        matches = self._match(boxes)  # {track_index: box_index}
        matched_boxes = set(matches.values())

        for t_idx, track in enumerate(self._open):
            if t_idx in matches:
                track.box = boxes[matches[t_idx]]
                track.last_seen = timestamp
                track.frames_seen += 1
                track.missed_frames = 0
            else:
                track.missed_frames += 1

        # Spawn tracks for detections that matched nothing.
        for b_idx, box in enumerate(boxes):
            if b_idx not in matched_boxes:
                self._open.append(
                    _Track(track_id=self._next_id, box=box, first_seen=timestamp, last_seen=timestamp)
                )
                self._next_id += 1

        # Age out stale tracks (do this after spawning so brand-new tracks stay).
        still_open = []
        for track in self._open:
            if track.missed_frames > self.max_missed_frames:
                self._finished.append(track)
            else:
                still_open.append(track)
        self._open = still_open

        # Schedule OCR for eligible tracks seen this frame.
        due = []
        for track in self._open:
            if track.missed_frames == 0 and track.needs_ocr(frame_idx):
                track.ocr_attempts += 1
                track.last_ocr_frame = frame_idx
                due.append((track.track_id, track.box))
        return due

    def add_reading(self, track_id, text, confidence):
        """Record an OCR result for a track; may trigger its early stop."""
        for track in self._open:
            if track.track_id == track_id:
                track.readings.append((text, confidence))
                if confidence >= self.early_stop_confidence:
                    track.done = True
                return
        # Track may have aged out between scheduling and reporting - find it there.
        for track in self._finished:
            if track.track_id == track_id:
                track.readings.append((text, confidence))
                return

    def finish(self):
        """Close all open tracks and return one event dict per resolved plate."""
        self._finished.extend(self._open)
        self._open = []
        events = [t.to_event() for t in self._finished]
        return [e for e in events if e is not None]

    def _match(self, boxes):
        """Greedy IOU matching of current tracks to this frame's detections.

        Returns {track_index_in_self._open: box_index}. Highest-IOU pairs are
        assigned first; each track and each box is used at most once.
        """
        pairs = []
        for t_idx, track in enumerate(self._open):
            for b_idx, box in enumerate(boxes):
                score = iou(track.box, box)
                if score >= self.iou_threshold:
                    pairs.append((score, t_idx, b_idx))

        pairs.sort(reverse=True)
        matches = {}
        used_boxes = set()
        for _score, t_idx, b_idx in pairs:
            if t_idx in matches or b_idx in used_boxes:
                continue
            matches[t_idx] = b_idx
            used_boxes.add(b_idx)
        return matches
