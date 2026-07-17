"""Tests for license_plate_pipeline.tracking - pure synthetic boxes, no models.

Covers the IOU function, track identity across frames/dropouts, and the
selective-OCR scheduling policy. See IMPLEMENTATION_PLAN.md Step 1 for the
design these lock in.
"""

from license_plate_pipeline.tracking import PlateTracker, iou


# --- iou -------------------------------------------------------------------

def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


# --- track identity --------------------------------------------------------

def _tracker():
    # Loose OCR gating off by default here; identity tests only care about events.
    return PlateTracker()


def test_single_box_drifting_is_one_event():
    t = _tracker()
    for i in range(6):
        box = (100 + i, 100 + i, 200 + i, 150 + i)  # drifts a few px/frame
        for track_id, _b in t.update([box], timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "ABC123", 0.99)
    events = t.finish()
    assert len(events) == 1
    assert events[0]["plate_text"] == "ABC123"
    assert events[0]["frame_count"] == 6


def test_two_parallel_boxes_stay_separate():
    t = _tracker()
    for i in range(6):
        left = (0 + i, 0, 50 + i, 40)
        right = (500 + i, 0, 550 + i, 40)
        for track_id, box in t.update([left, right], timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "LEFT" if box[0] < 250 else "RIGHT", 0.99)
    events = t.finish()
    assert len(events) == 2
    assert {e["plate_text"] for e in events} == {"LEFT", "RIGHT"}


def test_brief_dropout_keeps_same_track():
    t = PlateTracker(max_missed_frames=15)
    box = (100, 100, 200, 150)
    # seen, gone for a few frames (< max_missed), then back at ~same spot
    seq = [box, box, box, None, None, None, box, box]
    for i, b in enumerate(seq):
        boxes = [] if b is None else [b]
        for track_id, _bb in t.update(boxes, timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "SAME1", 0.99)
    events = t.finish()
    assert len(events) == 1


def test_long_gap_splits_into_two_tracks():
    t = PlateTracker(max_missed_frames=2)
    box = (100, 100, 200, 150)
    seq = [box, box, box, None, None, None, None, box, box, box]
    for i, b in enumerate(seq):
        boxes = [] if b is None else [b]
        for track_id, _bb in t.update(boxes, timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "SPLIT", 0.99)
    events = t.finish()
    assert len(events) == 2


# --- selective OCR policy --------------------------------------------------

def test_ocr_not_scheduled_before_min_frames():
    # MIN_TRACK_FRAMES default is 3, so frames 0 and 1 must not request OCR.
    t = _tracker()
    box = (100, 100, 200, 150)
    scheduled_frames = []
    for i in range(3):
        due = t.update([box], timestamp=i * 0.1, frame_idx=i)
        if due:
            scheduled_frames.append(i)
    # frame_idx 0 -> frames_seen 1, frame 1 -> 2, frame 2 -> 3 (first eligible)
    assert scheduled_frames == [2]


def test_ocr_call_count_capped_and_early_stops():
    # Long track: without early stop it would OCR MAX_OCR_PER_TRACK times; a
    # high-confidence read should stop it sooner.
    t = PlateTracker(early_stop_confidence=0.95)
    box = (100, 100, 200, 150)
    ocr_count = 0
    for i in range(80):
        for track_id, _b in t.update([box], timestamp=i * 0.1, frame_idx=i):
            ocr_count += 1
            # Return a strong read on the 2nd OCR call to trigger early stop.
            conf = 0.96 if ocr_count == 2 else 0.5
            t.add_reading(track_id, "STOP99", conf)
    assert ocr_count == 2  # stopped early rather than running the full cap
    events = t.finish()
    assert events[0]["plate_text"] == "STOP99"


def test_track_with_no_readings_is_dropped():
    t = _tracker()
    box = (100, 100, 200, 150)
    for i in range(10):
        t.update([box], timestamp=i * 0.1, frame_idx=i)  # never add_reading
    assert t.finish() == []


# --- streaming drain (live deployment) -------------------------------------

def test_drain_emits_closed_track_then_clears():
    t = PlateTracker(max_missed_frames=2)
    box = (100, 100, 200, 150)
    # Present for 4 frames, then gone long enough to close.
    seq = [box, box, box, box, None, None, None, None]
    drained = []
    for i, b in enumerate(seq):
        boxes = [] if b is None else [b]
        for track_id, _bb in t.update(boxes, timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "DRN123", 0.99)
        drained.extend(t.drain_finished())
    assert len(drained) == 1
    assert drained[0]["plate_text"] == "DRN123"
    # Once drained, it isn't returned again by finish().
    assert t.finish() == []


def test_drain_returns_empty_while_track_still_open():
    t = _tracker()
    box = (100, 100, 200, 150)
    for i in range(5):
        for track_id, _bb in t.update([box], timestamp=i * 0.1, frame_idx=i):
            t.add_reading(track_id, "OPEN12", 0.99)
        assert t.drain_finished() == []  # nothing closed yet
    assert len(t.finish()) == 1  # still open until finish()
