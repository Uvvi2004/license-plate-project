from license_plate_pipeline.dedup import cluster_by_time_and_text, merge_finished_clusters


def _event(text, first_seen, last_seen, confidence, frame_count):
    return {
        "plate_text": text,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "best_confidence": confidence,
        "frame_count": frame_count,
    }


def _cluster(text, first_seen, last_seen, confidence, frame_count):
    c = _event(text, first_seen, last_seen, confidence, frame_count)
    c["readings"] = [(text, confidence, frame_count)]
    return c


# --- cluster_by_time_and_text (pass 1) ---
# Real numbers from PROJECT_CONTEXT.md "Dedup Clustering" (demo.mp4 run).

def test_merges_flicker_of_the_same_plate():
    events = [
        _event("R-183-JF", 0.00, 1.63, 0.99, 46),
        _event("R-183JF", 1.80, 1.90, 0.93, 4),
    ]
    clusters = cluster_by_time_and_text(events)
    assert len(clusters) == 1
    assert clusters[0]["plate_text"] == "R-183-JF"  # kept the higher-confidence reading


def test_time_proximity_alone_is_not_enough_to_merge():
    # This is the "attempt 1" failure this function was built to avoid: two
    # different vehicles passing close together must NOT merge just because
    # they're close in time.
    events = [
        _event("R-183-JF", 0.00, 1.90, 0.99, 55),
        _event("N-894-JV", 2.13, 3.73, 0.98, 47),
    ]
    clusters = cluster_by_time_and_text(events)
    assert len(clusters) == 2


# --- merge_finished_clusters (pass 2) ---

def test_merges_same_plate_split_by_a_gap_just_over_the_cutoff():
    # Real case: L605-HZ / L-605-HZ, 1.73s gap, 93.3 similarity.
    clusters = [
        _cluster("L605-HZ", 13.97, 14.40, 0.97, 9),
        _cluster("L-605-HZ", 16.13, 19.13, 0.98, 87),
    ]
    merged = merge_finished_clusters(clusters)
    assert len(merged) == 1
    assert merged[0]["plate_text"] == "L-605-HZ"


def test_does_not_merge_overlapping_but_textually_different_plates():
    # Real case: ZH-509-1 (similarity to L-605-HZ is only 37.5) must stay
    # separate even though its time window overlaps L-605-HZ's - otherwise a
    # truck's tractor and trailer (legally separate plates) could get merged.
    clusters = [
        _cluster("L-605-HZ", 16.13, 19.13, 0.98, 87),
        _cluster("ZH-509-1", 16.17, 16.33, 0.87, 3),
    ]
    merged = merge_finished_clusters(clusters)
    assert len(merged) == 2


def test_leftover_duplicate_from_processing_order_still_gets_merged():
    # Real case: N894-N (71.4 similarity to N-894-JV) sat as its own cluster
    # after pass 1 due to processing order - pass 2 must catch it.
    clusters = [
        _cluster("N894-N", 2.13, 2.43, 0.78, 5),
        _cluster("N-894-JV", 2.30, 3.73, 0.98, 42),
    ]
    merged = merge_finished_clusters(clusters)
    assert len(merged) == 1
    assert merged[0]["plate_text"] == "N-894-JV"
