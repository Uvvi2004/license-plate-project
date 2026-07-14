"""Dedup clustering: merge repeated sightings of the same physical plate into one event.

Two passes, both required - see PROJECT_CONTEXT.md "Dedup Clustering" for the full
history of why each one exists and the real failures that led to this design:

1. cluster_by_time_and_text: groups raw per-frame readings into sub-events, requiring
   BOTH time proximity and text similarity (pure time-proximity over-merges badly when
   different vehicles pass close together).
2. merge_finished_clusters: re-compares already-finished clusters against each other,
   since pass 1 only merges new readings into existing clusters and can leave genuine
   duplicates sitting side by side due to processing order. Allows a wider time gap
   only when similarity is very high (likely a brief detection dropout, not a new
   vehicle).
"""

import logging

from rapidfuzz import fuzz

from license_plate_pipeline.config import (
    CLUSTER_GAP_SECONDS,
    EXTENDED_GAP_SECONDS,
    HIGH_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def cluster_by_time_and_text(events, gap_seconds=CLUSTER_GAP_SECONDS, similarity_threshold=SIMILARITY_THRESHOLD):
    """Merge sub-events into one physical-plate cluster only if they're BOTH close in time
    AND textually similar - time alone isn't enough when vehicles pass close together.

    Checks against every cluster still within the time window (not just the most recent one) -
    otherwise an unrelated reading interleaving between two truly-same-plate readings would
    incorrectly split them into separate clusters.
    """
    clusters = []
    for event in sorted(events, key=lambda e: e["first_seen"]):
        best_match, best_score = None, -1
        for cluster in clusters:
            if event["first_seen"] - cluster["last_seen"] > gap_seconds:
                continue
            score = fuzz.ratio(event["plate_text"], cluster["plate_text"])
            if score >= similarity_threshold and score > best_score:
                best_match, best_score = cluster, score

        if best_match is not None:
            best_match["last_seen"] = max(best_match["last_seen"], event["last_seen"])
            best_match["frame_count"] += event["frame_count"]
            best_match["readings"].append((event["plate_text"], event["best_confidence"], event["frame_count"]))
            if event["best_confidence"] > best_match["best_confidence"]:
                best_match["plate_text"] = event["plate_text"]
                best_match["best_confidence"] = event["best_confidence"]
        else:
            clusters.append({
                "plate_text": event["plate_text"],
                "first_seen": event["first_seen"],
                "last_seen": event["last_seen"],
                "best_confidence": event["best_confidence"],
                "frame_count": event["frame_count"],
                "readings": [(event["plate_text"], event["best_confidence"], event["frame_count"])],
            })
    return clusters


def merge_finished_clusters(
    clusters,
    gap_seconds=CLUSTER_GAP_SECONDS,
    similarity_threshold=SIMILARITY_THRESHOLD,
    extended_gap_seconds=EXTENDED_GAP_SECONDS,
    high_similarity_threshold=HIGH_SIMILARITY_THRESHOLD,
):
    """Re-compare already-finished clusters against each other and merge any that qualify."""
    clusters = sorted(clusters, key=lambda c: c["first_seen"])
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if clusters[i] is None:
                continue
            for j in range(i + 1, len(clusters)):
                if clusters[j] is None:
                    continue
                a, b = clusters[i], clusters[j]
                gap = max(0, b["first_seen"] - a["last_seen"])
                score = fuzz.ratio(a["plate_text"], b["plate_text"])
                allowed_gap = extended_gap_seconds if score >= high_similarity_threshold else gap_seconds
                if score >= similarity_threshold and gap <= allowed_gap:
                    a["last_seen"] = max(a["last_seen"], b["last_seen"])
                    a["frame_count"] += b["frame_count"]
                    a["readings"].extend(b["readings"])
                    if b["best_confidence"] > a["best_confidence"]:
                        a["plate_text"] = b["plate_text"]
                        a["best_confidence"] = b["best_confidence"]
                    clusters[j] = None
                    changed = True
        clusters = [c for c in clusters if c is not None]
    return clusters


def dedup_events(raw_events):
    """Run both clustering passes in sequence. See module docstring for why both are needed."""
    clusters = cluster_by_time_and_text(raw_events)
    logger.info("Pass 1 (time + text similarity): %d sub-events -> %d clusters", len(raw_events), len(clusters))
    clusters = merge_finished_clusters(clusters)
    logger.info("Pass 2 (merge finished clusters): -> %d clusters", len(clusters))
    return clusters
