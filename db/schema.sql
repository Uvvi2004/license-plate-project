-- Plate event store (Phase 10). Applied automatically by run_live.py --db, but
-- kept here for reference and for provisioning a cloud mirror (e.g. Cloud SQL)
-- later. Image-ready (image_path) and sync-ready (synced_to_cloud) from the start.

CREATE TABLE IF NOT EXISTS plate_events (
    id              BIGSERIAL   PRIMARY KEY,
    plate_text      TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    first_seen      TIMESTAMPTZ NOT NULL,
    last_seen       TIMESTAMPTZ NOT NULL,
    frame_count     INTEGER     NOT NULL,
    camera_id       TEXT,
    image_path      TEXT,                             -- local path or GCS URI; NULL until images added
    synced_to_cloud BOOLEAN     NOT NULL DEFAULT FALSE, -- true once archived to GCP
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_plate_events_unsynced ON plate_events (synced_to_cloud, created_at);
CREATE INDEX IF NOT EXISTS idx_plate_events_plate    ON plate_events (plate_text);
