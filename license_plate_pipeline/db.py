"""PostgreSQL storage for plate events - the Pi's local "hot" store (Phase 10).

Design is image-ready and sync-ready from day one so later phases need no schema
migration:
  - `image_path`      - NULL for now; will hold a local path or GCS URI once we
                        start saving a photo per vehicle.
  - `synced_to_cloud` - FALSE until a row has been archived to GCP (the next
                        phase). The cleanup job will only ever delete rows that
                        are already synced, so nothing is lost when the pen drive
                        fills up.

psycopg2 is imported lazily (only in get_connection), so this module imports fine
on machines without the driver - and the pure logic (event_row) stays unit-testable.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Local Postgres on the Pi by default; override with the DATABASE_URL env var.
DEFAULT_DSN = "dbname=plates user=plates host=localhost"

SCHEMA = """
CREATE TABLE IF NOT EXISTS plate_events (
    id              BIGSERIAL   PRIMARY KEY,
    plate_text      TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    first_seen      TIMESTAMPTZ NOT NULL,
    last_seen       TIMESTAMPTZ NOT NULL,
    frame_count     INTEGER     NOT NULL,
    camera_id       TEXT,
    image_path      TEXT,
    synced_to_cloud BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_plate_events_unsynced ON plate_events (synced_to_cloud, created_at);
CREATE INDEX IF NOT EXISTS idx_plate_events_plate    ON plate_events (plate_text);
"""

# first_seen/last_seen arrive as epoch seconds (the live pipeline's timestamps),
# so to_timestamp() turns them into proper TIMESTAMPTZ values.
INSERT_SQL = """
INSERT INTO plate_events
    (plate_text, confidence, first_seen, last_seen, frame_count, camera_id, image_path)
VALUES
    (%s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s)
RETURNING id;
"""


def get_dsn():
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def get_connection(dsn=None):
    """Connect to Postgres. Imports psycopg2 lazily so the module loads without it."""
    import psycopg2

    return psycopg2.connect(dsn or get_dsn())


def init_schema(conn):
    """Create the plate_events table + indexes if they don't exist. Idempotent."""
    with conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)


def event_row(event, camera_id=None, image_path=None):
    """Map a pipeline event dict to the INSERT parameter tuple (pure - unit-tested)."""
    return (
        event["plate_text"],
        float(event["best_confidence"]),
        float(event["first_seen"]),
        float(event["last_seen"]),
        int(event["frame_count"]),
        camera_id,
        image_path,
    )


def insert_event(conn, event, camera_id=None, image_path=None):
    """Insert one plate event and return its new id. Commits on success."""
    with conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, event_row(event, camera_id, image_path))
            return cur.fetchone()[0]
