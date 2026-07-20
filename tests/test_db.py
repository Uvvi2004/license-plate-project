"""Tests for the DB layer's pure logic. No real Postgres needed - psycopg2 is
imported lazily in db.get_connection, and here we drive insert_event with a fake
connection so the SQL parameter mapping and commit behavior are locked in."""

from license_plate_pipeline.db import INSERT_SQL, event_row, insert_event


def test_event_row_maps_pipeline_event_to_insert_tuple():
    event = {
        "plate_text": "ABC1234",
        "best_confidence": 0.95,
        "first_seen": 1000.0,
        "last_seen": 1005.5,
        "frame_count": 12,
    }
    assert event_row(event, camera_id="gate1") == ("ABC1234", 0.95, 1000.0, 1005.5, 12, "gate1", None)


def test_event_row_defaults_camera_and_image_to_none():
    event = {"plate_text": "XY9876", "best_confidence": 0.8, "first_seen": 1.0, "last_seen": 2.0, "frame_count": 3}
    assert event_row(event) == ("XY9876", 0.8, 1.0, 2.0, 3, None, None)


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return (42,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self):
        self._cur = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.committed = True  # psycopg2 commits on successful `with conn`
        return False


def test_insert_event_executes_insert_and_returns_id():
    conn = _FakeConn()
    event = {"plate_text": "ABC1234", "best_confidence": 0.9, "first_seen": 1.0, "last_seen": 2.0, "frame_count": 5}

    new_id = insert_event(conn, event, camera_id="gate1")

    assert new_id == 42
    assert conn.committed
    sql, params = conn._cur.executed[0]
    assert sql == INSERT_SQL
    assert params == ("ABC1234", 0.9, 1.0, 2.0, 5, "gate1", None)
