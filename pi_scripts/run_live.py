"""Live deployment entry point: read plates from a camera and record each vehicle.

This is what runs at the gate. It opens the camera, runs the tracked pipeline
(detection every frame, OCR only a few times per tracked vehicle), and records
one row per vehicle as it leaves frame. Detection/OCR use the Pi engine
(ONNX + RapidOCR); output is validated (no junk, no non-Latin glyphs).

Two record destinations:
    - CSV (default) - simple, no setup.
    - PostgreSQL (--db) - the local "hot" store; on any DB error it falls back to
      the CSV so a truck is never lost. See license_plate_pipeline/db.py and
      POSTGRES_SETUP.md.

Run from the repo root:
    python pi_scripts/run_live.py                       # camera 0 -> plate_events.csv
    python pi_scripts/run_live.py --db                  # -> PostgreSQL (DATABASE_URL or local default)
    python pi_scripts/run_live.py --db --camera-id gate1
    python pi_scripts/run_live.py --camera 1
    python pi_scripts/run_live.py --preview             # live window (needs a display)

Stop with Ctrl-C (or 'q' in the preview window).
"""

import argparse
import csv
import datetime as dt
import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from license_plate_pipeline import db  # noqa: E402
from license_plate_pipeline.pi.pipeline import process_camera  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_live")

_stop = {"flag": False}


def main():
    parser = argparse.ArgumentParser(description="Live license-plate recording")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--db", action="store_true", help="record to PostgreSQL instead of CSV")
    parser.add_argument("--camera-id", default=None, help="tag rows with this camera/site id")
    parser.add_argument("--log", default="plate_events.csv", help="CSV path (primary, or DB fallback)")
    parser.add_argument("--preview", action="store_true", help="show a live preview window")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: _stop.update(flag=True))

    on_frame = None
    if args.preview:
        import cv2

        def on_frame(frame, boxes):
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Live plates - Ctrl-C to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _stop["flag"] = True

    # Optional Postgres connection (primary store when --db). If it can't connect
    # at all, we don't crash the gate - we log a warning and fall back to CSV.
    conn = None
    if args.db:
        try:
            conn = db.get_connection()
            db.init_schema(conn)
            print(f"Recording to PostgreSQL ({db.get_dsn()}). CSV fallback: {args.log}")
        except Exception:
            logger.exception("Could not connect to PostgreSQL - falling back to CSV only")
            conn = None

    log_path = Path(args.log)
    new_file = not log_path.exists()
    if conn is None:
        print(f"Recording to {log_path.resolve()} (camera {args.camera}). Ctrl-C to stop.")

    count = 0
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["iso_time", "plate_text", "confidence", "first_seen_epoch", "last_seen_epoch", "frames"])
            fh.flush()

        def write_csv(event, iso):
            writer.writerow([
                iso, event["plate_text"], f"{event['best_confidence']:.3f}",
                f"{event['first_seen']:.2f}", f"{event['last_seen']:.2f}", event["frame_count"],
            ])
            fh.flush()  # persist immediately - a power cut shouldn't lose logged trucks

        for event in process_camera(camera_index=args.camera, on_frame=on_frame, stop=lambda: _stop["flag"]):
            iso = dt.datetime.fromtimestamp(event["timestamp"]).isoformat(timespec="seconds")

            stored = False
            if conn is not None:
                try:
                    db.insert_event(conn, event, camera_id=args.camera_id)
                    stored = True
                except Exception:
                    logger.exception("DB insert failed - writing CSV fallback row instead")
            if not stored:
                write_csv(event, iso)

            count += 1
            print(f"[{iso}] {event['plate_text']} ({event['best_confidence']:.2f})")

    if conn is not None:
        conn.close()
    if args.preview:
        import cv2

        cv2.destroyAllWindows()
    dest = "PostgreSQL" if conn is not None else str(log_path)
    print(f"\nStopped. Recorded {count} vehicles to {dest}.")


if __name__ == "__main__":
    main()
