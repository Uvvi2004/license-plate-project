"""Live deployment entry point: read plates from a camera and log each vehicle.

This is what actually runs at the gate. It opens the camera, runs the tracked
pipeline (detection every frame, OCR only a few times per tracked vehicle), and
appends one row per vehicle to a CSV as it leaves frame. Detection/OCR use the
Pi engine (ONNX + RapidOCR); output is validated (no junk, no Chinese glyphs).

Run from the repo root:
    python pi_scripts/run_live.py                 # camera 0, logs to plate_events.csv
    python pi_scripts/run_live.py --camera 1
    python pi_scripts/run_live.py --preview       # also show a live window (needs a display)
    python pi_scripts/run_live.py --log trucks.csv

Stop with Ctrl-C (or 'q' in the preview window). Each logged row:
    iso_time, plate_text, confidence, first_seen_epoch, last_seen_epoch, frames

This is inference + logging only. Pushing rows to PostgreSQL is Phase 10; a CSV
is deliberately the first step so the pipeline can be validated on real trucks
before adding a database.
"""

import argparse
import csv
import datetime as dt
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from license_plate_pipeline.pi.pipeline import process_camera  # noqa: E402

_stop = {"flag": False}


def main():
    parser = argparse.ArgumentParser(description="Live license-plate logging")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--log", default="plate_events.csv", help="CSV output path")
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

    log_path = Path(args.log)
    new_file = not log_path.exists()
    print(f"Logging to {log_path.resolve()} (camera {args.camera}). Ctrl-C to stop.")

    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["iso_time", "plate_text", "confidence", "first_seen_epoch", "last_seen_epoch", "frames"])
            fh.flush()

        count = 0
        for event in process_camera(camera_index=args.camera, on_frame=on_frame, stop=lambda: _stop["flag"]):
            iso = dt.datetime.fromtimestamp(event["timestamp"]).isoformat(timespec="seconds")
            writer.writerow([
                iso,
                event["plate_text"],
                f"{event['best_confidence']:.3f}",
                f"{event['first_seen']:.2f}",
                f"{event['last_seen']:.2f}",
                event["frame_count"],
            ])
            fh.flush()  # persist immediately - a power cut shouldn't lose logged trucks
            count += 1
            print(f"[{iso}] {event['plate_text']} ({event['best_confidence']:.2f})")

    if args.preview:
        import cv2

        cv2.destroyAllWindows()
    print(f"\nStopped. Logged {count} vehicles to {log_path}.")


if __name__ == "__main__":
    main()
