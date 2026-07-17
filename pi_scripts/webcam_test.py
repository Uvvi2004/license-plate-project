"""Live webcam test on the Pi - same pattern as the notebook's step 10 on the
dev machine, but using the ONNX/RapidOCR pi/ pipeline.

Run from the repo root: python pi_scripts/webcam_test.py
Press 'q' in the video window to stop.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from license_plate_pipeline.pi.detection import detect_boxes  # noqa: E402
from license_plate_pipeline.pi.ocr import read_crop, select_plate_text  # noqa: E402

CAMERA_INDEX = 0  # first connected camera - adjust if you have more than one


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera (index {CAMERA_INDEX})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Capture resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Webcam open. Press 'q' in the video window to stop.")

    start_time = time.time()
    readings_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam - stopping.")
                break

            for x1, y1, x2, y2 in detect_boxes(frame):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = frame[y1:y2, x1:x2]
                picked = select_plate_text(read_crop(crop))
                if picked:
                    text, confidence = picked
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:6.2f}s] '{text}' (confidence: {confidence:.2f})")
                    readings_count += 1
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Live webcam (Pi) - press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nStopped. Captured {readings_count} raw readings.")


if __name__ == "__main__":
    main()
