# Raspberry Pi Setup — License Plate Pipeline

Instructions for setting up the Pi deployment of this project. Everything here
uses `license_plate_pipeline/pi/` (ONNX Runtime + RapidOCR), **not** the
dev-machine `ultralytics`/`paddleocr` path used by the notebook — that path
can't install on the Pi's ARM64 architecture (see `PROJECT_CONTEXT.md` →
"Open question for Phase 8" for why).

Run these commands directly on the Pi (over SSH, or via Raspberry Pi Connect's
terminal).

## 1. Verify architecture and OS

```bash
uname -m
cat /etc/os-release
```

You need **`aarch64`** (64-bit) — `onnxruntime`'s ARM wheel requires it. If this
shows `armv7l` or anything 32-bit, **stop here**: you'll need to reflash with the
64-bit Raspberry Pi OS image (Raspberry Pi Imager → choose "Raspberry Pi OS
(64-bit)") before continuing.

## 2. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-venv python3-pip python3-dev \
    libgl1 libglib2.0-0 libgtk-3-0 libatlas-base-dev
```

(`libgl1`/`libglib2.0-0`/`libgtk-3-0` are common missing-library errors for
OpenCV's GUI functions on a fresh Pi install — installing them up front avoids
a `cv2.imshow` crash later.)

## 3. Clone the repo

```bash
git clone https://github.com/Uvvi2004/license-plate-project.git
cd license-plate-project
```

## 4. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

This installs only `onnxruntime`, `opencv-python`, `rapidocr-onnxruntime`,
`numpy`, `pandas`, `rapidfuzz`, `tqdm` — **not** `ultralytics`, `torch`,
`paddleocr`, or `paddlepaddle`. Much lighter than the dev-machine setup, and
avoids the ARM64 wheel problem entirely.

**If this fails** with something like "no matching distribution" for one of
the pinned versions: these exact version numbers were validated on x86_64
Windows, not confirmed against ARM64 wheel availability (I don't have a way
to check that from here). Retry without exact pins:
```bash
pip install numpy opencv-python onnxruntime rapidocr-onnxruntime pandas rapidfuzz tqdm
```
and report back which versions actually installed — worth re-running the
smoke test (step 5) to confirm nothing behaves differently on whatever
versions land.

## 5. Run the smoke test

```bash
python pi_scripts/smoke_test.py
```

This runs detection + OCR on the bundled `test_images/demo2.jpg` and should
print a plate reading of `6FVZ747` at roughly 0.97 confidence — the same
result already validated on the dev machine. If this doesn't match, stop and
report back before continuing (something about this Pi's environment differs
from what was tested).

## 6. Test the webcam

```bash
python pi_scripts/webcam_test.py
```

Opens a live preview window (viewable through Raspberry Pi Connect's
screen-share) showing your webcam feed with detected plates boxed and labeled,
same as the notebook's step 10 on the dev machine. Press `q` in the window to
stop.

## What's Deliberately Not Done Yet

- No `systemd` service / auto-restart supervision (Phase 9 remaining item)
- No PostgreSQL connection or dashboard (Phase 10)
- No physical camera mounting/placement decision (Phase 11)

This gets inference working and validated on the actual hardware — the next
step after that is deciding what "success" looks like for the first live demo.
