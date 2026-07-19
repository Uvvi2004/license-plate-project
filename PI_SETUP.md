# Raspberry Pi Setup — License Plate Pipeline

Battle-tested on a **Raspberry Pi 5** running **Debian 13 (trixie)**, 2026-07-18.
This runs `license_plate_pipeline/pi/` (ONNX Runtime + RapidOCR) — no
PyTorch/PaddlePaddle, so it installs on the Pi's ARM64 architecture.

Run these on the Pi (over SSH, or a terminal in Raspberry Pi Connect).

## 1. Verify architecture

```bash
uname -m          # must print: aarch64
cat /etc/os-release
```

You need **`aarch64`** (64-bit). If it shows `armv7l` or 32-bit, reflash with the
64-bit Raspberry Pi OS first — `onnxruntime`'s ARM wheel requires 64-bit.

## 2. System dependencies

```bash
sudo apt update
sudo apt install -y git curl libgl1 libglib2.0-0 libgtk-3-0
```

Notes from the real install:
- **No `libatlas-base-dev`** — it was removed in Debian 13, and isn't needed
  (the pip `numpy` wheel bundles its own math library).
- If `apt update` errors with **"The package cache file is corrupted"**, clear
  and re-sync: `sudo rm -rf /var/lib/apt/lists/* && sudo apt update`, then retry.

## 3. Clone the repo

```bash
cd ~
git clone https://github.com/Uvvi2004/license-plate-project.git
cd license-plate-project
```

## 4. Create a Python 3.12 environment with `uv`

**Why 3.12 and not the system Python:** the validated OCR engine
(`rapidocr-onnxruntime==1.4.4`, which uses the accurate PP-OCRv4 model) requires
**Python < 3.13**, but current Pi OS ships Python 3.13. On 3.13, pip is forced to
an older RapidOCR (1.2.3) with a weaker model that misreads plates. `uv` fetches
a prebuilt CPython 3.12 (no compiling) and sidesteps this cleanly.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 .venv312
source .venv312/bin/activate
uv pip install -r requirements-pi.txt
```

Confirm the install list shows **`rapidocr-onnxruntime==1.4.4`** (not 1.2.x).

## 5. Smoke test — the go/no-go

```bash
python pi_scripts/smoke_test.py
```

Should print `'6FVZ747' (confidence: ~0.98)` and **PASSED**, matching the dev
machine. (The `Failed to detect devices under /sys/class/drm` warnings are
harmless — that's onnxruntime looking for a GPU and using the CPU.) If it reads
something else or a much lower confidence, stop and check you're in `.venv312`
with RapidOCR 1.4.4 — don't continue until this passes.

## 6. Run the live logger — the deployment

```bash
python pi_scripts/run_live.py                 # camera 0 -> plate_events.csv
python pi_scripts/run_live.py --camera 1      # if the USB webcam is a different index
python pi_scripts/run_live.py --preview       # live window (see note below)
```

It opens the camera at 720p, runs the tracked pipeline, and appends **one
validated row per vehicle** to `plate_events.csv` (`iso_time, plate_text,
confidence, first_seen, last_seen, frames`) as it leaves frame. Output is
filtered — no junk, no non-Latin characters, confidence floor applied. Stop with
**Ctrl-C** (or `q` in the preview window). Check results with `cat plate_events.csv`.

Notes:
- **`--preview` needs a display.** Over SSH it errors (`could not connect to
  display`); that's expected — either drop `--preview` (headless, terminal +
  CSV), or run it from a terminal on the Pi's desktop via Raspberry Pi Connect.
- **Camera index:** `ls /dev/video*` — the Pi lists many internal video nodes; a
  USB webcam is usually `video0`. If camera 0 won't open, try `--camera 1`.
- **It's slow (~2–3 fps, ~2s per read).** That's the Pi CPU, and it's fine for a
  gate — a truck sits in view for seconds. Reading is most reliable on a flat,
  straight-on, well-lit plate that fills the frame.

## Starting a later session

Each time you reconnect:
```bash
cd ~/license-plate-project
source .venv312/bin/activate
python pi_scripts/run_live.py
```

## What's deliberately not done yet

- **`systemd` auto-start** so `run_live.py` launches on boot and restarts on
  crash (next step, once live logging is confirmed by hand).
- **PostgreSQL + dashboard** (Phase 10) — `run_live.py` writes a CSV on purpose,
  so the pipeline can be proven on real trucks before adding a database.
- **Physical camera mounting/placement.**
