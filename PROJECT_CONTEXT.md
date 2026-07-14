# Warehouse Truck License Plate Tracking System — Project Context

## Goal

Camera-based system to track incoming/outgoing trucks at a warehouse: truck
status (arrived / at dock / departed), timestamp, license plate number.

**Hard requirement:** all data stays on-premises — no cloud dependency for the
core data pipeline (business/privacy requirement).

Current phase: personal/learning prototype on a Raspberry Pi 5, built by a
self-described tech beginner learning concepts alongside implementation.

## Roadmap & Status (expanded toward enterprise-ready, 2026-07-14)

Original plan was 10 phases to get a working prototype. Expanded below to
capture the full path discussed toward a genuinely polished, enterprise-fit
deployment — added phases 9 and 11 specifically for that, rather than
treating "prototype works" and "enterprise-ready" as the same finish line.

1. ✅ Learn just enough Python
2. ✅ Understand tools via existing YOLO tutorial notebook
3. ✅ Dataset acquired from Roboflow Universe
4. ✅ Model trained (see Training below)
5. ✅ Detection tested on images/video — works well
6. ✅ OCR added — switched to PaddleOCR after head-to-head test (see below)
7. 🔶 Video pipeline + data cleaning (in progress)
   - ✅ Frame-by-frame detect→crop→OCR on video, outputs a timestamped events table
   - ✅ **Data cleaning: confidence-based dedup clustering** — see "Dedup
     Clustering" section below for the full (honest, two-failed-attempts)
     writeup
   - ⬜ Validate against additional real-world test videos (to be uploaded)
   - ⬜ Live webcam feed test on this laptop, before moving to the Pi
8. ⬜ Raspberry Pi 5 + Camera Module 3 deployment
   - ⬜ Export YOLO to NCNN or ONNX (ARM-friendly runtime)
   - ⬜ Export PaddleOCR to ONNX (via `paddle2onnx`) or Paddle-Lite — no
     official ARM64 `paddlepaddle` pip wheel exists, see gotcha below
   - ⬜ Re-validate converted models against known test cases (e.g. `6FVZ747`)
     to confirm conversion didn't silently change accuracy
   - ⬜ Live camera integration (Camera Module 3 or an upgraded global-shutter
     camera — see Camera Hardware note below) feeding the same pipeline
   - ⬜ Physical placement decision (indoor mounting, dock-door angle, tractor
     vs. trailer per TN legal section below)
   - ⬜ Live demo: real trucks passing, live output — table/log for now,
     Postgres once Phase 10 exists
9. 🔶 **Production hardening** (closes the "notebook prototype" vs.
    "enterprise code" gap explicitly) — see "Phase 9 Hardening" section
    below for the full writeup
   - ✅ Extract the pipeline out of the notebook into `license_plate_pipeline/`
   - ✅ Error handling around detection/OCR (logs and skips a bad
     frame/crop instead of crashing the whole run)
   - ✅ Automated test suite (`pad_box`, `select_plate_text`, both dedup
     passes) using the real calibration cases as regression tests
   - ✅ Config management — constants moved to `config.py`
   - ⬜ Structured logging beyond basic error logging (log levels, format,
     destination) — not yet needed at this scale, revisit if it becomes hard
     to follow
   - ⬜ systemd service with auto-restart/supervision — Pi-specific, waits
     for Phase 8
   - ⬜ Reliability for 24/7 operation — SD card wear mitigation, watchdog —
     Pi-specific, waits for Phase 8
10. ⬜ Server (FastAPI or Spring Boot) + PostgreSQL + dashboard
    - ⬜ Schema design: plate number, first-seen, last-seen, confidence,
      status, camera/site id
    - ⬜ Pi → Postgres write path (on-prem network only — hard requirement)
    - ⬜ Data validation/sanitization before insert
    - ⬜ Dashboard: current trucks on-site + dwell time
11. ⬜ **Enterprise-fit polish** (new phase — explicitly scoped, see honest
    caveat below)
    - ⬜ Camera hardware evaluation: global shutter (fixes motion blur/skew
      the stock Camera Module 3's rolling shutter can't avoid on a moving
      truck), IR illumination, dynamic range
    - ⬜ Indoor mounting to remove weatherproofing/rain-ingress concerns
      entirely (still need to check placement relative to dock-door openings
      — indoor doesn't fully block wind-driven rain or condensation there)
    - ⬜ Redundancy/failover considerations (backup camera/compute, UPS)
    - **Honest scope note:** this phase closes *some* of the gap between a
      hobbyist Pi rig and true enterprise ALPR (Genetec/Rekor/Axis-class
      systems bring hardware redundancy, weatherproof purpose-built cameras,
      and vendor support contracts a Pi can't fully replicate). The
      on-prem/no-cloud requirement actually favors this self-hosted approach
      over several cloud-tied "enterprise" products — but this is a
      single/few-site self-hosted system, not a competitor to 24/7
      mission-critical multi-site ALPR vendors, and shouldn't be sold to
      anyone (including future-you) as equivalent to one.

## Hardware / Environment Decisions

- **Raspberry Pi 5 (8GB) + Camera Module 3** for eventual deployment — more
  CPU headroom than Pi 4, strong community support, low power. Considered
  Jetson Nano/Orin (steeper learning curve) and an old laptop (fine for
  prototyping, not 24/7 deployment).
- **Training** happened on Google Colab (free T4 GPU) — CPU training would be
  far slower. Local machine is fine for *inference* (no GPU needed).
- Local dev machine has **no CUDA GPU** — confirmed via `torch.cuda.is_available() == False`.
  All local inference runs on CPU. This is expected and fine for prototyping.
- Enterprise reality check (context, not the build target): real ALPR
  deployments use purpose-built systems (Genetec AutoVu, Rekor, Plate
  Recognizer, OpenALPR, Axis/Vaxtor) for edge-case data, reliability, and
  support contracts. This project is the "learn how it works" path, not a
  competitor to those products.

## Tennessee Legal Detail (camera placement)

Under TN Code § 55-4-110 (TN is a one-plate state):
- Passenger vehicles & trailers/semi-trailers: **rear plate only**
- Truck tractors (cab unit): **front plate only**

A semi (tractor + trailer) may have plates in two different locations on two
different registered units. May need cameras at both an entry point
(front-facing, for the tractor) and dock/exit (rear-facing, for the trailer),
depending on whether the goal is tracking the tractor, trailer, or both.
Plate height on trucks varies more than cars (law sets a 12" minimum, no
fixed max) — camera framing needs a wider vertical tolerance than a standard
car-plate ALPR setup. **This decision is still open.**

## Dataset

- Source: Roboflow Universe, `roboflow-universe-projects` workspace (chosen
  over a personal account for longevity — the original tutorial's dataset
  from a student account did in fact disappear)
- Project: `license-plate-recognition-rxg4e`, version 4
- ~21,173 train images, 2,046 valid images, plus a test split
- `data.yaml`: `nc: 1`, `names: ['License_Plate']`
- **Known caveat:** upstream dataset has some train/test leakage (near-dupe
  images across splits per an external review) — real-world accuracy may be
  somewhat lower than reported, especially on trucks (dataset skews toward
  cars) and non-US plate formats.
- ⚠️ A Roboflow API key was pasted in plain text in earlier chat history —
  **confirm it's been regenerated** (Roboflow → Settings → API Key) before
  reusing any code referencing it.

## Model / Training (verified present locally)

- Base: `yolo26n.pt` (YOLO26 nano, ~2.5M params, ~5.4MB) — chosen for
  Pi-friendly CPU inference speed and simplified export (no separate NMS
  step). YOLO12 was considered but reportedly less accurate for small-object
  detection like this.
- Trained on Colab T4 GPU, 25 epochs, `imgsz=640`, `patience=5`,
  `batch=16` (~1,324 batches/epoch), 3.156 hours total.
- **Lesson learned the hard way:** first training run wasn't saved to Drive
  and was lost when the Colab session disconnected. All runs now save via
  `project=".../license_plate_project"` — treat this as mandatory for any
  long run.
- **Final results** (confirmed against `runs/train/results.csv` locally):
  mAP50 = 0.982, mAP50-95 = 0.694, Precision = 0.984, Recall = 0.953.
- Weights: [models/best.pt](models/best.pt) (copy for easy script access) and
  [runs/train/weights/](runs/train/weights/) (`best.pt` + `last.pt`, full
  training run with PR curves, confusion matrix, results.csv, etc.)
- To resume interrupted training: load `last.pt` and call
  `model.train(resume=True)` — **never** restart from `yolo26n.pt`, that
  discards all progress.

## OCR Accuracy — RESOLVED (switched to PaddleOCR)

Original pipeline (detection → crop → EasyOCR) produced garbage text on the
Indian-plate test case: `MH01AY8866` read back as `'Eaho 18266'` at
confidence 0.15. Detection/cropping was confirmed correct — the failure was
isolated to OCR.

Applied fixes first (upscale 3x, grayscale + histogram equalization, 10%
bbox padding — all still in `scripts/main.py`'s `preprocess_for_ocr`/
`pad_box`), which improved the same test to `'HHo14y8866'` at 0.29 confidence
— better but still wrong on the letters.

**Then tested on a real US plate** (California, Getty stock photo, ground
truth `6FVZ747`) — the actual representative case for this project's
deployment target — comparing EasyOCR vs PaddleOCR head to head in
[notebooks/license_plate_pipeline.ipynb](notebooks/license_plate_pipeline.ipynb):

| Engine | Result | Confidence |
|---|---|---|
| EasyOCR | `'6FVZ7LZ'` (fragmented, wrong) | 0.53 |
| **PaddleOCR** | **`'6FVZ747'` (exact match)** | **0.97** |

**Decision: PaddleOCR replaces EasyOCR as the OCR engine** in the pipeline.
Chosen both for this accuracy win and because PP-OCRv4
"mobile" models (~10-15MB combined det+rec) have a much lighter footprint
than EasyOCR's CRAFT+recognition stack (100MB+) and a real edge-deployment
story via PaddleLite — relevant for the eventual Pi target.

**Cleanup (2026-07-14):** `easyocr` was removed entirely — from the
notebook (no more side-by-side comparison, just the single PaddleOCR call),
from the venv (`pip uninstall`), and from `requirements.txt`. Once a
decision like this is made, running both engines every time is pure wasted
compute/footprint, especially with a Raspberry Pi target — the comparison
numbers are preserved here for the record, no need to keep re-deriving them
live.

### Environment gotchas hit while setting this up (document for the Pi port)

1. **PaddleOCR's model host doesn't resolve via this machine's default DNS.**
   `paddleocr.bj.bcebos.com` (Baidu Cloud) times out on the default resolver
   but resolves fine via Google/Cloudflare DNS (8.8.8.8 / 1.1.1.1) to
   `103.235.47.176`, and the server itself responds instantly once reached.
   Worked around it by downloading the 3 model tarballs (det/rec/cls) with
   `curl --resolve` and extracting them by hand into
   `~/.paddleocr/whl/{det,rec,cls}/...` — see git history of this file /
   PROJECT_CONTEXT for the exact commands if this needs redoing on another
   machine (e.g. the Pi) with the same DNS problem.
2. **`paddlepaddle==3.3.1` (latest, pip default) cannot run these pretrained
   PP-OCR inference models** — errors with `NotFoundError: OneDnnContext
   does not have the input Filter [operator < fused_conv2d >]`. This is
   PaddlePaddle 3.x's rewritten inference engine (PIR) choking on models
   exported years ago with an older Paddle version. `enable_mkldnn=False`
   does **not** fix it. **Fix: pinned `paddlepaddle==2.6.2`** (see
   requirements.txt) — matches the era these models were exported in.
3. **Import order matters on Windows:** `ultralytics` (which pulls in
   `torch`) must be imported **before** `paddleocr` (which pulls in
   `paddle`), or torch fails with `OSError: ... shm.dll or one of its
   dependencies`. The notebook imports `ultralytics` first — keep that
   order in any new script.

## Video Pipeline + Dedup (Phase 7/9 first pass, 2026-07-14)

Section 8 of `notebooks/license_plate_pipeline.ipynb` runs the full
detect→crop→OCR pipeline frame-by-frame over `test_images/demo.mp4` (631
frames @ 30fps, ~2 min on this CPU) and outputs a pandas table: one row per
detected *event* (plate text, first-seen/last-seen seconds into video, frame
count, best confidence) — previewing the shape of the future Postgres events
table.

**Two things had to be handled that single-image mode didn't:**
- Plate crops also pick up nearby text (registration stickers, dealer frame
  text) — `select_plate_text()` picks the fragment that looks plate-shaped
  (5-8 alphanumeric chars) instead of just the highest-confidence fragment.
- The same vehicle appears across many consecutive frames — deduped via the
  approach already planned in this doc: group by **exact plate text**, treat
  a re-appearance as a new event only after a `GAP_SECONDS` (1.5s) gap.

**Confirmed empirically (not just a hypothetical caveat):** `demo.mp4` turns
out to show non-US/European-style plates (e.g. `R-183-JF`), so this run is a
**pipeline mechanics test, not an accuracy benchmark for the real TN truck
target**. It surfaced real instances of the known dedup limitation — e.g.
`R-183-JF` (46 frames) and `R-183JF` (4 frames, dash dropped) were almost
certainly the same physical plate, split into two table rows because OCR
flickered on punctuation. Same pattern with `66-HH-07` vs `66-HH-O7`
(0/O confusion). **This is real evidence, not speculation, that Phase 9
eventually needs box-position/motion tracking, not just exact-text
matching**, to avoid fragmenting one vehicle sighting into several rows.

**Why this matters for Postgres (Phase 10):** the `events_table` columns
(plate number, first-seen, last-seen, confidence) are meant to map directly
onto that table's schema — building this now previews what those rows will
look like, and the flicker-fragmentation problem needs solving *before*
writing to the DB, or duplicate rows will pile up there too.

## Dedup Clustering — DONE (Phase 7, 2026-07-14)

Section 9 of `notebooks/license_plate_pipeline.ipynb` fixes the fragmentation
problem flagged above. Took two failed attempts before landing on a working
approach — worth recording exactly why, so this doesn't get re-tried blindly:

1. **Time proximity alone:** merged 71 sub-events into just 2 clusters,
   combining several genuinely *different* plates. `demo.mp4` turns out to be
   continuous traffic with sub-second gaps between different vehicles — about
   the same gap size as OCR flicker within one sighting. Time alone can't
   distinguish the two.
2. **Time + text similarity (`rapidfuzz`), but only checked the
   most-recently-created cluster:** better (27 clusters), but the flagship
   example (`R-183-JF`/`R-183JF`) still didn't merge — an unrelated reading
   interleaving in time between two truly-same-plate readings caused a split.
3. **Fix: check every cluster still within the time window, merge into the
   best text match.** Similarity threshold (70 on `rapidfuzz.fuzz.ratio`)
   calibrated against real data: same-plate flicker pairs scored 87-94,
   different-plate pairs scored 25-50. Result: 71 → 25 clusters,
   `R-183-JF`/`R-183JF` and `H-044-LX`/`H-644-LX` correctly merge, genuinely
   different plates stay separate.

This is a heuristic improvement, not full box/motion tracking — still
documented as a real Phase-9-territory upgrade if this heuristic starts
failing on new footage (to be uploaded).

**Follow-up fix (same day):** manually checked the resulting table row by
row against the source data rather than trusting it, and found two more
real issues:
- `N894-N` sat as its own row next to `N-894-JV` despite scoring 71.4
  similarity (above threshold) — caused by a processing-order quirk: pass 1
  only merges new raw events into existing clusters, it never re-compares
  two already-finished clusters against each other.
- `L605-HZ`/`L-605-HZ` (93.3 similarity) and `ZH-509-7`/`ZH-509-1` (87.5
  similarity) are the same plates, split by a time gap just over the 1.5s
  cutoff (1.73s/1.90s) — likely a brief detection dropout.

Fix: a second pass re-compares every pair of finished clusters, with a
wider allowed gap (3s) specifically when similarity is very high (≥85).
**Explicitly checked this doesn't overreach:** `ZH-509-1` vs `L-605-HZ`
scores only 37.5 and correctly stayed separate, despite overlapping in
time — important because this project's trucks may have two legally
separate plates (tractor + trailer, see TN section above), so a rule that
merges low-confidence overlapping reads *without* a text-similarity check
would risk silently deleting a real second plate. Result: 25 → 21 clusters,
confirmed correct on manual inspection.

### Open question for Phase 8 (Pi deployment)

The `paddlepaddle` pip wheel is x86_64-only (Windows/Linux/macOS) — there is
**no official pip wheel for Raspberry Pi's ARM64**. Real Pi deployment will
likely need either **Paddle-Lite** (Baidu's dedicated ARM/mobile inference
runtime) or exporting the PP-OCR models to ONNX and running them via
`onnxruntime` (which does have ARM64 wheels). This wasn't resolved yet —
flag it before starting Phase 8, don't assume today's PaddleOCR setup
ports directly to the Pi.

## Concepts Already Covered (don't re-explain from scratch)

Epochs/batches, loss, mAP50 vs mAP50-95, overfitting & `patience`,
train/valid/test split roles, what a `.pt` file is (binary weights, not
human-editable — only interact via `YOLO("best.pt")`), fine-tuning vs.
training from scratch, why training vs. "memory" are different (a model
can't have individual facts added/removed via fine-tuning — that's what the
planned PostgreSQL layer is for), general model file-size scaling, Colab
Drive persistence gotchas.

## Local Environment (this migration, 2026-07-13/14)

- Project root: `c:\license_plate_project`
- Python 3.10.11 venv at `venv/` (ultralytics needs 3.10–3.12; the machine's
  default `python`/`py` is 3.14, which is untested with ultralytics/torch —
  use the venv, not system Python)
- Registered as a Jupyter kernel: **"License Plate Project (venv)"**
  (`license_plate_venv`) — select this kernel when opening any notebook here.
- Installed (see [requirements.txt](requirements.txt), final working pins):
  `ultralytics==8.4.95`, `torch==2.13.0` (CPU build), `torchvision==0.28.0`,
  `paddleocr==2.9.1`, `paddlepaddle==2.6.2` (pinned — see OCR gotchas above),
  `numpy==1.26.4` (pinned <2 — paddleocr's `imgaug` dependency errors on
  numpy 2.0), `opencv-python==4.10.0.84`, `jupyter`, `ipykernel`.
  `easyocr` was installed during the comparison, then removed entirely once
  PaddleOCR was confirmed the winner — not a project dependency.
- Verified: `models/best.pt` loads correctly (`model.names == {0:
  'License_Plate'}`), `model.predict()` runs end-to-end, and the full
  detect→crop→OCR pipeline correctly reads a real US plate (`6FVZ747`, 0.97
  confidence) via PaddleOCR — all confirmed by executing
  `notebooks/license_plate_pipeline.ipynb` top to bottom.
- **Structure decision (2026-07-14):** everything lives in one self-contained
  notebook (`notebooks/license_plate_pipeline.ipynb`), matching the original
  Colab workflow of running/understanding one step at a time, rather than
  splitting logic across separate `.py` scripts. The earlier `scripts/`
  folder (`main.py`, `smoke_test.py`) and a first draft comparison notebook
  were deleted once their content was folded into this one notebook. When
  Phase 8 (Pi deployment) starts, the pipeline function in this notebook
  will get extracted back into a plain script — the Pi runs it as a
  headless/systemd service, not a notebook.
- `test_images/demo.mp4` and `test_images/demo2.jpg` (US plate, ground truth
  `6FVZ747`) are populated. `demo.mp4` and the original Indian-plate
  `demo2.jpeg` came from the tutorial's GitHub repo
  ([Arijit1080/Licence-Plate-Detection-using-YOLO-V8](https://github.com/Arijit1080/Licence-Plate-Detection-using-YOLO-V8));
  `demo2.jpg` (US plate) was swapped in later and is a Getty stock photo —
  fine for local testing, don't use in anything public-facing.

## Project Structure

```
license_plate_project/
├── models/
│   └── best.pt                     # copy of trained weights, for easy access
├── runs/train/                      # full Ultralytics training run (curves, results.csv, weights/)
├── license_plate_pipeline/          # Phase 9: the tested, hardened pipeline implementation
│   ├── config.py                     # all tunable constants, calibrated against real data
│   ├── detection.py                  # model loading, pad_box, detect_boxes (error-handled)
│   ├── ocr.py                        # OCR engine loading, preprocessing, select_plate_text
│   ├── dedup.py                      # both clustering passes
│   └── pipeline.py                   # orchestration: read_plates_from_frame/_image, process_video
├── tests/                           # pytest suite - pure logic, real calibration cases as regression tests
├── notebooks/
│   └── license_plate_pipeline.ipynb  # interactive walkthrough; imports from license_plate_pipeline/, doesn't duplicate it
├── test_images/
│   ├── demo.mp4                       # tutorial demo video
│   └── demo2.jpg                      # US plate (California), ground truth 6FVZ747
├── venv/                            # Python 3.10 virtual environment
├── requirements.txt
└── PROJECT_CONTEXT.md               # this file
```

## Phase 9 Hardening (2026-07-14)

Extracted the notebook's detect/OCR/dedup logic into `license_plate_pipeline/`,
a proper package with error handling and a pytest suite — started this now
(while waiting on the Pi connection and better test footage) rather than
waiting for Phase 8, since it doesn't need either.

- **Error handling:** `detection.detect_boxes` and `ocr.read_crop` catch and
  log exceptions per-frame/per-crop instead of letting one bad frame crash
  a whole video run.
- **Tests lock in real behavior, not hypotheticals:** `tests/test_dedup.py`
  uses the *actual* calibration cases found while building this (the
  `R-183-JF`/`R-183JF` flicker merge, the `N894-N` processing-order bug, the
  `ZH-509-1` vs `L-605-HZ` non-merge) as regression tests — if a future
  change breaks any of these, the tests catch it immediately.
- **Notebook now imports from the package** instead of duplicating function
  definitions — re-ran it end to end after the refactor and confirmed
  byte-identical results (71 → 25 → 21 clusters, same final table). One
  deliberate exception: the notebook's step 7 (`detect_and_read`) still
  shows *raw, unfiltered* OCR output (via `detection.detect_boxes` +
  `ocr.read_crop` directly, bypassing `select_plate_text`) — that's
  intentional, preserving the pedagogical point that step 8 exists to solve
  the sticker/dealer-text noise problem visible in step 7.
- **Not done yet:** the systemd-supervision and SD-card-wear items are
  genuinely Pi-specific and still wait for Phase 8.

## Immediate Next Steps

1. **Validate against additional real-world test videos** — blocked on
   user upload of more representative (US/TN, ideally truck) footage.
2. **Live webcam test locally** — postponed by user choice, not blocked on
   anything; actionable whenever.
3. Both Pi-related items (ONNX/NCNN export, live camera integration) are
   paused until the user explicitly reconnects Pi work — don't start them
   unprompted.

Everything after that follows the phase order in the Roadmap section above;
don't re-derive next steps from scratch, that list is the source of truth.

Also still outstanding, not urgent: confirm the Roboflow API key was
regenerated (a key was pasted in plain text early in this project's history
— see Dataset section above).

## Working Style Notes

- User is a genuine beginner, learning concepts hands-on — prefers conceptual
  explanation alongside practical steps, not just code.
- Prefers structured, phase-by-phase progress over jumping ahead.
