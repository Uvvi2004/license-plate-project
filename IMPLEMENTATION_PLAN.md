# Implementation Plan — Fast, Enterprise-Grade Detection → Pi → Truck/Trailer

Written 2026-07-16, after root-causing the OCR speed problem. This is the
authoritative "what to build next, in what order" document. Each step has a
**validation gate** — do not move to the next step until the gate passes.
`PROJECT_CONTEXT.md` stays the source of truth for history/status; this file
is the forward plan.

---

## Root-cause diagnosis (measured, not guessed)

Benchmarked on the dev machine against the real `demo2.jpg` plate crop
(185x109 native), 6-8 runs averaged:

| Configuration | Time/call | Correct read? |
|---|---|---|
| RapidOCR full: det + angle-cls + rec | **1363 ms** | yes (`6FVZ747` @ 0.97) |
| RapidOCR det + rec (no angle-cls) | **966 ms** | yes |
| RapidOCR rec-only (no internal det) | **22 ms** | no — garbage (multi-line crop) |
| Upscale 1x vs 2x vs 3x (det+rec) | ~915 / ~1050 / ~980 ms | all correct — upscale doesn't help RapidOCR |

**The finding that changes everything:** ~95% of OCR time is the engine's
*internal text-detection* stage — re-finding text inside a crop where YOLO
already found the plate. Character recognition itself is ~22 ms. PaddleOCR
has the same architecture (det+cls+rec), which is why it was also "laggy" —
both engines pay ~1s/crop, and the pipeline calls OCR on **every frame a
plate is visible** (~600+ calls for a 21s video).

**Conclusion: the OCR engine is not the bottleneck. The number of OCR calls
is.** Swapping engines again would be solving the wrong problem. The fix is
to track plates across frames and OCR each physical plate only a few times.

---

## Step 0 — Quick win: trim RapidOCR config (~30 min, tiny)

In `license_plate_pipeline/pi/ocr.py`:

1. `read_crop`: call the reader with `use_cls=False` (angle classifier adds
   ~400 ms/call and does nothing for us — a fixed, upright gate camera never
   produces rotated plates). RapidOCR's `__call__` accepts this kwarg.
2. Change `preprocess_for_ocr` to return the crop **unchanged** (no 3x
   upscale). Benchmarks above show upscale gains RapidOCR nothing in accuracy
   and costs a cubic resize per call. (Keep the function — the interface and
   its test stay; it just becomes identity for the pi path. Docstring should
   say why, citing the 1x/2x/3x benchmark.)

**Validation gate:** `venv/Scripts/python.exe pi_scripts/smoke_test.py` still
prints `6FVZ747` and PASSES. `pytest tests/` still green.

---

## Step 1 — THE core fix: IOU box tracking + selective OCR (main effort)

> **DONE (2026-07-16). Validated on `demo.mp4` — both pipelines now emit a
> clean 8-event table with all 7 clearly-readable plates correct and zero
> fragment/misread rows:**
>
> | Pipeline | Before (per-frame) | After (tracking) | Speedup |
> |---|---|---|---|
> | Pi / RapidOCR | 1171.7s, 14 events (fragmented) | **103s, 8 events (clean)** | **~11x** |
> | Dev / PaddleOCR | ~120s, 10 events | **64s, 8 events (clean)** | ~2x |
>
> Implemented as `tracking.py` (pure `PlateTracker`) + `video.py` (shared cv2
> driver, engine funcs injected) + a 3rd dedup pass `absorb_fragments` (folds a
> partial read like `HZ` into the `L-605-HZ` it's a substring of — YOLO
> sometimes emits a second partial box of one plate) + a `MIN_PLATE_LEN` stub
> filter. Both `pipeline.py` and `pi/pipeline.py` `process_video` now delegate
> to the shared driver. 10 new tracking tests; full suite 25 passed. Only gap
> vs. the old reference: the marginal `ZH-509-1` trailer-plate (conf 0.87 in the
> old run) didn't surface as its own event — revisit if it turns out to matter
> for the tractor/trailer dual-plate case (Step 5).

### New module: `license_plate_pipeline/tracking.py`

Pure logic, **no ML imports** (same philosophy as `dedup.py`) — takes boxes
in, decides which need OCR, emits events. Fully unit-testable without models.

Design:

- **Track** dict/dataclass: `track_id`, `box` (last known), `first_seen`,
  `last_seen` (timestamps), `frames_seen`, `missed_frames`, `readings`
  (list of `(text, confidence)`), `ocr_attempts`.
- **Per frame:** caller runs YOLO (`detect_boxes`, cheap — keep every frame),
  passes boxes + timestamp to `tracker.update(boxes, timestamp)`:
  - Greedy IOU matching of detections to open tracks (standard IOU on
    `(x1,y1,x2,y2)`; match if IOU ≥ `IOU_THRESHOLD`, highest-IOU pairs first).
  - Unmatched detection → new track. Unmatched track → `missed_frames += 1`.
  - Track with `missed_frames > MAX_MISSED_FRAMES` → closed, emitted as an
    event (same dict shape as today's `finished_events` rows: `plate_text`
    from best reading via `select_plate_text` logic, `first_seen`,
    `last_seen`, `frame_count`, `best_confidence`) so the existing two-pass
    dedup (`dedup.py`) runs downstream **unchanged**.
  - `update()` returns the list of tracks that **need OCR this frame** (the
    caller does the actual OCR and reports back via
    `tracker.add_reading(track_id, text, confidence)`) — this keeps
    tracking.py free of OCR imports and testable.
- **Selective-OCR policy** (when does a track need OCR?):
  - Not before `frames_seen >= MIN_TRACK_FRAMES` (filters 1-2 frame noise —
    replaces today's `MIN_FRAMES` post-filter role at the source).
  - Then at most once every `OCR_FRAME_INTERVAL` frames.
  - Stop entirely once `ocr_attempts >= MAX_OCR_PER_TRACK` **or** a reading
    matching the plate pattern reached `EARLY_STOP_CONFIDENCE`.
- **New constants in `config.py`** (starting values — calibrate in the gate):
  `IOU_THRESHOLD = 0.3`, `MAX_MISSED_FRAMES = 15` (≈0.5s @30fps),
  `MIN_TRACK_FRAMES = 3`, `OCR_FRAME_INTERVAL = 10`,
  `MAX_OCR_PER_TRACK = 5`, `EARLY_STOP_CONFIDENCE = 0.95`.

### Wiring

- `license_plate_pipeline/pipeline.py` **and** `pi/pipeline.py`: replace the
  per-frame OCR loop inside `process_video` with the tracker flow. The
  tracker module is shared; only the injected `detect_boxes`/`read_crop`
  differ between the two — if it's cleaner, refactor `process_video` into one
  shared implementation taking the detect/ocr functions as parameters.
- Notebook step 8 keeps working (it imports `read_plates_from_frame` — leave
  that function in place for single-frame use; the notebook's video cell can
  later be updated to the tracker API in a separate pass).

### Tests (`tests/test_tracking.py` — pure synthetic, no models)

- One box drifting a few px/frame → exactly one track, one event.
- Two boxes far apart moving in parallel → two tracks (never cross-matched).
- Box vanishes for < MAX_MISSED_FRAMES then returns at ~same spot → same track.
- Box vanishes for > MAX_MISSED_FRAMES → two events.
- OCR policy: given a 60-frame track, `update()` requests OCR on the expected
  frames only; stops after a 0.96-confidence plate-pattern reading.
- IOU function itself: identical boxes → 1.0, disjoint → 0.0, half-overlap case.

### Validation gate (the important one)

1. Dev pipeline (`pipeline.py` + PaddleOCR) on `demo.mp4`: final deduped
   table must still contain the validated 10 plates (`R-183-JF`, `N-894-JV`,
   `L-656-XH`, `H-644-LX`, `K-884-RS`, `66-HH-07`, `L-605-HZ`, `ZH-509-1`,
   `R-197-G3` family). Runtime should drop from ~120s to well under 60s.
2. Pi pipeline (`pi/pipeline.py` + RapidOCR) on `demo.mp4`: runtime should
   drop from 1171s to ~2 min or less (~600 OCR calls → ~30-60). Compare its
   table against the same reference; expect the fragment rows (`94-JV`,
   `4-LX`, `-RS`, `66`, `197-G9`) to mostly disappear, because each physical
   plate now gets a best-of-N-reads selection within one track.
3. Report actual before/after numbers in `PROJECT_CONTEXT.md` (same rigor as
   every previous phase).

**Expected outcome:** OCR calls drop ~15x; engine speed becomes a non-issue;
track-identity (same box = same plate) also fixes the OCR-flicker dedup
fragmentation more fundamentally than text similarity ever could — this IS
the "box-tracking before Postgres" item already on the roadmap.

---

## Step 2 — Re-validate RapidOCR accuracy after tracking

Only after Step 1's gate: check whether the pi pipeline's misread problem is
gone. Tracking should fix fragmentation (fragments came from many separate
noisy per-frame reads). If the per-track best reads are still wrong where
PaddleOCR's were right, proceed to Step 3; otherwise **skip Step 3 entirely**.

> **DONE (2026-07-16).** Tested on three uploaded 4K real-world clips (a US
> dealership lot, a UK street, and traffic CCTV). Tracking held, but RapidOCR's
> default **Chinese** models produced real junk on this harder footage: numeric
> fragments (`619879`), state-name misreads (`IDAHO`→`HDAHO`), and a Chinese
> province glyph (`皖EKH9211`). Two responses:
> - **Step 2.5 (below), done now:** a plate-validation layer that rejects that
>   junk at the output. Reliable, offline, ships immediately.
> - **Step 3 (English models):** now clearly warranted, and confirmed feasible
>   (GitHub + ModelScope reachable from here). But best tuned/validated on real
>   *warehouse-gate* footage, not stock clips of dealerships/streets that don't
>   resemble the deployment — so it's the next focused task, ideally alongside
>   first real Pi footage.

## Step 2.5 — Plate-validation layer (DONE 2026-07-16)

New `license_plate_pipeline/validation.py`: `is_valid_plate` (5-8 alphanumeric
chars, ignoring separators, **must contain a letter AND a digit**),
`canonical_plate`/`display_plate` (strip non-ASCII, so `皖EKH9211`→`EKH9211` is
recovered rather than dropped). Wired in three places:
- `ocr.py` + `pi/ocr.py` `select_plate_text`: return the best *valid* plate, or
  `None` — no more "fall back to highest-confidence anything" (that was the junk
  source).
- `video.py`: final events must pass `is_valid_plate` AND clear
  `MIN_PLATE_CONFIDENCE` (0.75) — kills low-confidence shape-valid misreads.

Kills numeric fragments, state-name reads, Chinese glyphs, and low-conf junk;
keeps every real plate in the test footage (all have a letter+digit, ≥0.87
conf). Tests in `tests/test_validation.py` built from the actual failures.

## Step 3 — English OCR models (warranted; validate on real Pi footage)

The paddle2onnx conversion on Windows is dead-ended (DLL/ABI error,
documented in PROJECT_CONTEXT.md). Do **not** resume debugging it.

> **DONE / DECIDED (2026-07-17): keep the Chinese model — English is worse.**
> Downloaded the English PP-OCRv4 and PP-OCRv5 `rec` models + dicts from
> ModelScope (RapidAI/RapidOCR) and wired them into RapidOCR via
> `rec_model_path`/`rec_keys_path`. Then benchmarked head-to-head on real plate
> crops (demo2 + four casey-video plates). **The bundled Chinese PP-OCRv4 rec
> model won decisively — 4/4 plates correct vs. English-v4 1/4, English-v5 2/4**
> (the English "mobile" models are smaller/weaker and truncate/misread:
> `CG4457T`→`C644571`, `BFE3975`→`BEE3975`). So the intuitive "swap to English"
> actually *regresses* accuracy.
>
> **Resolution:** keep RapidOCR's default (Chinese PP-OCRv4) recognition model —
> it's simply the better OCR. Its only downside, occasional Chinese glyphs, is
> already fully removed by the Step 2.5 validation layer, which strips non-ASCII
> before any plate is emitted (`皖EKH9211`→`EKH9211`). Net: best available Latin
> accuracy AND a guaranteed Chinese-free output. English model files were
> downloaded, tested, and deleted (not committed). If a *stronger* English/Latin
> plate model appears later (or real warehouse footage shows the Chinese model
> failing specifically on US plates), revisit — but not with the mobile models.
>
> The two fallback options below were never needed; kept for the record.
>
> - **WSL2 or a Linux docker container** for a one-shot paddle2onnx run (the
>   Windows DLL failure is Windows-specific).
> - Hand-rolled conversion — last resort only.

## Step 4 — Pi deployment (existing handoff, now actually viable)

- `PI_SETUP.md` flow as written (user runs it themselves on the Pi).
- Add to `requirements-pi.txt`/setup only if Step 3 happened (model files).
- Collect real on-device numbers: FPS for detection-only, per-OCR-call time,
  end-to-end. **Lever to remember:** a dock-door camera does not need 30fps —
  5-10 fps is plenty for a truck pulling in, which multiplies all headroom.
- Then the remaining Phase 9 items (systemd service, watchdog) apply.

## Step 5 — Later: truck/trailer detection (enterprise expansion)

Goal: know *which vehicle* a plate belongs to (tractor vs. trailer — see the
TN two-plate legal section in PROJECT_CONTEXT.md) and log vehicle-level
events.

- **Phase A (no training needed):** run a COCO-pretrained YOLO (`truck`/`car`
  classes, ONNX-exported the same way as `best.onnx`) alongside plate
  detection. Associate each plate box to the vehicle box that contains it
  (containment / max overlap). Two models per frame — fine at 5-10 fps,
  and vehicle detection can run on a sparser frame interval than plates.
  The tracking layer from Step 1 extends naturally: track vehicles, attach
  plate tracks to vehicle tracks.
- **Phase B (when real site footage exists):** retrain ONE multi-class model
  (plate + tractor + trailer) on Colab, same workflow as the original
  training. One inference pass, site-tuned classes. Phase A's association
  logic carries over unchanged.
- Recommendation: A first (zero training cost, proves the association logic),
  B once there's labeled real-warehouse footage worth training on.

---

## Sequencing summary

| Step | What | Size | Blocks Pi? |
|---|---|---|---|
| 0 | `use_cls=False`, drop upscale | tiny | no — do immediately |
| 1 | tracking + selective OCR | the main build | **yes — this is the fix** |
| 2 | re-validate accuracy | small (run + compare) | gate for 3 |
| 3 | English models via model zoo | small-medium | only if 2 fails |
| 4 | real Pi run (user-executed) | small (already written) | — |
| 5 | truck/trailer | later, after Pi works | no |

## Explicitly do NOT

- Do not resume the Windows `paddle2onnx` DLL debugging (superseded by the
  model-zoo option, and likely unnecessary at all after tracking).
- Do not switch OCR engines again — measured proof says call count, not
  engine, is the problem.
- Do not chase the 22 ms rec-only path yet — it needs tight single-line
  crops (US plates carry state-name/sticker text that currently needs the
  internal det stage to separate). Legitimate future optimization, but only
  after tracking lands and only if the Pi still needs more speed.
