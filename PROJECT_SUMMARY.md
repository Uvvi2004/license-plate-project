# Warehouse Truck License Plate Tracking System

---

## Part 1: For Leadership — Full Detail

### 1. The Goal

Track every truck that arrives at, docks at, and departs from the warehouse — automatically, using a camera and AI, with no staff needing to watch a gate or type anything by hand. For each truck, the system captures three things: the **license plate number**, the **timestamp** of the event, and (eventually) the **status** (arrived / at dock / departed).

**Hard requirement from day one:** all data stays on our own hardware. Nothing is sent to an outside cloud service at any point. This is a business and privacy decision, not a technical limitation — it means we retain full control of the data, there's no third party with access to our site's traffic patterns, and there are no ongoing cloud fees.

### 2. How the System Works, Step by Step

1. **A camera captures a frame** (`OpenCV`/`cv2.VideoCapture`) — a video file or a live webcam feed today; a dedicated USB/CSI camera on the deployed hardware eventually.
2. **YOLO26n (nano)** — a lightweight object-detection model from the Ultralytics YOLO family — scans the frame and outputs a bounding box for anything it identifies as a license plate. It was fine-tuned (transfer learning from the pretrained `yolo26n.pt` checkpoint) specifically for this one class (`License_Plate`), which is what keeps it fast and small enough for eventual edge deployment.
3. **The boxed region is cropped** and run through a preprocessing pipeline before OCR: the box is padded 10% per side (`pad_box`, in case the detection clipped an edge), the crop is upscaled 3x with `cv2.resize` (cubic interpolation), then converted to grayscale and contrast-enhanced with `cv2.equalizeHist`. Plates are small relative to a full frame, so this step is what makes OCR actually reliable.
4. **PaddleOCR** (Baidu's PP-OCR, specifically the PP-OCRv3 detection + PP-OCRv4 recognition models, `lang="en"`) reads the enhanced crop and returns each detected text fragment with a confidence score.
5. **A regex-based filter** (`select_plate_text`) picks the fragment matching a plate-like pattern (5–8 alphanumeric characters) over surrounding noise — plate crops often also contain registration-sticker or dealer-frame text (e.g. "STALCHRYSLER.COM") that OCR happily reads too.
6. **A two-pass deduplication algorithm** (detailed in Section 5) tracks sightings across frames and merges repeated readings of the same physical truck into a single event.
7. **(Upcoming)** Each finalized event — plate number, first-seen timestamp, last-seen timestamp, confidence — gets written to a PostgreSQL database and surfaced on a live dashboard (FastAPI or Spring Boot backend, still to be decided).

### 3. The AI Models Themselves

**Detection model — YOLO26n (Ultralytics), fine-tuned:**
- Base checkpoint: `yolo26n.pt`, the nano variant (~2.5M parameters, ~5.4MB) — chosen specifically for CPU/edge-device inference speed over larger YOLO variants.
- Training data: [Roboflow Universe](https://universe.roboflow.com) dataset `license-plate-recognition-rxg4e` (workspace `roboflow-universe-projects`), version 4 — 21,173 training images, 2,046 validation images, single class (`License_Plate`).
- Training run: 25 epochs, `imgsz=640`, `batch=16`, `patience=5` (early stopping), on a Google Colab T4 GPU (~3.16 hours).
- **Validated results:** mAP50 = **0.982**, mAP50-95 = **0.694**, Precision = **0.984**, Recall = **0.953** — confirmed against the training run's own validation split.
- Known caveat carried forward honestly: the upstream dataset skews toward passenger cars and has some documented train/test near-duplicate leakage, so real-world truck-specific accuracy may run somewhat lower than these validation numbers until tested against our own footage.

**Reading model — PaddleOCR (PP-OCRv3/v4):**
- Chosen after a head-to-head benchmark against EasyOCR on a real US plate: PaddleOCR returned an **exact character match** (`6FVZ747`) at **0.97 confidence**, where EasyOCR fragmented and misread the same plate.
- Also chosen with edge deployment in mind: PaddleOCR's mobile-optimized models are roughly **10–15MB** combined, versus EasyOCR's 100MB+ detector+recognizer stack — a meaningful difference for a resource-constrained embedded device.

### Technical Stack (Reference)

| Layer | Technology |
|---|---|
| Detection | YOLO26n, `ultralytics==8.4.95`, PyTorch (`torch==2.13.0`, CPU) |
| OCR | PaddleOCR `2.9.1`, PaddlePaddle `2.6.2` (pinned — see engineering notes) |
| Image processing | OpenCV (`opencv-python`, `opencv-contrib-python`) |
| Dedup / fuzzy matching | `rapidfuzz` (Levenshtein-based similarity scoring) |
| Data handling | `pandas`, `tqdm` |
| Language / environment | Python 3.10, isolated virtual environment |
| Testing | `pytest`, regression suite built from real calibration cases |
| Development interface | Jupyter notebook (interactive, step-by-step) |
| Source control | Git, hosted on GitHub (private history, incremental commits) |
| Target deployment hardware | Raspberry Pi 5 (8GB) + Camera Module 3 |

### 4. Engineering Rigor Applied

This isn't just a prototype script — real software engineering practices have been applied throughout:

- **Modular package structure:** the logic lives in a proper Python package (`license_plate_pipeline/`), split into `config.py` (tunable constants), `detection.py`, `ocr.py`, `dedup.py`, and `pipeline.py` (orchestration) — not one monolithic script. The interactive notebook imports from this package rather than duplicating logic.
- **Automated test suite (`pytest`):** dedicated tests cover `pad_box`, `select_plate_text`, and both dedup-clustering passes, built directly as regression tests from real cases found during testing (see Section 5) — so future changes can't silently break something that already works.
- **Error handling:** detection and OCR calls are wrapped in try/except with logging (Python's `logging` module) at the per-frame and per-crop level, so a single bad frame or a momentary read failure is logged and skipped rather than crashing the run.
- **Dependency hygiene:** runtime dependencies (`requirements.txt`) are kept separate from development-only tools (`requirements-dev.txt`: Jupyter, pytest) — the eventual Pi deployment installs only what it actually needs to run.
- **Version control:** the full codebase and its commit history are tracked in Git and hosted on GitHub, with incremental, per-feature commits rather than one undifferentiated blob.

### 5. Avoiding Duplicate Records — Why It Mattered

A truck doesn't appear for a fraction of a second — it may sit in the camera's view for many seconds at a dock. Naively, that produces dozens of near-identical readings per truck instead of one clean record. Two layers of logic solve this, both calibrated against real measured data rather than guessed:

- **Pass 1 — time + text similarity:** sightings of the *same* plate text within a 1.5-second gap are grouped into one sub-event. Different-text sightings are then compared using `rapidfuzz`'s Levenshtein-ratio similarity score — a threshold of 70/100 was calibrated against real data, where genuinely same-plate OCR misreads scored 87–94 and genuinely different plates scored 25–50.
- **Pass 2 — cluster re-merge:** a second pass re-compares already-finished clusters against each other (catching cases the first pass's processing order missed), with a wider allowed time gap (3 seconds, vs. 1.5) specifically when text similarity is very high (≥85/100) — reasoning that near-exact matches separated by a longer gap likely indicate a brief detection dropout, not a new vehicle.

**Explicitly tested for over-merging:** a truck's tractor and trailer can legally carry two different, separate license plates. A test case with two genuinely different, time-overlapping plates was checked and confirmed to score only 37.5 on the similarity metric — well below threshold, correctly staying as two separate records. This wasn't assumed; it was directly verified against real test data.

### 6. Compliance Consideration

Tennessee law affects camera placement: Tennessee is a one-plate state where standard vehicles and trailers carry a rear plate, but truck tractors (the cab/engine unit) carry a **front** plate. A semi-truck (tractor + trailer) may therefore have two plates in two different locations belonging to two separately registered units. This has been factored into camera placement planning — depending on whether the goal is tracking the tractor, the trailer, or both, one or two cameras may be needed.

### 7. Current Status vs. Remaining Work

**Done and validated:**
- Detection and reading AI models, both tested against real data with strong, confirmed results.
- Full video-processing pipeline with intelligent duplicate-merging, tested against real footage.
- Live camera feed tested successfully on a standard laptop webcam.
- Full automated test coverage and error handling for the core logic.

**Remaining:**
1. **Edge deployment (Raspberry Pi 5, ARM64):** PaddlePaddle (the OCR runtime) has no official ARM64 pip wheel, and Ultralytics/PyTorch benefit from a lighter runtime on ARM too — so both models need exporting to a Pi-friendly format (ONNX via `onnxruntime`, or Baidu's Paddle-Lite for OCR specifically, and NCNN or ONNX for the detector) before deployment. This is scoped and understood, not yet executed — pending physical access to the hardware.
2. **Permanent record storage** — a PostgreSQL database so every event is saved and searchable indefinitely; schema (plate, first-seen, last-seen, confidence, site/camera ID) is already designed around the event-table shape the pipeline produces today.
3. **Live dashboard** — a lightweight web frontend (FastAPI or Spring Boot backend, undecided) showing current trucks on-site and dwell time.
4. **Production polish** — running as a `systemd`-supervised, auto-restarting background service on the Pi, plus additional reliability work for continuous 24/7 operation (e.g. storage-wear mitigation for the SD card/boot media).

### Bottom Line

The hardest and riskiest part of this project — proving the AI can reliably find and read license plates, and can tell repeated sightings of one truck apart from two different trucks, without any human assistance — is done, tested, and verified against real data. What remains is largely deployment and connecting to permanent storage, not new invention.

---

## Part 2: Explain It Like I'm 5 — Full Detail

Imagine you have two robot friends whose whole job is to watch trucks go by and write down what they see.

### Robot #1: The Spotter

Robot #1's only job is to play "I Spy" — but instead of spying colors or shapes, it spies license plates. To learn this, we showed it about **21,000 pictures** of license plates — way more than you could count — until it got really, really good at pointing to exactly where a plate is in any picture, almost like it memorized what plates "look like" in general, not just the ones it practiced on. Now, when we show it a brand new picture it's never seen, it can point to the plate correctly about **98 times out of 100**. It doesn't read anything yet — it just draws an invisible box around the plate and says "it's right there!"

### Making the Picture Easier to Read

Once Robot #1 finds the box, the little picture inside that box is usually pretty small and blurry — like trying to read a street sign from far away. So before we let anyone try to read it, we:
- **Zoom in** and make the picture 3 times bigger.
- **Turn up the contrast**, like adjusting brightness on a TV so the letters stand out better against the background.
- **Give it a little extra wiggle room** around the edges, just in case Robot #1's box was drawn a tiny bit too tight and accidentally cut off part of a letter.

### Robot #2: The Reader

Now we hand that nice, big, clear picture to Robot #2, whose superpower is reading. It looks at the letters and numbers and tells us exactly what they say. We tested it on a real license plate and it read every single letter and number correctly. Sometimes the little picture also has other writing on it, like a sticker or a dealership name — Robot #2 is smart enough to know that the short, blocky text (like "6FVZ747") is the real plate, and the longer decorative writing nearby isn't.

### The Tricky Part: Videos Aren't Just One Picture

A video isn't one picture — it's *hundreds* of pictures shown super fast, one after another. And a truck doesn't just appear for one picture — it might stay in view for **ten seconds**, which is like 300 pictures in a row! If Robot #1 and Robot #2 looked at every single one of those pictures separately, they'd write down the same truck's plate 300 times. That would be like introducing yourself over and over every few seconds even though you're talking to the same person the whole time — silly and unhelpful.

So we taught them two smart rules:

**Rule 1 — "I already know you."** If the robots just wrote down a plate a moment ago, and they see that *exact same* plate again right away, they know it's still the same truck sitting there — they don't write it down again.

**Rule 2 — "That's probably still you, just a little blurry."** Sometimes Robot #2 makes a tiny mistake, like misreading a zero as the letter "O." So the robots learned a second, smarter rule: if two readings happen at *almost* the same moment and look *almost* the same, they're probably the same truck with one tiny misread — so those get combined into one, too.

But we were **very careful** here, because sometimes a big truck really does have two *different, real* license plates — one on the front cab part and one on the back trailer part! So the robots only combine readings when they're both close in time **and** look really similar — they won't accidentally squish two genuinely different real plates into one just because they showed up around the same time.

### Proving It Actually Works

We didn't just build this and hope — we tested it. We ran a real practice video through both robots, looked very closely at everything they wrote down, found a few spots where they'd made small mistakes, and fixed those mistakes one at a time until the list looked right. Then we tested it again with a **real, live camera** (a laptop's webcam) instead of just a pre-recorded video, to make sure the robots work just as well when they're watching something happening *right now*, not just a video from before.

### What's Left

Right now, both robots are working on a regular computer, just to prove they're good at their jobs. The next steps are:

- Give the robots their **own** tiny computer and camera, small enough to live right at the truck gate — like giving them their own little house to work in instead of borrowing ours.
- Give them a magic notebook that **never runs out of pages**, so every truck they ever see gets written down forever, not just remembered for a little while and then forgotten.
- Build a simple window (a screen) so people can peek in and see "which trucks are here right now?" without having to ask the robots themselves.

### Why This Is a Big Deal

You built two AI "brains" that each do one job really well, taught them to work together, taught them not to repeat themselves, and — even trickier — taught them *when not to* combine things that only look similar but are actually different. That's genuinely the hardest part of a project like this, and it's done, tested, and proven to work on real data.
