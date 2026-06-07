<img width="1374" height="709" alt="image" src="https://github.com/user-attachments/assets/edef5127-84a8-4229-b29d-188b93983f66" />
<img width="1438" height="804" alt="image" src="https://github.com/user-attachments/assets/ce4da768-b945-4ee6-b339-1b855e8d21a8" />

# computer-vision-toolkit

Production-grade computer vision tools for CCTV, surveillance, and real-time inference.
Built on YOLOv8, ByteTrack, EasyOCR, and MSS — no cloud dependency, runs on-premise.

```
computer-vision-toolkit/
├── rtsp/
│   └── rtsp_inference.py      # Real-time detection on RTSP streams
├── tracking/
│   └── zone_counter.py        # People counting in polygonal zones
├── lpr/
│   └── lpr_pipeline.py        # License plate recognition (2-stage)
├── batch/
│   └── batch_processor.py     # Archive video analysis
└── utils/
    ├── auto_label_hsv.py      # HSV-based dataset auto-labeling
    ├── pascal_voc_to_yolo.py  # Pascal VOC XML → YOLO format converter
    ├── mouse_aim.py           # Smooth mouse movement with jitter
    └── screen_capture.py      # High-FPS screen capture (MSS + threading)
```

---

## Core Tools

### 1. RTSP Inference

Real-time object detection on IP camera streams with automatic reconnect.

```bash
python rtsp/rtsp_inference.py --source rtsp://192.168.1.100:554/stream
python rtsp/rtsp_inference.py --source rtsp://admin:pass@192.168.1.100/cam1 --classes 0 --fps-limit 15 --save
python rtsp/rtsp_inference.py --source rtsp://... --no-show --save  # headless server mode
```

Features: auto-reconnect on stream drop · EMA FPS counter · optional output recording · class filtering

---

### 2. Zone Counter

Count objects entering and exiting polygonal zones. Tracks unique IDs — counts events, not instantaneous presence.

```bash
python tracking/zone_counter.py --source rtsp://... --zones zones.json
python tracking/zone_counter.py --source rtsp://... --draw-zones  # interactive polygon drawing
python tracking/zone_counter.py --source video.mp4 --persons-only
```

**Zone config (`zones.json`):**
```json
[
  {
    "name": "entrance",
    "polygon": [[120, 300], [450, 300], [450, 580], [120, 580]],
    "color": [0, 255, 150]
  }
]
```

Output: `events_TIMESTAMP.json` with per-track timestamps and entry/exit events.
Tracker: ByteTrack (built into Ultralytics) — handles occlusion and re-identification.

---

### 3. LPR Pipeline

Two-stage license plate recognition:
1. YOLOv8 → detect vehicle + tracking
2. YOLOv8 (plate weights) → detect plate in vehicle crop
3. EasyOCR → read text with Otsu binarization preprocessing

```bash
python lpr/lpr_pipeline.py --source rtsp://...
python lpr/lpr_pipeline.py --source rtsp://... --lang en es --region mx --save-crops
python lpr/lpr_pipeline.py --source video.mp4 --lang en ru --region ru
```

| Region | Pattern | Example |
|---|---|---|
| `mx` | 3 letters + 3 digits | `ABC-123` |
| `ru` | Cyrillic + digits | `А123ВС77` |
| `us` | 5–8 alphanumeric | `ABC1234` |

Results: `lpr_results_TIMESTAMP.json`

---

### 4. Batch Processor

Scan hours of archive footage — extract only frames with detections.

```bash
python batch/batch_processor.py --input /recordings --output /results --strategy interval --interval 5
python batch/batch_processor.py --input /recordings --find person car --strategy motion
python batch/batch_processor.py --input /recordings --workers 4 --recursive
```

| Strategy | When to use |
|---|---|
| `interval` | Regular sampling, predictable coverage |
| `motion` | Skip static scenes |
| `keyframes` | Maximum speed, I-frames only |

Output: annotated JPEG keyframes + `report_TIMESTAMP.json` with wall-clock timestamps per event.

---

## Utils

### `auto_label_hsv.py` — Dataset Auto-Labeling

Automatic YOLO dataset preparation via HSV color filtering. Full pipeline: video → frames → labels → verify → split.

```bash
# Step 1: calibrate HSV range interactively (drag sliders)
python utils/auto_label_hsv.py calibrate --image samples/frame_001.jpg

# Step 2: label all images
python utils/auto_label_hsv.py label \
    --input dataset/raw \
    --output dataset/train \
    --hsv 64,54,120,180,255,255 \
    --class-id 0

# Step 3: visual verification (SPACE=next, DEL=delete label)
python utils/auto_label_hsv.py verify \
    --images dataset/train/images \
    --labels dataset/train/labels

# All-in-one: video → frames → labels
python utils/auto_label_hsv.py from-video \
    --video recording.mp4 \
    --output dataset/train \
    --hsv 64,54,120,180,255,255

# Split into train/val/test (70/20/10)
python utils/auto_label_hsv.py split --dataset dataset/
```

Works well for objects with distinctive colors (safety vests, specific vehicles, product packaging).

---

### `pascal_voc_to_yolo.py` — Annotation Converter

Convert Pascal VOC XML labels to YOLO format. Handles coordinate rescaling when changing image resolution.

```bash
# Discover class names in your XML files
python utils/pascal_voc_to_yolo.py --discover --input dataset/voc_labels

# Convert directory
python utils/pascal_voc_to_yolo.py \
    --input dataset/voc_labels \
    --output dataset/yolo_labels \
    --classes CT T CT_head T_head

# Rescale bounding boxes to new resolution (e.g. 1366x768 → 640x640)
python utils/pascal_voc_to_yolo.py \
    --input dataset/voc_labels \
    --output dataset/yolo_labels \
    --classes CT T CT_head T_head \
    --resize 640 640

# Auto-generate data.yaml and classes.txt
python utils/pascal_voc_to_yolo.py \
    --input dataset/voc_labels \
    --output dataset/yolo_labels \
    --classes CT T CT_head T_head \
    --generate-yaml dataset/
```

Pascal VOC stores coordinates in pixels — rescaling without this converter silently corrupts all labels.

---

### `screen_capture.py` — High-FPS Screen Capture

MSS-based screen capture with threading. Separates capture and inference into two threads for higher throughput.

```
Single-threaded MSS:          ~20 FPS
Threaded (capture + process): ~30–40 FPS
With YOLOv8 on GPU:           20–25 FPS (GPU-bound)
```

```python
from utils.screen_capture import ScreenCapture, CaptureRegion, CaptureInferencePipeline
from ultralytics import YOLO

# Simple iterator
region = CaptureRegion(left=0, top=0, width=1280, height=720)
cap = ScreenCapture(region=region)
for frame in cap.frames():
    results = model(frame)

# Threaded — producer thread keeps capturing while you process
cap = ScreenCapture(region=region, threaded=True)
cap.start()
while True:
    frame = cap.get_frame()
    if frame is not None:
        process(frame)

# Full two-thread pipeline
def my_processor(frame):
    return model(frame, verbose=False)[0].plot()

pipeline = CaptureInferencePipeline(processor=my_processor, region=region)
pipeline.run(show=True)
```

```bash
# Demo: display capture FPS
python utils/screen_capture.py --threaded

# Demo with YOLO inference
python utils/screen_capture.py --yolo yolov8n.pt --resize 640 640
```

---

### `mouse_aim.py` — Smooth Mouse Movement

Programmatic mouse control with jitter — mimics human hand motion. Useful for RPA automation and CV-driven cursor control.

```python
from utils.mouse_aim import MouseController, AimConfig, config_human_fast

ctrl = MouseController(config=config_human_fast())

ctrl.move_to(960, 540)           # smooth move to screen coords
ctrl.move_by(dx=100, dy=-50)     # smooth relative move
ctrl.snap_to(960, 540)           # instant teleport

# From YOLO bounding box
box = (x1, y1, x2, y2)
ctrl.aim_at_box(box, target='top_third')  # upper third = head zone
```

**Preset configs:**

| Config | smoothing | jitter | Use case |
|---|---|---|---|
| `config_human_slow()` | 0.20 | 2.0 | UI automation |
| `config_human_fast()` | 0.08 | 1.5 | Real-time CV |
| `config_precise()` | 0.05 | 0.5 | Precise targeting |

**Backends:** `win32api` (Windows, recommended) · `pynput` (cross-platform) · `pyautogui`

```bash
python utils/mouse_aim.py --mode demo    # move to corners
python utils/mouse_aim.py --mode circle  # draw a circle
python utils/mouse_aim.py --mode follow  # jitter demo
```

---

## Requirements

```bash
pip install ultralytics opencv-python easyocr mss
pip install pywin32       # Windows — for mouse_aim win32 backend
pip install pynput        # cross-platform mouse backend alternative
```

GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Python 3.10+ · CUDA optional (CPU works, slower)

---

## Performance Notes

**RTSP latency:** `CAP_PROP_BUFFERSIZE=1` minimizes buffering. For sub-500ms, use GStreamer backend.

**Screen capture FPS:** Threading separates I/O (MSS grab) from compute (YOLO). On a single GPU, the bottleneck shifts from capture to inference — threading adds ~10 FPS headroom.

**ByteTrack vs DeepSORT:** ByteTrack used by default — no re-ID model required, faster. For cross-camera tracking, DeepSORT with appearance features is needed.

**Pascal VOC → YOLO rescaling:** When training resolution differs from annotation resolution, coordinates must be rescaled proportionally. The converter handles this automatically via `--resize`.

**HSV labeling limitations:** Works best for objects with distinctive, consistent colors under stable lighting. For complex scenes or multiple classes, use manual labeling or Roboflow.

---

*Part of [DenisZDV](https://github.com/DenisZDV) automation toolkit*
