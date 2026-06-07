"""
lpr_pipeline.py
Two-stage License Plate Recognition pipeline.

Stage 1: YOLOv8 — detect vehicle bounding box
Stage 2: YOLOv8 (plate model) — detect plate within vehicle crop
Stage 3: EasyOCR — read plate text

Supports: Latin plates (default), Cyrillic (ru), Spanish/Latin America (es)

Usage:
    python lpr_pipeline.py --source rtsp://...
    python lpr_pipeline.py --source video.mp4 --lang en es --save-crops
    python lpr_pipeline.py --source 0 --show-conf
"""

import cv2
import re
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from ultralytics import YOLO

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning('easyocr not installed. pip install easyocr')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ── DATA STRUCTURES ────────────────────────────────────────────────────────

@dataclass
class PlateReading:
    timestamp:   str
    frame_id:    int
    track_id:    int | None
    plate_text:  str
    confidence:  float
    vehicle_box: list[int]   # [x1,y1,x2,y2]
    plate_box:   list[int]   # [x1,y1,x2,y2] relative to vehicle crop
    crop_path:   str | None = None


# ── PLATE TEXT CLEANING ────────────────────────────────────────────────────

# Common OCR confusion pairs
OCR_CORRECTIONS = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2',
}

# Mexican/LATAM plate format: 3 letters + 3 digits  (ABC-123)
# Russian: 3+3 with region code
PLATE_PATTERNS = {
    'mx': re.compile(r'^[A-Z]{3}[-\s]?\d{3}$'),
    'ru': re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$'),
    'us': re.compile(r'^[A-Z0-9]{5,8}$'),
    'generic': re.compile(r'^[A-Z0-9]{4,9}$'),
}


def clean_plate_text(raw: str, region: str = 'generic') -> tuple[str, bool]:
    """
    Normalize OCR output to plate format.
    Returns (cleaned_text, is_valid).
    """
    text = raw.upper().strip()
    text = re.sub(r'[^A-Z0-9\-]', '', text)

    # Apply common corrections
    corrected = ''
    for ch in text:
        corrected += OCR_CORRECTIONS.get(ch, ch)

    pattern = PLATE_PATTERNS.get(region, PLATE_PATTERNS['generic'])
    is_valid = bool(pattern.match(corrected))

    return corrected, is_valid


# ── PIPELINE ───────────────────────────────────────────────────────────────

class LPRPipeline:
    """
    Two-stage LPR:
      1. Detect vehicles with YOLOv8 (tracking optional)
      2. For each vehicle: detect plate with second YOLO model
      3. Run EasyOCR on plate crop
    """

    VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

    def __init__(
        self,
        vehicle_model: str = 'yolov8n.pt',
        plate_model:   str = 'yolov8n.pt',  # replace with plate-specific weights
        ocr_langs:     list[str] = None,
        conf_vehicle:  float = 0.45,
        conf_plate:    float = 0.40,
        conf_ocr:      float = 0.35,
        region:        str = 'generic',
        use_tracking:  bool = True,
        save_crops:    bool = False,
        crops_dir:     str = 'plate_crops',
    ):
        self.conf_vehicle = conf_vehicle
        self.conf_plate   = conf_plate
        self.conf_ocr     = conf_ocr
        self.region       = region
        self.use_tracking = use_tracking
        self.save_crops   = save_crops
        self.crops_dir    = Path(crops_dir)

        log.info(f'Loading vehicle model: {vehicle_model}')
        self.vehicle_model = YOLO(vehicle_model)

        log.info(f'Loading plate model: {plate_model}')
        self.plate_model = YOLO(plate_model)

        if EASYOCR_AVAILABLE:
            langs = ocr_langs or ['en']
            log.info(f'Initializing EasyOCR: {langs}')
            self.reader = easyocr.Reader(langs, gpu=True)
        else:
            self.reader = None

        if save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.readings: list[PlateReading] = []
        self.frame_id = 0

        # Deduplicate: avoid logging same plate per track_id repeatedly
        self._seen: dict[int, str] = {}  # track_id → last plate text

    def _detect_plate(self, vehicle_crop: cv2.Mat) -> tuple[cv2.Mat | None, list[int]]:
        """Run plate detector on vehicle crop. Returns (plate_crop, box)."""
        results = self.plate_model.predict(
            vehicle_crop, conf=self.conf_plate, verbose=False
        )
        if not results[0].boxes:
            return None, []

        # Take highest-confidence detection
        best = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best.xyxy[0])

        # Add padding
        pad = 4
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(vehicle_crop.shape[1], x2 + pad)
        y2 = min(vehicle_crop.shape[0], y2 + pad)

        crop = vehicle_crop[y1:y2, x1:x2]
        return crop, [x1, y1, x2, y2]

    def _ocr_plate(self, plate_crop: cv2.Mat) -> tuple[str, float]:
        """Run OCR on plate crop. Returns (text, confidence)."""
        if self.reader is None:
            return '', 0.0

        # Preprocess: upscale + grayscale for better OCR
        h, w = plate_crop.shape[:2]
        if h < 32:
            scale = 32 / h
            plate_crop = cv2.resize(plate_crop, (int(w * scale), 32),
                                    interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = self.reader.readtext(thresh, detail=1, paragraph=False)
        if not results:
            return '', 0.0

        # Concatenate all text regions
        texts = [(r[1], r[2]) for r in results if r[2] >= self.conf_ocr]
        if not texts:
            return '', 0.0

        combined = ''.join(t for t, _ in texts)
        avg_conf = sum(c for _, c in texts) / len(texts)
        return combined, avg_conf

    def process_frame(self, frame: cv2.Mat) -> tuple[cv2.Mat, list[PlateReading]]:
        """Process one frame. Returns annotated frame and new readings."""
        frame_readings = []
        self.frame_id += 1

        if self.use_tracking:
            results = self.vehicle_model.track(
                frame,
                conf=self.conf_vehicle,
                classes=self.VEHICLE_CLASSES,
                tracker='bytetrack.yaml',
                persist=True,
                verbose=False,
            )
        else:
            results = self.vehicle_model.predict(
                frame,
                conf=self.conf_vehicle,
                classes=self.VEHICLE_CLASSES,
                verbose=False,
            )

        if not results[0].boxes:
            return frame, []

        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = (results[0].boxes.id.int().cpu().tolist()
                     if results[0].boxes.id is not None else [None] * len(boxes))

        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            plate_crop, plate_box = self._detect_plate(vehicle_crop)
            if plate_crop is None or plate_crop.size == 0:
                continue

            raw_text, ocr_conf = self._ocr_plate(plate_crop)
            if not raw_text:
                continue

            text, valid = clean_plate_text(raw_text, self.region)
            if not text:
                continue

            # Deduplicate per track
            if tid is not None and self._seen.get(tid) == text:
                # Still draw, just don't log again
                pass
            else:
                reading = PlateReading(
                    timestamp=datetime.now().isoformat(timespec='milliseconds'),
                    frame_id=self.frame_id,
                    track_id=tid,
                    plate_text=text,
                    confidence=round(ocr_conf, 3),
                    vehicle_box=[x1, y1, x2, y2],
                    plate_box=plate_box,
                )

                if self.save_crops:
                    crop_name = f'{self.frame_id:06d}_{text}.jpg'
                    crop_path = self.crops_dir / crop_name
                    cv2.imwrite(str(crop_path), plate_crop)
                    reading.crop_path = str(crop_path)

                self.readings.append(reading)
                frame_readings.append(reading)
                if tid is not None:
                    self._seen[tid] = text

                log.info(f'Plate: {text}  conf={ocr_conf:.2f}  track={tid}')

            # Draw plate annotation
            color = (0, 255, 80) if valid else (0, 180, 255)
            px1, py1 = x1 + plate_box[0], y1 + plate_box[1]
            px2, py2 = x1 + plate_box[2], y1 + plate_box[3]
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(frame, text, (px1, py1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return frame, frame_readings

    def run(self, source: str, show: bool = True, save_path: str = None):
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open: {source}')

        writer = None
        if save_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, _ = self.process_frame(frame)

                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow('LPR', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    def save_results(self, path: str = 'lpr_results.json'):
        data = [asdict(r) for r in self.readings]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
        log.info(f'Saved {len(data)} readings → {path}')

    def unique_plates(self) -> list[str]:
        return list({r.plate_text for r in self.readings})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',        default='0')
    p.add_argument('--vehicle-model', default='yolov8n.pt')
    p.add_argument('--plate-model',   default='yolov8n.pt',
                   help='YOLOv8 weights trained on license plates')
    p.add_argument('--lang',          nargs='+', default=['en'],
                   help='OCR languages: en es ru (space-separated)')
    p.add_argument('--region',        default='generic',
                   choices=['mx', 'ru', 'us', 'generic'])
    p.add_argument('--conf',          type=float, default=0.45)
    p.add_argument('--save',          action='store_true')
    p.add_argument('--save-crops',    action='store_true')
    p.add_argument('--no-tracking',   action='store_true')
    args = p.parse_args()

    pipeline = LPRPipeline(
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        ocr_langs=args.lang,
        conf_vehicle=args.conf,
        region=args.region,
        use_tracking=not args.no_tracking,
        save_crops=args.save_crops,
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline.run(
        source=args.source,
        save_path=f'lpr_output_{ts}.mp4' if args.save else None,
    )

    print(f'\nUnique plates detected: {pipeline.unique_plates()}')
    pipeline.save_results(f'lpr_results_{ts}.json')


if __name__ == '__main__':
    main()
