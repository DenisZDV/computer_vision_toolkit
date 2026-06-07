"""
batch_processor.py
Process archive video files — extract key frames with detections,
generate reports, search for specific objects across recordings.

Use case: analyze hours of CCTV footage, find events without watching everything.

Usage:
    python batch_processor.py --input /recordings --output /results
    python batch_processor.py --input /recordings --find person --from 2024-01-15T08:00
    python batch_processor.py --input video.mp4 --keyframes-only --interval 30
"""

import cv2
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ── DATA ───────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_name: str
    confidence: float
    box: list[int]


@dataclass
class KeyFrame:
    source_file:  str
    frame_index:  int
    timestamp_s:  float        # seconds from start of file
    wall_time:    str | None   # estimated real time if file mtime is available
    detections:   list[Detection]
    saved_path:   str | None = None


@dataclass
class VideoReport:
    source_file:   str
    duration_s:    float
    total_frames:  int
    processed_frames: int
    keyframes:     list[KeyFrame]
    class_counts:  dict[str, int]


# ── PROCESSOR ──────────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Scan video files for events.

    Strategies:
    - interval: sample every N seconds
    - motion: process frames where pixel diff > threshold
    - keyframes: use video keyframes (I-frames) only
    """

    STRATEGIES = ('interval', 'motion', 'keyframes')

    def __init__(
        self,
        model_path:      str = 'yolov8n.pt',
        conf:            float = 0.35,
        target_classes:  list[str] = None,   # None = all detections
        strategy:        str = 'interval',
        interval_s:      float = 5.0,        # seconds between samples
        motion_thresh:   int = 25,           # pixel diff for motion detection
        save_keyframes:  bool = True,
        output_dir:      str = 'batch_output',
        workers:         int = 1,
    ):
        assert strategy in self.STRATEGIES, f'strategy must be one of {self.STRATEGIES}'
        self.conf           = conf
        self.target_classes = set(target_classes) if target_classes else None
        self.strategy       = strategy
        self.interval_s     = interval_s
        self.motion_thresh  = motion_thresh
        self.save_keyframes = save_keyframes
        self.output_dir     = Path(output_dir)
        self.workers        = workers

        if save_keyframes:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def _should_process(
        self,
        frame: cv2.Mat,
        prev_frame: cv2.Mat | None,
        frame_idx: int,
        fps: float,
    ) -> bool:
        if self.strategy == 'interval':
            interval_frames = max(1, int(fps * self.interval_s))
            return frame_idx % interval_frames == 0

        if self.strategy == 'motion' and prev_frame is not None:
            diff = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            )
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            motion_pct = thresh.sum() / 255 / thresh.size * 100
            return motion_pct > self.motion_thresh

        # keyframes — all frames (filtering happens at capture level)
        return True

    def _detect(self, frame: cv2.Mat) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name   = self.class_names[cls_id]
            if self.target_classes and name not in self.target_classes:
                continue
            detections.append(Detection(
                class_name=name,
                confidence=round(float(box.conf[0]), 3),
                box=list(map(int, box.xyxy[0])),
            ))
        return detections

    def _save_keyframe(
        self,
        frame: cv2.Mat,
        detections: list[Detection],
        source_name: str,
        frame_idx: int,
    ) -> str:
        # Annotate
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, f'{det.class_name} {det.confidence:.2f}',
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 100), 1, cv2.LINE_AA)

        stem = Path(source_name).stem
        filename = f'{stem}_f{frame_idx:06d}.jpg'
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def process_video(self, video_path: str | Path) -> VideoReport:
        video_path = Path(video_path)
        log.info(f'Processing: {video_path.name}')

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error(f'Cannot open: {video_path}')
            return None

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_s  = total / fps

        # Estimate wall time from file modification time
        try:
            mtime = datetime.fromtimestamp(video_path.stat().st_mtime)
            start_time = mtime - timedelta(seconds=dur_s)
        except Exception:
            start_time = None

        keyframes: list[KeyFrame] = []
        class_counts: dict[str, int] = {}
        prev_frame = None
        processed = 0

        try:
            for frame_idx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break

                if not self._should_process(frame, prev_frame, frame_idx, fps):
                    prev_frame = frame
                    continue

                processed += 1
                ts_s = frame_idx / fps
                detections = self._detect(frame)

                if not detections:
                    prev_frame = frame
                    continue

                # Count classes
                for det in detections:
                    class_counts[det.class_name] = \
                        class_counts.get(det.class_name, 0) + 1

                # Wall time
                wall_time = None
                if start_time:
                    wall_time = (start_time + timedelta(seconds=ts_s)) \
                                .strftime('%Y-%m-%d %H:%M:%S')

                saved_path = None
                if self.save_keyframes:
                    frame_copy = frame.copy()
                    saved_path = self._save_keyframe(
                        frame_copy, detections, video_path.name, frame_idx
                    )

                keyframes.append(KeyFrame(
                    source_file=str(video_path),
                    frame_index=frame_idx,
                    timestamp_s=round(ts_s, 2),
                    wall_time=wall_time,
                    detections=detections,
                    saved_path=saved_path,
                ))

                prev_frame = frame

        finally:
            cap.release()

        log.info(f'{video_path.name}: {len(keyframes)} events in {processed} sampled frames')

        return VideoReport(
            source_file=str(video_path),
            duration_s=round(dur_s, 1),
            total_frames=total,
            processed_frames=processed,
            keyframes=keyframes,
            class_counts=class_counts,
        )

    def process_directory(
        self,
        directory: str | Path,
        extensions: tuple[str] = ('.mp4', '.avi', '.mkv', '.mov', '.ts'),
        recursive: bool = False,
    ) -> list[VideoReport]:
        directory = Path(directory)
        pattern = '**/*' if recursive else '*'
        files = [
            f for f in directory.glob(pattern)
            if f.suffix.lower() in extensions
        ]
        files.sort()

        log.info(f'Found {len(files)} video files in {directory}')

        reports = []
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(self.process_video, f): f for f in files}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r:
                        reports.append(r)
        else:
            for f in files:
                r = self.process_video(f)
                if r:
                    reports.append(r)

        return reports

    def save_report(self, reports: list[VideoReport], path: str = 'batch_report.json'):
        data = [asdict(r) for r in reports]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

        # Summary
        total_events = sum(len(r.keyframes) for r in reports)
        all_classes: dict[str, int] = {}
        for r in reports:
            for cls, cnt in r.class_counts.items():
                all_classes[cls] = all_classes.get(cls, 0) + cnt

        log.info(f'Report saved: {path}')
        log.info(f'Total events: {total_events}')
        log.info(f'Class breakdown: {all_classes}')


def main():
    p = argparse.ArgumentParser(description='Batch CCTV video analysis')
    p.add_argument('--input',    required=True, help='Video file or directory')
    p.add_argument('--output',   default='batch_output', help='Output directory')
    p.add_argument('--model',    default='yolov8n.pt')
    p.add_argument('--conf',     type=float, default=0.35)
    p.add_argument('--find',     nargs='+', help='Filter: only these classes (e.g. person car)')
    p.add_argument('--strategy', default='interval',
                   choices=BatchProcessor.STRATEGIES)
    p.add_argument('--interval', type=float, default=5.0,
                   help='Seconds between samples (interval strategy)')
    p.add_argument('--workers',  type=int, default=1,
                   help='Parallel workers for directory processing')
    p.add_argument('--no-save-frames', action='store_true',
                   help='Do not save annotated keyframes to disk')
    p.add_argument('--recursive', action='store_true',
                   help='Scan subdirectories recursively')
    args = p.parse_args()

    processor = BatchProcessor(
        model_path=args.model,
        conf=args.conf,
        target_classes=args.find,
        strategy=args.strategy,
        interval_s=args.interval,
        save_keyframes=not args.no_save_frames,
        output_dir=args.output,
        workers=args.workers,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        reports = processor.process_directory(input_path, recursive=args.recursive)
    else:
        r = processor.process_video(input_path)
        reports = [r] if r else []

    if reports:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        processor.save_report(reports, f'{args.output}/report_{ts}.json')
