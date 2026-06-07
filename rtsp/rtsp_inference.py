"""
rtsp_inference.py
Real-time object detection on RTSP video stream using YOLOv8.

Usage:
    python rtsp_inference.py --source rtsp://192.168.1.100:554/stream
    python rtsp_inference.py --source rtsp://user:pass@192.168.1.100/cam1 --conf 0.4
    python rtsp_inference.py --source 0  # webcam fallback
"""

import cv2
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 RTSP inference')
    p.add_argument('--source',  default='0',         help='RTSP URL or webcam index')
    p.add_argument('--model',   default='yolov8n.pt', help='YOLOv8 model weights')
    p.add_argument('--conf',    type=float, default=0.35, help='Confidence threshold')
    p.add_argument('--iou',     type=float, default=0.45, help='IoU threshold (NMS)')
    p.add_argument('--classes', nargs='+', type=int,  help='Filter classes, e.g. 0 2 (person, car)')
    p.add_argument('--save',    action='store_true',  help='Save annotated output video')
    p.add_argument('--no-show', action='store_true',  help='Disable display window')
    p.add_argument('--fps-limit', type=int, default=0, help='Limit processing FPS (0=unlimited)')
    return p.parse_args()


class RTSPInference:
    def __init__(self, source, model_path, conf=0.35, iou=0.45, classes=None):
        self.source  = source
        self.conf    = conf
        self.iou     = iou
        self.classes = classes

        log.info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)

        self.cap = self._open_stream()
        self.writer = None

        # Stats
        self.frame_count = 0
        self.fps_ema     = 0.0
        self.t_prev      = time.time()

    def _open_stream(self):
        """Open RTSP stream with reconnect-friendly settings."""
        source = int(self.source) if self.source.isdigit() else self.source

        # RTSP-specific OpenCV flags
        if isinstance(source, str) and source.startswith('rtsp'):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # minimize latency
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f'Cannot open stream: {source}')

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        log.info(f'Stream opened: {w}x{h} @ {fps:.1f}fps')
        self.frame_wh  = (w, h)
        self.stream_fps = fps
        return cap

    def setup_writer(self, output_path: str):
        """Initialize video writer for saving output."""
        w, h = self.frame_wh
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.stream_fps, (w, h))
        log.info(f'Saving output to: {output_path}')

    def _update_fps(self):
        t = time.time()
        inst_fps = 1.0 / max(t - self.t_prev, 1e-6)
        self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst_fps
        self.t_prev = t
        return self.fps_ema

    def annotate(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = f'{self.model.names[cls_id]} {conf:.2f}'

            color = (0, 255, 100) if cls_id == 0 else (100, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return frame

    def run(self, show=True, fps_limit=0, save_path=None):
        if save_path:
            self.setup_writer(save_path)

        min_interval = 1.0 / fps_limit if fps_limit else 0
        reconnect_delay = 2

        log.info('Starting inference loop. Press Q to quit.')
        try:
            while True:
                t0 = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    log.warning('Frame read failed — reconnecting...')
                    time.sleep(reconnect_delay)
                    self.cap.release()
                    self.cap = self._open_stream()
                    continue

                # Inference
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    verbose=False,
                )

                frame = self.annotate(frame, results)
                fps   = self._update_fps()
                self.frame_count += 1

                # HUD
                n_det = len(results[0].boxes)
                ts    = datetime.now().strftime('%H:%M:%S')
                cv2.putText(frame, f'{ts}  FPS:{fps:.1f}  Det:{n_det}',
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 1, cv2.LINE_AA)

                if self.writer:
                    self.writer.write(frame)

                if show:
                    cv2.imshow('YOLOv8 RTSP', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # FPS limiter
                elapsed = time.time() - t0
                if min_interval and elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        log.info(f'Done. Processed {self.frame_count} frames.')


def main():
    args = parse_args()
    save_path = None
    if args.save:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'output_{ts}.mp4'

    inf = RTSPInference(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
    )
    inf.run(
        show=not args.no_show,
        fps_limit=args.fps_limit,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
