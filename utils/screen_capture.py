"""
screen_capture.py
High-performance screen capture using MSS with threading.

Single-threaded MSS: ~20 FPS
Multi-threaded (capture + process): ~30–40 FPS
With YOLOv8 inference on GPU: depends on GPU, typically 20–25 FPS

Architecture:
    CaptureThread  →  frame_queue  →  ProcessThread
    (MSS grab)                        (YOLO / any processor)

Usage:
    # Basic — capture and display
    python utils/screen_capture.py

    # With YOLO inference
    from utils.screen_capture import ScreenCapture, CaptureRegion

    region = CaptureRegion(left=0, top=0, width=1280, height=720)
    cap = ScreenCapture(region=region)

    for frame in cap.frames():
        # frame is a numpy BGR array ready for cv2 / YOLO
        results = model(frame)
        ...

    # Threaded producer-consumer
    cap = ScreenCapture(region=region, threaded=True)
    cap.start()
    while True:
        frame = cap.get_frame()
        if frame is not None:
            process(frame)
"""

import cv2
import time
import threading
import argparse
import numpy as np
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Iterator, Callable

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print('Warning: mss not installed. pip install mss')


# ── REGION ─────────────────────────────────────────────────────────────────

@dataclass
class CaptureRegion:
    """Screen region to capture."""
    left:   int = 0
    top:    int = 0
    width:  int = 1280
    height: int = 720

    def to_mss(self) -> dict:
        return {
            'left':   self.left,
            'top':    self.top,
            'width':  self.width,
            'height': self.height,
        }

    @staticmethod
    def full_screen() -> 'CaptureRegion':
        """Capture entire primary monitor."""
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                return CaptureRegion(
                    left=mon['left'], top=mon['top'],
                    width=mon['width'], height=mon['height']
                )
        return CaptureRegion(0, 0, 1920, 1080)

    @staticmethod
    def center(width: int = 640, height: int = 640) -> 'CaptureRegion':
        """Capture center of primary screen."""
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                sw, sh = mon['width'], mon['height']
        else:
            sw, sh = 1920, 1080
        left = (sw - width) // 2
        top  = (sh - height) // 2
        return CaptureRegion(left, top, width, height)


# ── SINGLE-THREADED CAPTURE ────────────────────────────────────────────────

class ScreenCapture:
    """
    Screen capture with optional threading.

    Args:
        region:    Screen region to capture
        threaded:  Use producer thread for higher FPS
        maxsize:   Max frames in queue (threaded mode)
        resize:    Resize captured frames to (width, height) or None
    """

    def __init__(
        self,
        region: CaptureRegion = None,
        threaded: bool = False,
        maxsize: int = 1,
        resize: tuple[int, int] = None,
    ):
        if not MSS_AVAILABLE:
            raise ImportError('pip install mss')

        self.region   = region or CaptureRegion.full_screen()
        self.threaded = threaded
        self.resize   = resize
        self._running = False

        self._queue: Queue = Queue(maxsize=maxsize)
        self._thread: threading.Thread = None
        self._latest_frame = None
        self._lock = threading.Lock()

        # FPS tracking
        self._fps_ema   = 0.0
        self._t_prev    = time.time()
        self._frame_cnt = 0

    def _grab_frame(self, sct) -> np.ndarray:
        """Grab one frame from screen and convert to BGR numpy array."""
        raw = sct.grab(self.region.to_mss())
        frame = np.array(raw)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self.resize:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)
        return frame

    def _update_fps(self):
        t = time.time()
        inst = 1.0 / max(t - self._t_prev, 1e-6)
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst
        self._t_prev = t
        self._frame_cnt += 1

    @property
    def fps(self) -> float:
        return self._fps_ema

    @property
    def frame_count(self) -> int:
        return self._frame_cnt

    # ── SIMPLE ITERATOR ──────────────────────────────────────────────────

    def frames(self) -> Iterator[np.ndarray]:
        """
        Single-threaded generator. Yields frames as fast as possible.

        Usage:
            cap = ScreenCapture()
            for frame in cap.frames():
                process(frame)
                if done: break
        """
        with mss.mss() as sct:
            while True:
                frame = self._grab_frame(sct)
                self._update_fps()
                yield frame

    # ── THREADED MODE ─────────────────────────────────────────────────────

    def _capture_loop(self):
        """Producer thread: grab frames and push to queue."""
        with mss.mss() as sct:
            while self._running:
                frame = self._grab_frame(sct)
                self._update_fps()

                # Always keep latest frame accessible
                with self._lock:
                    self._latest_frame = frame

                # Non-blocking put — drop frame if queue full
                try:
                    self._queue.put_nowait(frame)
                except Exception:
                    # Queue full — discard old frame, push new one
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except Exception:
                        pass

    def start(self):
        """Start background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name='ScreenCapture',
        )
        self._thread.start()

    def stop(self):
        """Stop background capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.05) -> np.ndarray | None:
        """
        Get latest frame from queue (threaded mode).
        Returns None if no frame available within timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            # Fallback to latest grabbed even if queue was full
            with self._lock:
                return self._latest_frame

    def get_latest(self) -> np.ndarray | None:
        """Get most recently captured frame without waiting."""
        with self._lock:
            return self._latest_frame

    def __enter__(self):
        if self.threaded:
            self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ── PIPELINE HELPER ────────────────────────────────────────────────────────

class CaptureInferencePipeline:
    """
    Two-thread pipeline: capture thread + inference thread.
    Designed for YOLOv8 on GPU — keeps GPU busy without blocking capture.

    Usage:
        def my_processor(frame):
            results = model(frame, verbose=False)
            return results[0].plot()

        pipeline = CaptureInferencePipeline(
            processor=my_processor,
            region=CaptureRegion.center(640, 640),
        )
        pipeline.run(show=True)
    """

    def __init__(
        self,
        processor: Callable[[np.ndarray], np.ndarray],
        region: CaptureRegion = None,
        resize: tuple[int, int] = None,
        queue_size: int = 2,
    ):
        self.processor = processor
        self.capture   = ScreenCapture(
            region=region or CaptureRegion.center(),
            threaded=True,
            maxsize=queue_size,
            resize=resize,
        )
        self._output_frame = None
        self._out_lock = threading.Lock()

    def _inference_loop(self):
        while self.capture._running:
            frame = self.capture.get_frame(timeout=0.1)
            if frame is None:
                continue
            try:
                result = self.processor(frame)
                with self._out_lock:
                    self._output_frame = result
            except Exception as e:
                print(f'Processor error: {e}')

    def run(self, show: bool = True, window: str = 'Capture'):
        self.capture.start()
        inf_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name='Inference'
        )
        inf_thread.start()

        print('Pipeline running. Press Q to quit.')
        try:
            while True:
                with self._out_lock:
                    frame = self._output_frame

                if frame is not None and show:
                    # FPS overlay
                    cv2.putText(frame,
                                f'FPS: {self.capture.fps:.1f}  '
                                f'Frames: {self.capture.frame_count}',
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 100), 1, cv2.LINE_AA)
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        finally:
            self.capture.stop()
            cv2.destroyAllWindows()


# ── CLI DEMO ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Screen capture demo')
    p.add_argument('--region',   nargs=4, type=int, metavar=('L','T','W','H'),
                   help='Capture region: left top width height')
    p.add_argument('--resize',   nargs=2, type=int, metavar=('W','H'))
    p.add_argument('--threaded', action='store_true',
                   help='Use producer thread (higher FPS)')
    p.add_argument('--yolo',     help='YOLOv8 model path for live inference demo')
    args = p.parse_args()

    region = CaptureRegion(*args.region) if args.region else CaptureRegion.center(640, 640)
    resize = tuple(args.resize) if args.resize else None

    if args.yolo:
        from ultralytics import YOLO
        model = YOLO(args.yolo)

        def process(frame):
            results = model(frame, verbose=False)
            return results[0].plot()

        pipeline = CaptureInferencePipeline(
            processor=process,
            region=region,
            resize=resize,
        )
        pipeline.run()
        return

    # Plain capture display
    cap = ScreenCapture(region=region, threaded=args.threaded, resize=resize)
    print(f'Region: {region}  Threaded: {args.threaded}')
    print('Press Q to quit.')

    if args.threaded:
        cap.start()
        try:
            while True:
                frame = cap.get_frame()
                if frame is None:
                    continue
                cv2.putText(frame, f'FPS: {cap.fps:.1f}', (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                cv2.imshow('Screen Capture (threaded)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.stop()
            cv2.destroyAllWindows()
    else:
        for frame in cap.frames():
            cv2.putText(frame, f'FPS: {cap.fps:.1f}', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.imshow('Screen Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
