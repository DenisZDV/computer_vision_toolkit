"""
zone_counter.py
People/object counting in polygonal zones using YOLOv8 + ByteTrack.

Tracks unique IDs across frames — counts entries/exits per zone,
not just instantaneous presence.

Usage:
    python zone_counter.py --source rtsp://... --zones zones.json
    python zone_counter.py --source video.mp4 --draw-zones  # interactive setup
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO


# ── ZONE DEFINITION ────────────────────────────────────────────────────────

class Zone:
    """Polygonal counting zone."""

    def __init__(self, name: str, polygon: list[list[int]], color=(0, 255, 150)):
        self.name    = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.color   = color

        # Tracking state
        self.inside:  set[int] = set()   # track IDs currently inside
        self.entered: int = 0            # total entries
        self.exited:  int = 0            # total exits

    def contains(self, point: tuple[float, float]) -> bool:
        """Check if point (cx, cy) is inside polygon."""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def update(self, track_id: int, point: tuple[float, float]):
        """Update zone state for one tracked object."""
        was_inside = track_id in self.inside
        is_inside  = self.contains(point)

        if is_inside and not was_inside:
            self.inside.add(track_id)
            self.entered += 1
        elif not is_inside and was_inside:
            self.inside.discard(track_id)
            self.exited += 1

    def remove_track(self, track_id: int):
        """Clean up when track is lost."""
        self.inside.discard(track_id)

    def draw(self, frame: np.ndarray, alpha=0.25):
        """Draw zone overlay on frame."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.polygon], True, self.color, 2)

        # Label
        cx = int(self.polygon[:, 0].mean())
        cy = int(self.polygon[:, 1].mean())
        label = f'{self.name}  in:{len(self.inside)}  +{self.entered}/-{self.exited}'
        cv2.putText(frame, label, (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def load_zones(path: str) -> list[Zone]:
    """
    Load zones from JSON file.

    Format:
    [
      {"name": "entrance", "polygon": [[100,200],[400,200],[400,500],[100,500]]},
      {"name": "checkout", "polygon": [...], "color": [255, 100, 0]}
    ]
    """
    data = json.loads(Path(path).read_text())
    zones = []
    default_colors = [(0,255,150), (255,150,0), (0,150,255), (200,0,255)]
    for i, z in enumerate(data):
        color = tuple(z.get('color', default_colors[i % len(default_colors)]))
        zones.append(Zone(z['name'], z['polygon'], color))
    return zones


def draw_zones_interactive(frame: np.ndarray) -> list[list[int]]:
    """Click to define polygon points. Press ENTER to finish, ESC to cancel."""
    points = []
    clone  = frame.copy()

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            cv2.imshow('Draw Zone', clone)

    cv2.namedWindow('Draw Zone')
    cv2.setMouseCallback('Draw Zone', mouse_cb)
    cv2.imshow('Draw Zone', clone)

    while True:
        key = cv2.waitKey(0)
        if key == 13:   # ENTER — close polygon
            if len(points) >= 3:
                cv2.line(clone, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                cv2.imshow('Draw Zone', clone)
            break
        if key == 27:   # ESC — cancel
            points = []
            break

    cv2.destroyWindow('Draw Zone')
    return points


# ── TRACKER WRAPPER ────────────────────────────────────────────────────────

class ZoneCounter:
    def __init__(
        self,
        source: str,
        model_path: str = 'yolov8n.pt',
        zones: list[Zone] = None,
        conf: float = 0.35,
        track_classes: list[int] = None,  # None = all, [0] = persons only
    ):
        self.source        = source
        self.zones         = zones or []
        self.conf          = conf
        self.track_classes = track_classes

        self.model = YOLO(model_path)
        self.cap   = cv2.VideoCapture(
            int(source) if source.isdigit() else source
        )
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open: {source}')

        # Track lifetime monitoring
        self.active_tracks: set[int] = set()

        # Event log
        self.events: list[dict] = []

    def _log_event(self, zone: Zone, event: str, track_id: int):
        self.events.append({
            'ts':       datetime.now().isoformat(timespec='seconds'),
            'zone':     zone.name,
            'event':    event,
            'track_id': track_id,
        })

    def run(self, show: bool = True, save_path: str = None):
        writer = None
        if save_path:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # YOLOv8 tracking (ByteTrack built-in)
                results = self.model.track(
                    frame,
                    conf=self.conf,
                    classes=self.track_classes,
                    tracker='bytetrack.yaml',
                    persist=True,
                    verbose=False,
                )

                current_tracks: set[int] = set()

                if results[0].boxes.id is not None:
                    boxes    = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes  = results[0].boxes.cls.int().cpu().tolist()

                    for box, tid, cls_id in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        current_tracks.add(tid)

                        # Update all zones
                        for zone in self.zones:
                            was_in = tid in zone.inside
                            zone.update(tid, (cx, cy))
                            now_in = tid in zone.inside
                            if now_in and not was_in:
                                self._log_event(zone, 'enter', tid)
                            elif not now_in and was_in:
                                self._log_event(zone, 'exit', tid)

                        # Draw track
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (100, 220, 255), 2)
                        cv2.putText(frame, f'#{tid}', (int(x1), int(y1) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)

                # Clean lost tracks from zones
                lost = self.active_tracks - current_tracks
                for tid in lost:
                    for zone in self.zones:
                        zone.remove_track(tid)
                self.active_tracks = current_tracks

                # Draw zones
                for zone in self.zones:
                    zone.draw(frame)

                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow('ZoneCounter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    def summary(self) -> dict:
        return {
            z.name: {'entered': z.entered, 'exited': z.exited, 'current': len(z.inside)}
            for z in self.zones
        }

    def save_events(self, path: str):
        import json
        Path(path).write_text(json.dumps(self.events, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',      default='0')
    p.add_argument('--model',       default='yolov8n.pt')
    p.add_argument('--zones',       help='JSON file with zone definitions')
    p.add_argument('--conf',        type=float, default=0.35)
    p.add_argument('--persons-only', action='store_true')
    p.add_argument('--save',        action='store_true')
    p.add_argument('--draw-zones',  action='store_true',
                   help='Interactively draw zones on first frame')
    args = p.parse_args()

    zones = []
    if args.zones:
        zones = load_zones(args.zones)
    elif args.draw_zones:
        cap   = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print('Draw zone polygon. Click points, press ENTER when done.')
            pts = draw_zones_interactive(frame)
            if pts:
                zones = [Zone('zone_1', pts)]

    counter = ZoneCounter(
        source=args.source,
        model_path=args.model,
        zones=zones,
        conf=args.conf,
        track_classes=[0] if args.persons_only else None,
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    counter.run(
        save_path=f'output_{ts}.mp4' if args.save else None
    )

    print('\nSummary:')
    for zone, stats in counter.summary().items():
        print(f'  {zone}: entered={stats["entered"]} exited={stats["exited"]}')

    if counter.events:
        log_path = f'events_{ts}.json'
        counter.save_events(log_path)
        print(f'Events saved: {log_path}')


if __name__ == '__main__':
    main()
