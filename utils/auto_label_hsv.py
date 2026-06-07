"""
auto_label_hsv.py
Automatic dataset labeling using HSV color filtering.

Workflow:
  1. Pick HSV range interactively on a sample frame
  2. Apply filter to all images → generate YOLO-format label files
  3. Verify labels visually before training

Usage:
    # Step 1: calibrate HSV range on a sample image
    python utils/auto_label_hsv.py calibrate --image samples/frame_001.jpg

    # Step 2: label all images in a folder
    python utils/auto_label_hsv.py label \
        --input dataset/raw_images \
        --output dataset/train \
        --hsv 64,54,120,180,255,255 \
        --class-id 0

    # Step 3: verify labels visually
    python utils/auto_label_hsv.py verify \
        --images dataset/train/images \
        --labels dataset/train/labels

    # All-in-one: extract frames from video + label
    python utils/auto_label_hsv.py from-video \
        --video recording.mp4 \
        --output dataset/train \
        --hsv 64,54,120,180,255,255 \
        --interval 5
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# ── HSV CALIBRATOR ─────────────────────────────────────────────────────────

@dataclass
class HSVRange:
    h1: int; s1: int; v1: int
    h2: int; s2: int; v2: int

    def lower(self): return np.array([self.h1, self.s1, self.v1], np.uint8)
    def upper(self): return np.array([self.h2, self.s2, self.v2], np.uint8)

    def __str__(self):
        return f'{self.h1},{self.s1},{self.v1},{self.h2},{self.s2},{self.v2}'

    @staticmethod
    def from_str(s: str) -> 'HSVRange':
        v = list(map(int, s.split(',')))
        return HSVRange(*v)


def calibrate_hsv(image_path: str) -> HSVRange:
    """
    Interactive HSV calibration using trackbars.
    Drag sliders until the target object is fully white in the mask window.
    Press ESC or Q to confirm and save values.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Cannot open: {image_path}')

    # Downscale for display if large
    h, w = image.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    WIN_RESULT   = 'Mask (white = detected)'
    WIN_ORIGINAL = 'Original'
    WIN_SETTINGS = 'HSV Settings'

    cv2.namedWindow(WIN_SETTINGS)
    cv2.namedWindow(WIN_RESULT)
    cv2.namedWindow(WIN_ORIGINAL)

    def nothing(_): pass

    cv2.createTrackbar('H min', WIN_SETTINGS,   0, 180, nothing)
    cv2.createTrackbar('S min', WIN_SETTINGS,   0, 255, nothing)
    cv2.createTrackbar('V min', WIN_SETTINGS,   0, 255, nothing)
    cv2.createTrackbar('H max', WIN_SETTINGS, 180, 180, nothing)
    cv2.createTrackbar('S max', WIN_SETTINGS, 255, 255, nothing)
    cv2.createTrackbar('V max', WIN_SETTINGS, 255, 255, nothing)

    cv2.imshow(WIN_ORIGINAL, image)

    print('\nHSV Calibration:')
    print('  Drag sliders until the target object is WHITE in the mask window')
    print('  Press ESC or Q to confirm\n')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = HSVRange(0, 0, 0, 180, 255, 255)

    while True:
        h1 = cv2.getTrackbarPos('H min', WIN_SETTINGS)
        s1 = cv2.getTrackbarPos('S min', WIN_SETTINGS)
        v1 = cv2.getTrackbarPos('V min', WIN_SETTINGS)
        h2 = cv2.getTrackbarPos('H max', WIN_SETTINGS)
        s2 = cv2.getTrackbarPos('S max', WIN_SETTINGS)
        v2 = cv2.getTrackbarPos('V max', WIN_SETTINGS)

        mask = cv2.inRange(hsv,
                           np.array([h1, s1, v1], np.uint8),
                           np.array([h2, s2, v2], np.uint8))

        # Show contour on original
        preview = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(preview, (x, y), (x + bw, y + bh), (0, 255, 100), 2)

        cv2.imshow(WIN_RESULT, mask)
        cv2.imshow(WIN_ORIGINAL, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):
            result = HSVRange(h1, s1, v1, h2, s2, v2)
            break

    cv2.destroyAllWindows()
    print(f'HSV range: {result}')
    print(f'Use with: --hsv {result}')
    return result


# ── LABELING ENGINE ────────────────────────────────────────────────────────

def find_objects_hsv(
    image: np.ndarray,
    hsv_range: HSVRange,
    min_area: int = 500,
    multi_object: bool = False,
) -> list[tuple[float, float, float, float]]:
    """
    Detect objects by HSV color range.
    Returns list of YOLO-format boxes: [(x_center, y_center, width, height), ...]
    All values normalized 0..1.
    """
    h, w = image.shape[:2]
    resized = cv2.resize(image, (640, 640))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_range.lower(), hsv_range.upper())

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if not multi_object:
        # Single largest object
        contours = [max(contours, key=cv2.contourArea)]

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x_center = (x + bw / 2) / 640
        y_center = (y + bh / 2) / 640
        width    = bw / 640
        height   = bh / 640
        boxes.append((x_center, y_center, width, height))

    return boxes


def label_images(
    input_dir: str,
    output_dir: str,
    hsv_range: HSVRange,
    class_id: int = 0,
    min_area: int = 500,
    multi_object: bool = False,
    img_size: int = 640,
    skip_empty: bool = True,
) -> tuple[int, int]:
    """
    Label all images in input_dir → save to output_dir/images + output_dir/labels.
    Returns (labeled_count, skipped_count).
    """
    input_path  = Path(input_dir)
    images_out  = Path(output_dir) / 'images'
    labels_out  = Path(output_dir) / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    files.sort()

    labeled = 0
    skipped = 0

    for i, fp in enumerate(files):
        image = cv2.imread(str(fp))
        if image is None:
            print(f'  [skip] Cannot read: {fp.name}')
            skipped += 1
            continue

        boxes = find_objects_hsv(image, hsv_range, min_area, multi_object)

        if not boxes:
            if skip_empty:
                skipped += 1
                continue
            # Save image without label (background sample)
            out_img = cv2.resize(image, (img_size, img_size))
            cv2.imwrite(str(images_out / fp.name), out_img)
            (labels_out / fp.with_suffix('.txt').name).write_text('')
            labeled += 1
            continue

        # Save resized image
        out_img = cv2.resize(image, (img_size, img_size))
        cv2.imwrite(str(images_out / fp.name), out_img)

        # Save YOLO label file
        label_path = labels_out / fp.with_suffix('.txt').name
        with open(label_path, 'w') as f:
            for (xc, yc, bw, bh) in boxes:
                f.write(f'{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

        labeled += 1
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{i+1}/{len(files)}] {fp.name} → {len(boxes)} object(s)')

    print(f'\nDone: {labeled} labeled, {skipped} skipped')
    return labeled, skipped


# ── VERIFICATION ───────────────────────────────────────────────────────────

def verify_labels(images_dir: str, labels_dir: str):
    """
    Visual verification: shows each image with drawn bounding boxes.
    Controls:
      SPACE / D — next image
      A         — previous image
      ESC / Q   — quit
      DEL       — delete label file for current image
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    files = sorted(images_path.glob('*.jpg')) + \
            sorted(images_path.glob('*.png'))
    if not files:
        print('No images found.')
        return

    idx = 0
    print(f'\nVerification: {len(files)} images')
    print('  SPACE/D = next   A = prev   DEL = delete label   ESC/Q = quit\n')

    while 0 <= idx < len(files):
        fp = files[idx]
        image = cv2.imread(str(fp))
        if image is None:
            idx += 1
            continue

        h, w = image.shape[:2]
        label_fp = labels_path / fp.with_suffix('.txt').name

        if label_fp.exists():
            with open(label_fp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
                    cv2.putText(image, f'cls {cls}', (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
        else:
            cv2.putText(image, 'NO LABEL', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        status = f'[{idx+1}/{len(files)}] {fp.name}'
        cv2.putText(image, status, (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Verify Labels', image)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord('q')):
            break
        elif key in (ord(' '), ord('d'), 83):   # next
            idx += 1
        elif key in (ord('a'), 81):             # prev
            idx = max(0, idx - 1)
        elif key == 255:                        # DEL
            if label_fp.exists():
                label_fp.unlink()
                print(f'Deleted: {label_fp.name}')
            idx += 1

    cv2.destroyAllWindows()


# ── VIDEO FRAME EXTRACTOR ──────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    interval_s: float = 1.0,
) -> int:
    """Extract frames from video every interval_s seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps * interval_s))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0
    print(f'Extracting every {interval_s}s ({step} frames) from {Path(video_path).name}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            filename = out_path / f'frame_{frame_idx:06d}.jpg'
            cv2.imwrite(str(filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f'Extracted {saved} frames → {output_dir}')
    return saved


# ── SPLIT DATASET ──────────────────────────────────────────────────────────

def split_dataset(
    dataset_dir: str,
    train: float = 0.7,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 42,
):
    """Split train/images into train/val/test sets."""
    import shutil, random

    random.seed(seed)
    base = Path(dataset_dir)

    src_images = base / 'train' / 'images'
    src_labels = base / 'train' / 'labels'

    for split in ('val', 'test'):
        (base / split / 'images').mkdir(parents=True, exist_ok=True)
        (base / split / 'labels').mkdir(parents=True, exist_ok=True)

    files = [f.stem for f in src_images.glob('*.jpg')]
    files += [f.stem for f in src_images.glob('*.png')]
    random.shuffle(files)

    n = len(files)
    n_val  = int(n * val)
    n_test = int(n * test)

    splits = {
        'val':  files[:n_val],
        'test': files[n_val:n_val + n_test],
    }

    moved = {'val': 0, 'test': 0}
    for split, stems in splits.items():
        for stem in stems:
            for ext in ('.jpg', '.png'):
                img_src = src_images / f'{stem}{ext}'
                if img_src.exists():
                    shutil.move(str(img_src), str(base / split / 'images' / f'{stem}{ext}'))
            lbl_src = src_labels / f'{stem}.txt'
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(base / split / 'labels' / f'{stem}.txt'))
            moved[split] += 1

    n_train = n - moved['val'] - moved['test']
    print(f'Split: train={n_train}  val={moved["val"]}  test={moved["test"]}')


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='HSV-based auto-labeling for YOLOv8 dataset preparation'
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    # calibrate
    cal = sub.add_parser('calibrate', help='Pick HSV range interactively')
    cal.add_argument('--image', required=True)

    # label
    lbl = sub.add_parser('label', help='Label images using HSV range')
    lbl.add_argument('--input',        required=True, help='Folder with raw images')
    lbl.add_argument('--output',       required=True, help='Output dataset folder')
    lbl.add_argument('--hsv',          required=True,
                     help='HSV range: h1,s1,v1,h2,s2,v2 (from calibrate step)')
    lbl.add_argument('--class-id',     type=int, default=0)
    lbl.add_argument('--min-area',     type=int, default=500,
                     help='Minimum contour area in pixels (640x640 space)')
    lbl.add_argument('--multi',        action='store_true',
                     help='Detect multiple objects per image')
    lbl.add_argument('--keep-empty',   action='store_true',
                     help='Save images with no detections (background samples)')

    # verify
    ver = sub.add_parser('verify', help='Visual label verification')
    ver.add_argument('--images', required=True)
    ver.add_argument('--labels', required=True)

    # from-video
    vid = sub.add_parser('from-video', help='Extract frames from video + label')
    vid.add_argument('--video',    required=True)
    vid.add_argument('--output',   required=True)
    vid.add_argument('--hsv',      required=True)
    vid.add_argument('--interval', type=float, default=1.0,
                     help='Seconds between extracted frames')
    vid.add_argument('--class-id', type=int, default=0)

    # split
    spl = sub.add_parser('split', help='Split dataset into train/val/test')
    spl.add_argument('--dataset', required=True)
    spl.add_argument('--train',   type=float, default=0.7)
    spl.add_argument('--val',     type=float, default=0.2)
    spl.add_argument('--test',    type=float, default=0.1)

    args = p.parse_args()

    if args.cmd == 'calibrate':
        result = calibrate_hsv(args.image)
        print(f'\nCopy this for labeling:')
        print(f'  --hsv {result}')

    elif args.cmd == 'label':
        hsv = HSVRange.from_str(args.hsv)
        label_images(
            input_dir=args.input,
            output_dir=args.output,
            hsv_range=hsv,
            class_id=args.class_id,
            min_area=args.min_area,
            multi_object=args.multi,
            skip_empty=not args.keep_empty,
        )

    elif args.cmd == 'verify':
        verify_labels(args.images, args.labels)

    elif args.cmd == 'from-video':
        hsv = HSVRange.from_str(args.hsv)
        tmp_dir = Path(args.output) / '_raw_frames'
        extract_frames(args.video, str(tmp_dir), args.interval)
        label_images(
            input_dir=str(tmp_dir),
            output_dir=args.output,
            hsv_range=hsv,
            class_id=args.class_id,
        )
        # Clean up raw frames
        import shutil
        shutil.rmtree(str(tmp_dir))

    elif args.cmd == 'split':
        split_dataset(args.dataset, args.train, args.val, args.test)


if __name__ == '__main__':
    main()
