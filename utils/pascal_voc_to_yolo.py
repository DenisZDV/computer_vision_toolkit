"""
pascal_voc_to_yolo.py
Convert Pascal VOC XML annotation files to YOLO format (.txt).

Pascal VOC format (XML):
    <annotation>
      <size><width>1366</width><height>768</height></size>
      <object>
        <name>CT</name>
        <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>300</xmax><ymax>500</ymax></bndbox>
      </object>
    </annotation>

YOLO format (TXT, one line per object):
    class_id  x_center  y_center  width  height   (all normalized 0..1)

Usage:
    # Convert single file
    python utils/pascal_voc_to_yolo.py --input labels/frame_001.xml --classes CT T CT_head T_head

    # Convert entire directory
    python utils/pascal_voc_to_yolo.py \
        --input dataset/voc_labels \
        --output dataset/yolo_labels \
        --classes CT T CT_head T_head

    # With image resize: rescale bounding boxes to new resolution
    python utils/pascal_voc_to_yolo.py \
        --input dataset/voc_labels \
        --output dataset/yolo_labels \
        --classes CT T CT_head T_head \
        --resize 640 640

    # Generate classes.txt and data.yaml automatically
    python utils/pascal_voc_to_yolo.py \
        --input dataset/voc_labels \
        --output dataset/yolo_labels \
        --classes CT T CT_head T_head \
        --generate-yaml dataset/
"""

import os
import xml.etree.ElementTree as ET
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass


# ── DATA STRUCTURES ────────────────────────────────────────────────────────

@dataclass
class BBox:
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class Annotation:
    filename:  str
    width:     int
    height:    int
    objects:   list[BBox]


# ── PARSER ─────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: str) -> Annotation:
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext('filename') or Path(xml_path).stem
    size     = root.find('size')
    width    = int(size.findtext('width'))
    height   = int(size.findtext('height'))

    objects = []
    for obj in root.findall('object'):
        name   = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin   = int(float(bndbox.findtext('xmin')))
        ymin   = int(float(bndbox.findtext('ymin')))
        xmax   = int(float(bndbox.findtext('xmax')))
        ymax   = int(float(bndbox.findtext('ymax')))
        objects.append(BBox(name, xmin, ymin, xmax, ymax))

    return Annotation(filename, width, height, objects)


# ── CONVERTER ──────────────────────────────────────────────────────────────

def bbox_to_yolo(
    box: BBox,
    img_width: int,
    img_height: int,
    class_map: dict[str, int],
    target_width: int = None,
    target_height: int = None,
) -> str | None:
    """
    Convert one BBox to a YOLO format line.
    Optionally rescales coordinates to target resolution.
    Returns None if class not in class_map.
    """
    if box.class_name not in class_map:
        return None

    class_id = class_map[box.class_name]

    # Rescale if target resolution given
    if target_width and target_height:
        scale_x = target_width  / img_width
        scale_y = target_height / img_height
        xmin = box.xmin * scale_x
        ymin = box.ymin * scale_y
        xmax = box.xmax * scale_x
        ymax = box.ymax * scale_y
        w = target_width
        h = target_height
    else:
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
        w, h = img_width, img_height

    # Clamp to image bounds
    xmin = max(0, min(xmin, w))
    ymin = max(0, min(ymin, h))
    xmax = max(0, min(xmax, w))
    ymax = max(0, min(ymax, h))

    if xmax <= xmin or ymax <= ymin:
        return None  # degenerate box

    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    bw       = (xmax - xmin) / w
    bh       = (ymax - ymin) / h

    return f'{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}'


def convert_file(
    xml_path: str,
    output_path: str,
    class_map: dict[str, int],
    target_size: tuple[int, int] = None,
) -> tuple[int, int]:
    """
    Convert one XML file to YOLO txt.
    Returns (converted_objects, skipped_objects).
    """
    ann = parse_voc_xml(xml_path)
    lines = []
    skipped = 0

    tw, th = target_size if target_size else (None, None)

    for box in ann.objects:
        line = bbox_to_yolo(box, ann.width, ann.height, class_map, tw, th)
        if line:
            lines.append(line)
        else:
            skipped += 1

    Path(output_path).write_text('\n'.join(lines) + ('\n' if lines else ''))
    return len(lines), skipped


def convert_directory(
    input_dir: str,
    output_dir: str,
    class_map: dict[str, int],
    target_size: tuple[int, int] = None,
    verbose: bool = True,
) -> dict:
    """
    Convert all XML files in input_dir to YOLO txt files in output_dir.
    Returns stats dict.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = list(input_path.glob('*.xml'))
    if not xml_files:
        print(f'No XML files found in {input_dir}')
        return {}

    total_files    = 0
    total_objects  = 0
    total_skipped  = 0
    unknown_classes: set[str] = set()

    for xml_file in sorted(xml_files):
        out_file = output_path / xml_file.with_suffix('.txt').name
        try:
            ann = parse_voc_xml(str(xml_file))
            lines = []
            sk = 0
            tw, th = target_size if target_size else (None, None)

            for box in ann.objects:
                if box.class_name not in class_map:
                    unknown_classes.add(box.class_name)
                    sk += 1
                    continue
                line = bbox_to_yolo(box, ann.width, ann.height, class_map, tw, th)
                if line:
                    lines.append(line)
                else:
                    sk += 1

            out_file.write_text('\n'.join(lines) + ('\n' if lines else ''))
            total_files   += 1
            total_objects += len(lines)
            total_skipped += sk

            if verbose and total_files % 100 == 0:
                print(f'  Processed {total_files}/{len(xml_files)} files...')

        except Exception as e:
            print(f'  [error] {xml_file.name}: {e}')

    stats = {
        'files':    total_files,
        'objects':  total_objects,
        'skipped':  total_skipped,
        'unknown':  list(unknown_classes),
    }

    print(f'\nConverted: {total_files} files, {total_objects} objects')
    if total_skipped:
        print(f'Skipped:   {total_skipped} objects')
    if unknown_classes:
        print(f'Unknown classes (not in map): {unknown_classes}')

    return stats


# ── YAML / CLASSES GENERATOR ───────────────────────────────────────────────

def generate_yaml(
    dataset_dir: str,
    class_names: list[str],
    yaml_path: str = None,
):
    """Generate data.yaml and classes.txt for YOLOv8 training."""
    base = Path(dataset_dir)
    yaml_path = yaml_path or str(base / 'data.yaml')
    classes_path = base / 'classes.txt'

    data = {
        'train': str(base / 'train' / 'images'),
        'val':   str(base / 'val'   / 'images'),
        'test':  str(base / 'test'  / 'images'),
        'nc':    len(class_names),
        'names': class_names,
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    classes_path.write_text('\n'.join(class_names) + '\n')

    print(f'Generated: {yaml_path}')
    print(f'Generated: {classes_path}')
    print(f'Classes ({len(class_names)}): {class_names}')


# ── DISCOVERY: auto-detect classes from XML files ──────────────────────────

def discover_classes(xml_dir: str) -> list[str]:
    """Scan all XML files and return sorted unique class names."""
    classes: set[str] = set()
    for xml_file in Path(xml_dir).glob('*.xml'):
        try:
            tree = ET.parse(str(xml_file))
            for obj in tree.getroot().findall('object'):
                name = obj.findtext('name')
                if name:
                    classes.add(name)
        except Exception:
            pass
    return sorted(classes)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Convert Pascal VOC XML annotations to YOLO format'
    )
    p.add_argument('--input',    required=True,
                   help='Input XML file or directory with XML files')
    p.add_argument('--output',   default=None,
                   help='Output directory for TXT files (default: same as input)')
    p.add_argument('--classes',  nargs='+',
                   help='Class names in order, e.g.: CT T CT_head T_head. '
                        'If omitted, auto-detected from XML files.')
    p.add_argument('--resize',   nargs=2, type=int, metavar=('W', 'H'),
                   help='Rescale bounding boxes to this resolution, e.g. 640 640')
    p.add_argument('--generate-yaml', metavar='DATASET_DIR',
                   help='Also generate data.yaml and classes.txt in this directory')
    p.add_argument('--discover', action='store_true',
                   help='List all class names found in XML files and exit')
    args = p.parse_args()

    input_path = Path(args.input)
    target_size = tuple(args.resize) if args.resize else None

    # Discovery mode
    if args.discover:
        xml_dir = str(input_path) if input_path.is_dir() else str(input_path.parent)
        classes = discover_classes(xml_dir)
        print(f'Found {len(classes)} classes:')
        for i, c in enumerate(classes):
            print(f'  {i}: {c}')
        return

    # Resolve class list
    if args.classes:
        class_names = args.classes
    else:
        xml_dir = str(input_path) if input_path.is_dir() else str(input_path.parent)
        class_names = discover_classes(xml_dir)
        if not class_names:
            print('No classes found. Use --classes or --discover first.')
            return
        print(f'Auto-detected classes: {class_names}')

    class_map = {name: i for i, name in enumerate(class_names)}

    # Convert
    if input_path.is_dir():
        output_dir = args.output or str(input_path)
        convert_directory(
            input_dir=str(input_path),
            output_dir=output_dir,
            class_map=class_map,
            target_size=target_size,
        )
    elif input_path.suffix.lower() == '.xml':
        out_dir = Path(args.output) if args.output else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / input_path.with_suffix('.txt').name
        converted, skipped = convert_file(
            str(input_path), str(out_file), class_map, target_size
        )
        print(f'Converted: {converted} objects, skipped: {skipped}')
        print(f'Output: {out_file}')
    else:
        print(f'Error: {input_path} is not an XML file or directory')
        return

    # Generate YAML
    if args.generate_yaml:
        generate_yaml(args.generate_yaml, class_names)


if __name__ == '__main__':
    main()
