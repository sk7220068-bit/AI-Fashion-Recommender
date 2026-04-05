"""
preprocess_deepfashion2.py
==========================
Converts the DeepFashion2 JSON annotations into the YOLO format
required for YOLOv8 training.

DeepFashion2 has 13 clothing categories with per-image JSON annotations:
  {
    "source": "shop",
    "pair_id": 1,
    "item1": {
      "category_name": "short_sleeved_shirt",
      "category_id": 1,
      "bounding_box": [x1, y1, x2, y2],
      "landmarks": [...],
      "segmentation": [...],
      "scale": "small",
      "occlusion": "slight",
      "zoom_in": "no",
      "viewpoint": "no"
    }
  }

YOLO format (per image, one .txt per image):
  <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)

Usage:
  python training/preprocess_deepfashion2.py \
    --input  ../data/deepfashion2 \
    --output ../data/deepfashion2_yolo

Output structure:
  deepfashion2_yolo/
    train/images/  train/labels/
    val/images/    val/labels/
    data.yaml
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── DeepFashion2 category ID (1-indexed) → YOLO class ID (0-indexed) ────────
CATEGORY_MAP = {
    1:  (0,  "short_sleeved_shirt"),
    2:  (1,  "long_sleeved_shirt"),
    3:  (2,  "short_sleeved_outwear"),
    4:  (3,  "long_sleeved_outwear"),
    5:  (4,  "vest"),
    6:  (5,  "sling"),
    7:  (6,  "shorts"),
    8:  (7,  "trousers"),
    9:  (8,  "skirt"),
    10: (9,  "short_sleeved_dress"),
    11: (10, "long_sleeved_dress"),
    12: (11, "vest_dress"),
    13: (12, "sling_dress"),
}

# Display names for data.yaml
CLASS_NAMES = [name for _, name in sorted(CATEGORY_MAP.values())]


def load_annotation(anno_path: Path) -> dict:
    """Loads a single DeepFashion2 JSON annotation file."""
    with open(anno_path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """
    Converts [x1, y1, x2, y2] absolute bbox to YOLO normalized format.

    Args:
        bbox:  [x1, y1, x2, y2] in pixel coordinates
        img_w: image width in pixels
        img_h: image height in pixels

    Returns:
        (x_center, y_center, width, height) all normalized [0, 1]
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))
    return x_center, y_center, w, h


def process_annotation(anno_path: Path, img_dir: Path,
                        out_img_dir: Path, out_lbl_dir: Path) -> int:
    """
    Processes one annotation JSON + its image.
    Returns number of objects processed (0 if skipped).
    """
    stem = anno_path.stem  # e.g., "000001"

    # Find corresponding image (jpg or jpeg)
    img_path = img_dir / f"{stem}.jpg"
    if not img_path.exists():
        img_path = img_dir / f"{stem}.jpeg"
    if not img_path.exists():
        return 0  # Image missing — skip

    # Load annotation
    try:
        anno = load_annotation(anno_path)
    except (json.JSONDecodeError, OSError):
        return 0

    # Get image dimensions
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception:
        return 0

    if img_w == 0 or img_h == 0:
        return 0

    # Collect all clothing items in this image
    label_lines = []
    for item_key in anno.keys():
        if not item_key.startswith("item"):
            continue
        item = anno[item_key]
        cat_id = item.get("category_id")
        bbox = item.get("bounding_box")

        if cat_id not in CATEGORY_MAP or not bbox or len(bbox) != 4:
            continue

        yolo_class_id = CATEGORY_MAP[cat_id][0]
        x_c, y_c, w, h = bbox_to_yolo(bbox, img_w, img_h)
        label_lines.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    if not label_lines:
        return 0  # No valid items in this image

    # Copy image to output
    dest_img = out_img_dir / img_path.name
    if not dest_img.exists():
        shutil.copy2(img_path, dest_img)

    # Write YOLO label file
    dest_lbl = out_lbl_dir / f"{stem}.txt"
    with open(dest_lbl, "w") as f:
        f.write("\n".join(label_lines))

    return len(label_lines)


def process_split(split: str, df2_root: Path, out_root: Path,
                  max_workers: int = 8) -> dict:
    """
    Converts one dataset split (train/validation) to YOLO format.

    Returns statistics dict.
    """
    img_dir  = df2_root / split / "image"
    anno_dir = df2_root / split / "annos"

    if not img_dir.exists() or not anno_dir.exists():
        logger.warning(f"Split '{split}' not found at {df2_root} — skipping")
        return {"images": 0, "labels": 0, "objects": 0}

    yolo_split = "train" if split == "train" else "val"
    out_img_dir = out_root / yolo_split / "images"
    out_lbl_dir = out_root / yolo_split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    anno_files = sorted(anno_dir.glob("*.json"))
    logger.info(f"Processing {split}: {len(anno_files):,} annotation files")

    total_objects = 0
    processed_images = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_annotation, af, img_dir, out_img_dir, out_lbl_dir): af
            for af in anno_files
        }
        with tqdm(total=len(anno_files), desc=f"  {split}", unit="img") as pbar:
            for future in as_completed(futures):
                obj_count = future.result()
                if obj_count > 0:
                    processed_images += 1
                    total_objects += obj_count
                pbar.update(1)

    return {
        "images":  processed_images,
        "labels":  processed_images,
        "objects": total_objects,
    }


def write_data_yaml(out_root: Path):
    """Writes the YOLO data.yaml configuration file."""
    yaml_content = f"""# DeepFashion2 YOLO Dataset Configuration
# Generated by preprocess_deepfashion2.py

path: {out_root.resolve()}
train: train/images
val:   val/images

nc: {len(CLASS_NAMES)}  # number of classes

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"Written: {yaml_path}")
    return yaml_path


def write_stats_report(out_root: Path, stats: dict):
    """Writes a conversion statistics report."""
    import json
    report_path = out_root / "conversion_stats.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 JSON annotations to YOLO format")
    parser.add_argument("--input",  default="../data/deepfashion2",
                        help="DeepFashion2 root directory")
    parser.add_argument("--output", default="../data/deepfashion2_yolo",
                        help="Output directory for YOLO-formatted dataset")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    df2_root = Path(args.input)
    out_root = Path(args.output)

    if not df2_root.exists():
        logger.error(f"Input directory not found: {df2_root}")
        logger.info("Download DeepFashion2 first: python training/data_downloader.py --source gdrive")
        return

    logger.info("=" * 60)
    logger.info("DeepFashion2 → YOLO Format Conversion")
    logger.info("=" * 60)
    logger.info(f"  Input:   {df2_root}")
    logger.info(f"  Output:  {out_root}")
    logger.info(f"  Classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES[:5])}...)")

    all_stats = {}

    for split in ["train", "validation"]:
        stats = process_split(split, df2_root, out_root, args.workers)
        all_stats[split] = stats
        logger.info(f"  {split}: {stats['images']:,} images, {stats['objects']:,} objects")

    # Write YOLO data.yaml
    yaml_path = write_data_yaml(out_root)

    # Write stats
    write_stats_report(out_root, all_stats)

    logger.info("\n✓ Conversion complete!")
    logger.info(f"  YOLO dataset at: {out_root}")
    logger.info(f"  data.yaml:       {yaml_path}")
    logger.info("\nNext step:")
    logger.info(f"  python training/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
