"""
preprocess_deepfashion.py
=========================
Preprocesses the DeepFashion Category & Attribute Prediction Benchmark
for ResNet50 fine-tuning.

Dataset structure (Kaggle mirror):
  deepfashion/
    img/                         ← 289,222 images in subdirs
    Anno/
      list_bbox.txt              ← per-image bbox [x1 y1 x2 y2]
      list_category_img.txt      ← image → category (1-50)
      list_category_cloth.txt    ← category id → name
      list_attr_img.txt          ← image → 1000 binary attributes
      list_attr_cloth.txt        ← attribute id → name + type
    Eval/
      list_eval_partition.txt    ← image → train/val/test split

Output (for ResNet training):
  deepfashion_processed/
    train/  val/  test/
      <category_name>/           ← organized by category for ImageFolder
    metadata/
      category_map.json          ← category id → name mapping
      attribute_map.json         ← attribute id → name mapping
      image_attributes.json      ← image → attribute vector

Usage:
  python training/preprocess_deepfashion.py \
    --input  ../data/deepfashion \
    --output ../data/deepfashion_processed
"""

import os
import shutil
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_list_file(path: Path, skip_rows: int = 2) -> list:
    """
    Parses a DeepFashion list annotation file.
    First line: count. Second line: header. Remaining: data.
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_rows:
                continue
            stripped = line.strip()
            if stripped:
                lines.append(stripped.split())
    return lines


def parse_category_names(category_cloth_path: Path) -> dict:
    """
    Parses list_category_cloth.txt.
    Returns dict: category_id (1-50) → category_name.
    """
    cat_map = {}
    with open(category_cloth_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip count + header lines
    for i, line in enumerate(lines[2:], start=1):
        parts = line.strip().split()
        if len(parts) >= 2:
            name = parts[0]
            # cat_type = int(parts[1])  # 1=upper-body, 2=lower-body, 3=full-body
            cat_map[i] = name

    return cat_map


def parse_attribute_names(attr_cloth_path: Path) -> dict:
    """
    Parses list_attr_cloth.txt.
    Returns dict: attribute_id (1-1000) → (name, type).
    """
    attr_map = {}
    with open(attr_cloth_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines[2:], start=1):
        parts = line.strip().split()
        if len(parts) >= 2:
            name = parts[0]
            attr_type = int(parts[1])  # 1-5 attribute type groups
            attr_map[i] = {"name": name, "type": attr_type}

    return attr_map


def parse_category_img(category_img_path: Path) -> dict:
    """
    Parses list_category_img.txt.
    Returns dict: image_path → category_id (1-50).
    """
    img_to_cat = {}
    rows = parse_list_file(category_img_path)
    for row in rows:
        if len(row) >= 2:
            img_path = row[0]
            cat_id = int(row[1])
            img_to_cat[img_path] = cat_id
    return img_to_cat


def parse_eval_partition(eval_path: Path) -> dict:
    """
    Parses list_eval_partition.txt.
    Returns dict: image_path → split ("train" | "val" | "test").
    """
    img_to_split = {}
    rows = parse_list_file(eval_path)
    for row in rows:
        if len(row) >= 2:
            img_path = row[0]
            split = row[1].lower()
            img_to_split[img_path] = split
    return img_to_split


def parse_attr_img(attr_img_path: Path) -> dict:
    """
    Parses list_attr_img.txt.
    Returns dict: image_path → list of attribute IDs (1-indexed) present.
    This file has format: image_path  attr1_val  attr2_val  ... attr1000_val
    where values are 1 (present) or -1 (absent).
    """
    img_to_attrs = {}

    with open(attr_img_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[2:]:  # skip count + header
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_path = parts[0]
        attr_values = [int(v) for v in parts[1:]]
        # Store as binary list (1=present, 0=absent)
        binary_attrs = [1 if v == 1 else 0 for v in attr_values]
        img_to_attrs[img_path] = binary_attrs

    return img_to_attrs


def copy_image_with_category(args):
    """Worker function: copies an image to split/category/ subdirectory."""
    img_path, src_root, dst_root, split, category_name, img_name = args
    dst_dir = dst_root / split / category_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / img_name
    if not dst_file.exists():
        src_file = src_root / img_path
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DeepFashion Category & Attribute dataset for ResNet training")
    parser.add_argument("--input",   default="../data/deepfashion",
                        help="DeepFashion root directory")
    parser.add_argument("--output",  default="../data/deepfashion_processed",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits",  nargs="+", default=["train", "val", "test"],
                        help="Which splits to process")
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)

    if not src_root.exists():
        logger.error(f"Input directory not found: {src_root}")
        logger.info("Download first: python training/data_downloader.py --source kaggle")
        return

    anno_dir = src_root / "Anno"
    eval_dir = src_root / "Eval"

    logger.info("=" * 60)
    logger.info("DeepFashion Category & Attribute Preprocessing")
    logger.info("=" * 60)

    # ── Parse all annotation files ──────────────────────────────
    logger.info("Parsing annotation files...")

    # Category names (50 categories)
    category_cloth_path = anno_dir / "list_category_cloth.txt"
    if not category_cloth_path.exists():
        # Some versions have different names
        for alt in ["list_category_cloth.txt", "Category_and_Attribute/Anno/list_category_cloth.txt"]:
            p = src_root / alt
            if p.exists():
                category_cloth_path = p
                break

    cat_names = {}
    if category_cloth_path.exists():
        cat_names = parse_category_names(category_cloth_path)
        logger.info(f"  Loaded {len(cat_names)} category names")
    else:
        logger.warning("  list_category_cloth.txt not found — using generic names")
        cat_names = {i: f"category_{i}" for i in range(1, 51)}

    # Image → category
    cat_img_path = anno_dir / "list_category_img.txt"
    img_to_cat = {}
    if not cat_img_path.exists():
        # Try alternative paths
        for alt in src_root.rglob("list_category_img.txt"):
            cat_img_path = alt
            break
    if cat_img_path.exists():
        img_to_cat = parse_category_img(cat_img_path)
        logger.info(f"  Loaded category labels for {len(img_to_cat):,} images")

    # Train/val/test partition
    eval_partition_path = eval_dir / "list_eval_partition.txt"
    if not eval_partition_path.exists():
        for alt in src_root.rglob("list_eval_partition.txt"):
            eval_partition_path = alt
            break
    img_to_split = {}
    if eval_partition_path.exists():
        img_to_split = parse_eval_partition(eval_partition_path)
        split_counts = defaultdict(int)
        for sp in img_to_split.values():
            split_counts[sp] += 1
        for sp, count in split_counts.items():
            logger.info(f"  {sp}: {count:,} images")

    # Attribute labels (optional — large file, ~289K rows × 1000 cols)
    attr_img_path = anno_dir / "list_attr_img.txt"
    img_to_attrs = {}
    if attr_img_path.exists():
        logger.info("  Parsing attribute labels (289K × 1000)... (may take a minute)")
        img_to_attrs = parse_attr_img(attr_img_path)
        logger.info(f"  Loaded attributes for {len(img_to_attrs):,} images")

    # Attribute names
    attr_cloth_path = anno_dir / "list_attr_cloth.txt"
    attr_names = {}
    if attr_cloth_path.exists():
        attr_names = parse_attribute_names(attr_cloth_path)
        logger.info(f"  Loaded {len(attr_names)} attribute names")

    # ── Save metadata ────────────────────────────────────────────
    metadata_dir = dst_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / "category_map.json", "w") as f:
        json.dump({str(k): v for k, v in cat_names.items()}, f, indent=2)

    if attr_names:
        with open(metadata_dir / "attribute_map.json", "w") as f:
            json.dump({str(k): v for k, v in attr_names.items()}, f, indent=2)

    if img_to_attrs:
        logger.info("  Saving attribute index...")
        with open(metadata_dir / "image_attributes.json", "w") as f:
            json.dump(img_to_attrs, f)

    logger.info(f"  Metadata saved → {metadata_dir}")

    # ── Organize images into split/category/ subdirectories ──────
    logger.info("\nOrganizing images by split + category...")

    copy_tasks = []
    skipped = 0

    for img_path, cat_id in img_to_cat.items():
        split = img_to_split.get(img_path, "train")
        if split not in args.splits:
            continue
        category_name = cat_names.get(cat_id, f"category_{cat_id}")
        img_name = Path(img_path).name
        copy_tasks.append((img_path, src_root, dst_root, split, category_name, img_name))

    logger.info(f"  Tasks: {len(copy_tasks):,} images to organize")

    copied = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(copy_image_with_category, task): task for task in copy_tasks}
        with tqdm(total=len(copy_tasks), desc="  Organizing", unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    copied += 1
                pbar.update(1)

    logger.info(f"  Copied {copied:,} images (skipped existing: {len(copy_tasks) - copied:,})")

    # ── Print final stats ─────────────────────────────────────────
    logger.info("\n✓ Preprocessing complete!")
    for split in args.splits:
        split_dir = dst_root / split
        if split_dir.exists():
            n_cats = len(list(split_dir.iterdir()))
            n_imgs = sum(1 for _ in split_dir.rglob("*.jpg"))
            logger.info(f"  {split}: {n_imgs:,} images in {n_cats} categories")

    logger.info("\nNext step:")
    logger.info("  python training/train_resnet.py --data-dir " + str(dst_root))


if __name__ == "__main__":
    main()
