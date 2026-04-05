"""
data_downloader.py
==================
Downloads and organizes the DeepFashion datasets from:
  1. Kaggle:       vishalbsadanand/deepfashion-1  (Category & Attribute benchmark)
  2. Google Drive: DeepFashion2 (detection dataset with bounding boxes)

Prerequisites:
  pip install kaggle gdown tqdm
  kaggle.json placed at ~/.kaggle/kaggle.json  (get from kaggle.com/account)

Usage:
  python training/data_downloader.py --source kaggle   # downloads from Kaggle
  python training/data_downloader.py --source gdrive   # downloads DeepFashion2
  python training/data_downloader.py --source both     # downloads everything
"""

import os
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Google Drive folder ID for DeepFashion2 ───────────────────────────────────
# Source: https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2
DEEPFASHION2_GDRIVE_FILES = {
    # File name → Google Drive file ID for individual zip parts
    "train.zip": "1BPqIzI1elWmEd0oeKfZvOFKcZfBhVh-_",
    "validation.zip": "1FqBqT1WuW4ItMVRHOqCR9JhVkTMvJ5qb",
    "test.zip": "1jTtLqGLh6y3rMbdBMrJkYZb1q4YbQ9gm",
}

# ── Kaggle dataset identifier ─────────────────────────────────────────────────
KAGGLE_DATASET = "vishalbsadanand/deepfashion-1"


def download_kaggle(output_dir: Path):
    """
    Downloads the DeepFashion Category & Attribute benchmark from Kaggle.
    Requires kaggle.json API credentials.
    """
    try:
        import kaggle
        logger.info(f"Downloading DeepFashion from Kaggle: {KAGGLE_DATASET}")
        output_dir.mkdir(parents=True, exist_ok=True)

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=True,
            quiet=False
        )
        logger.info(f"Kaggle download complete → {output_dir}")

    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        raise
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info("Manual download: https://www.kaggle.com/datasets/vishalbsadanand/deepfashion-1")
        raise


def download_deepfashion2(output_dir: Path):
    """
    Downloads DeepFashion2 detection dataset from Google Drive.
    Uses gdown for large file download with resume support.
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading DeepFashion2 from Google Drive → {output_dir}")

    for filename, file_id in DEEPFASHION2_GDRIVE_FILES.items():
        dest = output_dir / filename
        if dest.exists():
            logger.info(f"  Skipping (already exists): {filename}")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info(f"  Downloading {filename}...")
        try:
            gdown.download(url, str(dest), quiet=False, resume=True)
        except Exception as e:
            logger.warning(f"  gdown failed for {filename}: {e}")
            logger.info(f"  Download manually from: https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2")

    # Extract zip files
    for filename in DEEPFASHION2_GDRIVE_FILES.keys():
        zip_path = output_dir / filename
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            split = filename.replace(".zip", "")
            extract_path = output_dir / split
            if extract_path.exists():
                logger.info(f"  Already extracted: {split}")
                continue
            logger.info(f"  Extracting {filename}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
            logger.info(f"  Extracted → {extract_path}")


def verify_deepfashion_structure(root: Path):
    """
    Verifies the expected DeepFashion Category & Attribute directory structure.
    Expected after Kaggle download:
      root/
        img/          ← all images organized by category
        Anno/
          list_bbox.txt
          list_category_img.txt
          list_attr_img.txt
          list_eval_partition.txt
        Eval/
          list_eval_partition.txt
    """
    required = [
        root / "img",
        root / "Anno" / "list_bbox.txt",
        root / "Anno" / "list_category_img.txt",
        root / "Anno" / "list_attr_img.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(f"Missing expected files: {missing}")
        logger.info("The Kaggle dataset may organize files differently — check the root directory")
        return False
    logger.info("✓ DeepFashion Category & Attribute structure verified")
    return True


def verify_deepfashion2_structure(root: Path):
    """
    Verifies the DeepFashion2 directory structure.
    Expected:
      root/
        train/
          image/    ← JPEG images
          annos/    ← JSON annotation files (one per image)
        validation/
          image/
          annos/
    """
    required = [
        root / "train" / "image",
        root / "train" / "annos",
        root / "validation" / "image",
        root / "validation" / "annos",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(f"Missing DeepFashion2 paths: {missing}")
        return False
    logger.info("✓ DeepFashion2 structure verified")
    return True


def print_dataset_stats(root: Path, name: str):
    """Counts images and annotation files in a dataset directory."""
    image_count = sum(1 for _ in root.rglob("*.jpg")) + \
                  sum(1 for _ in root.rglob("*.jpeg")) + \
                  sum(1 for _ in root.rglob("*.png"))
    logger.info(f"{name}: {image_count:,} images found in {root}")


def main():
    parser = argparse.ArgumentParser(description="Download DeepFashion datasets")
    parser.add_argument("--source", choices=["kaggle", "gdrive", "both"], default="both",
                        help="Download source: kaggle, gdrive, or both")
    parser.add_argument("--data-dir", default="../data",
                        help="Root directory for downloaded data")
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    if args.source in ("kaggle", "both"):
        df_dir = data_root / "deepfashion"
        logger.info("=" * 60)
        logger.info("Downloading DeepFashion Category & Attribute (Kaggle)")
        logger.info("=" * 60)

        if (df_dir / "img").exists():
            logger.info("DeepFashion already downloaded — skipping")
        else:
            try:
                download_kaggle(df_dir)
            except Exception:
                logger.warning("Kaggle download failed. Please download manually:")
                logger.warning("  1. Go to: https://www.kaggle.com/datasets/vishalbsadanand/deepfashion-1")
                logger.warning(f"  2. Download and extract to: {df_dir}")

        verify_deepfashion_structure(df_dir)
        if df_dir.exists():
            print_dataset_stats(df_dir, "DeepFashion")

    if args.source in ("gdrive", "both"):
        df2_dir = data_root / "deepfashion2"
        logger.info("=" * 60)
        logger.info("Downloading DeepFashion2 (Google Drive)")
        logger.info("=" * 60)

        try:
            download_deepfashion2(df2_dir)
        except Exception:
            logger.warning("Google Drive download failed. Please download manually:")
            logger.warning("  https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2")
            logger.warning(f"  Extract to: {df2_dir}")

        if df2_dir.exists():
            verify_deepfashion2_structure(df2_dir)
            print_dataset_stats(df2_dir, "DeepFashion2")

    logger.info("\nNext steps:")
    logger.info("  python training/preprocess_deepfashion2.py  # Convert to YOLO format")
    logger.info("  python training/preprocess_deepfashion.py   # Convert for ResNet")
    logger.info("  python training/train_yolo.py               # Train YOLOv8")
    logger.info("  python training/train_resnet.py             # Train ResNet50")


if __name__ == "__main__":
    main()
