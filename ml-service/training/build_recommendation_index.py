"""
build_recommendation_index.py
==============================
Builds a FAISS approximate nearest-neighbor index from the DeepFashion
In-Shop Clothes Retrieval benchmark.

Pipeline:
  1. Load fine-tuned ResNet50 weights
  2. Batch-extract 2048-dim feature vectors from all 52,712 In-Shop images
  3. L2-normalize vectors (for cosine similarity via inner product)
  4. Build a FAISS IndexFlatIP (exact cosine similarity) or IVFFlat (fast ANN)
  5. Save index + metadata JSON

This index powers the backend's /find-similar endpoint for fast retrieval.

Usage:
  python training/build_recommendation_index.py
  python training/build_recommendation_index.py --images-dir ../data/deepfashion/img/In-shop

Output:
  models/faiss/fashion.index
  models/faiss/metadata.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Output paths
DEFAULT_INDEX_DIR = Path("../models/faiss")
DEFAULT_WEIGHTS   = Path("../models/resnet/fashion_resnet50.pth")


def load_resnet_backbone(weights_path: Path, device):
    """
    Loads the fine-tuned ResNet50 and returns the feature extractor
    (backbone without the classification heads).
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    logger.info(f"Loading ResNet50 weights from: {weights_path}")

    if weights_path.exists():
        checkpoint = torch.load(str(weights_path), map_location=device)
        num_cats  = checkpoint.get("num_categories", 50)
        num_attrs = checkpoint.get("num_attributes", 1000)

        # Reconstruct the FashionResNet50 architecture
        from train_resnet import build_model
        full_model = build_model(num_cats, num_attrs, pretrained=False)
        full_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"  Loaded fine-tuned weights (val Top-1: {checkpoint.get('val_top1', 'N/A')}%)")
        class_names = checkpoint.get("class_names", [])
    else:
        logger.warning(f"Fine-tuned weights not found — using ImageNet pretrained ResNet50")
        logger.warning("  For best accuracy, train first: python training/train_resnet.py")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        class_names = []

        class FallbackBackbone(nn.Module):
            def __init__(self, resnet):
                super().__init__()
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                self.flatten  = nn.Flatten()
            def forward(self, x):
                return self.flatten(self.backbone(x)), None, self.flatten(self.backbone(x))

        full_model = FallbackBackbone(resnet)

    # Feature extractor: backbone only (returns 2048-dim vectors)
    class FeatureExtractorOnly(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            _, _, feats = self.model(x)
            return feats

    extractor = FeatureExtractorOnly(full_model)
    extractor = extractor.to(device)
    extractor.eval()

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return extractor, transform, class_names


def extract_features_batch(extractor, transform, image_paths: list, device,
                            batch_size: int = 64) -> np.ndarray:
    """
    Extracts 2048-dim feature vectors for a list of image paths in batches.
    Returns: float32 array of shape (N, 2048).
    """
    import torch
    from PIL import Image
    from tqdm import tqdm

    all_features = []
    valid_paths  = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        batch_valid  = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)
                batch_valid.append(str(path))
            except Exception as e:
                logger.debug(f"Skipping {path}: {e}")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = extractor(batch)
        feats_np = feats.cpu().numpy().astype(np.float32)
        all_features.append(feats_np)
        valid_paths.extend(batch_valid)

        if (i // batch_size) % 20 == 0:
            logger.info(f"  Features extracted: {len(valid_paths):,} / {len(image_paths):,}")

    if not all_features:
        return np.array([]), []

    return np.vstack(all_features), valid_paths


def build_faiss_index(features: np.ndarray, use_ivf: bool = False, nlist: int = 100):
    """
    Builds a FAISS index for fast nearest-neighbor search.

    Args:
        features:  (N, D) float32 array, L2-normalized
        use_ivf:   Use IVFFlat (faster) vs FlatIP (exact)
        nlist:     IVF cluster count (higher = faster but less accurate)

    Returns:
        faiss.Index
    """
    try:
        import faiss
    except ImportError:
        logger.error("FAISS not installed. Run: pip install faiss-cpu  (or faiss-gpu)")
        return None

    n, d = features.shape
    logger.info(f"Building FAISS index: {n:,} vectors × {d} dims")

    # L2 normalize for cosine similarity via inner product
    faiss.normalize_L2(features)

    if use_ivf and n > 1000:
        logger.info(f"  Using IVFFlat (nlist={nlist}) for fast approximate search")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(features)
    else:
        logger.info("  Using IndexFlatIP (exact cosine similarity)")
        index = faiss.IndexFlatIP(d)

    index.add(features)
    logger.info(f"  Index built: {index.ntotal:,} vectors")
    return index


def save_index(index, index_dir: Path):
    """Saves the FAISS index to disk."""
    try:
        import faiss
    except ImportError:
        return None
    index_path = index_dir / "fashion.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"  FAISS index saved → {index_path}")
    return index_path


def save_metadata(valid_paths: list, image_dir: Path,
                  class_names: list, index_dir: Path) -> Path:
    """
    Saves a metadata JSON mapping FAISS vector index → image info.
    """
    metadata = []
    for i, img_path in enumerate(valid_paths):
        path = Path(img_path)
        # Derive category from directory name (In-shop structure: img/category/item_id/img.jpg)
        parts = path.relative_to(image_dir).parts if image_dir in path.parents else path.parts
        category = parts[-3] if len(parts) >= 3 else "unknown"
        item_id  = parts[-2] if len(parts) >= 2 else path.stem

        metadata.append({
            "index":    i,
            "path":     img_path,
            "category": category,
            "item_id":  item_id,
            "filename": path.name,
        })

    meta_path = index_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    logger.info(f"  Metadata saved ({len(metadata):,} items) → {meta_path}")
    return meta_path


def test_index(index, features: np.ndarray, metadata: list, k: int = 5):
    """Runs a quick sanity-check query on the built index."""
    try:
        import faiss
    except ImportError:
        return

    if features.shape[0] < k:
        return

    query = features[0:1].copy()
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k + 1)  # +1 because query itself is in index

    logger.info(f"\n✓ Index sanity check (query: {metadata[0].get('item_id', '?')}):")
    for rank, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:]), start=1):
        m = metadata[idx] if idx < len(metadata) else {}
        logger.info(f"  Rank {rank}: {m.get('item_id', idx)} (score: {dist:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS recommendation index from DeepFashion")
    parser.add_argument("--images-dir",  default=None,
                        help="Directory containing In-Shop images (auto-detected if not given)")
    parser.add_argument("--weights",   default=str(DEFAULT_WEIGHTS),
                        help="Path to fine-tuned ResNet50 .pth file")
    parser.add_argument("--output",    default=str(DEFAULT_INDEX_DIR),
                        help="Output directory for FAISS index and metadata")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-ivf",   action="store_true",
                        help="Use IVFFlat for faster (approximate) search")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images (for testing)")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch torchvision")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    index_dir = Path(args.output)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── Find images ───────────────────────────────────────────────
    if args.images_dir:
        image_dir = Path(args.images_dir)
    else:
        # Auto-detect In-Shop directory
        candidates = [
            Path("../data/deepfashion/img/In-shop"),
            Path("../data/deepfashion/In-shop"),
            Path("../data/deepfashion/img"),
        ]
        image_dir = next((p for p in candidates if p.exists()), None)
        if image_dir is None:
            logger.warning("No image directory found. Using sample fallback.")
            logger.info("  Provide path with: --images-dir /path/to/deepfashion/img")
            # Build empty index to allow service startup
            _build_empty_index(index_dir)
            return

    logger.info(f"Scanning images in: {image_dir}")
    image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    logger.info(f"  Found {len(image_paths):,} images")

    if args.max_images:
        image_paths = image_paths[:args.max_images]
        logger.info(f"  Limited to {len(image_paths):,} images (--max-images)")

    if not image_paths:
        logger.error("No images found. Check the images directory.")
        return

    # ── Load model ────────────────────────────────────────────────
    extractor, transform, class_names = load_resnet_backbone(
        Path(args.weights), device)

    # ── Extract features ──────────────────────────────────────────
    logger.info(f"\nExtracting features from {len(image_paths):,} images...")
    start = time.time()
    features, valid_paths = extract_features_batch(
        extractor, transform, image_paths, device, args.batch_size)

    logger.info(f"  Extracted {features.shape[0]:,} feature vectors in {time.time()-start:.1f}s")

    if features.shape[0] == 0:
        logger.error("No features extracted — check image directory and model.")
        return

    # ── Build FAISS index ─────────────────────────────────────────
    logger.info("\nBuilding FAISS index...")
    index = build_faiss_index(features.copy(), use_ivf=args.use_ivf)
    if index is None:
        return

    # ── Save ──────────────────────────────────────────────────────
    logger.info("\nSaving index and metadata...")
    save_index(index, index_dir)
    metadata = save_metadata(valid_paths, image_dir, class_names, index_dir)

    # Quick test
    meta_list = json.loads((index_dir / "metadata.json").read_text())
    test_index(index, features, meta_list)

    # Summary
    summary = {
        "num_vectors":    features.shape[0],
        "feature_dim":    features.shape[1],
        "index_type":     "IVFFlat" if args.use_ivf else "FlatIP",
        "index_path":     str(index_dir / "fashion.index"),
        "metadata_path":  str(index_dir / "metadata.json"),
        "build_time_s":   round(time.time() - start, 1),
    }
    with open(index_dir / "index_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✓ FAISS index built: {features.shape[0]:,} items")
    logger.info(f"  Index:    {index_dir / 'fashion.index'}")
    logger.info(f"  Metadata: {index_dir / 'metadata.json'}")
    logger.info("\nNext: python training/evaluate_models.py")


def _build_empty_index(index_dir: Path):
    """Creates an empty placeholder index so the service can start."""
    try:
        import faiss
        d = 2048
        index = faiss.IndexFlatIP(d)
        faiss.write_index(index, str(index_dir / "fashion.index"))
        with open(index_dir / "metadata.json", "w") as f:
            json.dump([], f)
        logger.info("Created empty placeholder FAISS index.")
    except ImportError:
        # Just create the metadata file
        with open(index_dir / "metadata.json", "w") as f:
            json.dump([], f)


if __name__ == "__main__":
    main()
