"""
Feature Extractor — Fine-tuned ResNet50 for fashion feature extraction.

Model priority:
  1. Fine-tuned DeepFashion weights  (../models/resnet/fashion_resnet50.pth)
  2. ImageNet pretrained ResNet50                    (fallback)
  3. Deterministic mock extractor                    (dev fallback)

Fine-tuned model outputs:
  - 2048-dim L2-normalized feature vector (for cosine similarity)
  - 50-category classification probabilities
  - Top-5 category predictions

FAISS index (../models/faiss/fashion.index) enables fast retrieval
of visually similar items from the 52K In-Shop Clothes benchmark.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).parent
_MODELS_DIR  = _THIS_DIR.parent / "models"
RESNET_WEIGHTS = _MODELS_DIR / "resnet" / "fashion_resnet50.pth"
LABEL_MAP_PATH = _MODELS_DIR / "resnet" / "category_labels.json"
FAISS_INDEX_PATH = _MODELS_DIR / "faiss" / "fashion.index"
FAISS_META_PATH  = _MODELS_DIR / "faiss" / "metadata.json"

# ── Standard ImageNet normalization ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FeatureExtractor:
    """
    Fine-tuned ResNet50 feature extractor + FAISS retrieval engine.

    On load:
      1. Loads fine-tuned fashion_resnet50.pth weights if available
      2. Builds FAISS index for in-shop retrieval if index file exists
    """

    def __init__(self, weights_path: str = None):
        self.model       = None
        self.transform   = None
        self.device      = None
        self.is_finetuned= False
        self.class_names : List[str] = []
        self.label_map   : Dict[str, str] = {}
        self.faiss_index = None
        self.faiss_meta  : List[Dict] = []

        self._load_model(weights_path)
        self._load_faiss_index()

    # ── Model loading ─────────────────────────────────────────────

    def _load_model(self, weights_path: str = None):
        """Loads ResNet50. Tries fine-tuned → pretrained → mock."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Resolve weights path
            target = Path(weights_path) if weights_path else None
            if target is None or not target.exists():
                target = RESNET_WEIGHTS if RESNET_WEIGHTS.exists() else None

            if target and target.exists():
                # Fine-tuned model
                checkpoint = torch.load(str(target), map_location=self.device)
                num_cats  = checkpoint.get("num_categories",  50)
                num_attrs = checkpoint.get("num_attributes", 1000)
                self.class_names = checkpoint.get("class_names", [])

                # Re-build architecture and load weights
                import sys
                sys.path.insert(0, str(_THIS_DIR.parent / "training"))
                from train_resnet import build_model
                self.model = build_model(num_cats, num_attrs, pretrained=False)
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_finetuned = True
                logger.info(f"Fine-tuned ResNet50 loaded: {target}")
                logger.info(f"  Categories: {num_cats}, Val Top-1: {checkpoint.get('val_top1', 'N/A')}%")
            else:
                # Fall back to ImageNet pretrained backbone
                logger.info("Fine-tuned weights not found → using ImageNet ResNet50 backbone")
                logger.info("  To train: python training/train_resnet.py")
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                # Backbone only (no classification head)
                self.model = nn.Sequential(*list(resnet.children())[:-1])
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_finetuned = False

            # Standard preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        except Exception as e:
            logger.warning(f"PyTorch unavailable ({e}). Using mock extractor.")
            self.model = None

        # Load label map
        if LABEL_MAP_PATH.exists():
            with open(LABEL_MAP_PATH) as f:
                self.label_map = json.load(f)

    def _load_faiss_index(self):
        """Loads the pre-built FAISS index for fast visual retrieval."""
        if not FAISS_INDEX_PATH.exists():
            logger.info("FAISS index not built yet.")
            logger.info("  To build: python training/build_recommendation_index.py")
            return

        try:
            import faiss
            self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f"FAISS index loaded: {self.faiss_index.ntotal:,} items")

            if FAISS_META_PATH.exists():
                with open(FAISS_META_PATH) as f:
                    self.faiss_meta = json.load(f)
                logger.info(f"FAISS metadata loaded: {len(self.faiss_meta):,} entries")
        except ImportError:
            logger.warning("faiss not installed. Install: pip install faiss-cpu")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")

    # ── Public API ────────────────────────────────────────────────

    def extract(self, image: Image.Image) -> List[float]:
        """
        Extracts a 2048-dim L2-normalized feature vector from a PIL Image.

        Returns:
            List[float] of length 2048
        """
        if self.model is not None:
            return self._extract_with_model(image)
        return self._mock_extract(image)

    def classify_category(self, image: Image.Image,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predicts clothing category probabilities using the classification head.

        Returns:
            List of {category, probability} dicts sorted by probability desc.
            Returns empty list if fine-tuned model not available.
        """
        if not self.is_finetuned or self.model is None:
            return []

        import torch
        import torch.nn.functional as F

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cat_logits, _, _ = self.model(tensor)
            probs = F.softmax(cat_logits, dim=1).squeeze()

        top_k = min(top_k, len(probs))
        top_probs, top_ids = probs.topk(top_k)

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_ids.cpu().numpy()):
            cat_name = self.label_map.get(str(idx),
                       self.class_names[idx] if idx < len(self.class_names) else f"category_{idx}")
            results.append({
                "category": cat_name,
                "probability": round(float(prob), 4),
                "rank": len(results) + 1,
            })
        return results

    def find_similar(self, image: Image.Image,
                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Finds the most visually similar items using FAISS nearest-neighbor search.

        Returns:
            List of {item_id, category, score, path} dicts.
            Returns empty list if FAISS index not available.
        """
        if self.faiss_index is None or len(self.faiss_meta) == 0:
            return []

        try:
            import faiss
            feature_vec = self.extract(image)
            query = np.array([feature_vec], dtype=np.float32)
            faiss.normalize_L2(query)

            k = min(top_k + 1, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.faiss_meta):
                    continue
                meta = self.faiss_meta[idx]
                results.append({
                    "item_id":  meta.get("item_id", str(idx)),
                    "category": meta.get("category", "unknown"),
                    "score":    round(float(dist), 4),
                    "path":     meta.get("path", ""),
                    "rank":     len(results) + 1,
                })
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def model_info(self) -> Dict:
        """Returns info about the currently loaded models."""
        info = {
            "feature_extractor": {
                "type":       "fine-tuned ResNet50 (DeepFashion)" if self.is_finetuned else
                              "pretrained ResNet50 (ImageNet)" if self.model else "mock",
                "weights":    str(RESNET_WEIGHTS) if self.is_finetuned else "ImageNet",
                "categories": len(self.label_map) or len(self.class_names),
                "feature_dim": 2048,
            },
            "retrieval_index": {
                "type":       "FAISS IndexFlatIP" if self.faiss_index else "not available",
                "items":      self.faiss_index.ntotal if self.faiss_index else 0,
                "path":       str(FAISS_INDEX_PATH),
            },
        }
        # Include training metrics if available
        metrics_path = _MODELS_DIR / "resnet" / "resnet_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                info["feature_extractor"]["training_metrics"] = json.load(f)
        return info

    # ── Private helpers ───────────────────────────────────────────

    def _extract_with_model(self, image: Image.Image) -> List[float]:
        """Runs actual ResNet50 inference to get 2048-dim features."""
        import torch

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_finetuned:
                # Fine-tuned model: use the shared feature from the last layer
                _, _, feats = self.model(tensor)
                vec = feats.squeeze().cpu().numpy()
            else:
                # Backbone only: avgpool output
                feats = self.model(tensor)
                vec = feats.squeeze().cpu().numpy()

        # L2-normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def _mock_extract(self, image: Image.Image) -> List[float]:
        """Deterministic mock feature vector seeded by image pixel statistics."""
        small  = image.resize((32, 32)).convert("RGB")
        pixels = np.array(small, dtype=np.float32) / 255.0
        mean_r = pixels[:, :, 0].mean()
        mean_g = pixels[:, :, 1].mean()
        mean_b = pixels[:, :, 2].mean()
        seed   = int((mean_r * 1000 + mean_g * 100 + mean_b * 10) * 1000) % (2**31)
        rng    = np.random.default_rng(seed)
        vec    = rng.normal(loc=np.mean([mean_r, mean_g, mean_b]), scale=0.15, size=2048).astype(np.float32)
        norm   = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
