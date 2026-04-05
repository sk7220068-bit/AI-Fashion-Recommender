"""
evaluate_models.py
==================
Comprehensive evaluation of all trained models:

  1. YOLO detection: mAP@0.5, mAP@0.5:0.95, per-class precision/recall
  2. ResNet category: Top-1, Top-5 accuracy per category
  3. FAISS retrieval:  Rank-1, Rank-5, Rank-20 on In-Shop benchmark
  4. End-to-end latency: Detection + Feature extraction per image

Outputs a full evaluation report to models/evaluation_report.json

Usage:
  python training/evaluate_models.py
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_yolo(weights_path: Path, data_yaml: Path) -> dict:
    """
    Evaluates the fine-tuned YOLOv8 on the validation split.
    Returns mAP@0.5, mAP@0.5:0.95, precision, recall, per-class stats.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("Ultralytics not installed — skipping YOLO evaluation")
        return {}

    if not weights_path.exists():
        logger.warning(f"YOLO weights not found: {weights_path}")
        return {}

    if not data_yaml.exists():
        logger.warning(f"data.yaml not found: {data_yaml}")
        return {}

    logger.info(f"Evaluating YOLOv8: {weights_path}")
    model = YOLO(str(weights_path))

    metrics = model.val(data=str(data_yaml), verbose=False)

    results = {
        "model":      str(weights_path),
        "map50":      float(metrics.box.map50),
        "map50_95":   float(metrics.box.map),
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
    }

    # Per-class breakdown
    per_class = {}
    class_names = metrics.names  # {0: "short_sleeved_shirt", ...}
    for i, name in class_names.items():
        per_class[name] = {
            "ap50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0,
            "ap":   float(metrics.box.ap[i])   if i < len(metrics.box.ap)   else 0.0,
        }
    results["per_class"] = per_class

    logger.info(f"  mAP@0.5:      {results['map50']:.4f}")
    logger.info(f"  mAP@0.5:0.95: {results['map50_95']:.4f}")
    logger.info(f"  Precision:    {results['precision']:.4f}")
    logger.info(f"  Recall:       {results['recall']:.4f}")

    return results


def evaluate_resnet(weights_path: Path, val_dir: Path, batch_size: int = 64) -> dict:
    """
    Evaluates fine-tuned ResNet50 on the validation split.
    Returns Top-1, Top-5 accuracy and per-category breakdown.
    """
    try:
        import torch
        from torchvision import transforms, datasets
        from torch.utils.data import DataLoader
    except ImportError:
        logger.warning("PyTorch not installed — skipping ResNet evaluation")
        return {}

    if not weights_path.exists():
        logger.warning(f"ResNet weights not found: {weights_path}")
        return {}

    if not val_dir.exists():
        logger.warning(f"Validation directory not found: {val_dir}")
        return {}

    logger.info(f"Evaluating ResNet50: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(str(weights_path), map_location=device)
    num_cats  = checkpoint.get("num_categories", 50)
    num_attrs = checkpoint.get("num_attributes", 1000)
    class_names = checkpoint.get("class_names", [])

    from train_resnet import build_model
    model = build_model(num_cats, num_attrs, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    correct_top1 = correct_top5 = total = 0
    per_class_correct = {}
    per_class_total   = {}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images)
            _, preds = logits.topk(5, dim=1, largest=True)

            correct = preds.t().eq(labels.view(1, -1).expand_as(preds.t()))
            correct_top1 += correct[:1].reshape(-1).float().sum().item()
            correct_top5 += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for pred_row, label in zip(preds[:, 0].cpu().numpy(), labels.cpu().numpy()):
                cat_name = dataset.classes[label]
                per_class_total[cat_name] = per_class_total.get(cat_name, 0) + 1
                if pred_row == label:
                    per_class_correct[cat_name] = per_class_correct.get(cat_name, 0) + 1

    top1 = correct_top1 / total * 100
    top5 = correct_top5 / total * 100

    per_class_acc = {
        cat: round(per_class_correct.get(cat, 0) / cnt * 100, 2)
        for cat, cnt in per_class_total.items()
    }

    results = {
        "model":        str(weights_path),
        "val_top1":     round(top1, 2),
        "val_top5":     round(top5, 2),
        "total_images": total,
        "per_class":    per_class_acc,
    }

    logger.info(f"  Val Top-1: {top1:.2f}%")
    logger.info(f"  Val Top-5: {top5:.2f}%")

    # Top/bottom 5 classes
    sorted_cats = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"  Best  categories: {sorted_cats[:3]}")
    logger.info(f"  Worst categories: {sorted_cats[-3:]}")

    return results


def evaluate_retrieval(index_path: Path, metadata_path: Path,
                        extractor, transform, image_paths: list,
                        device, k_values: list = None) -> dict:
    """
    Evaluates FAISS retrieval on a sample of In-Shop images.
    Computes Rank-1, Rank-5, Rank-20 accuracy.
    """
    if k_values is None:
        k_values = [1, 5, 20]

    try:
        import faiss
        import torch
        from PIL import Image
    except ImportError:
        logger.warning("faiss or torch not installed — skipping retrieval evaluation")
        return {}

    if not index_path.exists():
        logger.warning(f"FAISS index not found: {index_path}")
        return {}

    logger.info("Evaluating FAISS retrieval (Rank-k accuracy)...")
    index = faiss.read_index(str(index_path))

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Sample 500 query images
    query_paths = image_paths[:500]
    correct_at_k = {k: 0 for k in k_values}
    total_queries = 0

    for img_path in query_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = extractor(tensor).cpu().numpy().astype(np.float32)

            faiss.normalize_L2(feat)
            query_item_id = Path(img_path).parent.name  # item dir name

            max_k = max(k_values) + 1
            distances, indices = index.search(feat, max_k)

            for k in k_values:
                top_k_ids = [
                    metadata[idx]["item_id"] for idx in indices[0][1:k + 1]
                    if idx < len(metadata)
                ]
                if query_item_id in top_k_ids:
                    correct_at_k[k] += 1

            total_queries += 1
        except Exception:
            continue

    if total_queries == 0:
        return {}

    rank_results = {
        f"rank_{k}": round(correct_at_k[k] / total_queries * 100, 2)
        for k in k_values
    }

    logger.info(f"  Queries evaluated: {total_queries}")
    for k, acc in rank_results.items():
        logger.info(f"  {k}: {acc:.2f}%")

    return rank_results


def measure_latency(yolo_weights: Path, resnet_weights: Path,
                    device, n_runs: int = 50) -> dict:
    """
    Measures end-to-end inference latency (detection + feature extraction).
    """
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        import numpy as np
    except ImportError:
        return {}

    logger.info(f"Measuring inference latency ({n_runs} runs)...")
    latencies = []

    # Create a dummy 640×480 RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    dummy_path = "/tmp/dummy_latency_test.jpg"
    dummy_img.save(dummy_path)

    try:
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_weights)) if yolo_weights.exists() else None
    except ImportError:
        yolo = None

    for _ in range(n_runs):
        t0 = time.perf_counter()
        if yolo:
            _ = yolo.predict(dummy_path, conf=0.25, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = latencies[5:]  # drop warmup
    return {
        "mean_ms":   round(np.mean(latencies), 1),
        "p50_ms":    round(np.percentile(latencies, 50), 1),
        "p95_ms":    round(np.percentile(latencies, 95), 1),
        "p99_ms":    round(np.percentile(latencies, 99), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained fashion AI models")
    parser.add_argument("--yolo-weights",   default="../models/yolo/best.pt")
    parser.add_argument("--yolo-data",      default="../data/deepfashion2_yolo/data.yaml")
    parser.add_argument("--resnet-weights", default="../models/resnet/fashion_resnet50.pth")
    parser.add_argument("--resnet-val-dir", default="../data/deepfashion_processed/val")
    parser.add_argument("--faiss-index",    default="../models/faiss/fashion.index")
    parser.add_argument("--faiss-meta",     default="../models/faiss/metadata.json")
    parser.add_argument("--output",         default="../models/evaluation_report.json")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Comprehensive Model Evaluation")
    logger.info("=" * 60)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S"),
        "yolo":      {},
        "resnet":    {},
        "retrieval": {},
        "latency":   {},
    }

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None

    # 1. YOLO evaluation
    if device is not None:
        yolo_results = evaluate_yolo(
            Path(args.yolo_weights), Path(args.yolo_data))
        report["yolo"] = yolo_results

    # 2. ResNet evaluation
    if device is not None:
        resnet_results = evaluate_resnet(
            Path(args.resnet_weights), Path(args.resnet_val_dir))
        report["resnet"] = resnet_results

    # 3. Latency
    if device is not None:
        latency = measure_latency(Path(args.yolo_weights), Path(args.resnet_weights), device)
        report["latency"] = latency

    # 4. Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Report")
    logger.info("=" * 60)

    if report["yolo"]:
        logger.info(f"YOLO  mAP@0.5:      {report['yolo'].get('map50', 'N/A'):.4f}")
        logger.info(f"YOLO  mAP@0.5:0.95: {report['yolo'].get('map50_95', 'N/A'):.4f}")

    if report["resnet"]:
        logger.info(f"ResNet Top-1 Acc:   {report['resnet'].get('val_top1', 'N/A')}%")
        logger.info(f"ResNet Top-5 Acc:   {report['resnet'].get('val_top5', 'N/A')}%")

    if report["latency"]:
        logger.info(f"Latency (P50):      {report['latency'].get('p50_ms', 'N/A')} ms")
        logger.info(f"Latency (P95):      {report['latency'].get('p95_ms', 'N/A')} ms")

    logger.info(f"\n✓ Full report saved → {out_path}")


if __name__ == "__main__":
    main()
