"""
train_yolo.py
=============
Fine-tunes YOLOv8s on DeepFashion2 for clothing item detection.

Model: YOLOv8s (small — best accuracy/speed balance for fashion)
Data:  DeepFashion2 YOLO format (13 clothing categories, ~491K images)
Goal:  mAP@0.5 > 0.80, mAP@0.5:0.95 > 0.55

Prerequisites:
  pip install ultralytics pyyaml
  python training/preprocess_deepfashion2.py  ← run first

Usage:
  python training/train_yolo.py
  python training/train_yolo.py --model yolov8m.pt  # larger model
  python training/train_yolo.py --resume             # resume checkpoint
  python training/train_yolo.py --data /custom/data.yaml

On Kaggle (free GPU):
  Use notebooks/deepfashion_training.ipynb with GPU accelerator enabled.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_training_config(config_path: Path) -> dict:
    """Loads the centralized training YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_data_yaml(data_yaml_path: Path) -> bool:
    """Checks that the YOLO data.yaml exists and has required fields."""
    if not data_yaml_path.exists():
        logger.error(f"data.yaml not found: {data_yaml_path}")
        logger.info("Run first: python training/preprocess_deepfashion2.py")
        return False
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["path", "train", "val", "names"]
    missing = [k for k in required if k not in cfg]
    if missing:
        logger.error(f"data.yaml missing keys: {missing}")
        return False
    nc = cfg.get("nc", 0)
    logger.info(f"✓ data.yaml verified: {nc} classes")
    return True


def train(args):
    """Main YOLOv8 fine-tuning function."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        return None

    # ── Load config ───────────────────────────────────────────────
    config_path = Path(__file__).parent / "config" / "training_config.yaml"
    cfg = {}
    if config_path.exists():
        cfg = load_training_config(config_path).get("yolo", {})

    # ── Resolve parameters (CLI overrides config) ─────────────────
    data_yaml  = Path(args.data)   if args.data   else Path(cfg.get("data", "../data/deepfashion2_yolo/data.yaml"))
    model_name = args.model        if args.model  else cfg.get("base_model", "yolov8s.pt")
    epochs     = args.epochs       if args.epochs else cfg.get("epochs", 100)
    batch      = args.batch        if args.batch  else cfg.get("batch", 16)
    imgsz      = args.imgsz        if args.imgsz  else cfg.get("imgsz", 640)
    output_dir = Path(args.output) if args.output else Path(cfg.get("project", "../models/yolo"))

    logger.info("=" * 60)
    logger.info("YOLOv8 Fine-Tuning on DeepFashion2")
    logger.info("=" * 60)
    logger.info(f"  Model:    {model_name}")
    logger.info(f"  Data:     {data_yaml}")
    logger.info(f"  Epochs:   {epochs}")
    logger.info(f"  Batch:    {batch}")
    logger.info(f"  ImgSz:    {imgsz}")
    logger.info(f"  Output:   {output_dir}")

    if not verify_data_yaml(data_yaml):
        return None

    # ── Load pre-trained model ────────────────────────────────────
    if args.resume:
        # Resume from last checkpoint
        resume_path = output_dir / cfg.get("name", "deepfashion2_finetune") / "weights" / "last.pt"
        if resume_path.exists():
            logger.info(f"Resuming from: {resume_path}")
            model = YOLO(str(resume_path))
        else:
            logger.warning(f"Resume checkpoint not found ({resume_path}), starting fresh")
            model = YOLO(model_name)
    else:
        logger.info(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────
    logger.info("\nStarting training...")
    start_time = time.time()

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=cfg.get("lr0", 0.01),
            lrf=cfg.get("lrf", 0.01),
            momentum=cfg.get("momentum", 0.937),
            weight_decay=cfg.get("weight_decay", 0.0005),
            warmup_epochs=cfg.get("warmup_epochs", 3),
            cos_lr=cfg.get("cos_lr", True),
            hsv_h=cfg.get("hsv_h", 0.015),
            hsv_s=cfg.get("hsv_s", 0.7),
            hsv_v=cfg.get("hsv_v", 0.4),
            fliplr=cfg.get("fliplr", 0.5),
            mosaic=cfg.get("mosaic", 1.0),
            mixup=cfg.get("mixup", 0.1),
            patience=cfg.get("patience", 30),
            project=str(output_dir),
            name=cfg.get("name", "deepfashion2_finetune"),
            save_period=cfg.get("save_period", 10),
            val=True,
            plots=True,
            verbose=True,
            device=cfg.get("device", ""),
            workers=cfg.get("workers", 8),
        )

        elapsed = time.time() - start_time
        logger.info(f"\nTraining completed in {elapsed/3600:.1f} hours")

        # ── Extract and save metrics ──────────────────────────────
        best_weights = output_dir / cfg.get("name", "deepfashion2_finetune") / "weights" / "best.pt"

        metrics = {
            "model": model_name,
            "epochs_trained": epochs,
            "training_time_hours": round(elapsed / 3600, 2),
            "best_weights": str(best_weights),
            "map50":      float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95":   float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision":  float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":     float(results.results_dict.get("metrics/recall(B)", 0)),
        }

        metrics_path = output_dir / "yolo_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("\n📊 Training Results:")
        logger.info(f"  mAP@0.5:      {metrics['map50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        logger.info(f"  Precision:    {metrics['precision']:.4f}")
        logger.info(f"  Recall:       {metrics['recall']:.4f}")
        logger.info(f"  Best weights: {best_weights}")

        # ── Copy best.pt to standard location ─────────────────────
        standard_dest = Path("../models/yolo/best.pt")
        standard_dest.parent.mkdir(parents=True, exist_ok=True)
        if best_weights.exists():
            import shutil
            shutil.copy2(best_weights, standard_dest)
            logger.info(f"  Copied best.pt → {standard_dest}")

        return metrics

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return None


def export_model(weights_path: Path, formats: list = None):
    """
    Exports trained YOLO model to additional formats for deployment.
    Supported: onnx, tflite, coreml, torchscript
    """
    if formats is None:
        formats = ["onnx"]

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        for fmt in formats:
            logger.info(f"Exporting to {fmt}...")
            model.export(format=fmt)
            logger.info(f"  Exported {fmt}")
    except Exception as e:
        logger.warning(f"Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on DeepFashion2")
    parser.add_argument("--model",   default=None, help="Base model (e.g., yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--data",    default=None, help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--imgsz",   type=int, default=None)
    parser.add_argument("--output",  default=None, help="Output directory")
    parser.add_argument("--resume",  action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--export",  nargs="*", default=None,
                        help="Export formats after training (e.g., --export onnx tflite)")
    args = parser.parse_args()

    metrics = train(args)

    if metrics and args.export is not None:
        weights = Path(metrics["best_weights"])
        if weights.exists():
            export_model(weights, args.export or ["onnx"])

    if metrics:
        logger.info("\n✓ YOLOv8 fine-tuning pipeline complete!")
        logger.info("Next: python training/train_resnet.py")


if __name__ == "__main__":
    main()
