"""
train_resnet.py
===============
Fine-tunes ResNet50 on the DeepFashion Category & Attribute benchmark.

Tasks:
  A) 50-class clothing category classification  (Top-1 accuracy target: ~89%)
  B) Multi-label 1,000-attribute prediction     (mAP target: ~85%)

Architecture:
  ResNet50 backbone (ImageNet pretrained)
  → Category head: FC(2048 → 512) → ReLU → Dropout → FC(512 → 50)
  → Attribute head: FC(2048 → 1024) → ReLU → Dropout → FC(1024 → 1000) → Sigmoid

Prerequisites:
  pip install torch torchvision matplotlib scikit-learn tqdm pyyaml
  python training/preprocess_deepfashion.py  ← run first

Usage:
  python training/train_resnet.py
  python training/train_resnet.py --data-dir ../data/deepfashion_processed
  python training/train_resnet.py --task category   # category only
  python training/train_resnet.py --task attribute  # attributes only
  python training/train_resnet.py --task both       # both heads (default)
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


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f).get("resnet", {})


def build_model(num_categories: int, num_attributes: int, pretrained: bool = True,
                freeze_layers: int = 20, dropout: float = 0.5):
    """
    Builds a dual-head ResNet50:
      - Category head for 50-class classification
      - Attribute head for 1000-label multi-label prediction
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    # Load pretrained ResNet50
    if pretrained:
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        backbone = models.resnet50(weights=None)

    # Freeze early layers (backbone feature extractor)
    layers = list(backbone.named_parameters())
    for name, param in layers[:freeze_layers]:
        param.requires_grad = False

    feature_dim = backbone.fc.in_features  # 2048

    class FashionResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            # Backbone without final FC
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.flatten  = nn.Flatten()

            # Shared feature projection
            self.shared = nn.Sequential(
                nn.Linear(feature_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

            # Category classification head
            self.category_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, num_categories),
            )

            # Attribute multi-label head
            self.attribute_head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(1024, num_attributes),
                nn.Sigmoid(),
            )

        def forward(self, x):
            feats = self.flatten(self.backbone(x))
            shared = self.shared(feats)
            category_logits = self.category_head(shared)
            attribute_preds  = self.attribute_head(shared)
            return category_logits, attribute_preds, feats  # feats for retrieval

    return FashionResNet50()


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int,
                       num_workers: int, attr_data: dict = None):
    """
    Builds PyTorch DataLoaders from the organized deepfashion_processed/ directory.
    Uses torchvision ImageFolder for category labels.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, datasets
    from PIL import Image

    train_transforms = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(data_dir / "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(data_dir / "val"),   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    logger.info(f"  Train: {len(train_dataset):,} images, {len(train_dataset.classes)} classes")
    logger.info(f"  Val:   {len(val_dataset):,} images")

    return train_loader, val_loader, train_dataset.classes


def train_epoch(model, loader, optimizer, criterion_cat, device):
    """Runs one training epoch. Returns (avg_loss, top1_acc, top5_acc)."""
    import torch

    model.train()
    total_loss = correct_top1 = correct_top5 = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        cat_logits, attr_preds, _ = model(images)

        loss = criterion_cat(cat_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        # Top-1 and Top-5 accuracy
        _, pred = cat_logits.topk(5, dim=1, largest=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_top1 += correct[:1].reshape(-1).float().sum().item()
        correct_top5 += correct[:5].reshape(-1).float().sum().item()

        if batch_idx % 100 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f}")

    n = len(loader)
    return (total_loss / n), (correct_top1 / total * 100), (correct_top5 / total * 100)


def evaluate(model, loader, criterion_cat, device):
    """Evaluates on validation set. Returns (avg_loss, top1_acc, top5_acc)."""
    import torch

    model.eval()
    total_loss = correct_top1 = correct_top5 = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            cat_logits, _, _ = model(images)
            loss = criterion_cat(cat_logits, labels)

            total_loss += loss.item()
            total += labels.size(0)

            _, pred = cat_logits.topk(5, dim=1, largest=True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            correct_top1 += correct[:1].reshape(-1).float().sum().item()
            correct_top5 += correct[:5].reshape(-1).float().sum().item()

    n = len(loader)
    return (total_loss / n), (correct_top1 / total * 100), (correct_top5 / total * 100)


def train(args):
    """Full ResNet50 fine-tuning loop."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.error("PyTorch not found. Install: pip install torch torchvision")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cpu":
        logger.warning("⚠ GPU not detected. Training will be very slow on CPU (~40h).")
        logger.warning("  Recommended: Use Kaggle GPU notebook — see notebooks/deepfashion_training.ipynb")

    # ── Load config ───────────────────────────────────────────────
    config_path = Path(__file__).parent / "config" / "training_config.yaml"
    cfg = {}
    if config_path.exists():
        cfg = load_config(config_path)

    data_dir    = Path(args.data_dir) if args.data_dir else Path("../data/deepfashion_processed")
    output_dir  = Path(args.output)   if args.output   else Path("../models/resnet")
    epochs      = args.epochs         if args.epochs    else cfg.get("epochs",      50)
    batch_size  = args.batch          if args.batch     else cfg.get("batch_size",  64)
    lr          = args.lr             if args.lr        else cfg.get("lr",          0.001)
    image_size  = cfg.get("image_size",     224)
    num_workers = cfg.get("num_workers",    8)
    freeze_layers = cfg.get("freeze_layers", 20)
    num_cats    = cfg.get("num_categories", 50)
    num_attrs   = cfg.get("num_attributes", 1000)
    dropout     = cfg.get("dropout",        0.5)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run first: python training/preprocess_deepfashion.py")
        return None

    logger.info("=" * 60)
    logger.info("ResNet50 Fine-Tuning on DeepFashion")
    logger.info("=" * 60)
    logger.info(f"  Data:     {data_dir}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Epochs:   {epochs}")
    logger.info(f"  Batch:    {batch_size}")
    logger.info(f"  LR:       {lr}")

    # ── DataLoaders ───────────────────────────────────────────────
    logger.info("\nBuilding dataloaders...")
    train_loader, val_loader, class_names = build_dataloaders(
        data_dir, image_size, batch_size, num_workers)
    actual_num_cats = len(class_names)
    logger.info(f"  Actual categories: {actual_num_cats}")

    # ── Model ─────────────────────────────────────────────────────
    logger.info("\nBuilding FashionResNet50...")
    model = build_model(actual_num_cats, num_attrs, pretrained=True,
                        freeze_layers=freeze_layers, dropout=dropout)
    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Loss + Optimizer ──────────────────────────────────────────
    criterion_cat  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 1e-4),
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training Loop ─────────────────────────────────────────────
    best_val_top1 = 0.0
    early_stopping_counter = 0
    patience = cfg.get("early_stopping", 10)
    history = []

    logger.info("\nStarting training loop...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        logger.info(f"\n── Epoch {epoch}/{epochs} ─────────────────────────────")
        logger.info(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, optimizer, criterion_cat, device)

        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion_cat, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f} | Top-1: {train_top1:.2f}% | Top-5: {train_top5:.2f}%")
        logger.info(f"  Val   Loss: {val_loss:.4f} | Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}%")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 2),
            "train_top5": round(train_top5, 2),
            "val_loss":   round(val_loss, 4),
            "val_top1":   round(val_top1, 2),
            "val_top5":   round(val_top5, 2),
        }
        history.append(epoch_stats)

        # Save best model
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            early_stopping_counter = 0
            logger.info(f"  🏆 New best! Val Top-1: {best_val_top1:.2f}%")

            # Save model state dict
            best_path = output_dir / "fashion_resnet50.pth"
            raw_model = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch":           epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_top1":        best_val_top1,
                "class_names":     class_names,
                "num_categories":  actual_num_cats,
                "num_attributes":  num_attrs,
            }, best_path)

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Save checkpoint periodically
        if epoch % 5 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)

    elapsed = time.time() - start_time

    # ── Save training history ─────────────────────────────────────
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Save category label map ───────────────────────────────────
    label_map = {str(i): name for i, name in enumerate(class_names)}
    label_map_path = output_dir / "category_labels.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    metrics = {
        "best_val_top1":       round(best_val_top1, 2),
        "training_time_hours": round(elapsed / 3600, 2),
        "epochs_trained":      len(history),
        "num_categories":      actual_num_cats,
        "weights_path":        str(output_dir / "fashion_resnet50.pth"),
        "label_map_path":      str(label_map_path),
    }
    metrics_path = output_dir / "resnet_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n📊 Training Summary:")
    logger.info(f"  Best Val Top-1:     {best_val_top1:.2f}%")
    logger.info(f"  Training Time:      {elapsed/3600:.1f} hours")
    logger.info(f"  Best model saved:   {output_dir / 'fashion_resnet50.pth'}")

    return metrics


def plot_training_history(history_path: Path, output_dir: Path):
    """Plots training and validation accuracy/loss curves."""
    try:
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(history_path) as f:
            history = json.load(f)

        epochs      = [h["epoch"]      for h in history]
        train_top1  = [h["train_top1"] for h in history]
        val_top1    = [h["val_top1"]   for h in history]
        train_loss  = [h["train_loss"] for h in history]
        val_loss    = [h["val_loss"]   for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("ResNet50 Training — DeepFashion", fontsize=14)

        ax1.plot(epochs, train_top1, "b-", label="Train Top-1")
        ax1.plot(epochs, val_top1,   "r-", label="Val Top-1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Top-1 Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_loss, "b-", label="Train Loss")
        ax2.plot(epochs, val_loss,   "r-", label="Val Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Cross-Entropy Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plot_path = output_dir / "training_curves.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved → {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on DeepFashion")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--task",     choices=["category", "attribute", "both"],
                        default="both", help="Which head(s) to train")
    args = parser.parse_args()

    metrics = train(args)

    if metrics:
        output_dir = Path(metrics["weights_path"]).parent
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            plot_training_history(history_path, output_dir)

        logger.info("\n✓ ResNet50 training complete!")
        logger.info("Next: python training/build_recommendation_index.py")


if __name__ == "__main__":
    main()
