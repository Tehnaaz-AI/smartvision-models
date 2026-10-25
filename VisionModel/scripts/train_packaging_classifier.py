"""Supervised Training Script for SmartVision Packaging Form-Factor Classifier.

Trains a PyTorch neural network to categorize pharmaceutical packaging
(TABLET_STRIP, SYRUP_BOTTLE, CARTON_BOX, OINTMENT_TUBE) from wearable camera visual features.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_vision.classifier import PackageType
from smartvision_vision.neural_classifier import (
    CLASS_MAP,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    PackagingClassifierNet,
    extract_package_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vision")


def generate_synthetic_packaging_dataset(samples_per_class: int = 300, seed: int = 42):
    """Generates synthetic pharmaceutical packaging geometry features with real-world noise."""
    random.seed(seed)
    np.random.seed(seed)

    features = []
    labels = []

    # 1. TABLET_STRIP (wide blister packs: aspect ratio 1.5 - 3.8)
    for _ in range(samples_per_class):
        h = np.random.uniform(70, 160)
        ar = np.random.uniform(1.45, 3.6) + np.random.normal(0, 0.1)
        w = h * ar
        feat = extract_package_features(int(w), int(h))
        features.append(feat)
        labels.append(CLASS_TO_IDX[PackageType.TABLET_STRIP])

    # 2. SYRUP_BOTTLE (tall vertical bottles: aspect ratio 0.35 - 0.68)
    for _ in range(samples_per_class):
        h = np.random.uniform(180, 420)
        ar = np.random.uniform(0.35, 0.65) + np.random.normal(0, 0.05)
        w = h * ar
        feat = extract_package_features(int(w), int(h))
        features.append(feat)
        labels.append(CLASS_TO_IDX[PackageType.SYRUP_BOTTLE])

    # 3. CARTON_BOX (square-ish to moderately rectangular boxes: aspect ratio 0.72 - 1.35)
    for _ in range(samples_per_class):
        h = np.random.uniform(130, 320)
        ar = np.random.uniform(0.72, 1.35) + np.random.normal(0, 0.06)
        w = h * ar
        feat = extract_package_features(int(w), int(h))
        features.append(feat)
        labels.append(CLASS_TO_IDX[PackageType.CARTON_BOX])

    # 4. OINTMENT_TUBE (elongated tubes: aspect ratio 2.6 - 5.5)
    for _ in range(samples_per_class):
        h = np.random.uniform(40, 85)
        ar = np.random.uniform(2.5, 5.2) + np.random.normal(0, 0.15)
        w = h * ar
        feat = extract_package_features(int(w), int(h))
        features.append(feat)
        labels.append(CLASS_TO_IDX[PackageType.OINTMENT_TUBE])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4):
    """Computes comprehensive classification metrics."""
    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []
    support = []
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1

    per_class = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = np.sum(confusion[:, c]) - tp
        fn = np.sum(confusion[c, :]) - tp
        sup = np.sum(confusion[c, :])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        support.append(sup)

        per_class[IDX_TO_CLASS[c].value] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "support": int(sup),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total = float(np.sum(support))
    weighted_f1 = float(np.sum([f1s[c] * (support[c] / total) for c in range(num_classes)]))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision Packaging Classifier Training")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating pharmaceutical packaging dataset (1200 samples across 4 classes)...")
    X, y = generate_synthetic_packaging_dataset(samples_per_class=300)

    # Train / Val split (80% / 20%)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    logger.info("Dataset split: Train=%d, Validation=%d", len(X_train), len(X_val))

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = PackagingClassifierNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info("Starting training across %d epochs...", args.epochs)
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(X_train)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %2d/%2d | Training Loss: %.4f", epoch, args.epochs, avg_loss)

    # Validation evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val))
        val_preds = torch.argmax(val_logits, dim=-1).numpy()

    metrics = compute_metrics(y_val, val_preds)
    metrics["loss_start"] = round(loss_history[0], 4)
    metrics["loss_final"] = round(loss_history[-1], 4)
    metrics["epochs"] = args.epochs

    logger.info("\n=== Training & Evaluation Results ===")
    logger.info("Overall Accuracy     : %.2f%%", metrics["accuracy"] * 100)
    logger.info("Macro Precision      : %.2f%%", metrics["macro_precision"] * 100)
    logger.info("Macro Recall         : %.2f%%", metrics["macro_recall"] * 100)
    logger.info("Macro F1-Score       : %.2f%%", metrics["macro_f1"] * 100)
    logger.info("Loss Drop            : %.4f -> %.4f (%.1f%% reduction)", metrics["loss_start"], metrics["loss_final"], (metrics["loss_start"] - metrics["loss_final"]) / metrics["loss_start"] * 100)

    for cls_name, cls_m in metrics["per_class"].items():
        logger.info("  [%-14s] Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%% (n=%d)", cls_name, cls_m["precision"] * 100, cls_m["recall"] * 100, cls_m["f1_score"] * 100, cls_m["support"])

    # Save weights & metadata
    weights_path = out_dir / "packaging_classifier.pt"
    torch.save(model.state_dict(), weights_path)
    logger.info("Saved model weights to %s", weights_path)

    metrics_path = out_dir / "packaging_classifier_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
