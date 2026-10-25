"""Adaptive Keyframe Quality Scorer Calibration & Training for Smart Glasses Video.

Calibrates an optimal multi-variate quality estimator on smart-glasses camera frames
under motion blur, blister-pack specular glare, underexposure, and illumination shifts.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_video.frame_quality import FrameQualityAssessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_video")


def generate_synthetic_frame_dataset(samples: int = 1000, seed: int = 42):
    """Generates synthetic camera frames across 5 camera capture regimes."""
    random.seed(seed)
    np.random.seed(seed)

    base = np.full((240, 320, 3), 220, dtype=np.uint8)
    cv2.putText(base, "PARACETAMOL 500 MG", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(base, "Exp: 12/2027  Batch: T942", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    X_features = []
    y_labels = []

    assessor = FrameQualityAssessor()

    for i in range(samples):
        # 50% usable, 50% degraded across 4 defect regimes
        is_usable = (i % 2 == 0)

        if is_usable:
            # 1. High quality sharp frame (Usable = 1)
            noise = np.random.normal(0, 4, base.shape).astype(np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            label = 1
        else:
            defect = (i // 2) % 4
            if defect == 0:
                # 2. Motion blur / tremor (Usable = 0)
                ksize = random.choice([7, 9, 11, 13, 17])
                frame = cv2.GaussianBlur(base, (ksize, ksize), 0)
                label = 0
            elif defect == 1:
                # 3. Blister pack specular glare (Usable = 0)
                frame = base.copy()
                cx, cy = np.random.randint(60, 260), np.random.randint(50, 190)
                cv2.ellipse(frame, (cx, cy), (60, 30), np.random.randint(0, 180), 0, 360, (255, 255, 255), -1)
                label = 0
            elif defect == 2:
                # 4. Underexposed / dim lighting (Usable = 0)
                scale = np.random.uniform(0.08, 0.22)
                frame = (base.astype(np.float32) * scale).astype(np.uint8)
                label = 0
            else:
                # 5. Overexposed / fluorescent washout (Usable = 0)
                offset = np.random.uniform(70, 120)
                frame = np.clip(base.astype(np.float32) + offset, 0, 255).astype(np.uint8)
                label = 0

        # Extract features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        glare_ratio = float(np.count_nonzero(gray >= 250)) / float(gray.size)
        contrast = float(np.std(gray))

        # Normalized feature vector: [lap_norm, brightness_center_dist, glare_ratio, contrast]
        lap_norm = min(1.0, laplacian_var / 300.0)
        bright_dist = abs(mean_brightness - 128.0) / 128.0
        contrast_norm = min(1.0, contrast / 80.0)

        feat = np.array([lap_norm, bright_dist, glare_ratio, contrast_norm], dtype=np.float32)
        X_features.append(feat)
        y_labels.append(label)

    return np.array(X_features), np.array(y_labels, dtype=np.float32)


class AdaptiveKeyframeLogisticModel(nn.Module):
    """Calibrated logistic scoring network for frame readability probability."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision Adaptive Keyframe Scorer Calibration")
    parser.add_argument("--samples", type=int, default=1000, help="Number of simulated calibration frames")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d simulated video camera frames under 5 real-world regimes...", args.samples)
    X, y = generate_synthetic_frame_dataset(samples=args.samples)

    # Train / Val Split (80% / 20%)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    model = AdaptiveKeyframeLogisticModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    logger.info("Training adaptive quality calibration model over %d epochs...", args.epochs)
    loss_start = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch == 1:
            loss_start = loss.item()
        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Epoch %2d/%2d | BCE Loss: %.4f", epoch, args.epochs, loss.item())

    loss_final = loss.item()

    # Evaluation on held-out validation set
    model.eval()
    with torch.no_grad():
        val_probs = model(torch.from_numpy(X_val)).numpy()
        val_preds = (val_probs >= 0.5).astype(np.float32)

    acc = float(np.mean(val_preds == y_val))
    tp = int(np.sum((val_preds == 1) & (y_val == 1)))
    fp = int(np.sum((val_preds == 1) & (y_val == 0)))
    fn = int(np.sum((val_preds == 0) & (y_val == 1)))
    tn = int(np.sum((val_preds == 0) & (y_val == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # False Blurry Rate (sharp frames discarded as blurry)
    fbr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    # False Sharp Rate (blurry frames falsely accepted)
    fsr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    weights = model.linear.weight.data.squeeze().tolist()
    bias = float(model.linear.bias.data.item())

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_sharp_rate_fsr": round(fsr, 4),
        "false_blurry_rate_fbr": round(fbr, 4),
        "loss_start": round(loss_start, 4),
        "loss_final": round(loss_final, 4),
        "calibrated_weights": {
            "laplacian_weight": round(weights[0], 4),
            "brightness_penalty_weight": round(weights[1], 4),
            "glare_penalty_weight": round(weights[2], 4),
            "contrast_weight": round(weights[3], 4),
            "bias": round(bias, 4),
        },
    }

    logger.info("\n=== VideoModel Keyframe Quality Calibration Results ===")
    logger.info("Overall Accuracy     : %.2f%%", metrics["accuracy"] * 100)
    logger.info("Precision (Usable)   : %.2f%%", metrics["precision"] * 100)
    logger.info("Recall (Usable)      : %.2f%%", metrics["recall"] * 100)
    logger.info("F1-Score             : %.2f%%", metrics["f1_score"] * 100)
    logger.info("False Sharp Rate (FSR): %.2f%% (Goal < 5%%)", metrics["false_sharp_rate_fsr"] * 100)
    logger.info("Loss Drop            : %.4f -> %.4f (%.1f%% reduction)", loss_start, loss_final, (loss_start - loss_final) / loss_start * 100)

    # Save model weights and metrics
    scorer_path = out_dir / "quality_scorer.json"
    scorer_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved calibrated quality scorer weights to %s", scorer_path)


if __name__ == "__main__":
    main()
