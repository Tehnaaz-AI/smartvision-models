"""Neural Packaging Form-Factor Classifier for Smart Glasses Vision.

PyTorch Multi-Layer Perceptron trained on geometric, aspect ratio,
and visual shape descriptors of pharmaceutical packaging items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import PackageType

# Class index mapping
CLASS_MAP: List[PackageType] = [
    PackageType.TABLET_STRIP,
    PackageType.SYRUP_BOTTLE,
    PackageType.CARTON_BOX,
    PackageType.OINTMENT_TUBE,
]
CLASS_TO_IDX: Dict[PackageType, int] = {cls: idx for idx, cls in enumerate(CLASS_MAP)}
IDX_TO_CLASS: Dict[int, PackageType] = {idx: cls for idx, cls in enumerate(CLASS_MAP)}


class PackagingClassifierNet(nn.Module):
    """Feedforward neural network for pharmaceutical package classification."""

    def __init__(self, in_features: int = 6, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass emitting unnormalized logits."""
        return self.net(x)


def extract_package_features(
    width: int,
    height: int,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> np.ndarray:
    """Extracts invariant normalized geometric feature vectors."""
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    f_w = float(frame_width)
    f_h = float(frame_height)

    aspect_ratio = w / h
    inv_aspect_ratio = h / w
    normalized_area = (w * h) / (f_w * f_h)
    perimeter_area_ratio = 2.0 * (w + h) / (np.sqrt(w * h) + 1e-5)
    eccentricity = abs(w - h) / (w + h + 1e-5)
    relative_dim = np.sqrt(w**2 + h**2) / np.sqrt(f_w**2 + f_h**2)

    return np.array(
        [
            aspect_ratio,
            inv_aspect_ratio,
            normalized_area,
            perimeter_area_ratio,
            eccentricity,
            relative_dim,
        ],
        dtype=np.float32,
    )


class NeuralPackageClassifier:
    """Inference wrapper for trained PackagingClassifierNet."""

    def __init__(self, weights_path: Optional[str | Path] = None):
        self.model = PackagingClassifierNet()
        self.model.eval()
        self.is_trained = False

        if weights_path is not None and Path(weights_path).exists():
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
                self.is_trained = True
            except Exception:
                self.is_trained = False

    def predict(
        self,
        width: int,
        height: int,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> Tuple[PackageType, float, Dict[str, float]]:
        """Predicts packaging class and confidence probabilities."""
        if width <= 0 or height <= 0:
            return PackageType.UNKNOWN, 0.0, {}

        features = extract_package_features(width, height, frame_width, frame_height)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = IDX_TO_CLASS[pred_idx]

        class_probs = {IDX_TO_CLASS[i].value: float(probs[i]) for i in range(len(CLASS_MAP))}
        return pred_class, round(confidence, 4), class_probs
