"""Medicine Packaging Classifier."""

from __future__ import annotations

from enum import Enum
from typing import Tuple


class PackageType(str, Enum):
    """Common visual categories of pharmaceutical packaging."""

    TABLET_STRIP = "TABLET_STRIP"
    SYRUP_BOTTLE = "SYRUP_BOTTLE"
    CARTON_BOX = "CARTON_BOX"
    OINTMENT_TUBE = "OINTMENT_TUBE"
    GENERIC_PACKAGE = "GENERIC_PACKAGE"
    UNKNOWN = "UNKNOWN"


class PackageClassifier:
    """Classifies packaging form-factor using geometric and visual descriptors."""

    _neural_classifier = None

    @classmethod
    def _get_neural_classifier(cls):
        if cls._neural_classifier is None:
            from pathlib import Path
            from .neural_classifier import NeuralPackageClassifier
            weights_path = Path(__file__).resolve().parent.parent.parent / "models" / "packaging_classifier.pt"
            cls._neural_classifier = NeuralPackageClassifier(weights_path=weights_path if weights_path.exists() else None)
        return cls._neural_classifier

    @staticmethod
    def classify_by_geometry(width: int, height: int, area_ratio: float = 1.0) -> Tuple[PackageType, float]:
        """Infers package type based on bounding box aspect ratio and dimensions."""
        if width <= 0 or height <= 0:
            return PackageType.UNKNOWN, 0.0

        aspect_ratio = float(width) / float(height)

        # Tablet strips are typically wide horizontal rectangles (aspect ratio >= 1.4)
        if aspect_ratio >= 1.4:
            return PackageType.TABLET_STRIP, 0.88
        # Bottles are vertical cylinders (aspect ratio between 0.35 and 0.68)
        elif 0.35 <= aspect_ratio <= 0.68:
            return PackageType.SYRUP_BOTTLE, 0.85
        # Boxes/cartons are typically square-ish to moderately rectangular (0.69 to 1.39)
        elif 0.69 < aspect_ratio < 1.4:
            return PackageType.CARTON_BOX, 0.82
        else:
            return PackageType.GENERIC_PACKAGE, 0.70

    @classmethod
    def classify(
        cls,
        width: int,
        height: int,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> Tuple[PackageType, float]:
        """Infers package type using neural classifier if weights available, else fallback."""
        neural = cls._get_neural_classifier()
        if neural.is_trained:
            pkg_type, conf, _ = neural.predict(width, height, frame_width, frame_height)
            return pkg_type, conf
        return cls.classify_by_geometry(width, height)
