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
