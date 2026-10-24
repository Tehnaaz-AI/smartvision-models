"""Computer Vision Package Detector for SmartVision."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .classifier import PackageClassifier, PackageType

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected objects."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return float(self.width) / float(self.height) if self.height > 0 else 0.0

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class DetectionResult:
    """Result of package detection on an input image."""

    bbox: Optional[BoundingBox]
    package_type: PackageType
    confidence: float
    cropped_image: Optional[np.ndarray] = None
    success: bool = True
    error_message: Optional[str] = None


class PackageDetector:
    """Locates and extracts medication packaging from smart-glasses frames."""

    def __init__(self, min_area_threshold: float = 0.02):
        self.min_area_threshold = min_area_threshold
        self.classifier = PackageClassifier()

    def detect(self, image_input: np.ndarray | str | Path) -> DetectionResult:
        """Detects the dominant medicine package in the frame."""
        image = self._load_image(image_input)
        if image is None:
            return DetectionResult(
                bbox=None,
                package_type=PackageType.UNKNOWN,
                confidence=0.0,
                success=False,
                error_message="Failed to load or decode image.",
            )

        h, w = image.shape[:2]
        total_frame_area = h * w

        # Convert to LAB to extract color saliency (packaging stands out from background skin/table)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if len(image.shape) == 3 else None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-scale edge detection
        edges = cv2.Canny(blurred, 30, 140)

        # Morphological closing to seal packaging contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = 0.0
        center_x, center_y = w / 2.0, h / 2.0

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            area_ratio = area / float(total_frame_area)
            if area_ratio >= self.min_area_threshold:
                # Foveal proximity prior: objects centered in glasses field of view have higher priority
                box_center_x = x + (bw / 2.0)
                box_center_y = y + (bh / 2.0)
                dist_from_center = np.hypot(box_center_x - center_x, box_center_y - center_y)
                max_diag = np.hypot(w, h) / 2.0
                centrality = max(0.2, 1.0 - (dist_from_center / max_diag))

                # Combined package saliency score
                score = (area_ratio * 0.5) + (centrality * 0.5)
                if score > best_score:
                    best_score = score
                    best_bbox = BoundingBox(x=x, y=y, width=bw, height=bh)

        # Fallback to center region if no sharp external contours found
        if best_bbox is None:
            # Smart-glasses wearer generally holds package near the center
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            best_bbox = BoundingBox(
                x=margin_x,
                y=margin_y,
                width=w - (2 * margin_x),
                height=h - (2 * margin_y),
            )
            conf = 0.50
        else:
            conf = min(0.95, round(0.60 + (best_bbox.area / float(total_frame_area)) * 0.4, 2))

        # Crop package
        x1, y1, x2, y2 = best_bbox.as_xyxy()
        cropped = image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy()

        # Classify package type
        pkg_type, type_conf = self.classifier.classify_by_geometry(
            best_bbox.width, best_bbox.height
        )

        return DetectionResult(
            bbox=best_bbox,
            package_type=pkg_type,
            confidence=round((conf + type_conf) / 2.0, 2),
            cropped_image=cropped,
            success=True,
        )

    def _load_image(self, img_input: np.ndarray | str | Path) -> Optional[np.ndarray]:
        """Safely loads image input into BGR numpy array."""
        if isinstance(img_input, np.ndarray):
            return img_input
        try:
            path_str = str(img_input)
            img = cv2.imread(path_str)
            return img
        except Exception as err:
            logger.error(f"Image load exception: {err}")
            return None


def get_package_detector() -> PackageDetector:
    """Factory helper to obtain a PackageDetector instance."""
    return PackageDetector()
