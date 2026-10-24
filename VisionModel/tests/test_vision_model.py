"""Unit Tests for SmartVision Computer Vision (VisionModel)."""

import cv2
import numpy as np
import pytest

from smartvision_vision.classifier import PackageClassifier, PackageType
from smartvision_vision.detector import PackageDetector, BoundingBox, get_package_detector


def test_package_classification_aspect_ratios():
    classifier = PackageClassifier()

    # Wide tablet strip (300 x 150 -> aspect 2.0)
    t1, conf1 = classifier.classify_by_geometry(300, 150)
    assert t1 == PackageType.TABLET_STRIP
    assert conf1 > 0.8

    # Tall syrup bottle (100 x 200 -> aspect 0.5)
    t2, conf2 = classifier.classify_by_geometry(100, 200)
    assert t2 == PackageType.SYRUP_BOTTLE
    assert conf2 > 0.75

    # Square carton box (200 x 200 -> aspect 1.0)
    t3, conf3 = classifier.classify_by_geometry(200, 200)
    assert t3 == PackageType.CARTON_BOX
    assert conf3 > 0.75


def test_bounding_box_methods():
    box = BoundingBox(x=10, y=20, width=100, height=50)
    assert box.area == 5000
    assert box.aspect_ratio == 2.0
    assert box.as_xyxy() == (10, 20, 110, 70)


def test_package_detector_on_synthetic_frame():
    detector = get_package_detector()

    # Generate synthetic image with a rectangular object in the middle
    frame = np.full((400, 500, 3), 200, dtype=np.uint8)
    cv2.rectangle(frame, (100, 80), (400, 280), (50, 50, 50), -1)

    result = detector.detect(frame)
    assert result.success is True
    assert result.bbox is not None
    assert result.bbox.width > 150
    assert result.cropped_image is not None
    assert result.cropped_image.shape[0] > 0
    assert result.cropped_image.shape[1] > 0


def test_package_detector_invalid_input():
    detector = get_package_detector()
    result = detector.detect("non_existent_image_file_999.jpg")
    assert result.success is False
    assert "Failed to load" in result.error_message
