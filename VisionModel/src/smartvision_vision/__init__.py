"""SmartVision Computer Vision (CV) Module.

Detects and classifies medicine packaging (strips, bottles, boxes)
to isolate regions of interest for OCR text extraction.
"""

from .classifier import PackageType, PackageClassifier
from .detector import PackageDetector, DetectionResult, get_package_detector

__all__ = [
    "PackageType",
    "PackageClassifier",
    "PackageDetector",
    "DetectionResult",
    "get_package_detector",
]
