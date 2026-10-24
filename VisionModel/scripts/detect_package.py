"""CLI Demonstration script for VisionModel Package Detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_vision.detector import get_package_detector


def create_synthetic_blister_pack() -> np.ndarray:
    """Generates a synthetic tablet blister pack image for demonstration."""
    img = np.full((400, 600, 3), 220, dtype=np.uint8)  # Light background table
    # Draw metallic blister strip
    cv2.rectangle(img, (150, 100), (450, 300), (180, 180, 185), -1)
    cv2.rectangle(img, (150, 100), (450, 300), (80, 80, 80), 2)
    # Draw pill pockets
    for row in [150, 250]:
        for col in [200, 300, 400]:
            cv2.circle(img, (col, row), 22, (240, 240, 250), -1)
            cv2.circle(img, (col, row), 22, (100, 100, 100), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="SmartVision Packaging Detector Demo")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="detected_package.jpg", help="Output path for crop")
    args = parser.parse_args()

    detector = get_package_detector()

    if args.image and Path(args.image).exists():
        img = cv2.imread(args.image)
        source = args.image
    else:
        print("No image provided. Using synthetic medicine blister pack for demonstration.")
        img = create_synthetic_blister_pack()
        source = "synthetic_blister_pack"

    result = detector.detect(img)

    if result.success and result.bbox:
        print(f"\n[SmartVision CV] Detection from {source}:")
        print(f"  Package Type : {result.package_type.value}")
        print(f"  Confidence   : {result.confidence:.2f}")
        print(f"  Bounding Box : x={result.bbox.x}, y={result.bbox.y}, w={result.bbox.width}, h={result.bbox.height}")

        if result.cropped_image is not None:
            cv2.imwrite(args.output, result.cropped_image)
            print(f"  Cropped Box  : Saved to {args.output} ({result.cropped_image.shape[1]}x{result.cropped_image.shape[0]}px)")
    else:
        print(f"Detection failed: {result.error_message}")


if __name__ == "__main__":
    main()
