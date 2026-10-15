"""Convenience launcher for the SmartVision OCR interactive demo."""

import sys
from pathlib import Path

# Add project src directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smartvision_ocr.demo.server import main

if __name__ == "__main__":
    main()
