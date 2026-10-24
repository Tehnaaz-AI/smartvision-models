# SmartVision — Computer Vision (VisionModel) Module

The **VisionModel** subproject performs medicine package detection, form-factor classification, and region-of-interest (ROI) localization for the SmartVision smart-glasses system.

---

## 1. Role in the Pipeline

When an elderly user holds a medicine package in front of the smart-glasses camera:
1. **Packaging Localization**: Distinguishes the medicine package from hands, tables, or background clutter.
2. **Form-Factor Classification**: Identifies whether the medication is a `TABLET_STRIP`, `SYRUP_BOTTLE`, `CARTON_BOX`, or `OINTMENT_TUBE`.
3. **Region Cropping**: Crops the high-resolution package area and passes it to `OCRModel` for focused multilingual text recognition.

---

## 2. Directory Structure

```text
VisionModel/
├── src/smartvision_vision/
│   ├── __init__.py           # Exports (PackageDetector, PackageClassifier)
│   ├── detector.py           # Contour & edge-based package locator
│   └── classifier.py         # Geometric packaging form-factor classifier
├── scripts/
│   └── detect_package.py     # CLI demo script
├── tests/
│   └── test_vision_model.py  # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd VisionModel
pip install -r requirements.txt
```

### CLI Demo
```bash
python scripts/detect_package.py
# Or with a real camera capture:
python scripts/detect_package.py -i my_medicine.jpg -o crop.jpg
```

### Python API
```python
import cv2
from smartvision_vision import get_package_detector

detector = get_package_detector()
frame = cv2.imread("frame.jpg")
result = detector.detect(frame)

print(f"Package: {result.package_type} (conf: {result.confidence})")
print(f"BBox: {result.bbox.as_xyxy()}")
```
