# SmartVision — Video & Frame Processing (VideoModel) Module

The **VideoModel** subproject implements intelligent camera frame selection and quality filtering for the SmartVision smart-glasses system.

---

## 1. Objectives & Approach

Instead of streaming expensive high-bandwidth video to cloud servers or running heavy 3D spatio-temporal video networks on wearable batteries:
1. **Burst Capture**: The smart glasses capture a 0.5–1.0 second burst of frames when a user holds up medicine.
2. **Sharpness Evaluation**: Computes the Laplacian variance to measure edge high-frequency energy and eliminate motion blur.
3. **Exposure & Glare Checking**: Flags dark frames or specular foil reflections from blister packages.
4. **Keyframe Selection**: Picks the single crispest, glare-free frame to send to `VisionModel` and `OCRModel`.

---

## 2. Directory Structure

```text
VideoModel/
├── src/smartvision_video/
│   ├── __init__.py           # Package exports
│   ├── frame_quality.py      # Laplacian sharpness, brightness, glare ratios
│   └── keyframe_selector.py  # Optimal keyframe burst buffer and ranking
├── scripts/
│   └── simulate_burst_selection.py # CLI simulation demo
├── tests/
│   └── test_video_model.py   # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd VideoModel
pip install -r requirements.txt
```

### CLI Demo
```bash
python scripts/simulate_burst_selection.py --burst-size 12
```

### Python API
```python
from smartvision_video import KeyframeSelector

selector = KeyframeSelector()
result = selector.select_best_frame(camera_frames_burst)

if result.success:
    print(f"Selected frame #{result.best_frame_index} (Quality: {result.best_quality.composite_score})")
    ocr_input = result.best_frame
```
