from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from smartvision_ocr.inference.pipeline import (
    OCRResult,
    RegionResult,
    detect_scripts,
    predict_full_image,
)


def _box(left, top, right, bottom):
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ]


class FakeDetector:
    def __init__(self, polygons=None, scores=None):
        self.polygons = polygons
        self.scores = scores

    def predict(self, image):
        if self.polygons is None:
            return [SimpleNamespace(json={"res": {"dt_polys": [], "dt_scores": []}})]
        return [
            SimpleNamespace(
                json={
                    "res": {
                        "dt_polys": np.array(self.polygons, dtype=np.int16),
                        "dt_scores": np.array(self.scores, dtype=np.float32)
                        if self.scores is not None
                        else None,
                    }
                }
            )
        ]


def test_detect_scripts_identifies_target_languages():
    assert detect_scripts("నమస్కారం") == ["Telugu"]
    assert detect_scripts("नमस्ते भारत") == ["Devanagari"]
    assert detect_scripts("SmartVision OCR") == ["Latin"]
    assert detect_scripts("SmartVision స్మార్ట్ విజన్") == ["Latin", "Telugu"]
    assert detect_scripts("12345 !?+-") == []


def test_ocr_result_serialization():
    region = RegionResult(
        region_index=0,
        reading_order=0,
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
        recognized_text="STOP",
        detector_score=0.98,
        scripts=["Latin"],
    )
    result = OCRResult(
        text="STOP",
        regions=[region],
        image_path="test.png",
        image_size=(100, 50),
        detected_scripts=["Latin"],
        timing_ms={"detection_ms": 12.5, "recognition_ms": 25.0, "total_ms": 38.0},
    )

    data = result.to_dict()
    assert data["text"] == "STOP"
    assert len(data["regions"]) == 1
    assert data["regions"][0]["recognized_text"] == "STOP"
    assert data["detected_scripts"] == ["Latin"]
    assert "STOP" in result.to_json()


def test_predict_full_image_missing_file():
    with pytest.raises(FileNotFoundError, match="Input image does not exist"):
        predict_full_image("non_existent_image_12345.jpg")


def test_predict_full_image_no_text_regions(monkeypatch):
    image = Image.new("RGB", (40, 20), (255, 255, 255))
    detector = FakeDetector()

    result = predict_full_image(
        image,
        detector=detector,
        model=object(),
        processor=object(),
    )

    assert result.status == "no_text_regions"
    assert result.text == ""
    assert result.regions == []
    assert result.timing_ms["recognition_ms"] == 0.0


def test_predict_full_image_orders_and_reconstructs_text(monkeypatch):
    image = Image.new("RGB", (60, 40), (255, 255, 255))
    # Two boxes: box 1 is top-left ("LINE 1"), box 2 is bottom-left ("LINE 2")
    polygons = [
        _box(2, 25, 30, 35),  # bottom
        _box(2, 5, 30, 15),   # top
    ]
    scores = [0.88, 0.95]
    detector = FakeDetector(polygons=polygons, scores=scores)

    # Mock predict_image
    predict_calls = []

    def fake_predict(crop, model=None, processor=None, device=None, model_name_or_path=None):
        predict_calls.append(crop.size)
        return "LINE 1" if len(predict_calls) == 1 else "LINE 2"

    monkeypatch.setattr("smartvision_ocr.inference.pipeline.predict_image", fake_predict)

    result = predict_full_image(
        image,
        detector=detector,
        model=object(),
        processor=object(),
    )

    assert result.status == "success"
    # Reading order sorts top first, then bottom
    assert result.text == "LINE 1\nLINE 2"
    assert len(result.regions) == 2
    assert result.regions[0].recognized_text == "LINE 1"
    assert result.regions[1].recognized_text == "LINE 2"
    assert result.regions[0].reading_order == 0
    assert result.regions[1].reading_order == 1
    assert result.detected_scripts == ["Latin"]
    assert "detection_ms" in result.timing_ms
    assert "recognition_ms" in result.timing_ms


def test_predict_full_image_handles_detector_failure():
    image = Image.new("RGB", (30, 30))
    detector = Mock()
    detector.predict.side_effect = RuntimeError("Detector crash")

    result = predict_full_image(
        image,
        detector=detector,
        model=object(),
        processor=object(),
    )

    assert result.status == "detection_failed"
    assert "Detector crash" in result.error
    assert result.regions == []
