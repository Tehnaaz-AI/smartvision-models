import csv
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from smartvision_ocr.inference import detect_and_ocr


def _box(left, top, right, bottom):
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ]


def _create_dataset(project_root):
    split_dir = project_root / "data" / "DrugSticker_data" / "test"
    split_dir.mkdir(parents=True)
    for number in range(4):
        Image.new("RGB", (30, 24), (number * 40, number * 20, 90)).save(
            split_dir / f"image_{number:02}.png"
        )
    return split_dir


def test_detector_results_are_sorted_rectified_and_keep_real_scores():
    image = Image.new("RGB", (20, 18))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x, y, 0))
    polygons = np.array(
        [
            _box(10, 10, 18, 15),
            _box(1, 2, 8, 7),
        ],
        dtype=np.int16,
    )
    result = SimpleNamespace(
        json={"res": {"dt_polys": polygons, "dt_scores": np.array([0.71, 0.93])}}
    )

    regions = detect_and_ocr.parse_detector_results([result], image)

    assert [region.polygon[0] for region in regions] == [(1.0, 2.0), (10.0, 10.0)]
    assert [region.detector_score for region in regions] == [0.93, 0.71]
    assert regions[0].crop.size == (7, 5)
    assert regions[0].crop.getpixel((0, 0)) == image.getpixel((1, 2))


def test_order_regions_uses_top_to_bottom_then_left_to_right():
    bottom_left = _box(2, 15, 8, 19)
    top_right = _box(12, 2, 18, 6)
    top_left = _box(1, 2, 7, 6)

    ordered = detect_and_ocr.order_regions(
        [bottom_left, top_right, top_left],
        image_size=(24, 24),
    )

    assert [polygon[0] for polygon in ordered] == [
        (1.0, 2.0),
        (12.0, 2.0),
        (2.0, 15.0),
    ]


def test_detector_polygon_touching_image_boundaries_is_kept():
    image = Image.new("RGB", (20, 18), "white")

    regions = detect_and_ocr.parse_detector_results(
        [SimpleNamespace(json={"res": {"dt_polys": [_box(0, 0, 20, 18)]}})],
        image,
    )

    assert len(regions) == 1
    assert set(regions[0].polygon) == {
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 18.0),
        (0.0, 18.0),
    }
    assert regions[0].crop.size == image.size


def test_partially_outside_polygon_is_clipped_without_inventing_crop_pixels():
    image = Image.new("RGB", (10, 10), "white")
    polygon = [[-2, 3], [5, 1], [8, 7], [1, 9]]

    regions = detect_and_ocr.parse_detector_results(
        [SimpleNamespace(json={"res": {"dt_polys": [polygon]}})],
        image,
    )

    assert len(regions) == 1
    assert len(regions[0].polygon) > 4
    assert all(0 <= x <= image.width and 0 <= y <= image.height
               for x, y in regions[0].polygon)
    assert regions[0].crop.size == (8, 8)
    assert regions[0].crop.getpixel((0, 0)) == image.getpixel((0, 1))


@pytest.mark.parametrize(
    "invalid_polygon",
    [
        [[1, 1], [5, 1], [4, 4]],
        [[1, 1], ["bad", 1], [5, 5], [1, 5]],
    ],
)
def test_malformed_polygon_is_skipped_without_discarding_valid_regions(
    invalid_polygon, caplog
):
    image = Image.new("RGB", (10, 10))
    detector_result = SimpleNamespace(
        json={
            "res": {
                "dt_polys": [invalid_polygon, _box(1, 1, 6, 6)],
                "dt_scores": [0.1, 0.9],
            }
        }
    )

    regions = detect_and_ocr.parse_detector_results([detector_result], image)

    assert len(regions) == 1
    assert regions[0].detector_score == 0.9
    assert "Skipping invalid detected polygon 0" in caplog.text


def test_zero_area_polygon_is_skipped_and_entirely_invalid_result_is_empty(caplog):
    image = Image.new("RGB", (10, 10))
    zero_area = [[1, 1], [3, 3], [5, 5], [7, 7]]
    detector_result = SimpleNamespace(
        json={"res": {"dt_polys": [zero_area], "dt_scores": [0.8]}}
    )

    regions = detect_and_ocr.parse_detector_results([detector_result], image)

    assert regions == []
    assert "degenerate" in caplog.text


def test_polygon_outside_image_is_skipped_instead_of_collapsing_image_parse(caplog):
    image = Image.new("RGB", (10, 10))
    outside = _box(12, 12, 18, 18)
    detector_result = SimpleNamespace(
        json={
            "res": {
                "dt_polys": [outside, _box(1, 1, 5, 5)],
                "dt_scores": [0.2, 0.9],
            }
        }
    )

    regions = detect_and_ocr.parse_detector_results([detector_result], image)

    assert len(regions) == 1
    assert regions[0].detector_score == 0.9
    assert "no visible area after clipping" in caplog.text


@pytest.mark.parametrize(
    "result",
    [
        [],
        [SimpleNamespace(json={"res": {"dt_scores": []}})],
        [
            SimpleNamespace(
                json={
                    "res": {
                        "dt_polys": [_box(1, 1, 6, 5)],
                        "dt_scores": [],
                    }
                }
            )
        ],
    ],
)
def test_invalid_or_empty_detector_result_is_reported(result):
    with pytest.raises(ValueError):
        detect_and_ocr.parse_detector_results(result, Image.new("RGB", (10, 10)))


def test_load_text_detector_uses_cpu_without_onednn(monkeypatch):
    fake_constructor = Mock(return_value="detector")
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(TextDetection=fake_constructor),
    )

    detector = detect_and_ocr.load_text_detector("test-detector")

    assert detector == "detector"
    fake_constructor.assert_called_once_with(
        model_name="test-detector",
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
    )


def test_run_detect_and_ocr_records_detection_recognition_and_empty_text(
    tmp_path, monkeypatch
):
    split_dir = _create_dataset(tmp_path)
    output_path = tmp_path / "data" / "predictions" / "detected.csv"
    original_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in split_dir.iterdir()
    }

    class FakeDetector:
        def predict(self, source_path):
            path = source_path.replace("\\", "/")
            if path.endswith("image_00.png"):
                polygons = [_box(2, 2, 12, 8)]
                scores = [0.9]
            elif path.endswith("image_01.png"):
                polygons = []
                scores = []
            elif path.endswith("image_02.png"):
                raise RuntimeError("synthetic detector failure")
            else:
                polygons = [_box(2, 2, 12, 8), _box(2, 12, 12, 18)]
                scores = [0.8, 0.7]
            return [
                SimpleNamespace(
                    json={"res": {"dt_polys": polygons, "dt_scores": scores}}
                )
            ]

    recognized = iter(["Verified-looking but unverified", "", None])

    def fake_predict(image, **kwargs):
        assert isinstance(image, Image.Image)
        assert kwargs["device"] == "cpu"
        result = next(recognized)
        if result is None:
            raise RuntimeError("synthetic recognition failure")
        return result

    monkeypatch.setattr(detect_and_ocr, "select_device", lambda requested=None: "cpu")
    monkeypatch.setattr(detect_and_ocr, "predict_image", fake_predict)

    rows = detect_and_ocr.run_detect_and_ocr(
        split_dir.parent,
        output_path,
        sample_size=4,
        seed=42,
        project_root=tmp_path,
        detector=FakeDetector(),
        model=object(),
        processor=object(),
    )

    assert {row["status"] for row in rows} == {
        "success",
        "no_text_regions",
        "detection_failed",
        "empty_text",
        "inference_failed",
    }
    assert all(row["prediction_verified"] == "false" for row in rows)
    assert all(row["split"] == "test" for row in rows)
    assert any(row["detector_score"] == "0.9" for row in rows)
    assert any(row["polygon_xy"] for row in rows)
    assert any("synthetic detector failure" in row["error"] for row in rows)
    assert any("synthetic recognition failure" in row["error"] for row in rows)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in split_dir.iterdir()
    } == original_hashes

    with output_path.open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert saved_rows == rows
    assert all(
        row["source_image"].startswith("data/DrugSticker_data/test/")
        for row in saved_rows
    )


def test_run_detect_and_ocr_records_per_image_loader_failure(tmp_path, monkeypatch):
    split_dir = _create_dataset(tmp_path)
    output_path = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        detect_and_ocr,
        "load_text_detector",
        lambda _name: (_ for _ in ()).throw(RuntimeError("detector unavailable")),
    )

    with pytest.raises(RuntimeError, match="failure records saved"):
        detect_and_ocr.run_detect_and_ocr(
            split_dir.parent,
            output_path,
            sample_size=2,
            seed=42,
            project_root=tmp_path,
        )

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    failures = [row for row in rows if row["status"] == "detector_load_failed"]
    assert len(failures) == 2
    assert all("detector unavailable" in row["error"] for row in failures)


def test_run_detect_and_ocr_refuses_dataset_output(tmp_path):
    split_dir = _create_dataset(tmp_path)

    with pytest.raises(ValueError, match="inside the source dataset"):
        detect_and_ocr.run_detect_and_ocr(
            split_dir.parent,
            split_dir / "predictions.csv",
            project_root=tmp_path,
        )
