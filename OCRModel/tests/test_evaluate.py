import csv
import json
from argparse import Namespace

import torch
from PIL import Image

from smartvision_ocr.training import evaluate


def _write_reference_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("image", "text"))
        writer.writeheader()
        writer.writerows(rows)


def test_run_evaluation_reports_each_row_and_does_not_score_failures(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 16), "white").save(image_dir / "valid.png")
    (image_dir / "unreadable.png").write_text("not an image", encoding="utf-8")
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(
        reference_csv,
        [
            {"image": "images/valid.png", "text": "Drug 500 mg!"},
            {"image": "images/missing.png", "text": "Missing line"},
            {"image": "images/unreadable.png", "text": "Unreadable line"},
            {"image": "images/valid.png", "text": "   "},
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "select_device",
        lambda requested=None: torch.device("cpu"),
    )
    monkeypatch.setattr(evaluate, "load_model", lambda model_id: (object(), object()))
    monkeypatch.setattr(
        evaluate,
        "predict_image",
        lambda *args, **kwargs: "DRUG 500 MG!",
    )
    output_csv = tmp_path / "results.csv"
    report_path = tmp_path / "report.json"
    args = Namespace(
        reference_csv=str(reference_csv),
        data_dir=str(tmp_path),
        output_csv=str(output_csv),
        report_json=str(report_path),
        device="cpu",
    )

    report = evaluate.run_evaluation(args)

    assert report["total_rows"] == 4
    assert report["images_evaluated"] == 1
    assert report["failed_predictions"] == 2
    assert report["invalid_rows"] == 1
    assert report["metrics"]["exact_match_rate"] == 0.0
    assert report["metrics"]["exact_match_rate_case_insensitive"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    with output_csv.open(encoding="utf-8", newline="") as file:
        output_rows = list(csv.DictReader(file))
    assert len(output_rows) == 4
    assert output_rows[0]["status"] == "evaluated"
    assert output_rows[0]["reference_text"] == "Drug 500 mg!"
    assert output_rows[0]["predicted_text"] == "DRUG 500 MG!"
    assert output_rows[1]["status"] == "failed"
    assert output_rows[1]["exact_match"] == ""
    assert output_rows[2]["status"] == "failed"
    assert output_rows[3]["status"] == "invalid_reference"


def test_empty_reference_csv_stops_before_loading_model(tmp_path, monkeypatch):
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(reference_csv, [])
    monkeypatch.setattr(
        evaluate,
        "load_model",
        lambda model_id: (_ for _ in ()).throw(AssertionError("model was loaded")),
    )
    args = Namespace(
        reference_csv=str(reference_csv),
        data_dir=str(tmp_path),
        output_csv=str(tmp_path / "results.csv"),
        report_json=str(tmp_path / "report.json"),
        device="cpu",
    )

    try:
        evaluate.run_evaluation(args)
    except ValueError as error:
        assert "No reference rows found" in str(error)
    else:
        raise AssertionError("An empty reference CSV should stop evaluation.")
    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "report.json").exists()


def test_reference_image_paths_are_resolved_from_data_dir(tmp_path):
    data_dir = tmp_path / "local-evaluation"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 16), "white").save(image_dir / "line.png")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    reference_csv = manifest_dir / "references.csv"
    _write_reference_csv(
        reference_csv,
        [{"image": "images/line.png", "text": "Drug 500 mg"}],
    )

    rows = evaluate._read_evaluation_csv(reference_csv, data_dir)

    assert len(rows) == 1
    assert rows[0].status == "pending"
    assert rows[0].image_path == (image_dir / "line.png").resolve()


def test_unsupported_reference_image_extension_is_reported(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "line.gif").write_bytes(b"not decoded")
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(
        reference_csv,
        [{"image": "images/line.gif", "text": "Drug 500 mg"}],
    )

    rows = evaluate._read_evaluation_csv(reference_csv, tmp_path)

    assert rows[0].status == "failed"
    assert "Unsupported image file extension" in rows[0].error


def test_prediction_error_is_recorded_not_scored(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 16), "white").save(image_dir / "valid.png")
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(
        reference_csv,
        [{"image": "images/valid.png", "text": "Drug 500 mg"}],
    )
    monkeypatch.setattr(
        evaluate,
        "select_device",
        lambda requested=None: torch.device("cpu"),
    )
    monkeypatch.setattr(evaluate, "load_model", lambda model_id: (object(), object()))

    def fail_prediction(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(evaluate, "predict_image", fail_prediction)
    args = Namespace(
        reference_csv=str(reference_csv),
        data_dir=str(tmp_path),
        output_csv=str(tmp_path / "results.csv"),
        report_json=str(tmp_path / "report.json"),
        device="cpu",
    )

    report = evaluate.run_evaluation(args)

    assert report["images_evaluated"] == 0
    assert report["failed_predictions"] == 1
    assert report["metrics"] is None
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["status"] == "failed"
    assert row["cer"] == ""
    assert row["error"] == "inference failed"


def test_duplicate_references_and_multiline_text_are_not_scored(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 16), "white").save(image_dir / "valid.png")
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(
        reference_csv,
        [
            {"image": "images/valid.png", "text": "Drug 500 mg"},
            {"image": "images/./valid.png", "text": "Repeated crop"},
            {"image": "images/valid.png", "text": "First line\nsecond line"},
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "select_device",
        lambda requested=None: torch.device("cpu"),
    )
    monkeypatch.setattr(evaluate, "load_model", lambda model_id: (object(), object()))
    monkeypatch.setattr(evaluate, "predict_image", lambda *args, **kwargs: "Drug 500 mg")
    args = Namespace(
        reference_csv=str(reference_csv),
        data_dir=str(tmp_path),
        output_csv=str(tmp_path / "results.csv"),
        report_json=str(tmp_path / "report.json"),
        device="cpu",
    )

    report = evaluate.run_evaluation(args)

    assert report["images_evaluated"] == 1
    assert report["invalid_rows"] == 2
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["status"] for row in rows] == [
        "evaluated",
        "invalid_row",
        "invalid_reference",
    ]


def test_validate_references_does_not_load_model_and_rejects_empty_set(tmp_path):
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(reference_csv, [])

    try:
        evaluate.validate_references(reference_csv, tmp_path)
    except ValueError as error:
        assert "No reference rows found" in str(error)
    else:
        raise AssertionError("An empty reference CSV must not validate.")


def test_run_evaluation_multilingual_paddle_model(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 16), "white").save(image_dir / "line.png")
    reference_csv = tmp_path / "references.csv"
    _write_reference_csv(
        reference_csv,
        [{"image": "images/line.png", "text": "నమస్కారం"}],
    )

    class MockRecognizer:
        def predict(self, img_array):
            return [{"rec_text": "నమస్కారం", "rec_score": 0.99}]

    monkeypatch.setattr(
        "smartvision_ocr.inference.pipeline.load_multilingual_recognizer",
        lambda *args, **kwargs: MockRecognizer(),
    )
    monkeypatch.setattr(
        evaluate,
        "select_device",
        lambda requested=None: torch.device("cpu"),
    )

    output_csv = tmp_path / "results.csv"
    report_path = tmp_path / "report.json"
    args = Namespace(
        reference_csv=str(reference_csv),
        data_dir=str(tmp_path),
        output_csv=str(output_csv),
        report_json=str(report_path),
        device="cpu",
        model="te_PP-OCRv5_mobile_rec",
    )

    report = evaluate.run_evaluation(args)
    assert report["images_evaluated"] == 1
    assert report["metrics"]["exact_match_rate"] == 1.0
    assert report["model"] == "te_PP-OCRv5_mobile_rec"
    assert "latency" in report
    assert report["latency"]["average_ms_per_image"] >= 0.0
