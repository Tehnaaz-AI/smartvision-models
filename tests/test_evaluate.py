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
