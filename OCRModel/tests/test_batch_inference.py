import csv
import hashlib

import pytest
from PIL import Image

from smartvision_ocr.inference import batch


def _create_dataset(project_root):
    split_dir = project_root / "data" / "DrugSticker_data" / "test"
    split_dir.mkdir(parents=True)
    for number in range(4):
        Image.new("RGB", (20, 12), (number * 40, 20, 90)).save(
            split_dir / f"image_{number:02}.jpg"
        )
    (split_dir / "duplicate.jpg").write_bytes(
        (split_dir / "image_02.jpg").read_bytes()
    )
    (split_dir / "broken.jpg").write_bytes(b"not an image")
    return split_dir


def test_sample_images_are_repeatable_and_seeded(tmp_path):
    split_dir = _create_dataset(tmp_path)
    paths = sorted(split_dir.glob("image_*.jpg"))

    first = batch._sample_images(paths, 2, 42, tmp_path)
    repeated = batch._sample_images(paths, 2, 42, tmp_path)
    different_seed = batch._sample_images(paths, 2, 21, tmp_path)

    assert first == repeated
    assert len(first) == 2
    assert first != different_seed


def test_run_batch_records_failures_skips_and_unavailable_confidence(
    tmp_path, monkeypatch
):
    split_dir = _create_dataset(tmp_path)
    output_csv = tmp_path / "data" / "predictions" / "batch.csv"
    original_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in split_dir.iterdir()
    }
    predict_calls = []

    monkeypatch.setattr(batch, "select_device", lambda requested=None: "cpu")
    monkeypatch.setattr(batch, "load_model", lambda model_name: ("processor", "model"))

    def fake_predict(image_path, **kwargs):
        predict_calls.append(image_path)
        assert kwargs["device"] == "cpu"
        if len(predict_calls) == 1:
            raise RuntimeError("simulated per-image failure")
        return "exploratory prediction"

    monkeypatch.setattr(batch, "predict_image", fake_predict)

    rows = batch.run_batch(
        tmp_path / "data" / "DrugSticker_data",
        output_csv,
        sample_size=2,
        seed=42,
        project_root=tmp_path,
    )

    assert len(predict_calls) == 2
    assert sum(row["status"] == "success" for row in rows) == 1
    assert sum(row["status"] == "inference_failed" for row in rows) == 1
    assert sum(row["status"] == "skipped_unreadable" for row in rows) == 1
    assert sum(row["status"] == "skipped_duplicate" for row in rows) == 1
    assert sum(row["status"] == "skipped_not_sampled" for row in rows) == 2
    assert all(row["input_mode"] == batch.INPUT_MODE for row in rows)
    assert all(row["confidence"] == "" for row in rows)
    assert all(row["confidence_method"] == batch.CONFIDENCE_METHOD for row in rows)
    assert any("simulated per-image failure" in row["error"] for row in rows)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in split_dir.iterdir()
    } == original_hashes

    with output_csv.open(encoding="utf-8", newline="") as file:
        written_rows = list(csv.DictReader(file))
    assert written_rows == rows
    assert all(
        row["source_image"].startswith("data/DrugSticker_data/test/")
        for row in written_rows
    )


def test_run_batch_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    split_dir = _create_dataset(tmp_path)
    output_csv = tmp_path / "predictions.csv"
    output_csv.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(
        batch,
        "load_model",
        lambda model_name: pytest.fail("model should not load"),
    )

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        batch.run_batch(
            split_dir.parent,
            output_csv,
            sample_size=2,
            project_root=tmp_path,
        )

    assert output_csv.read_text(encoding="utf-8") == "preserve me"


def test_run_batch_rejects_output_inside_source_dataset(tmp_path):
    split_dir = _create_dataset(tmp_path)

    with pytest.raises(ValueError, match="inside the source dataset"):
        batch.run_batch(
            split_dir.parent,
            split_dir / "predictions.csv",
            sample_size=2,
            project_root=tmp_path,
        )


def test_run_batch_reports_missing_split_directory(tmp_path):
    dataset_dir = tmp_path / "data" / "DrugSticker_data"

    with pytest.raises(FileNotFoundError, match="split directory"):
        batch.run_batch(
            dataset_dir,
            tmp_path / "predictions.csv",
            project_root=tmp_path,
        )
