import csv

import pytest
from PIL import Image

from smartvision_ocr.data.validation import read_manifest, validate_disjoint_splits


def _write_manifest(path, rows, fields=("image", "text")):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_missing_image_is_reported(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = tmp_path / "labels.csv"
    _write_manifest(manifest, [{"image": "missing.png", "text": "Medicine"}])

    with pytest.raises(FileNotFoundError, match="image does not exist"):
        read_manifest(manifest, data_dir)


def test_invalid_csv_columns_are_reported(tmp_path):
    manifest = tmp_path / "labels.csv"
    _write_manifest(manifest, [{"filename": "line.png", "label": "Medicine"}],
                    fields=("filename", "label"))

    with pytest.raises(ValueError, match="columns named 'image' and 'text'"):
        read_manifest(manifest, tmp_path)


def test_empty_label_is_reported(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    Image.new("RGB", (16, 8), "white").save(data_dir / "line.png")
    manifest = tmp_path / "labels.csv"
    _write_manifest(manifest, [{"image": "line.png", "text": "  "}])

    with pytest.raises(ValueError, match="text label is empty"):
        read_manifest(manifest, data_dir)


def test_rejects_shared_image_across_splits(tmp_path):
    image_path = tmp_path / "line.png"
    Image.new("RGB", (16, 8), "white").save(image_path)
    from smartvision_ocr.data.validation import ImageTextSample

    sample = ImageTextSample(image_path=image_path, text="label")
    with pytest.raises(ValueError, match="Data leakage"):
        validate_disjoint_splits({"train": [sample], "test": [sample]})


def test_rejects_byte_identical_images_across_splits(tmp_path):
    image_one = tmp_path / "one.png"
    image_two = tmp_path / "two.png"
    Image.new("RGB", (16, 8), "white").save(image_one)
    image_two.write_bytes(image_one.read_bytes())
    from smartvision_ocr.data.validation import ImageTextSample

    with pytest.raises(ValueError, match="byte-identical"):
        validate_disjoint_splits(
            {
                "train": [ImageTextSample(image_one, "one")],
                "test": [ImageTextSample(image_two, "two")],
            }
        )
