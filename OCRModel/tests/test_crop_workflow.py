import csv

import pytest
from PIL import Image

from scripts.select_evaluation_crops import (
    crop_bounds,
    fit_preview_size,
    load_candidates,
    save_selected_crop,
)


def _write_candidates(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("source_image", "split", "review_note")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_candidates_resolves_relative_paths_and_validates_images(tmp_path):
    image_path = tmp_path / "data" / "heldout" / "source.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 8), "white").save(image_path)
    candidate_csv = tmp_path / "candidates.csv"
    _write_candidates(
        candidate_csv,
        [
            {
                "source_image": "data/heldout/source.png",
                "split": "heldout",
                "review_note": "Review manually.",
            }
        ],
    )

    candidates = load_candidates(candidate_csv, tmp_path)

    assert len(candidates) == 1
    assert candidates[0].source_path == image_path
    assert candidates[0].split == "heldout"


@pytest.mark.parametrize(
    ("source_image", "split"),
    [
        ("../outside.png", "heldout"),
        ("data/missing.png", "heldout"),
        ("data/train/other.png", "train"),
        ("data/source.png", "train"),
    ],
)
def test_load_candidates_rejects_unsafe_missing_or_mixed_split_paths(
    tmp_path, source_image, split
):
    image_path = tmp_path / "data" / "source.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 8), "white").save(image_path)
    if source_image == "data/train/other.png":
        train_image = tmp_path / source_image
        train_image.parent.mkdir(parents=True)
        Image.new("RGB", (12, 8), "white").save(train_image)
    candidate_csv = tmp_path / "candidates.csv"
    rows = [
        {
            "source_image": "data/source.png",
            "split": "heldout",
            "review_note": "Review manually.",
        },
        {
            "source_image": source_image,
            "split": split,
            "review_note": "Review manually.",
        },
    ]
    if source_image in {"../outside.png", "data/missing.png", "data/source.png"}:
        rows = [rows[1]]
    _write_candidates(candidate_csv, rows)

    with pytest.raises((ValueError, FileNotFoundError)):
        load_candidates(candidate_csv, tmp_path)


def test_load_candidates_rejects_byte_identical_images(tmp_path):
    source_dir = tmp_path / "data" / "test"
    source_dir.mkdir(parents=True)
    image_path = source_dir / "one.png"
    Image.new("RGB", (12, 8), "white").save(image_path)
    (source_dir / "two.png").write_bytes(image_path.read_bytes())
    candidate_csv = tmp_path / "candidates.csv"
    _write_candidates(
        candidate_csv,
        [
            {
                "source_image": "data/test/one.png",
                "split": "test",
                "review_note": "Review manually.",
            },
            {
                "source_image": "data/test/two.png",
                "split": "test",
                "review_note": "Review manually.",
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicates an image"):
        load_candidates(candidate_csv, tmp_path)


def test_crop_bounds_maps_display_selection_to_original_pixels():
    bounds = crop_bounds(
        start=(20, 30),
        end=(70, 80),
        image_origin=(10, 10),
        scale=(2.0, 2.0),
        image_size=(100, 100),
    )
    assert bounds == (5, 10, 30, 35)


@pytest.mark.parametrize(
    ("image_size", "canvas_size", "expected_size"),
    [
        ((2000, 1000), (800, 600), (800, 400)),
        ((800, 1000), (400, 500), (400, 500)),
        ((100, 80), (400, 500), (100, 80)),
    ],
)
def test_fit_preview_size_respects_canvas_and_does_not_enlarge(
    image_size, canvas_size, expected_size
):
    assert fit_preview_size(image_size, canvas_size) == expected_size


def test_save_selected_crop_records_source_and_preserves_original(tmp_path):
    source_path = tmp_path / "data" / "test" / "source.png"
    source_path.parent.mkdir(parents=True)
    source_image = Image.new("RGB", (12, 8))
    for x in range(source_image.width):
        for y in range(source_image.height):
            source_image.putpixel((x, y), (x, y, 50))
    source_image.save(source_path)
    original_bytes = source_path.read_bytes()
    output_dir = tmp_path / "data" / "evaluation" / "images"

    crop_one = save_selected_crop(
        source_image,
        (2, 1, 8, 5),
        output_dir,
        source_path,
        "test",
        tmp_path,
    )
    crop_two = save_selected_crop(
        source_image,
        (1, 2, 7, 6),
        output_dir,
        source_path,
        "test",
        tmp_path,
    )

    assert crop_one.name == "line_001.png"
    assert crop_two.name == "line_002.png"
    assert source_path.read_bytes() == original_bytes
    with Image.open(crop_one) as saved_crop:
        assert list(saved_crop.getdata()) == list(
            source_image.crop((2, 1, 8, 5)).getdata()
        )
    with (output_dir.parent / "crop_sources.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert [row["crop"] for row in rows] == [
        "data/evaluation/images/line_001.png",
        "data/evaluation/images/line_002.png",
    ]
    assert all(row["source_image"] == "data/test/source.png" for row in rows)
    assert all(row["split"] == "test" for row in rows)


def test_save_selected_crop_rejects_bounds_outside_original_image(tmp_path):
    source_path = tmp_path / "source.png"
    source_image = Image.new("RGB", (8, 6), "white")
    source_image.save(source_path)

    with pytest.raises(ValueError, match="Crop bounds"):
        save_selected_crop(
            source_image,
            (-1, 0, 9, 6),
            tmp_path / "evaluation" / "images",
            source_path,
            "test",
            tmp_path,
        )


def test_crop_bounds_clamps_to_source_image_and_rejects_tiny_selection():
    bounds = crop_bounds(
        start=(-10, -10),
        end=(30, 30),
        image_origin=(0, 0),
        scale=(1.0, 1.0),
        image_size=(20, 20),
    )
    assert bounds == (0, 0, 20, 20)

    with pytest.raises(ValueError, match="at least 2 pixels"):
        crop_bounds(
            start=(4, 4),
            end=(5, 5),
            image_origin=(0, 0),
            scale=(1.0, 1.0),
            image_size=(20, 20),
        )
