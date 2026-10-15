import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from smartvision_ocr.data.manifest import export_trocr_split, validate_manifest
from smartvision_ocr.data.validation import read_manifest


def _record(**overrides):
    record = {
        "schema_version": "1",
        "sample_id": "sample-1",
        "source_dataset": "fixture",
        "image_path": "line.png",
        "transcription": "Medicine 10 mg",
        "region": {"type": "bbox_xyxy", "coordinates": [2, 1, 8, 5]},
        "language": "en",
        "script": "Latn",
        "text_scope": "line",
        "split": "train",
        "source_group_id": "source-image-1",
        "annotation_provenance": {"method": "human", "reference": "annotation-1"},
        "verification_status": "verified",
        "label_origin": "human",
        "license_status": "approved",
    }
    record.update(overrides)
    return record


def _write_manifest(path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return path


def _setup(tmp_path, records=None):
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (10, 6), "white").save(root / "line.png")
    manifest = tmp_path / "records.jsonl"
    _write_manifest(manifest, records or [_record()])
    return root, manifest


def test_validation_accepts_portable_record_and_exact_unicode(tmp_path):
    record = _record(transcription="ఔషధం 500 mg — दवा")
    root, manifest = _setup(tmp_path, [record])

    report = validate_manifest(manifest, root)

    assert report.valid
    assert report.records[0].transcription == "ఔషధం 500 mg — दवा"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transcription", "", "non-empty string"),
        ("transcription", "first line\nsecond line", "must not contain CR or LF"),
        ("split", "dev", "split must be one of"),
        ("image_path", "absent.png", "image does not exist"),
        ("image_path", "../outside.png", "escapes"),
    ],
)
def test_validation_reports_missing_or_invalid_fields(tmp_path, field, value, message):
    record = _record(**{field: value})
    root, manifest = _setup(tmp_path, [record])

    report = validate_manifest(manifest, root)

    assert not report.valid
    assert any(message in item for item in report.errors)


def test_validation_reports_unreadable_image(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "line.png").write_bytes(b"not an image")
    manifest = _write_manifest(tmp_path / "records.jsonl", [_record()])

    report = validate_manifest(manifest, root)

    assert any("unreadable image" in item for item in report.errors)


@pytest.mark.parametrize(
    "region",
    [
        {"type": "bbox_xyxy", "coordinates": [5, 1, 2, 4]},
        {"type": "bbox_xyxy", "coordinates": [0, 0, 11, 4]},
        {"type": "bbox_xyxy", "coordinates": [0, 0, float("nan"), 4]},
        {"type": "polygon", "coordinates": [[1, 1], [4, 4]]},
        {"type": "polygon", "coordinates": [[1, 1], [4, 1], [7, 1]]},
        {"type": "polygon", "coordinates": [[1, 1], [7, 5], [1, 5], [7, 1]]},
        {"type": "polygon", "coordinates": [[1, 1], [7, 1], [7, 5], [1, 5], [1, 1]]},
        {"type": "circle", "coordinates": []},
    ],
)
def test_invalid_geometry_is_reported(tmp_path, region):
    root, manifest = _setup(tmp_path, [_record(region=region)])

    report = validate_manifest(manifest, root)

    assert not report.valid
    assert any("region" in item or "bbox" in item or "polygon" in item for item in report.errors)


def test_duplicate_ids_and_image_transcription_records_are_reported(tmp_path):
    first = _record(sample_id="duplicate")
    duplicate_id = _record(sample_id="duplicate", transcription="different")
    duplicate_text = _record(sample_id="another")
    root, manifest = _setup(tmp_path, [first, duplicate_id, duplicate_text])

    report = validate_manifest(manifest, root)

    assert any("duplicate sample_id" in item for item in report.errors)
    assert any("duplicate image-transcription record" in item for item in report.errors)


def test_same_source_group_cannot_cross_splits(tmp_path):
    first = _record()
    second = _record(
        sample_id="sample-2",
        image_path="copy.png",
        split="test",
        transcription="Other line",
    )
    root, manifest = _setup(tmp_path, [first, second])
    (root / "copy.png").write_bytes((root / "line.png").read_bytes())

    report = validate_manifest(manifest, root)

    assert any("source group" in item and "leaks across" in item for item in report.errors)
    assert any("byte-identical image leaks" in item for item in report.errors)


def test_verified_export_preserves_unicode_and_records_crop_provenance(tmp_path):
    text = "ఔషధం 500 mg — दवा"
    root, manifest = _setup(tmp_path, [_record(transcription=text)])
    destination = tmp_path / "prepared"

    result = export_trocr_split(manifest, root, destination, "train")

    assert result.exported == 1
    with result.csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["text"] == text
    assert read_manifest(result.csv_path, destination)[0].text == text
    with Image.open(destination / rows[0]["image"]) as crop:
        assert crop.size == (6, 4)
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["sample_id"] == "sample-1"
    assert provenance["source_image"] == "line.png"
    assert provenance["crop_bounds_xyxy"] == [2, 1, 8, 5]


def test_csv_loader_and_export_preserve_edge_whitespace(tmp_path):
    text = "  ఔషధం 500 mg — दवा  "
    root, manifest = _setup(tmp_path, [_record(transcription=text)])

    result = export_trocr_split(manifest, root, tmp_path / "prepared", "train")
    samples = read_manifest(result.csv_path, tmp_path / "prepared")

    assert samples[0].text == text


def test_export_excludes_unverified_and_pseudo_labels(tmp_path):
    verified = _record(sample_id="verified")
    unverified = _record(
        sample_id="unverified",
        image_path="second.png",
        source_group_id="second",
        verification_status="unverified",
    )
    pseudo = _record(
        sample_id="pseudo",
        image_path="third.png",
        source_group_id="third",
        label_origin="pseudo",
    )
    root, manifest = _setup(tmp_path, [verified, unverified, pseudo])
    Image.new("RGB", (10, 6), "black").save(root / "second.png")
    Image.new("RGB", (10, 6), "gray").save(root / "third.png")

    result = export_trocr_split(manifest, root, tmp_path / "prepared", "train")

    assert result.exported == 1
    assert result.excluded == {"not_verified": 1, "pseudo_label": 1}


@pytest.mark.parametrize("license_status", ["unknown", "not_permitted"])
def test_export_excludes_unapproved_licenses(tmp_path, license_status):
    approved = _record()
    unapproved = _record(
        sample_id="unapproved",
        image_path="unapproved.png",
        source_group_id="unapproved",
        license_status=license_status,
    )
    root, manifest = _setup(tmp_path, [approved, unapproved])
    Image.new("RGB", (10, 6), "black").save(root / "unapproved.png")

    result = export_trocr_split(manifest, root, tmp_path / "prepared", "train")

    assert result.exported == 1
    assert result.excluded == {"license_not_approved": 1}


def test_export_rejects_full_image_transcription_without_region(tmp_path):
    full_image = _record(region=None, text_scope="full_image")
    root, manifest = _setup(tmp_path, [full_image])

    with pytest.raises(ValueError, match="No eligible verified line crops"):
        export_trocr_split(manifest, root, tmp_path / "prepared", "train")


def test_export_aborts_instead_of_skipping_invalid_manifest_records(tmp_path):
    valid = _record()
    invalid = _record(
        sample_id="invalid",
        image_path="second.png",
        source_group_id="second",
        split="development",
    )
    root, manifest = _setup(tmp_path, [valid, invalid])
    Image.new("RGB", (10, 6), "black").save(root / "second.png")
    destination = tmp_path / "prepared"

    with pytest.raises(ValueError, match="Manifest validation failed"):
        export_trocr_split(manifest, root, destination, "train")

    assert not destination.exists()


def test_export_refuses_existing_files_without_overwrite(tmp_path):
    root, manifest = _setup(tmp_path)
    destination = tmp_path / "prepared"
    first = export_trocr_split(manifest, root, destination, "train")
    old_csv = first.csv_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_trocr_split(manifest, root, destination, "train")

    assert first.csv_path.read_bytes() == old_csv
    replacement = export_trocr_split(
        manifest, root, destination, "train", overwrite=True
    )
    assert replacement.exported == 1


def test_overwrite_never_allows_source_image_collision(tmp_path):
    sample_id = "source-collision"
    crop_name = f"{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()}.png"
    root = tmp_path / "images"
    (root / "crops").mkdir(parents=True)
    Image.new("RGB", (10, 6), "white").save(root / "crops" / crop_name)
    record = _record(sample_id=sample_id, image_path=f"crops/{crop_name}")
    manifest = _write_manifest(tmp_path / "records.jsonl", [record])
    original_bytes = (root / "crops" / crop_name).read_bytes()

    with pytest.raises(ValueError, match="overwrite an input manifest or source image"):
        export_trocr_split(manifest, root, root, "train", overwrite=True)

    assert (root / "crops" / crop_name).read_bytes() == original_bytes


def test_overwrite_never_allows_input_manifest_collision(tmp_path):
    root, _ = _setup(tmp_path)
    destination = tmp_path / "prepared"
    destination.mkdir()
    manifest = _write_manifest(destination / "provenance.jsonl", [_record()])
    original_bytes = manifest.read_bytes()

    with pytest.raises(ValueError, match="overwrite an input manifest or source image"):
        export_trocr_split(manifest, root, destination, "train", overwrite=True)

    assert manifest.read_bytes() == original_bytes


def test_overwrite_never_follows_output_symlink_outside_destination(tmp_path):
    root, manifest = _setup(tmp_path)
    destination = tmp_path / "prepared"
    destination.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "marker").write_text("preserve", encoding="utf-8")
    try:
        (destination / "crops").symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="outside the output directory|symbolic link"):
        export_trocr_split(manifest, root, destination, "train", overwrite=True)

    assert (external / "marker").read_text(encoding="utf-8") == "preserve"


def test_overwrite_never_modifies_source_via_hard_link(tmp_path):
    sample_id = "hardlink-source"
    crop_name = f"{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()}.png"
    root = tmp_path / "images"
    (root / "crops").mkdir(parents=True)
    source_image = root / "line.png"
    Image.new("RGB", (10, 6), "white").save(source_image)
    crop_target = root / "crops" / crop_name
    try:
        os.link(source_image, crop_target)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")
    record = _record(sample_id=sample_id)
    manifest = _write_manifest(tmp_path / "records.jsonl", [record])
    original_bytes = source_image.read_bytes()

    with pytest.raises(ValueError, match="overwrite an input manifest or source image"):
        export_trocr_split(manifest, root, root, "train", overwrite=True)

    assert source_image.read_bytes() == original_bytes


def test_canonical_manifest_rejects_invalid_utf8(tmp_path):
    root, manifest = _setup(tmp_path)
    manifest.write_bytes(b"\xff\n")

    report = validate_manifest(manifest, root)

    assert not report.valid


def test_cli_returns_nonzero_for_invalid_manifest(tmp_path):
    root, manifest = _setup(tmp_path, [_record(split="development")])
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartvision_ocr.data.prepare",
            "validate",
            "--manifest",
            str(manifest),
            "--image-root",
            str(root),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "split must be one of" in result.stdout


def test_export_cli_returns_nonzero_for_invalid_manifest(tmp_path):
    root, manifest = _setup(tmp_path, [_record(split="development")])
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartvision_ocr.data.prepare",
            "export",
            "--manifest",
            str(manifest),
            "--image-root",
            str(root),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--split",
            "train",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Manifest validation failed" in result.stdout
