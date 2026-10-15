"""Canonical JSONL OCR manifest validation and TrOCR crop export."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image, UnidentifiedImageError

SPLITS = frozenset({"train", "validation", "test"})
VERIFICATION_STATUSES = frozenset({"verified", "unverified", "rejected"})
LABEL_ORIGINS = frozenset({"human", "source_annotation", "pseudo"})
REGION_TYPES = frozenset({"bbox_xyxy", "polygon"})
TEXT_SCOPES = frozenset({"line", "word", "full_image", "paragraph", "field"})
LICENSE_STATUSES = frozenset({"approved", "unknown", "not_permitted"})


@dataclass(frozen=True)
class OCRRecord:
    sample_id: str
    source_dataset: str
    image_path: str
    transcription: str
    split: str
    source_group_id: str
    annotation_provenance: dict[str, Any]
    verification_status: str
    label_origin: str
    license_status: str
    text_scope: str
    region: dict[str, Any] | None = None
    language: str | None = None
    script: str | None = None
    line_number: int = 0


@dataclass
class ValidationReport:
    records: list[OCRRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_lines: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def verified_count(self) -> int:
        return sum(row.verification_status == "verified" for row in self.records)

    @property
    def pseudo_count(self) -> int:
        return sum(row.label_origin == "pseudo" for row in self.records)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and (isinstance(value, int) or math.isfinite(value))
    )


def _parse_record(raw: Any, line_number: int) -> OCRRecord:
    if not isinstance(raw, dict):
        raise ValueError("record must be a JSON object")

    required = {
        "schema_version",
        "sample_id",
        "source_dataset",
        "image_path",
        "transcription",
        "split",
        "source_group_id",
        "annotation_provenance",
        "verification_status",
        "label_origin",
        "license_status",
        "text_scope",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if raw["schema_version"] != "1":
        raise ValueError("schema_version must be the string '1'")

    for name in (
        "sample_id",
        "source_dataset",
        "image_path",
        "source_group_id",
        "license_status",
        "text_scope",
    ):
        if not _nonempty_string(raw[name]):
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(raw["transcription"], str) or not raw["transcription"].strip():
        raise ValueError("transcription must be a non-empty string")
    if not isinstance(raw["split"], str) or raw["split"] not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(sorted(SPLITS))}")
    if (
        not isinstance(raw["verification_status"], str)
        or raw["verification_status"] not in VERIFICATION_STATUSES
    ):
        raise ValueError(
            "verification_status must be one of "
            f"{', '.join(sorted(VERIFICATION_STATUSES))}"
        )
    if not isinstance(raw["label_origin"], str) or raw["label_origin"] not in LABEL_ORIGINS:
        raise ValueError(f"label_origin must be one of {', '.join(sorted(LABEL_ORIGINS))}")
    if not isinstance(raw["license_status"], str) or raw["license_status"] not in LICENSE_STATUSES:
        raise ValueError(f"license_status must be one of {', '.join(sorted(LICENSE_STATUSES))}")
    if not isinstance(raw["text_scope"], str) or raw["text_scope"] not in TEXT_SCOPES:
        raise ValueError(f"text_scope must be one of {', '.join(sorted(TEXT_SCOPES))}")
    if raw["text_scope"] == "line" and any(
        line_break in raw["transcription"] for line_break in ("\r", "\n")
    ):
        raise ValueError("line transcription must not contain CR or LF line breaks")
    provenance = raw["annotation_provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("annotation_provenance must be a non-empty JSON object")

    for optional in ("language", "script"):
        if raw.get(optional) is not None and not isinstance(raw[optional], str):
            raise ValueError(f"{optional} must be a string or null")
    region = raw.get("region")
    if region is not None and not isinstance(region, dict):
        raise ValueError("region must be a JSON object or null")

    return OCRRecord(
        sample_id=raw["sample_id"],
        source_dataset=raw["source_dataset"],
        image_path=raw["image_path"],
        transcription=raw["transcription"],
        split=raw["split"],
        source_group_id=raw["source_group_id"],
        annotation_provenance=provenance,
        verification_status=raw["verification_status"],
        label_origin=raw["label_origin"],
        license_status=raw["license_status"],
        text_scope=raw["text_scope"],
        region=region,
        language=raw.get("language"),
        script=raw.get("script"),
        line_number=line_number,
    )


def _resolve_image(image_root: Path, image_value: str, label: str) -> Path:
    # Manifest paths are portable, root-relative POSIX paths.
    relative = Path(image_value)
    if (
        "\\" in image_value
        or relative.is_absolute()
        or PureWindowsPath(image_value).is_absolute()
        or PureWindowsPath(image_value).drive
    ):
        raise ValueError(f"{label}: image_path must be a relative path using '/' separators")
    path = (image_root / relative).resolve()
    if not path.is_relative_to(image_root):
        raise ValueError(f"{label}: image_path escapes the configured image root")
    if not path.is_file():
        raise ValueError(f"{label}: image does not exist: {image_value}")
    return path


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    ab_c = orientation(a, b, c)
    ab_d = orientation(a, b, d)
    cd_a = orientation(c, d, a)
    cd_b = orientation(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return (
        (ab_c == 0 and on_segment(a, c, b))
        or (ab_d == 0 and on_segment(a, d, b))
        or (cd_a == 0 and on_segment(c, a, d))
        or (cd_b == 0 and on_segment(c, b, d))
    )


def _validated_region(
    record: OCRRecord, width: int, height: int, label: str
) -> tuple[int, int, int, int] | None:
    region = record.region
    if region is None:
        return None
    kind = region.get("type")
    coordinates = region.get("coordinates")
    if not isinstance(kind, str) or kind not in REGION_TYPES:
        raise ValueError(f"{label}: region.type must be bbox_xyxy or polygon")

    if kind == "bbox_xyxy":
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError(f"{label}: bbox_xyxy coordinates must be [x1, y1, x2, y2]")
        if any(
            not _finite_number(value)
            for value in coordinates
        ):
            raise ValueError(f"{label}: bbox coordinates must be finite numbers")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{label}: bbox must have positive area and lie within "
                f"image bounds 0..{width} by 0..{height}"
            )
        bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    else:
        if not isinstance(coordinates, list) or len(coordinates) < 3:
            raise ValueError(f"{label}: polygon must contain at least three [x, y] points")
        points: list[tuple[float, float]] = []
        for point in coordinates:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    not _finite_number(value)
                    for value in point
                )
            ):
                raise ValueError(f"{label}: polygon points must be finite [x, y] pairs")
            x, y = point
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"{label}: polygon point lies outside image bounds")
            points.append((x, y))
        if len(set(points)) != len(points):
            raise ValueError(f"{label}: polygon vertices must be unique")
        if _polygon_area(points) <= 0:
            raise ValueError(f"{label}: polygon must have non-zero area")
        edge_count = len(points)
        for first in range(edge_count):
            for second in range(first + 1, edge_count):
                if second in {first, (first + 1) % edge_count}:
                    continue
                if first == 0 and second == edge_count - 1:
                    continue
                if _segments_intersect(
                    points[first],
                    points[(first + 1) % edge_count],
                    points[second],
                    points[(second + 1) % edge_count],
                ):
                    raise ValueError(f"{label}: polygon edges self-intersect")
        bounds = (
            math.floor(min(point[0] for point in points)),
            math.floor(min(point[1] for point in points)),
            math.ceil(max(point[0] for point in points)),
            math.ceil(max(point[1] for point in points)),
        )
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(f"{label}: polygon crop has zero area")
    return bounds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    manifest_path: str | Path, image_root: str | Path
) -> ValidationReport:
    """Validate a UTF-8 JSONL manifest without modifying its inputs."""
    source = Path(manifest_path)
    root = Path(image_root).expanduser().resolve()
    report = ValidationReport()
    if not source.is_file():
        report.errors.append(f"Manifest does not exist: {source}")
        return report
    seen_ids: dict[str, int] = {}
    seen_pairs: dict[tuple[str, str], int] = {}
    seen_groups: dict[tuple[str, str], tuple[str, int]] = {}
    seen_paths: dict[str, tuple[str, int]] = {}
    seen_hashes: dict[str, tuple[str, int]] = {}

    try:
        stream = source.open("rb")
    except OSError as error:
        report.errors.append(f"Could not open manifest {source}: {error}")
        return report

    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                report.errors.append(
                    f"{source}:{line_number}: invalid UTF-8 encoding: {error}"
                )
                report.total_lines += 1
                continue
            if not line.strip():
                continue
            report.total_lines += 1
            try:
                raw = json.loads(line)
                record = _parse_record(raw, line_number)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                report.errors.append(f"{source}:{line_number}: {error}")
                continue

            label = f"{source}:{line_number} [{record.sample_id}]"
            previous_line = seen_ids.get(record.sample_id)
            if previous_line is not None:
                report.errors.append(
                    f"{label}: duplicate sample_id (first seen on line {previous_line})"
                )
            else:
                seen_ids[record.sample_id] = line_number

            group_key = (record.source_dataset, record.source_group_id)
            previous_group = seen_groups.get(group_key)
            if previous_group and previous_group[0] != record.split:
                report.errors.append(
                    f"{label}: source group '{record.source_group_id}' leaks across "
                    f"'{previous_group[0]}' (line {previous_group[1]}) and '{record.split}'"
                )
            else:
                seen_groups[group_key] = (record.split, line_number)

            try:
                image_path = _resolve_image(root, record.image_path, label)
            except ValueError as error:
                report.errors.append(str(error))
                report.records.append(record)
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError, ValueError) as error:
                report.errors.append(f"{label}: unreadable image {record.image_path}: {error}")
                report.records.append(record)
                continue

            try:
                _validated_region(record, width, height, label)
            except ValueError as error:
                report.errors.append(str(error))

            resolved_key = str(image_path)
            previous_path = seen_paths.get(resolved_key)
            if previous_path and previous_path[0] != record.split:
                report.errors.append(
                    f"{label}: source image appears in both '{previous_path[0]}' "
                    f"(line {previous_path[1]}) and '{record.split}'"
                )
            else:
                seen_paths[resolved_key] = (record.split, line_number)

            digest = _sha256(image_path)
            pair_key = (digest, record.transcription)
            previous_line = seen_pairs.get(pair_key)
            if previous_line is not None:
                report.errors.append(
                    f"{label}: duplicate image-transcription record "
                    f"(first seen on line {previous_line})"
                )
            else:
                seen_pairs[pair_key] = line_number

            previous_digest = seen_hashes.get(digest)
            if previous_digest and previous_digest[0] != record.split:
                report.errors.append(
                    f"{label}: byte-identical image leaks across '{previous_digest[0]}' "
                    f"(line {previous_digest[1]}) and '{record.split}'"
                )
            else:
                seen_hashes[digest] = (record.split, line_number)
            report.records.append(record)

    if report.total_lines == 0:
        report.errors.append(f"Manifest contains no JSONL records: {source}")
    return report


@dataclass(frozen=True)
class ExportReport:
    split: str
    exported: int
    excluded: dict[str, int]
    csv_path: Path
    provenance_path: Path


def export_trocr_split(
    manifest_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    split: str,
    *,
    overwrite: bool = False,
) -> ExportReport:
    """Export verified line crops and exact labels to the existing trainer CSV."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(sorted(SPLITS))}")
    report = validate_manifest(manifest_path, image_root)
    if not report.valid:
        raise ValueError("Manifest validation failed:\n" + "\n".join(report.errors))

    root = Path(image_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    eligible: list[tuple[OCRRecord, Path, tuple[int, int, int, int]]] = []
    excluded: dict[str, int] = {}
    for record in report.records:
        if record.split != split:
            continue
        reason = None
        if record.text_scope == "full_image":
            reason = "full_image_requires_text_region"
        elif record.verification_status != "verified":
            reason = "not_verified"
        elif record.label_origin == "pseudo":
            reason = "pseudo_label"
        elif record.license_status != "approved":
            reason = "license_not_approved"
        elif record.text_scope != "line":
            reason = "not_line_scope"
        elif record.region is None:
            reason = "missing_region"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue

        image_path = _resolve_image(root, record.image_path, record.sample_id)
        with Image.open(image_path) as image:
            bounds = _validated_region(
                record, image.width, image.height, record.sample_id
            )
        if bounds is None:
            excluded["missing_region"] = excluded.get("missing_region", 0) + 1
            continue
        eligible.append((record, image_path, bounds))

    if not eligible:
        details = ", ".join(f"{key}={value}" for key, value in sorted(excluded.items()))
        raise ValueError(
            f"No eligible verified line crops in split '{split}'. "
            f"Full-image transcriptions require a text region; excluded: {details or 'none'}"
        )

    csv_path = destination / f"{split}.csv"
    provenance_path = destination / "provenance.jsonl"
    crop_paths = [
        destination / "crops" / f"{hashlib.sha256(r.sample_id.encode('utf-8')).hexdigest()}.png"
        for r, _, _ in eligible
    ]
    targets = [csv_path, provenance_path, *crop_paths]
    destination_paths = []
    for path in targets:
        if not path.resolve().is_relative_to(destination):
            raise ValueError(
                f"Export target resolves outside the output directory: {path}"
            )
        current = path
        while current != destination:
            if current.is_symlink():
                raise ValueError(f"Export target uses a symbolic link: {current}")
            current = current.parent
        destination_paths.append(path)

    protected_sources = {
        Path(manifest_path).expanduser().resolve(),
        *(
            _resolve_image(root, record.image_path, record.sample_id)
            for record in report.records
        ),
    }
    source_collisions = [
        path
        for path in destination_paths
        if path.resolve() in protected_sources
        or (
            path.exists()
            and any(
                source.exists() and path.samefile(source)
                for source in protected_sources
            )
        )
    ]
    if source_collisions:
        raise ValueError(
            "Export target would overwrite an input manifest or source image: "
            + ", ".join(str(path) for path in source_collisions)
        )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing export files; pass overwrite=True "
            f"to replace them: {', '.join(str(path) for path in existing)}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "crops").mkdir(exist_ok=True)
    rows: list[tuple[str, str]] = []
    provenance_rows: list[dict[str, Any]] = []
    for (record, image_path, bounds), crop_path in zip(eligible, crop_paths):
        with Image.open(image_path) as source_image:
            crop = source_image.convert("RGB").crop(bounds)
            crop.save(crop_path, format="PNG")
        relative_crop = crop_path.relative_to(destination).as_posix()
        rows.append((relative_crop, record.transcription))
        provenance_rows.append(
            {
                "sample_id": record.sample_id,
                "source_dataset": record.source_dataset,
                "source_image": record.image_path,
                "source_group_id": record.source_group_id,
                "source_image_sha256": _sha256(image_path),
                "region": record.region,
                "crop_path": relative_crop,
                "crop_bounds_xyxy": list(bounds),
                "transcription": record.transcription,
                "language": record.language,
                "script": record.script,
                "split": record.split,
                "annotation_provenance": record.annotation_provenance,
                "verification_status": record.verification_status,
                "label_origin": record.label_origin,
                "license_status": record.license_status,
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "text"))
        writer.writerows(rows)
    with provenance_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in provenance_rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return ExportReport(
        split=split,
        exported=len(rows),
        excluded=excluded,
        csv_path=csv_path,
        provenance_path=provenance_path,
    )
