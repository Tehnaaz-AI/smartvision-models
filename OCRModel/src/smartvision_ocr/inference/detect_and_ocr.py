"""Detect text regions in package photos, then recognize each region with TrOCR."""

import argparse
import csv
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from smartvision_ocr.config import DATA_DIR, MODEL_NAME_OR_PATH, PROJECT_ROOT
from smartvision_ocr.inference.batch import (
    _inspect_images,
    _sample_images,
    _source_reference,
)
from smartvision_ocr.inference.predict import load_model, predict_image, select_device
from smartvision_ocr.utils.logging_utils import configure_logging

DETECTOR_MODEL_NAME = "PP-OCRv5_mobile_det"
RESULT_FIELDS = (
    "source_image",
    "split",
    "region_index",
    "polygon_xy",
    "detector_score",
    "detector_model",
    "recognizer_model",
    "device",
    "recognized_text",
    "prediction_verified",
    "status",
    "error",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextRegion:
    polygon: tuple[tuple[float, float], ...]
    detector_score: float | None
    crop: Image.Image


def _to_number(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Detector returned a non-numeric {description}.") from error
    if not math.isfinite(number):
        raise ValueError(f"Detector returned a non-finite {description}.")
    return number


def _as_sequence(value: Any, description: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"Detector field '{description}' must be a sequence.")
    try:
        return list(value)
    except TypeError as error:
        raise ValueError(
            f"Detector field '{description}' must be a sequence."
        ) from error


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2


def _cross_product(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - second[1]) - (
        second[1] - first[1]
    ) * (third[0] - second[0])


def _clip_polygon(
    points: Sequence[tuple[float, float]],
    image_size: tuple[int, int],
) -> list[tuple[float, float]]:
    """Clip a polygon to the image rectangle without collapsing outside vertices."""
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")

    clipped = list(points)
    boundaries = (
        (
            lambda point: point[0] >= 0,
            lambda first, second: (
                0.0,
                first[1]
                + (second[1] - first[1])
                * (0.0 - first[0])
                / (second[0] - first[0]),
            ),
        ),
        (
            lambda point: point[0] <= width,
            lambda first, second: (
                float(width),
                first[1]
                + (second[1] - first[1])
                * (float(width) - first[0])
                / (second[0] - first[0]),
            ),
        ),
        (
            lambda point: point[1] >= 0,
            lambda first, second: (
                first[0]
                + (second[0] - first[0])
                * (0.0 - first[1])
                / (second[1] - first[1]),
                0.0,
            ),
        ),
        (
            lambda point: point[1] <= height,
            lambda first, second: (
                first[0]
                + (second[0] - first[0])
                * (float(height) - first[1])
                / (second[1] - first[1]),
                float(height),
            ),
        ),
    )
    for is_inside, intersection in boundaries:
        if not clipped:
            break
        output: list[tuple[float, float]] = []
        previous = clipped[-1]
        previous_inside = is_inside(previous)
        for current in clipped:
            current_inside = is_inside(current)
            if current_inside != previous_inside:
                output.append(intersection(previous, current))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        clipped = output

    distinct: list[tuple[float, float]] = []
    for point in clipped:
        if not distinct or not math.isclose(
            point[0], distinct[-1][0], abs_tol=1e-9
        ) or not math.isclose(point[1], distinct[-1][1], abs_tol=1e-9):
            distinct.append(point)
    if len(distinct) > 1 and math.isclose(
        distinct[0][0], distinct[-1][0], abs_tol=1e-9
    ) and math.isclose(distinct[0][1], distinct[-1][1], abs_tol=1e-9):
        distinct.pop()
    return distinct


def _crop_polygon(
    image: Image.Image,
    polygon: tuple[tuple[float, float], ...],
) -> Image.Image:
    if len(polygon) == 4:
        return _rectify_polygon(image, polygon)

    left = max(0, math.floor(min(point[0] for point in polygon)))
    top = max(0, math.floor(min(point[1] for point in polygon)))
    right = min(image.width, math.ceil(max(point[0] for point in polygon)))
    bottom = min(image.height, math.ceil(max(point[1] for point in polygon)))
    if right <= left or bottom <= top:
        raise ValueError("Detected text polygon has no visible image area.")
    return image.crop((left, top, right, bottom))


def _order_quad(
    points: Any,
    image_size: tuple[int, int],
) -> tuple[tuple[float, float], ...]:
    point_values = _as_sequence(points, "polygon")
    if len(point_values) != 4:
        raise ValueError("Each detected text region must contain exactly four points.")

    parsed: list[tuple[float, float]] = []
    for point in point_values:
        point_values = _as_sequence(point, "polygon point")
        if len(point_values) != 2:
            raise ValueError("Each detected polygon point must contain x and y.")
        parsed.append(
            (
                _to_number(point_values[0], "polygon coordinate"),
                _to_number(point_values[1], "polygon coordinate"),
            )
        )

    if len(set(parsed)) != 4:
        raise ValueError("Detected text polygon contains duplicate vertices.")
    center_x = sum(point[0] for point in parsed) / len(parsed)
    center_y = sum(point[1] for point in parsed) / len(parsed)
    parsed.sort(
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x)
    )
    crosses = [
        _cross_product(parsed[index - 1], parsed[index], parsed[(index + 1) % 4])
        for index in range(4)
    ]
    if any(abs(cross) <= 1e-9 for cross in crosses) or not (
        all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)
    ):
        raise ValueError("Detected text polygon is degenerate or not convex.")
    if _polygon_area(parsed) <= 1e-9:
        raise ValueError("Detected text polygon has zero area.")

    clipped = _clip_polygon(parsed, image_size)
    if len(clipped) < 3 or _polygon_area(clipped) <= 1e-9:
        raise ValueError("Detected text polygon has no visible area after clipping.")
    clipped_center_x = sum(point[0] for point in clipped) / len(clipped)
    clipped_center_y = sum(point[1] for point in clipped) / len(clipped)
    clipped.sort(
        key=lambda point: math.atan2(
            point[1] - clipped_center_y, point[0] - clipped_center_x
        )
    )

    if len(clipped) != 4:
        start = min(range(len(clipped)), key=lambda index: (clipped[index][1], clipped[index][0]))
        clipped = clipped[start:] + clipped[:start]
        return tuple(clipped)

    top_left = min(clipped, key=lambda point: (point[0] + point[1], point[1]))
    bottom_right = max(clipped, key=lambda point: (point[0] + point[1], point[1]))
    top_right = min(clipped, key=lambda point: (point[1] - point[0], point[0]))
    bottom_left = max(clipped, key=lambda point: (point[1] - point[0], point[0]))
    ordered = (top_left, top_right, bottom_right, bottom_left)
    if len(set(ordered)) != 4 or _polygon_area(ordered) <= 1e-9:
        ordered = tuple(clipped)
    return ordered


def _rectify_polygon(
    image: Image.Image,
    polygon: tuple[tuple[float, float], ...],
) -> Image.Image:
    """Perspective-rectify one four-corner polygon to a line crop."""
    top_left, top_right, bottom_right, bottom_left = polygon

    def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    crop_width = max(
        2,
        round(max(distance(top_left, top_right), distance(bottom_left, bottom_right))),
    )
    crop_height = max(
        2,
        round(max(distance(top_left, bottom_left), distance(top_right, bottom_right))),
    )
    source_quad = (
        top_left[0],
        top_left[1],
        bottom_left[0],
        bottom_left[1],
        bottom_right[0],
        bottom_right[1],
        top_right[0],
        top_right[1],
    )
    return image.transform(
        (crop_width, crop_height),
        Image.Transform.QUAD,
        source_quad,
        resample=Image.Resampling.BICUBIC,
    )


def _reading_order_key(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    return ((top + bottom) / 2, left, top, right)


def order_regions(
    polygons: Sequence[Any],
    image_size: tuple[int, int],
) -> list[tuple[tuple[float, float], ...]]:
    """Order detected quadrilaterals top-to-bottom, then left-to-right."""
    ordered = [_order_quad(polygon, image_size) for polygon in polygons]
    return sorted(ordered, key=_reading_order_key)


def _extract_detection_payload(result: Any) -> Mapping[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        raise ValueError("Text detector returned an unsupported result object.")
    if "res" in payload:
        payload = payload["res"]
    if not isinstance(payload, Mapping):
        raise ValueError("Text detector result is missing its 'res' mapping.")
    if "dt_polys" not in payload:
        raise ValueError("Text detector result is missing 'dt_polys'.")
    return payload


def parse_detector_results(
    results: Any,
    image: Image.Image,
) -> list[TextRegion]:
    """Parse PaddleOCR TextDetection outputs into ordered, rectified crops."""
    result_values = _as_sequence(results, "predict() results")
    if not result_values:
        raise ValueError("Text detector returned no result object for the image.")

    payload = _extract_detection_payload(result_values[0])
    polygons = payload["dt_polys"]
    polygon_values = [] if polygons is None else _as_sequence(polygons, "dt_polys")

    scores = payload.get("dt_scores")
    score_values = None if scores is None else _as_sequence(scores, "dt_scores")
    if score_values is not None and len(score_values) != len(polygon_values):
        raise ValueError("Detector 'dt_scores' count does not match 'dt_polys'.")

    indexed_polygons = []
    for index, polygon in enumerate(polygon_values):
        try:
            ordered = _order_quad(polygon, image.size)
            score = (
                _to_number(score_values[index], "detection score")
                if score_values is not None
                else None
            )
        except ValueError as error:
            LOGGER.warning("Skipping invalid detected polygon %d: %s", index, error)
            continue
        indexed_polygons.append((ordered, score))
    indexed_polygons.sort(key=lambda item: _reading_order_key(item[0]))
    return [
        TextRegion(
            polygon=polygon,
            detector_score=score,
            crop=_crop_polygon(image, polygon),
        )
        for polygon, score in indexed_polygons
    ]


def load_text_detector(model_name: str = DETECTOR_MODEL_NAME) -> Any:
    """Load the optional CPU PaddleOCR detector, downloading weights as needed."""
    try:
        from paddleocr import TextDetection
    except ImportError as error:
        raise RuntimeError(
            "Text detection requires the optional packages in "
            "requirements-detector.txt. Install them in the project .venv first."
        ) from error

    try:
        return TextDetection(
            model_name=model_name,
            device="cpu",
            engine="paddle_static",
            enable_mkldnn=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load PaddleOCR text detector '{model_name}'. "
            "Check the detector dependencies, model download, and network access."
        ) from error


def _base_row(
    image_path: Path,
    split: str,
    detector_model: str,
    recognizer_model: str,
    device: str,
    project_root: Path,
    status: str,
    error: str = "",
) -> dict[str, str]:
    return {
        "source_image": _source_reference(image_path, project_root),
        "split": split,
        "region_index": "",
        "polygon_xy": "",
        "detector_score": "",
        "detector_model": detector_model,
        "recognizer_model": recognizer_model,
        "device": device,
        "recognized_text": "",
        "prediction_verified": "false",
        "status": status,
        "error": error,
    }


def _row_for_region(
    image_path: Path,
    split: str,
    region_index: int,
    region: TextRegion,
    detector_model: str,
    recognizer_model: str,
    device: str,
    project_root: Path,
) -> dict[str, str]:
    row = _base_row(
        image_path,
        split,
        detector_model,
        recognizer_model,
        device,
        project_root,
        "inference_failed",
    )
    row["region_index"] = str(region_index)
    row["polygon_xy"] = json.dumps(region.polygon, separators=(",", ":"))
    if region.detector_score is not None:
        row["detector_score"] = str(region.detector_score)
    return row


def _write_results(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_detect_and_ocr(
    dataset_dir: str | Path,
    output_csv: str | Path,
    *,
    split: str = "test",
    sample_size: int = 10,
    seed: int = 42,
    device: str | None = None,
    detector_model: str = DETECTOR_MODEL_NAME,
    recognizer_model: str = MODEL_NAME_OR_PATH,
    project_root: str | Path = PROJECT_ROOT,
    detector: Any | None = None,
    model: Any | None = None,
    processor: Any | None = None,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Detect, rectify, and recognize a deterministic sample from one split."""
    if split not in {"train", "test"}:
        raise ValueError("Split must be either 'train' or 'test'.")
    if sample_size < 1:
        raise ValueError("Sample size must be at least 1.")
    if (model is None) != (processor is None):
        raise ValueError("Provide both recognizer model and processor, or neither.")

    root = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    if dataset_root.name != "DrugSticker_data":
        raise ValueError("Detection workflow is restricted to DrugSticker_data.")
    output_path = Path(output_csv).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    if output_path.is_relative_to(dataset_root):
        raise ValueError("Prediction CSV must not be written inside the source dataset.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prediction CSV already exists: {output_path}. "
            "Choose another path or pass --overwrite."
        )

    selected_device = select_device(device)
    device_name = str(selected_device)
    readable, skipped_rows = _inspect_images(
        dataset_root / split,
        split,
        detector_model,
        device_name,
        root,
    )
    selected = _sample_images(readable, sample_size, seed, root)
    selected_set = set(selected)
    rows: list[dict[str, str]] = []
    for skipped in skipped_rows:
        rows.append(
            _base_row(
                root / Path(skipped["source_image"]),
                split,
                detector_model,
                recognizer_model,
                device_name,
                root,
                skipped["status"],
                skipped["error"],
            )
        )
    for image_path in readable:
        if image_path not in selected_set:
            rows.append(
                _base_row(
                    image_path,
                    split,
                    detector_model,
                    recognizer_model,
                    device_name,
                    root,
                    "skipped_not_sampled",
                    f"Outside deterministic sample (size={sample_size}, seed={seed}).",
                )
            )

    global_load_error = ""
    global_load_status = "model_load_failed"
    if selected and detector is None:
        try:
            detector = load_text_detector(detector_model)
        except (RuntimeError, ValueError, OSError) as error:
            global_load_error = f"Detector load failed: {type(error).__name__}: {error}"
            global_load_status = "detector_load_failed"
    if selected and not global_load_error and (model is None or processor is None):
        try:
            processor, model = load_model(recognizer_model)
        except (RuntimeError, ValueError, OSError) as error:
            global_load_error = f"Recognizer load failed: {type(error).__name__}: {error}"
            global_load_status = "recognizer_load_failed"
    if global_load_error:
        rows.extend(
            _base_row(
                image_path,
                split,
                detector_model,
                recognizer_model,
                device_name,
                root,
                global_load_status,
                global_load_error,
            )
            for image_path in selected
        )
        _write_results(output_path, sorted(rows, key=lambda row: row["source_image"]))
        raise RuntimeError(
            f"{global_load_error}; per-image failure records saved to {output_path}."
        )

    for image_path in selected:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            rows.append(
                _base_row(
                    image_path,
                    split,
                    detector_model,
                    recognizer_model,
                    device_name,
                    root,
                    "skipped_unreadable",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        try:
            detected = detector.predict(str(image_path))
            regions = parse_detector_results(detected, image)
        except Exception as error:
            LOGGER.exception("Text detection failed for %s.", image_path)
            rows.append(
                _base_row(
                    image_path,
                    split,
                    detector_model,
                    recognizer_model,
                    device_name,
                    root,
                    "detection_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        if not regions:
            rows.append(
                _base_row(
                    image_path,
                    split,
                    detector_model,
                    recognizer_model,
                    device_name,
                    root,
                    "no_text_regions",
                )
            )
            continue

        for region_index, region in enumerate(regions, start=1):
            row = _row_for_region(
                image_path,
                split,
                region_index,
                region,
                detector_model,
                recognizer_model,
                device_name,
                root,
            )
            try:
                text = predict_image(
                    region.crop,
                    model=model,
                    processor=processor,
                    device=device_name,
                )
                row["recognized_text"] = text
                row["status"] = "success" if text.strip() else "empty_text"
            except Exception as error:
                LOGGER.exception(
                    "Recognition failed for %s region %d.",
                    image_path,
                    region_index,
                )
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["source_image"],
            int(row["region_index"]) if row["region_index"] else 0,
        )
    )
    _write_results(output_path, rows)
    LOGGER.info(
        "Detection/OCR complete: %d CSV rows written to %s. Predictions are "
        "unverified; no accuracy metrics were calculated.",
        len(rows),
        output_path,
    )
    return rows


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from error
    if result < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run text detection and OCR on an image file, or on a deterministic "
            "sample from DrugSticker_data."
        )
    )
    parser.add_argument(
        "--image",
        help="Path to an arbitrary scene, sign, or document image for end-to-end OCR.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format when using --image (default: text).",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sample-size", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--detector-model", default=DETECTOR_MODEL_NAME)
    parser.add_argument("--recognizer-model", default=MODEL_NAME_OR_PATH)
    parser.add_argument("--output-csv")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging()

    if args.image:
        from smartvision_ocr.inference.pipeline import predict_full_image

        try:
            result = predict_full_image(
                args.image,
                detector_model=args.detector_model,
                recognizer_model=args.recognizer_model,
                device=args.device,
            )
            if args.format == "json":
                print(result.to_json())
            else:
                print(f"Recognized text ({len(result.regions)} regions detected):")
                print("---")
                print(result.text if result.text else "(No text recognized)")
                print("---")
                if result.detected_scripts:
                    print(f"Detected scripts: {', '.join(result.detected_scripts)}")
                if result.timing_ms:
                    print(f"Latency: {result.timing_ms.get('total_ms', 0)} ms")
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            logger.error("%s", error)
            raise SystemExit(1) from error
        return

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else DATA_DIR
        / "predictions"
        / f"drugsticker_{args.split}_detected_text.csv"
    )
    try:
        run_detect_and_ocr(
            DATA_DIR / "DrugSticker_data",
            output_csv,
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
            device=args.device,
            detector_model=args.detector_model,
            recognizer_model=args.recognizer_model,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
