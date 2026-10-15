"""Run exploratory batch OCR on DrugSticker package photos."""

import argparse
import csv
import hashlib
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from smartvision_ocr.config import DATA_DIR, MODEL_NAME_OR_PATH, PROJECT_ROOT
from smartvision_ocr.data.validation import SUPPORTED_IMAGE_SUFFIXES
from smartvision_ocr.inference.predict import (
    load_model,
    predict_image,
    select_device,
)
from smartvision_ocr.utils.logging_utils import configure_logging

RESULT_FIELDS = (
    "source_image",
    "split",
    "input_mode",
    "model",
    "device",
    "predicted_text",
    "status",
    "confidence",
    "confidence_method",
    "error",
)
INPUT_MODE = "whole_image_exploratory_no_text_detection"
CONFIDENCE_METHOD = "unavailable: current generate() inference does not expose confidence"
LOGGER = logging.getLogger(__name__)


def _source_reference(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _new_row(
    path: Path,
    split: str,
    model_name: str,
    device: str,
    project_root: Path,
    status: str,
    error: str = "",
) -> dict[str, str]:
    return {
        "source_image": _source_reference(path, project_root),
        "split": split,
        "input_mode": INPUT_MODE,
        "model": model_name,
        "device": device,
        "predicted_text": "",
        "status": status,
        "confidence": "",
        "confidence_method": CONFIDENCE_METHOD,
        "error": error,
    }


def _inspect_images(
    split_dir: Path,
    split: str,
    model_name: str,
    device: str,
    project_root: Path,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Return readable, unique images and explicit rows for skipped images."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory does not exist: {split_dir}")

    readable: list[Path] = []
    skipped: list[dict[str, str]] = []
    seen_digests: dict[str, Path] = {}
    image_paths = sorted(
        (
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in {split_dir}.")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            skipped.append(
                _new_row(
                    image_path,
                    split,
                    model_name,
                    device,
                    project_root,
                    "skipped_unreadable",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        duplicate_of = seen_digests.get(digest)
        if duplicate_of is not None:
            skipped.append(
                _new_row(
                    image_path,
                    split,
                    model_name,
                    device,
                    project_root,
                    "skipped_duplicate",
                    f"Byte-identical to {_source_reference(duplicate_of, project_root)}.",
                )
            )
            continue

        seen_digests[digest] = image_path
        readable.append(image_path)

    return readable, skipped


def _sample_images(
    paths: list[Path],
    sample_size: int,
    seed: int,
    project_root: Path,
) -> list[Path]:
    """Select a stable sample independent of Python's random implementation."""
    if sample_size < 1:
        raise ValueError("Sample size must be at least 1.")

    def rank(path: Path) -> bytes:
        reference = _source_reference(path, project_root)
        return hashlib.sha256(f"{seed}\0{reference}".encode("utf-8")).digest()

    return sorted(paths, key=lambda path: (rank(path), path.as_posix()))[:sample_size]


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


def run_batch(
    dataset_dir: str | Path,
    output_csv: str | Path,
    *,
    split: str = "test",
    sample_size: int = 10,
    seed: int = 42,
    device: str | None = None,
    model_name: str = MODEL_NAME_OR_PATH,
    project_root: str | Path = PROJECT_ROOT,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Run TrOCR on a deterministic sample and write one auditable CSV."""
    if split not in {"train", "test"}:
        raise ValueError("Split must be either 'train' or 'test'.")
    if sample_size < 1:
        raise ValueError("Sample size must be at least 1.")

    root = Path(project_root).expanduser().resolve()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    if dataset_root.name != "DrugSticker_data":
        raise ValueError("Batch inference is restricted to DrugSticker_data.")
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
    split_dir = dataset_root / split
    readable, rows = _inspect_images(
        split_dir,
        split,
        model_name,
        device_name,
        root,
    )
    selected = _sample_images(readable, sample_size, seed, root)
    selected_set = set(selected)
    for image_path in readable:
        if image_path not in selected_set:
            rows.append(
                _new_row(
                    image_path,
                    split,
                    model_name,
                    device_name,
                    root,
                    "skipped_not_sampled",
                    f"Outside deterministic sample (size={sample_size}, seed={seed}).",
                )
            )

    if selected:
        try:
            processor, model = load_model(model_name)
        except (OSError, RuntimeError, ValueError) as error:
            for image_path in selected:
                rows.append(
                    _new_row(
                        image_path,
                        split,
                        model_name,
                        device_name,
                        root,
                        "model_load_failed",
                        f"{type(error).__name__}: {error}",
                    )
                )
            _write_results(output_path, sorted(rows, key=lambda row: row["source_image"]))
            LOGGER.error("Could not load OCR model; failure rows saved to %s: %s",
                         output_path, error)
            raise RuntimeError(
                f"Could not load OCR model; failure rows saved to {output_path}."
            ) from error

        for image_path in selected:
            row = _new_row(
                image_path,
                split,
                model_name,
                device_name,
                root,
                "inference_failed",
            )
            try:
                row["predicted_text"] = predict_image(
                    image_path,
                    model=model,
                    processor=processor,
                    device=device_name,
                )
                row["status"] = "success"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                LOGGER.exception(
                    "Inference failed for %s; continuing with the batch.",
                    row["source_image"],
                )
            rows.append(row)

    rows.sort(key=lambda row: row["source_image"])
    _write_results(output_path, rows)
    successful = sum(row["status"] == "success" for row in rows)
    failures = sum(row["status"] == "inference_failed" for row in rows)
    skipped = sum(row["status"].startswith("skipped_") for row in rows)
    LOGGER.info(
        "Batch complete: %d successful predictions, %d inference failures, "
        "%d skipped rows. Results: %s",
        successful,
        failures,
        skipped,
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
            "Run exploratory TrOCR inference on DrugSticker package photos. "
            "This does not detect text regions or calculate OCR accuracy."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Use one dataset split only (default: test; no training is performed).",
    )
    parser.add_argument(
        "--sample-size",
        type=_positive_int,
        default=10,
        help="Number of readable, unique images to infer (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--model", default=MODEL_NAME_OR_PATH)
    parser.add_argument(
        "--output-csv",
        help="Output CSV path; defaults to data/predictions/drugsticker_<split>_predictions.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output CSV if it already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging()
    dataset_dir = DATA_DIR / "DrugSticker_data"
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else DATA_DIR / "predictions" / f"drugsticker_{args.split}_predictions.csv"
    )
    try:
        run_batch(
            dataset_dir,
            output_csv,
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
            device=args.device,
            model_name=args.model,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
