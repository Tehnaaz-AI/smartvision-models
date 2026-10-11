"""Evaluate pretrained TrOCR on manually transcribed medicine-label crops."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from smartvision_ocr.config import BASE_MODEL_ID, DATA_DIR
from smartvision_ocr.data.validation import ImageTextSample
from smartvision_ocr.inference.predict import (
    load_model,
    predict_image,
    select_device,
)
from smartvision_ocr.utils.logging_utils import configure_logging
from smartvision_ocr.utils.metrics import (
    compute_pair_metrics,
    compute_text_metrics,
)

RESULT_FIELDS = (
    "image",
    "reference_text",
    "predicted_text",
    "cer",
    "wer",
    "exact_match",
    "exact_match_case_insensitive",
    "status",
    "error",
)


@dataclass
class EvaluationRow:
    image: str
    reference_text: str
    image_path: Path | None
    predicted_text: str = ""
    cer: float | None = None
    wer: float | None = None
    exact_match: bool | None = None
    exact_match_case_insensitive: bool | None = None
    status: str = "pending"
    error: str = ""


def _read_evaluation_csv(
    csv_path: str | Path,
    data_dir: str | Path,
) -> list[EvaluationRow]:
    manifest_path = Path(csv_path).expanduser()
    root = Path(data_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reference CSV does not exist: {manifest_path}")

    rows: list[EvaluationRow] = []
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["image", "text"]:
                raise ValueError(
                    f"{manifest_path} must contain exactly these CSV columns: image,text."
                )

            for row_number, row in enumerate(reader, start=2):
                image_value = (row.get("image") or "").strip()
                reference = row.get("text") or ""
                record = EvaluationRow(image=image_value, reference_text=reference, image_path=None)
                rows.append(record)

                if None in row:
                    record.status = "invalid_row"
                    record.error = f"Row {row_number} has more values than CSV columns."
                    continue
                if not image_value:
                    record.status = "failed"
                    record.error = f"Row {row_number} has an empty image path."
                    continue
                if not reference.strip():
                    record.status = "invalid_reference"
                    record.error = f"Row {row_number} has an empty reference transcription."
                    continue

                relative_path = Path(image_value)
                if relative_path.is_absolute():
                    record.status = "failed"
                    record.error = f"Image path must be relative to {root}: {image_value}"
                    continue

                image_path = (root / relative_path).resolve()
                if not image_path.is_relative_to(root):
                    record.status = "failed"
                    record.error = (
                        f"Image path escapes the evaluation data directory: {image_value}"
                    )
                    continue
                record.image_path = image_path
                if not image_path.is_file():
                    record.status = "failed"
                    record.error = f"Image does not exist: {image_path}"
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    record.status = "failed"
                    record.error = f"Unreadable image {image_path}: {error}"
    except csv.Error as error:
        raise ValueError(f"Could not parse reference CSV {manifest_path}: {error}") from error

    if not rows:
        raise ValueError(
            f"No reference rows found in {manifest_path}. Add permitted real images "
            "and manually verified transcriptions before evaluation."
        )
    return rows


def predict_samples(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    samples: list[ImageTextSample],
    batch_size: int,
    device: torch.device,
) -> list[str]:
    """Batch predictions used by the existing training validation workflow."""
    def collate(batch: list[ImageTextSample]) -> torch.Tensor:
        images = []
        for sample in batch:
            with Image.open(sample.image_path) as source:
                images.append(source.convert("RGB"))
        return processor(images=images, return_tensors="pt").pixel_values

    loader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model.to(device)
    model.eval()
    predictions: list[str] = []
    with torch.inference_mode():
        for pixel_values in loader:
            generated = model.generate(pixel_values.to(device))
            predictions.extend(
                processor.batch_decode(generated, skip_special_tokens=True)
            )
    return predictions


def _write_results(
    rows: list[EvaluationRow],
    output_csv: Path,
    report_path: Path,
    report: dict[str, Any],
    reference_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            result = asdict(row)
            result["image_path"] = None
            writer.writerow({field: result[field] for field in RESULT_FIELDS})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    logger = configure_logging()
    reference_csv = Path(args.reference_csv).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    report_path = Path(args.report_json).expanduser()
    if output_csv.resolve() == reference_csv.resolve():
        raise ValueError("Results CSV must not overwrite the reference CSV.")
    if report_path.resolve() in {reference_csv.resolve(), output_csv.resolve()}:
        raise ValueError("Report path must not overwrite an input or results CSV.")

    rows = _read_evaluation_csv(reference_csv, args.data_dir)
    candidates = [
        row
        for row in rows
        if row.status == "pending" and row.image_path is not None
    ]

    device = select_device(args.device)
    if candidates:
        logger.info("Loading %s on %s", BASE_MODEL_ID, device)
        processor, model = load_model(BASE_MODEL_ID)
        for row in candidates:
            try:
                row.predicted_text = predict_image(
                    row.image_path,
                    model=model,
                    processor=processor,
                    device=str(device),
                )
                metrics = compute_pair_metrics(row.reference_text, row.predicted_text)
                row.cer = float(metrics["cer"])
                row.wer = float(metrics["wer"])
                row.exact_match = bool(metrics["exact_match"])
                row.exact_match_case_insensitive = bool(
                    metrics["exact_match_case_insensitive"]
                )
                row.status = "evaluated"
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                row.status = "failed"
                row.error = str(error)
                logger.error("Prediction failed for %s: %s", row.image, error)

    successful_rows = [row for row in rows if row.status == "evaluated"]
    failed_predictions = sum(row.status == "failed" for row in rows)
    invalid_rows = sum(row.status in {"invalid_row", "invalid_reference"} for row in rows)
    aggregate_metrics = (
        compute_text_metrics(
            [row.reference_text for row in successful_rows],
            [row.predicted_text for row in successful_rows],
        )
        if successful_rows
        else None
    )
    report: dict[str, Any] = {
        "model": BASE_MODEL_ID,
        "device": str(device),
        "total_rows": len(rows),
        "images_evaluated": len(successful_rows),
        "failed_predictions": failed_predictions,
        "invalid_rows": invalid_rows,
        "metrics": aggregate_metrics,
        "case_insensitive_normalization": (
            "Unicode casefold only; no removal of units, numbers, punctuation, "
            "or whitespace."
        ),
        "word_error_rate_tokenization": (
            "Split on whitespace; punctuation and token content are retained."
        ),
        "empty_reference_denominator": (
            "Use one when a reference has zero characters/words so insertions "
            "are counted and the score is defined."
        ),
        "errors": [
            {"image": row.image, "status": row.status, "error": row.error}
            for row in rows
            if row.error
        ],
    }

    _write_results(
        rows,
        output_csv,
        report_path,
        report,
        reference_csv,
    )
    logger.info(
        "Evaluation summary: %d rows; %d images evaluated; %d failed predictions; "
        "%d invalid rows.",
        len(rows),
        len(successful_rows),
        failed_predictions,
        invalid_rows,
    )
    if aggregate_metrics is not None:
        logger.info(
            "CER %.4f; WER %.4f; exact match %.2f%%; case-insensitive exact "
            "match %.2f%% (successful predictions only).",
            aggregate_metrics["cer"],
            aggregate_metrics["wer"],
            aggregate_metrics["exact_match_rate"] * 100,
            aggregate_metrics["exact_match_rate_case_insensitive"] * 100,
        )
    logger.info("Per-image results: %s", output_csv)
    logger.info("Evaluation report: %s", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained TrOCR using verified text references."
    )
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--data-dir", default=str(DATA_DIR / "evaluation"))
    parser.add_argument(
        "--output-csv",
        default=str(DATA_DIR / "evaluation" / "evaluation_results.csv"),
    )
    parser.add_argument(
        "--report-json",
        default=str(DATA_DIR / "evaluation" / "evaluation_report.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging()
    try:
        run_evaluation(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
