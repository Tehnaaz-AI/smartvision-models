"""Explicit TrOCR fine-tuning command; importing this module does not train."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from smartvision_ocr.config import (
    BASE_MODEL_ID,
    CHECKPOINT_DIR,
    DATA_DIR,
    FINAL_MODEL_DIR,
)
from smartvision_ocr.data.dataset import OCRDataset
from smartvision_ocr.data.validation import read_manifest, validate_disjoint_splits
from smartvision_ocr.inference.predict import select_device
from smartvision_ocr.training.evaluate import predict_samples
from smartvision_ocr.utils.logging_utils import configure_logging
from smartvision_ocr.utils.metrics import compute_text_metrics


def _collate(processor: TrOCRProcessor, batch: list[tuple[Image.Image, str]]) -> dict[str, Any]:
    images, texts = zip(*batch)
    encoded = processor(
        images=list(images),
        text=list(texts),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = encoded.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": encoded.pixel_values, "labels": labels}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    completed_epoch: int,
) -> Path:
    target = checkpoint_dir / f"epoch-{completed_epoch:04d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    processor.save_pretrained(target)
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    (target / "training_state.json").write_text(
        json.dumps({"completed_epoch": completed_epoch}, indent=2),
        encoding="utf-8",
    )
    return target


def run_training(args: argparse.Namespace) -> None:
    logger = configure_logging()
    _set_seed(args.seed)
    data_dir = Path(args.data_dir).expanduser()
    train_samples = read_manifest(args.train_csv, data_dir)
    validation_samples = read_manifest(args.validation_csv, data_dir)
    test_samples = read_manifest(args.test_csv, data_dir)
    validate_disjoint_splits(
        {
            "train": train_samples,
            "validation": validation_samples,
            "test": test_samples,
        }
    )

    device = select_device(args.device)
    logger.info("Training on %s", device)
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).expanduser()
        processor = TrOCRProcessor.from_pretrained(checkpoint)
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
        state_path = checkpoint / "training_state.json"
        optimizer_state_path = checkpoint / "optimizer.pt"
        if not state_path.is_file() or not optimizer_state_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint lacks recovery state files: {checkpoint}"
            )
        start_epoch = int(
            json.loads(state_path.read_text(encoding="utf-8"))["completed_epoch"]
        )
    else:
        processor = TrOCRProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.resume_from_checkpoint:
        optimizer.load_state_dict(
            torch.load(
                Path(args.resume_from_checkpoint) / "optimizer.pt",
                map_location=device,
                weights_only=True,
            )
        )

    train_dataset = OCRDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: _collate(processor, batch),
    )
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.detach().cpu())

        predictions = predict_samples(
            model, processor, validation_samples, args.batch_size, device
        )
        metrics = compute_text_metrics(
            [sample.text for sample in validation_samples], predictions
        )
        average_loss = total_loss / len(train_loader)
        logger.info(
            "Epoch %d/%d - loss %.4f - validation CER %.4f - validation WER %.4f",
            epoch + 1,
            args.epochs,
            average_loss,
            metrics["cer"],
            metrics["wer"],
        )
        saved_path = _save_checkpoint(
            model, processor, optimizer, checkpoint_dir, epoch + 1
        )
        logger.info("Saved recovery checkpoint: %s", saved_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Saved fine-tuned model and processor to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on line crops.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--output-dir", default=str(FINAL_MODEL_DIR))
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.learning_rate <= 0 or args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("Learning rate, batch size, and epochs must be positive.")
    run_training(args)


if __name__ == "__main__":
    main()
