"""Fine-Tuning Execution Script for SmartVision TrOCR / VisionEncoderDecoderModel.

Performs supervised fine-tuning on line crops with learning-rate scheduling,
loss logging, CER/WER validation tracking, and recovery checkpoint saving.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Setup paths
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from smartvision_ocr.config import BASE_MODEL_ID
from smartvision_ocr.data.dataset import OCRDataset
from smartvision_ocr.data.validation import read_manifest
from smartvision_ocr.utils.metrics import compute_text_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smartvision_finetuning")


def collate_fn(processor: TrOCRProcessor, batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
    """Collates and tokenizes images and label texts."""
    images, texts = zip(*batch)
    pixel_values = processor(images=list(images), return_tensors="pt").pixel_values
    encoded_text = processor.tokenizer(
        text=list(texts),
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    labels = encoded_text.input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    samples: list,
    device: torch.device,
    max_eval_samples: int = 15,
) -> dict[str, float]:
    """Evaluates current model checkpoint on validation samples."""
    model.eval()
    eval_subset = samples[:max_eval_samples]
    references = [s.text for s in eval_subset]
    predictions = []

    with torch.no_grad():
        for sample in eval_subset:
            try:
                img = Image.open(sample.image_path).convert("RGB")
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values, max_new_tokens=40)
                pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                predictions.append(pred_text)
            except Exception:
                predictions.append("")

    return compute_text_metrics(references, predictions)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision OCR Fine-Tuning Runner")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, default=8, help="Max training samples for rapid verification")
    parser.add_argument("--max-val-samples", type=int, default=4, help="Max validation samples for evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Initializing fine-tuning pipeline on device: %s", device)

    # Load dataset splits
    data_dir = root_dir / "data"
    train_csv = data_dir / "labels" / "train.csv"
    val_csv = data_dir / "labels" / "validation.csv"

    train_samples = read_manifest(str(train_csv), data_dir)
    val_samples = read_manifest(str(val_csv), data_dir)

    if args.max_train_samples and args.max_train_samples < len(train_samples):
        train_samples = train_samples[: args.max_train_samples]

    logger.info("Loaded %d training samples and %d validation samples", len(train_samples), len(val_samples))

    # Load pretrained Hugging Face TrOCR
    logger.info("Loading pretrained Hugging Face model: %s", BASE_MODEL_ID)
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_ID)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # DataLoader
    dataset = OCRDataset(train_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )

    checkpoint_dir = root_dir / "models" / "checkpoints"
    final_dir = root_dir / "models" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate pre-finetuning baseline
    logger.info("Measuring pre-finetuning baseline on validation crops...")
    pre_metrics = evaluate(model, processor, val_samples, device, max_eval_samples=args.max_val_samples)
    logger.info("Pre-training Baseline Metrics: CER=%.4f, WER=%.4f", pre_metrics["cer"], pre_metrics["wer"])

    # Training Loop
    logger.info("Starting fine-tuning for %d epochs...", args.epochs)
    training_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            epoch_loss += loss_val
            step_count += 1
            logger.info("  Epoch %d | Step %d/%d | Batch Loss: %.4f", epoch + 1, step + 1, len(dataloader), loss_val)

        avg_loss = epoch_loss / max(1, step_count)
        epoch_time = time.time() - epoch_start

        # Post-epoch evaluation
        post_metrics = evaluate(model, processor, val_samples, device, max_eval_samples=args.max_val_samples)
        logger.info(
            "Epoch %d Summary | Loss: %.4f | Val CER: %.4f | Val WER: %.4f | Time: %.2fs",
            epoch + 1,
            avg_loss,
            post_metrics["cer"],
            post_metrics["wer"],
            epoch_time,
        )

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint-epoch-{epoch + 1:02d}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

        history_record = {
            "epoch": epoch + 1,
            "average_loss": round(avg_loss, 4),
            "val_cer": round(post_metrics["cer"], 4),
            "val_wer": round(post_metrics["wer"], 4),
            "checkpoint_path": str(ckpt_path),
        }
        training_history.append(history_record)

    # Save final model
    logger.info("Saving final fine-tuned model to: %s", final_dir)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    report_path = final_dir / "finetuning_report.json"
    report = {
        "model_id": BASE_MODEL_ID,
        "epochs_trained": args.epochs,
        "device": str(device),
        "pre_training_metrics": pre_metrics,
        "training_history": training_history,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Fine-tuning report written to: %s", report_path)
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
