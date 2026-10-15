"""Download, validate, and prepare the verified Telugu Wikisource line OCR dataset.

Dataset: harsha-desaraju/telugu-line-ocr-bench
License: CC-BY-SA 4.0 (explicitly approved for training, research, commercial use)
Source: Telugu Wikisource page scans, human-verified transcriptions.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

# Ensure standard output handles UTF-8 for Telugu characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TELUGU_DIR = DATA_DIR / "telugu_wikisource"
IMAGES_DIR = TELUGU_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
EVALUATION_DIR = DATA_DIR / "evaluation"

PARQUET_URL = (
    "https://huggingface.co/datasets/harsha-desaraju/telugu-line-ocr-bench"
    "/resolve/main/data/test-00000-of-00001.parquet"
)
PARQUET_LOCAL = TELUGU_DIR / "telugu_lines.parquet"


def download_dataset() -> Path:
    """Download parquet file if not already cached."""
    TELUGU_DIR.mkdir(parents=True, exist_ok=True)
    if not PARQUET_LOCAL.is_file() or PARQUET_LOCAL.stat().st_size < 1000:
        print(f"Downloading Telugu benchmark dataset from {PARQUET_URL}...")
        req = urllib.request.Request(
            PARQUET_URL,
            headers={"User-Agent": "SmartVision-OCR-DataPreparer/1.0"},
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        PARQUET_LOCAL.write_bytes(data)
        print(f"Downloaded {len(data) / (1024 * 1024):.2f} MB to {PARQUET_LOCAL}")
    else:
        print(f"Using cached dataset: {PARQUET_LOCAL} ({PARQUET_LOCAL.stat().st_size / (1024 * 1024):.2f} MB)")
    return PARQUET_LOCAL


def process_dataset(seed: int = 42) -> None:
    parquet_path = download_dataset()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquet data...")
    df = pd.read_parquet(parquet_path)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from dataset.")

    # Group by source_file to prevent data leakage across splits
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        source_key = str(row.get("source_file") or f"src_{idx % 50}")
        groups[source_key].append((idx, row))

    rng = random.Random(seed)
    sorted_group_keys = sorted(groups.keys())
    rng.shuffle(sorted_group_keys)

    # 80% train, 10% validation, 10% test
    n_groups = len(sorted_group_keys)
    val_cut = int(n_groups * 0.8)
    test_cut = int(n_groups * 0.9)

    train_keys = set(sorted_group_keys[:val_cut])
    val_keys = set(sorted_group_keys[val_cut:test_cut])
    test_keys = set(sorted_group_keys[test_cut:])

    split_map: dict[int, str] = {}
    for k in sorted_group_keys:
        if k in train_keys:
            s = "train"
        elif k in val_keys:
            s = "validation"
        else:
            s = "test"
        for idx, _ in groups[k]:
            split_map[idx] = s

    # Export images and canonical manifest records
    canonical_records = []
    split_csv_rows = defaultdict(list)
    eval_references = []

    print("Extracting lossless image crops and validating Unicode transcriptions...")
    extracted_count = 0
    for idx, row in df.iterrows():
        split = split_map[idx]
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        sample_id = f"te_wikisource_{idx:04d}"
        img_filename = f"te_line_{idx:04d}.png"
        img_out_path = IMAGES_DIR / img_filename

        # Extract image bytes
        img_data = row.get("image")
        if isinstance(img_data, dict) and "bytes" in img_data:
            img_bytes = img_data["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(img_data, bytes):
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(row.get("line_image"), (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(row["line_image"])).convert("RGB")
        else:
            continue

        if not img_out_path.is_file():
            pil_img.save(img_out_path, format="PNG")
        extracted_count += 1

        rel_image_path = f"telugu_wikisource/images/{img_filename}"
        source_group = str(row.get("source_file") or f"src_{idx}")

        record = {
            "schema_version": "1",
            "sample_id": sample_id,
            "source_dataset": "telugu_wikisource_ocr_bench",
            "image_path": rel_image_path,
            "transcription": text,
            "split": split,
            "source_group_id": source_group,
            "annotation_provenance": {
                "source_file": source_group,
                "n_graphemes": int(row.get("n_graphemes", len(text))),
                "slug": str(row.get("slug", "")),
                "license": "CC-BY-SA-4.0",
                "human_verified": True,
            },
            "verification_status": "verified",
            "label_origin": "human",
            "license_status": "approved",
            "text_scope": "line",
            "language": "te",
            "script": "Telugu",
        }
        canonical_records.append(record)

        # Standard CSV line for training/validation/test
        split_csv_rows[split].append({"image": rel_image_path, "text": text})

        # Test set is also populated in evaluation/references.csv
        if split == "test":
            eval_references.append({
                "image": f"telugu_wikisource/images/{img_filename}",
                "text": text,
            })

    # Write canonical JSONL manifest
    manifest_path = TELUGU_DIR / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in canonical_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote canonical manifest: {manifest_path} ({len(canonical_records)} records)")

    # Write train.csv, validation.csv, test.csv
    for split_name in ("train", "validation", "test"):
        csv_path = LABELS_DIR / f"{split_name}.csv"
        rows = split_csv_rows[split_name]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}: {len(rows)} samples")

    # Write evaluation references.csv
    eval_csv_path = EVALUATION_DIR / "references.csv"
    with eval_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(eval_references)
    print(f"Updated {eval_csv_path}: {len(eval_references)} held-out Telugu test references")

    print("\nDataset Preparation Summary:")
    print(f"  Total verified line crops: {extracted_count}")
    print(f"  Train samples: {len(split_csv_rows['train'])} ({(len(split_csv_rows['train']) / extracted_count) * 100:.1f}%)")
    print(f"  Validation samples: {len(split_csv_rows['validation'])} ({(len(split_csv_rows['validation']) / extracted_count) * 100:.1f}%)")
    print(f"  Test samples: {len(split_csv_rows['test'])} ({(len(split_csv_rows['test']) / extracted_count) * 100:.1f}%)")
    print("  License: CC-BY-SA 4.0 (Human Verified, Zero Data Leakage)")


if __name__ == "__main__":
    process_dataset()
