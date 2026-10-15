"""Convert SmartVision split manifests into PaddleOCR / CTC training format.

Outputs:
- data/paddle_format/rec_gt_train.txt: <image_path>\\t<text>
- data/paddle_format/rec_gt_val.txt:   <image_path>\\t<text>
- data/paddle_format/rec_gt_test.txt:  <image_path>\\t<text>
- data/paddle_format/dict_telugu.txt:  Unique character dictionary
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
OUT_DIR = DATA_DIR / "paddle_format"


def prepare_paddle_format() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_characters: set[str] = set()
    split_counts: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        src_csv = LABELS_DIR / f"{split}.csv"
        target_name = "val" if split == "validation" else split
        out_txt = OUT_DIR / f"rec_gt_{target_name}.txt"

        if not src_csv.is_file():
            print(f"Warning: {src_csv} not found, skipping.")
            continue

        lines: list[str] = []
        with src_csv.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row.get("image", "").strip()
                text = row.get("text", "").strip()
                if not img_rel or not text:
                    continue
                # Normalize path separators for cross-platform compatibility
                clean_img_rel = img_rel.replace("\\", "/")
                lines.append(f"{clean_img_rel}\t{text}")
                for char in text:
                    if char not in ("\n", "\r", "\t"):
                        all_characters.add(char)

        with out_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        split_counts[split] = len(lines)
        print(f"Generated {out_txt.name}: {len(lines)} samples")

    # Character dictionary sorted by codepoint
    sorted_chars = sorted(all_characters)
    dict_file = OUT_DIR / "dict_telugu.txt"
    with dict_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(sorted_chars) + "\n")

    print(f"\nExtracted Character Dictionary: {dict_file}")
    print(f"Total Unique Characters / Graphemes: {len(sorted_chars)}")
    print("\nDataset Preparation for PaddleOCR / CTC Training Complete:")
    for split, count in split_counts.items():
        print(f"  - {split}: {count} lines")


if __name__ == "__main__":
    prepare_paddle_format()
