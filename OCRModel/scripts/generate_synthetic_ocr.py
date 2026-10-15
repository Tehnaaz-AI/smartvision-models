"""Generate domain-specific synthetic OCR training and testing datasets.

Generates realistic printed text crops for:
1. Medicine names, formulations, and dosage instructions (Telugu, Hindi, English).
2. Assistive signage (Exits, entrances, hospital departments, warnings).
3. Realistic optical variations (light angle skew, contrast jitter, noise).

Outputs:
- data/synthetic_domain/images/*.png
- data/synthetic_domain/records.jsonl
- data/synthetic_domain/train.csv
- data/synthetic_domain/test.csv
- data/synthetic_domain/rec_gt_train.txt
- data/synthetic_domain/rec_gt_test.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synthetic_domain"
IMAGES_DIR = DATA_DIR / "images"

# Standard Windows Indic font
DEFAULT_FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"

# Curated domain text entries: (Text, Language, Category)
DOMAIN_TEXT_CORPUS: list[tuple[str, str, str]] = [
    # Telugu Medicines & Dosages
    ("పారాసిటమాల్ 500 మి.గ్రా", "te", "medicine"),
    ("డోలో 650 మాత్రలు", "te", "medicine"),
    ("సిట్రిజిన్ 10 మి.గ్రా", "te", "medicine"),
    ("అమోక్సిసిలిన్ 250 మి.గ్రా", "te", "medicine"),
    ("మెట్‌ఫార్మిన్ 500 మి.గ్రా", "te", "medicine"),
    ("ఆస్పిరిన్ 75 మి.గ్రా", "te", "medicine"),
    ("రోజుకు రెండు సార్లు", "te", "dosage"),
    ("భోజనం తర్వాత తీసుకోండి", "te", "dosage"),
    ("ఉదయం మరియు రాత్రి", "te", "dosage"),
    ("1 టాబ్లెట్ నీటితో", "te", "dosage"),
    ("కడుపు నిండా భోజనం తర్వాత", "te", "dosage"),
    ("డాక్టర్ సలహా మేరకు మాత్రమే", "te", "instruction"),
    ("పిల్లలకు దూరంగా ఉంచండి", "te", "instruction"),
    ("చల్లని మరియు పొడి ప్రదేశంలో ఉంచండి", "te", "instruction"),
    # Telugu Assistive Signage
    ("అత్యవసర నిష్క్రమణ", "te", "signage"),
    ("ప్రవేశ ద్వారం", "te", "signage"),
    ("నిష్క్రమణ ద్వారం", "te", "signage"),
    ("ఆసుపత్రి", "te", "signage"),
    ("మందుల దుకాణం", "te", "signage"),
    ("శౌచాలయం", "te", "signage"),
    ("మొదటి అంతస్తు", "te", "signage"),
    ("లిఫ్ట్", "te", "signage"),
    ("మెట్లు", "te", "signage"),
    ("జాగ్రత్త: తడి నేల", "te", "signage"),
    ("నిశ్శబ్దంగా ఉండండి", "te", "signage"),
    # Hindi Medicines & Signage
    ("पैरासिटामोल 500 मि.ग्रा.", "hi", "medicine"),
    ("डोलो 650 गोलियाँ", "hi", "medicine"),
    ("सिट्रिजिन 10 मि.ग्रा.", "hi", "medicine"),
    ("दिन में दो बार", "hi", "dosage"),
    ("भोजन के बाद लें", "hi", "dosage"),
    ("आपातकालीन निकास", "hi", "signage"),
    ("प्रवेश द्वार", "hi", "signage"),
    ("दवाई की दुकान", "hi", "signage"),
    ("अस्पताल", "hi", "signage"),
    ("कृपया शांति बनाए रखें", "hi", "signage"),
    # English Medicines & Signage
    ("Paracetamol 500 mg", "en", "medicine"),
    ("Dolo 650 Tablets", "en", "medicine"),
    ("Cetirizine 10 mg", "en", "medicine"),
    ("Amoxicillin 250 mg", "en", "medicine"),
    ("Metformin 500 mg", "en", "medicine"),
    ("Take 1 tablet after food", "en", "dosage"),
    ("Twice daily morning & night", "en", "dosage"),
    ("Keep out of reach of children", "en", "instruction"),
    ("Store in a cool dry place", "en", "instruction"),
    ("Emergency Exit", "en", "signage"),
    ("Entrance Only", "en", "signage"),
    ("Pharmacy", "en", "signage"),
    ("Hospital Reception", "en", "signage"),
    ("Restroom", "en", "signage"),
    ("Caution Wet Floor", "en", "signage"),
    ("Do Not Enter", "en", "signage"),
]


def render_synthetic_crop(
    text: str,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = 30,
    add_augmentation: bool = True,
) -> Image.Image:
    """Render a realistic single-line text crop with optional camera artifacts."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    # Calculate text bounding box
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(bbox[2] - bbox[0], 20)
    text_h = max(bbox[3] - bbox[1], 16)

    # Padding
    pad_x = random.randint(12, 24)
    pad_y = random.randint(8, 16)
    canvas_w = text_w + pad_x * 2
    canvas_h = text_h + pad_y * 2

    # Background color (subtle off-white, light gray, pale cream)
    bg_val = random.randint(238, 255)
    bg_color = (bg_val, bg_val - random.randint(0, 5), bg_val - random.randint(0, 8))

    # Text color (dark charcoal or black)
    fg_val = random.randint(10, 45)
    fg_color = (fg_val, fg_val, fg_val)

    img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    draw.text((pad_x, pad_y), text, font=font, fill=fg_color)

    if add_augmentation:
        # 1. Slight rotation (-2 to +2 degrees)
        angle = random.uniform(-2.5, 2.5)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=bg_color)

        # 2. Slight contrast / brightness jitter
        if random.random() > 0.4:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.85, 1.25))

        # 3. Slight sensor noise
        if random.random() > 0.5:
            arr = np.array(img, dtype=np.int16)
            noise = np.random.normal(0, random.uniform(2, 6), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # 4. Subtle blur occasionally
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img


def generate_synthetic_dataset(
    target_count: int = 200,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> None:
    """Generate synthetic samples, splits, and manifests."""
    random.seed(seed)
    np.random.seed(seed)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    paddle_train_lines: list[str] = []
    paddle_test_lines: list[str] = []

    corpus_len = len(DOMAIN_TEXT_CORPUS)
    print(f"Generating {target_count} synthetic domain text crops...")

    for i in range(target_count):
        text_entry = DOMAIN_TEXT_CORPUS[i % corpus_len]
        text, lang, category = text_entry

        # Add random variation occasionally (e.g. variation in numbers/dosages)
        if category in ("medicine", "dosage") and random.random() > 0.5:
            num = random.choice([100, 250, 500, 650, 1000])
            if "500" in text:
                text = text.replace("500", str(num))
            elif "650" in text:
                text = text.replace("650", str(num))

        sample_id = f"syn_{i:04d}"
        filename = f"{sample_id}.png"
        out_path = IMAGES_DIR / filename

        font_size = random.choice([26, 28, 30, 32, 34])
        crop_img = render_synthetic_crop(text, font_size=font_size)
        crop_img.save(out_path, format="PNG")

        is_test = (i % int(1 / test_ratio)) == 0 if test_ratio > 0 else False
        split = "test" if is_test else "train"
        rel_path = f"synthetic_domain/images/{filename}"

        record = {
            "sample_id": sample_id,
            "image_path": rel_path,
            "text": text,
            "language": lang,
            "category": category,
            "split": split,
            "origin": "synthetic",
        }
        records.append(record)

        csv_row = {"image": rel_path, "text": text}
        paddle_line = f"{rel_path}\t{text}"

        if split == "test":
            test_rows.append(csv_row)
            paddle_test_lines.append(paddle_line)
        else:
            train_rows.append(csv_row)
            paddle_train_lines.append(paddle_line)

    # Save JSONL manifest
    jsonl_path = DATA_DIR / "records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSVs
    for split_name, rows in [("train", train_rows), ("test", test_rows)]:
        csv_path = DATA_DIR / f"{split_name}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(rows)

    # Save PaddleOCR format
    with (DATA_DIR / "rec_gt_train.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(paddle_train_lines) + "\n")
    with (DATA_DIR / "rec_gt_test.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(paddle_test_lines) + "\n")

    print(f"\nSynthetic Dataset Generated successfully in {DATA_DIR}:")
    print(f"  - Total samples: {target_count}")
    print(f"  - Train set: {len(train_rows)} images")
    print(f"  - Test set: {len(test_rows)} images")
    print(f"  - Manifest: {jsonl_path}")
    print(f"  - Paddle format: {DATA_DIR / 'rec_gt_train.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic domain OCR crops.")
    parser.add_argument("--count", type=int, default=200, help="Number of crops to generate.")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Ratio for test split.")
    args = parser.parse_args()
    generate_synthetic_dataset(target_count=args.count, test_ratio=args.test_ratio)
