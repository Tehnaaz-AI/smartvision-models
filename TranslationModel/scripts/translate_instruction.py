"""CLI Demonstration Script for SmartVision TranslationModel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_translation.translator import get_translator

SAMPLE_INSTRUCTIONS = [
    ("Take 1 tablet twice daily after food", "en", "te"),
    ("Take with water at bedtime", "en", "hi"),
    ("Store in a cool dry place", "en", "te"),
    ("Paracetamol 500 mg for fever", "en", "te"),
    ("Dolo 650 take 1 tablet after meals", "en", "hi"),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision Medical Translation Demo")
    parser.add_argument("--text", "-t", type=str, default=None, help="Text to translate")
    parser.add_argument("--source", "-s", type=str, default="en", choices=["en", "te", "hi"], help="Source language")
    parser.add_argument("--target", "-r", type=str, default="te", choices=["en", "te", "hi"], help="Target language")
    args = parser.parse_args()

    translator = get_translator()

    if args.text:
        items = [(args.text, args.source, args.target)]
    else:
        items = SAMPLE_INSTRUCTIONS

    print("\n--- SmartVision Medical Translation Demo ---")
    for text, src, tgt in items:
        res = translator.translate(text, source_lang=src, target_lang=tgt)
        print(f"\nSource ({src}) : {text}")
        print(f"Target ({tgt}) : {res.translated_text}")
        if res.preserved_terms:
            print(f"Preserved Medicine Names: {res.preserved_terms}")


if __name__ == "__main__":
    main()
