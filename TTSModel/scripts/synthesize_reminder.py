"""CLI Demonstration Script for SmartVision Text-to-Speech (TTS).

Synthesizes sample medication reminders in Telugu, Hindi, or English.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_tts.engine import TTSEngine, get_tts_engine

SAMPLE_REMINDERS = {
    "en": "Reminder: It is time to take your Paracetamol 500 milligram tablet with water.",
    "te": "జ్ఞాపిక: నీటితో మీ పారాసిటమాల్ 500 మిల్లీగ్రాముల టాబ్లెట్ తీసుకోవడానికి సమయం అయింది.",
    "hi": "स्मरण: पानी के साथ पैरासिटामोल 500 मिलीग्राम की गोली लेने का समय हो गया है।",
}


def main():
    parser = argparse.ArgumentParser(description="SmartVision TTS Medication Reminder Synthesizer")
    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "te", "hi"],
        default="en",
        help="Language for spoken reminder (default: en)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Custom text to synthesize. Defaults to language sample reminder.",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        default=140,
        help="Speech speed in words per minute (default: 140 for elderly users)",
    )
    args = parser.parse_args()

    # Configure stdout for UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    text = args.text or SAMPLE_REMINDERS[args.language]
    print(f"\n[SmartVision TTS] Synthesizing ({args.language}):")
    print(f"Message: {text}")

    engine = get_tts_engine(language=args.language, speech_rate=args.rate)
    result = engine.synthesize(text=text)

    if result.success:
        print(f"Synthesis successful!")
        print(f"Audio file saved to: {result.audio_path}")
        print(f"Estimated duration: {result.duration_seconds} seconds")
        print(f"Engine: {result.engine_name}")
    else:
        print(f"Synthesis failed: {result.error_message}")


if __name__ == "__main__":
    main()
