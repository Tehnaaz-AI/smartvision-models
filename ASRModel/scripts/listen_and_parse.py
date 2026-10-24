"""CLI Demo for SmartVision Automatic Speech Recognition (ASR).

Parses sample spoken commands in Telugu, Hindi, and English into structured intents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_asr.engine import get_asr_engine
from smartvision_asr.intents import UserIntent

SAMPLE_COMMANDS = [
    "What is this medicine?",
    "ఈ మందు పేరు ఏమిటి?",
    "यह कौन सी दवा है?",
    "When should I take this tablet?",
    "ఎన్ని టాబ్లెట్లు వేసుకోవాలి?",
    "खाने के बाद कितनी गोली लेनी है?",
    "Help me please call doctor",
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision ASR Intent Parser Demo")
    parser.add_argument("--query", "-q", type=str, default=None, help="Custom query string")
    args = parser.parse_args()

    engine = get_asr_engine()

    queries = [args.query] if args.query else SAMPLE_COMMANDS

    print("\n--- SmartVision ASR Intent Recognition Demo ---")
    for q in queries:
        result = engine.process_text_command(q)
        print(f"\nUtterance : \"{q}\"")
        print(f"  Language: {result.language}")
        print(f"  Intent  : {result.intent.value} (Confidence: {result.confidence:.2f})")


if __name__ == "__main__":
    main()
