"""CLI Verification Demonstrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_backend.verifier import get_verifier
from smartvision_backend.scheduler import get_scheduler


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="SmartVision Medication Verification Demonstrator")
    parser.add_argument("--ocr-text", "-t", type=str, default="Dolo 500 mg", help="Detected OCR text")
    parser.add_argument("--strength", "-s", type=str, default="500 mg", help="Detected strength")
    parser.add_argument("--scheduled", type=str, default="Paracetamol", help="Expected scheduled medicine")
    args = parser.parse_args()

    scheduler = get_scheduler()
    verifier = get_verifier(patient=scheduler.patient)

    print(f"\n[SmartVision Backend] Cross-Verifying Capture against Prescription:")
    print(f"  Patient              : {scheduler.patient.patient_name} ({scheduler.patient.patient_id})")
    print(f"  Detected Text        : \"{args.ocr_text}\"")
    print(f"  Detected Strength    : {args.strength}")
    print(f"  Scheduled Due Dose   : {args.scheduled}")

    result = verifier.verify_capture(
        detected_text=args.ocr_text,
        detected_strength=args.strength,
        scheduled_medicine_name=args.scheduled,
    )

    print("\n--- Verification Verdict ---")
    print(f"  Status        : {result.status.value}")
    print(f"  Safe to Take? : {'YES - SAFE' if result.is_safe_to_take else 'NO - UNSAFE / DO NOT TAKE'}")
    print(f"  Action        : {result.action_required}")
    print(f"  Spoken (EN)   : {result.spoken_message}")
    if result.spoken_message_te:
        print(f"  Spoken (TE)   : {result.spoken_message_te}")


if __name__ == "__main__":
    main()
