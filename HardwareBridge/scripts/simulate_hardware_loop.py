"""CLI Simulation of the complete Smart Glasses Hardware Loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src directories to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "HardwareBridge" / "src"))
sys.path.insert(0, str(root_dir / "VideoModel" / "src"))
sys.path.insert(0, str(root_dir / "VisionModel" / "src"))
sys.path.insert(0, str(root_dir / "OCRModel" / "src"))
sys.path.insert(0, str(root_dir / "Backend" / "src"))
sys.path.insert(0, str(root_dir / "TTSModel" / "src"))

from smartvision_hardware.controller import SmartGlassesController


def generate_camera_burst() -> list[np.ndarray]:
    """Generates synthetic camera burst frames from glasses camera."""
    frames = []
    base = np.full((350, 550, 3), 220, dtype=np.uint8)
    # Medicine blister package
    cv2.rectangle(base, (100, 80), (450, 270), (180, 180, 190), -1)
    cv2.putText(base, "PARACETAMOL 500 MG", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    for i in range(5):
        kernel = max(1, i * 2 + 1)
        frames.append(cv2.GaussianBlur(base, (kernel, kernel), 0))
    return frames


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Smart Glasses Hardware Simulation")
    parser.add_argument("--language", "-l", choices=["en", "te", "hi"], default="te", help="Wearer preferred language")
    parser.add_argument("--scheduled", type=str, default="Paracetamol", help="Expected scheduled medicine")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  SmartVision Smart Glasses: Live Hardware Simulation  ")
    print(f"=======================================================")
    print(f"Wearer Language : {args.language}")
    print(f"Scheduled Dose  : {args.scheduled}")

    controller = SmartGlassesController(patient_language=args.language)
    print("\n[Peripherals] Wearer pressed capture button on glasses frame...")
    print("[Peripherals] Camera capturing 5-frame burst at 30 fps...")
    burst = generate_camera_burst()

    print("[Controller] Executing end-to-end SmartVision pipeline...")
    response = controller.process_medication_check(burst, scheduled_medicine=args.scheduled)

    print(f"\n--- Output Delivered to Glasses Hardware ---")
    print(f"Status        : {response.status}")
    print(f"Safe to take? : {'YES - SAFE' if response.is_safe else 'NO - WARNING'}")
    print(f"Display HUD   : {response.display_summary}")
    print(f"Speaker Spoken: {response.spoken_text}")
    if response.spoken_text_te:
        print(f"Telugu Audio  : {response.spoken_text_te}")
    if response.audio_file:
        print(f"Audio File    : {response.audio_file}")

    # Display GPS Telemetry and On-Lens Optical HUD Preview
    from smartvision_hardware.hud import SmartGlassesHUD
    from smartvision_backend.location import get_location_service

    loc = get_location_service().current_location
    maps_url = get_location_service().maps_client.get_google_maps_pin_url(loc.latitude, loc.longitude)

    print(f"\n--- Wearable GPS Telemetry & Google Maps ---")
    print(f"GPS Satellites: {loc.satellites} Locked")
    print(f"Location Pin  : {loc.latitude:.4f} N, {loc.longitude:.4f} E")
    print(f"Address       : {loc.address}")
    print(f"Google Maps   : {maps_url}")

    print(f"\n--- On-Lens Micro-OLED Optical HUD View ---")
    hud = SmartGlassesHUD(width=128, height=64)
    hud.display_verification_result(
        is_safe=response.is_safe,
        medicine_name=args.scheduled,
        dosage="1 Tablet",
    )
    print(hud.render_ascii_preview())
    print("=======================================================\n")


if __name__ == "__main__":
    main()
