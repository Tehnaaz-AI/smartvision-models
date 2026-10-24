"""CLI Demonstration Script for SmartVision Keyframe Selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_video.keyframe_selector import KeyframeSelector


def generate_simulated_burst(count: int = 10) -> list[np.ndarray]:
    """Simulates a 10-frame smart-glasses camera burst with varying motion blur."""
    frames = []
    base = np.full((300, 500, 3), 230, dtype=np.uint8)
    # Draw crisp medicine text
    cv2.putText(base, "PARACETAMOL 500 MG", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    for i in range(count):
        # Simulate shake: frames at beginning and end have heavy motion blur
        blur_kernel = max(1, abs(i - 4) * 4 + 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        noisy_frame = cv2.GaussianBlur(base, (blur_kernel, blur_kernel), 0)
        frames.append(noisy_frame)

    return frames


def main():
    parser = argparse.ArgumentParser(description="SmartVision Keyframe Selector Demo")
    parser.add_argument("--burst-size", type=int, default=10, help="Number of simulated burst frames")
    parser.add_argument("--output", type=str, default="best_keyframe.jpg", help="Path to save chosen frame")
    args = parser.parse_args()

    print(f"\n[SmartVision Video] Generating {args.burst_size}-frame camera burst with motion blur...")
    burst = generate_simulated_burst(args.burst_size)

    selector = KeyframeSelector()
    result = selector.select_best_frame(burst)

    print(f"\nResults:")
    print(f"  Status        : {result.status_message}")
    print(f"  Frames Tested : {result.frames_evaluated}")
    print(f"  Usable Frames : {result.usable_frames_count}")
    if result.best_quality:
        print(f"  Best Frame Idx: Frame #{result.best_frame_index}")
        print(f"  Sharpness Var : {result.best_quality.sharpness_score}")
        print(f"  Brightness    : {result.best_quality.brightness}")
        print(f"  Composite Q   : {result.best_quality.composite_score:.2f}")

    if result.best_frame is not None:
        cv2.imwrite(args.output, result.best_frame)
        print(f"  Saved Image   : {args.output}")


if __name__ == "__main__":
    main()
