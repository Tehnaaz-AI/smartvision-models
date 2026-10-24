"""Frame Quality Assessment for Smart Glasses Camera Streams."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FrameQuality:
    """Quantitative image metrics for camera frames."""

    sharpness_score: float  # Variance of Laplacian
    is_blurry: bool
    brightness: float  # Mean pixel intensity (0 - 255)
    is_underexposed: bool
    is_overexposed: bool
    glare_ratio: float  # Fraction of saturated white pixels (> 250)
    composite_score: float  # Overall suitability (0.0 to 1.0)


class FrameQualityAssessor:
    """Evaluates sharpness, exposure, and glare in camera frames."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        max_glare_ratio: float = 0.15,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_glare_ratio = max_glare_ratio

    def assess(self, frame: np.ndarray) -> FrameQuality:
        """Calculates image quality metrics on a single frame."""
        if frame is None or frame.size == 0:
            return FrameQuality(
                sharpness_score=0.0,
                is_blurry=True,
                brightness=0.0,
                is_underexposed=True,
                is_overexposed=False,
                glare_ratio=0.0,
                composite_score=0.0,
            )

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Sharpness via Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        is_blurry = sharpness < self.blur_threshold

        # 2. Exposure & Brightness
        brightness = float(np.mean(gray))
        is_underexposed = brightness < self.min_brightness
        is_overexposed = brightness > self.max_brightness

        # 3. Specular glare (e.g. foil reflections on blister packs)
        saturated_pixels = np.count_nonzero(gray >= 250)
        glare_ratio = float(saturated_pixels) / float(gray.size)

        # 4. Composite Quality Score (0.0 to 1.0)
        # S-curve normalization for sharpness
        norm_sharpness = min(1.0, sharpness / (self.blur_threshold * 3.0))

        # Penalty for under/over exposure
        exp_score = 1.0
        if is_underexposed:
            exp_score = max(0.1, brightness / self.min_brightness)
        elif is_overexposed:
            exp_score = max(0.1, (255.0 - brightness) / (255.0 - self.max_brightness))

        # Penalty for glare
        glare_penalty = max(0.0, (glare_ratio - 0.02) * 2.0)
        glare_score = max(0.0, 1.0 - glare_penalty)

        composite = round(0.55 * norm_sharpness + 0.25 * exp_score + 0.20 * glare_score, 3)

        return FrameQuality(
            sharpness_score=round(sharpness, 2),
            is_blurry=is_blurry,
            brightness=round(brightness, 2),
            is_underexposed=is_underexposed,
            is_overexposed=is_overexposed,
            glare_ratio=round(glare_ratio, 4),
            composite_score=max(0.0, min(1.0, composite)),
        )
