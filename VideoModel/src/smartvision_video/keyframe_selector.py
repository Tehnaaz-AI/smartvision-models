"""Optimal Keyframe Selection from camera frame bursts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .frame_quality import FrameQuality, FrameQualityAssessor


@dataclass
class KeyframeResult:
    """Outcome of keyframe selection on a sequence of captured frames."""

    best_frame: Optional[np.ndarray]
    best_frame_index: int
    best_quality: Optional[FrameQuality]
    frames_evaluated: int
    usable_frames_count: int
    success: bool
    status_message: str


class KeyframeSelector:
    """Buffers consecutive frames and extracts the single clearest, non-blurry candidate."""

    def __init__(self, blur_threshold: float = 100.0, min_quality_score: float = 0.35):
        self.assessor = FrameQualityAssessor(blur_threshold=blur_threshold)
        self.min_quality_score = min_quality_score

    def select_best_frame(self, frames: List[np.ndarray]) -> KeyframeResult:
        """Evaluates a list/burst of camera frames and returns the top keyframe."""
        if not frames:
            return KeyframeResult(
                best_frame=None,
                best_frame_index=-1,
                best_quality=None,
                frames_evaluated=0,
                usable_frames_count=0,
                success=False,
                status_message="No frames provided for selection.",
            )

        best_idx = -1
        best_score = -1.0
        best_quality = None
        usable_count = 0

        for i, frame in enumerate(frames):
            quality = self.assessor.assess(frame)
            if not quality.is_blurry and not quality.is_underexposed:
                usable_count += 1

            if quality.composite_score > best_score:
                best_score = quality.composite_score
                best_idx = i
                best_quality = quality

        if best_idx >= 0 and best_score >= self.min_quality_score:
            return KeyframeResult(
                best_frame=frames[best_idx],
                best_frame_index=best_idx,
                best_quality=best_quality,
                frames_evaluated=len(frames),
                usable_frames_count=usable_count,
                success=True,
                status_message=f"Optimal keyframe selected (Frame {best_idx}, score={best_score:.2f})",
            )
        elif best_idx >= 0:
            # Fallback frame exists but below quality threshold (e.g. user moved too fast)
            return KeyframeResult(
                best_frame=frames[best_idx],
                best_frame_index=best_idx,
                best_quality=best_quality,
                frames_evaluated=len(frames),
                usable_frames_count=usable_count,
                success=False,
                status_message="All frames are excessively blurry or poorly illuminated. Please hold medicine steady.",
            )

        return KeyframeResult(
            best_frame=None,
            best_frame_index=-1,
            best_quality=None,
            frames_evaluated=len(frames),
            usable_frames_count=0,
            success=False,
            status_message="Evaluation failed to produce a valid candidate.",
        )
