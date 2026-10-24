"""SmartVision Video Processing Module.

Handles smart-glasses camera frames, filters out motion blur and specular glare,
and selects optimal keyframes for downstream OCR text recognition.
"""

from .frame_quality import FrameQuality, FrameQualityAssessor
from .keyframe_selector import KeyframeSelector, KeyframeResult

__all__ = [
    "FrameQuality",
    "FrameQualityAssessor",
    "KeyframeSelector",
    "KeyframeResult",
]
