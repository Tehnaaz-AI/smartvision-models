"""Unit Tests for SmartVision Video Processing and Keyframe Selection."""

import cv2
import numpy as np
import pytest

from smartvision_video.frame_quality import FrameQualityAssessor, FrameQuality
from smartvision_video.keyframe_selector import KeyframeSelector, KeyframeResult


def test_frame_quality_sharp_vs_blurry():
    assessor = FrameQualityAssessor(blur_threshold=100.0)

    # Sharp image with high-frequency edges
    sharp_img = np.zeros((200, 300), dtype=np.uint8)
    cv2.putText(sharp_img, "SHARP TEXT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)

    sharp_q = assessor.assess(sharp_img)
    assert not sharp_q.is_blurry
    assert sharp_q.sharpness_score > 100.0

    # Heavily blurred image
    blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
    blurry_q = assessor.assess(blurry_img)
    assert blurry_q.is_blurry
    assert blurry_q.sharpness_score < sharp_q.sharpness_score


def test_frame_quality_exposure():
    assessor = FrameQualityAssessor(min_brightness=40.0, max_brightness=220.0)

    # Pitch black frame
    dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
    dark_q = assessor.assess(dark_img)
    assert dark_q.is_underexposed
    assert not dark_q.is_overexposed

    # Overexposed blown out frame
    bright_img = np.full((100, 100, 3), 245, dtype=np.uint8)
    bright_q = assessor.assess(bright_img)
    assert bright_q.is_overexposed
    assert not bright_q.is_underexposed


def test_keyframe_selector_picks_sharpest():
    selector = KeyframeSelector()

    base = np.zeros((150, 250), dtype=np.uint8)
    cv2.putText(base, "TEST", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

    # Create sequence: Blurry, Sharp, Blurry
    f0 = cv2.GaussianBlur(base, (15, 15), 0)
    f1 = base.copy()  # Sharpest!
    f2 = cv2.GaussianBlur(base, (21, 21), 0)

    result = selector.select_best_frame([f0, f1, f2])
    assert result.success is True
    assert result.best_frame_index == 1
    assert result.frames_evaluated == 3


def test_keyframe_selector_empty_list():
    selector = KeyframeSelector()
    result = selector.select_best_frame([])
    assert result.success is False
    assert result.best_frame is None
