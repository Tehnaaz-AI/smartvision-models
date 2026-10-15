"""Unit tests for SmartVision preprocessing module."""

import numpy as np
from PIL import Image

from smartvision_ocr.inference.preprocessing import (
    denoise_and_sharpen,
    enhance_contrast_clahe,
    normalize_crop_dimensions,
    preprocess_crop_for_ocr,
)


def test_enhance_contrast_clahe_preserves_dimensions():
    img = Image.new("RGB", (120, 48), color=(200, 200, 200))
    enhanced = enhance_contrast_clahe(img)
    assert enhanced.size == (120, 48)
    assert enhanced.mode == "RGB"


def test_normalize_crop_dimensions():
    # Crop with height 24 and width 100 should scale to height 48 and width 200
    img = Image.new("RGB", (100, 24), color="white")
    normalized = normalize_crop_dimensions(img, target_height=48)
    assert normalized.size == (200, 48)


def test_denoise_and_sharpen():
    arr = np.random.randint(0, 255, (48, 120, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    processed = denoise_and_sharpen(img)
    assert processed.size == (120, 48)
    assert processed.mode == "RGB"


def test_preprocess_crop_for_ocr_pipeline():
    img = Image.new("RGB", (64, 32), color=(220, 220, 220))
    result = preprocess_crop_for_ocr(img, target_height=48)
    assert result.size[1] == 48
    assert result.mode == "RGB"
