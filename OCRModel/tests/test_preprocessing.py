"""Unit tests for SmartVision preprocessing module."""

import numpy as np
from PIL import Image

from smartvision_ocr.inference.preprocessing import (
    denoise_and_sharpen,
    enhance_contrast_clahe,
    normalize_crop_dimensions,
    preprocess_crop_for_ocr,
    suppress_specular_glare_lens,
    correct_orientation_lens,
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


def test_suppress_specular_glare_lens():
    # Create image with high-glare foil hotspot
    arr = np.full((60, 100, 3), 120, dtype=np.uint8)
    arr[20:35, 30:50] = 255  # Specular glare
    img = Image.fromarray(arr)
    glare_reduced = suppress_specular_glare_lens(img, glare_threshold=240)
    assert glare_reduced.size == (100, 60)
    assert glare_reduced.mode == "RGB"


def test_correct_orientation_lens():
    # Vertical crop (height 120, width 40) should be rotated to horizontal (120, 40) -> (40, 120)
    img = Image.new("RGB", (40, 120), color="blue")
    oriented = correct_orientation_lens(img)
    assert oriented.size[0] > oriented.size[1]


def test_preprocess_crop_for_ocr_pipeline():
    img = Image.new("RGB", (64, 32), color=(220, 220, 220))
    result = preprocess_crop_for_ocr(img, target_height=48)
    assert result.size[1] == 48
    assert result.mode == "RGB"
