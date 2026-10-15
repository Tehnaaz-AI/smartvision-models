"""Image enhancement and normalization preprocessing for scene OCR crops.

Addresses smart-glasses real-world camera artifacts:
- Uneven lighting and low-contrast medicine packaging.
- Sensor grain and camera motion blur.
- Variable crop resolutions and aspect ratios.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def enhance_contrast_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> Image.Image:
    """Enhance local contrast using CLAHE in LAB color space.

    Preserves color balance while enhancing faint or shadowed text.
    """
    img_np = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)


def denoise_and_sharpen(image: Image.Image) -> Image.Image:
    """Remove high-frequency sensor noise while keeping character edges crisp."""
    img_np = np.array(image.convert("RGB"))
    # Bilateral filter preserves sharp character edges while smoothing flat noise
    denoised = cv2.bilateralFilter(img_np, d=5, sigmaColor=35, sigmaSpace=35)
    pil_denoised = Image.fromarray(denoised)

    # Subtle unsharp mask to accentuate Indic vowel matras and conjuncts
    enhancer = ImageEnhance.Sharpness(pil_denoised)
    return enhancer.enhance(1.2)


def normalize_crop_dimensions(
    image: Image.Image,
    target_height: int = 48,
    min_width: int = 16,
    max_width: int = 1200,
) -> Image.Image:
    """Proportionally resize line crop to canonical recognizer height (48px for PP-OCR)."""
    w, h = image.size
    if h <= 0 or w <= 0:
        return image

    scale = target_height / float(h)
    new_w = max(min_width, min(int(round(w * scale)), max_width))
    return image.resize((new_w, target_height), resample=Image.Resampling.BILINEAR)


def preprocess_crop_for_ocr(
    image: Image.Image,
    enhance_contrast: bool = True,
    denoise: bool = True,
    normalize_height: bool = True,
    target_height: int = 48,
) -> Image.Image:
    """Full preprocessing pipeline for text line crops before recognizer inference."""
    processed = image.convert("RGB")
    if enhance_contrast:
        processed = enhance_contrast_clahe(processed)
    if denoise:
        processed = denoise_and_sharpen(processed)
    if normalize_height:
        processed = normalize_crop_dimensions(processed, target_height=target_height)
    return processed
