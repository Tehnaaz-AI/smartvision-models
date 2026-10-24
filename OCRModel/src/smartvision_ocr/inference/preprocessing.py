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


def suppress_specular_glare_lens(image: Image.Image, glare_threshold: int = 240) -> Image.Image:
    """Google Lens-style specular glare suppression for metallic foil blister packs.

    Detects overexposed highlight regions on aluminum foil packaging and restores
    washed-out letter strokes using local neighborhood blending.
    """
    img_np = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Foil specular highlights have near-white intensity (high V, low S)
    glare_mask = ((v_channel >= glare_threshold) & (s_channel <= 50)).astype(np.uint8) * 255

    # If glare covers between 1% and 40% of the crop, apply inpaint restoration
    total_pixels = glare_mask.size
    glare_ratio = np.count_nonzero(glare_mask) / float(total_pixels)
    if 0.01 <= glare_ratio <= 0.40:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(glare_mask, kernel, iterations=1)
        restored = cv2.inpaint(img_np, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(restored)

    return image


def correct_orientation_lens(image: Image.Image) -> Image.Image:
    """Google Lens-style multi-angle orientation detection and auto-rectification.

    If a text line is captured vertically (e.g. medicine strip held sideways),
    rotates it 90 degrees to ensure horizontal baseline reading for OCR recognizers.
    """
    w, h = image.size
    # If crop is clearly taller than wide (vertical text or rotated strip)
    if h > w * 1.25:
        # Rotate 90 degrees counter-clockwise so characters align horizontally
        return image.rotate(270, expand=True)
    return image


def preprocess_crop_for_ocr(
    image: Image.Image,
    enhance_contrast: bool = True,
    denoise: bool = True,
    suppress_glare: bool = True,
    auto_orient: bool = True,
    normalize_height: bool = True,
    target_height: int = 48,
) -> Image.Image:
    """Full Google Lens-style preprocessing pipeline for text line crops before recognizer inference."""
    processed = image.convert("RGB")
    if auto_orient:
        processed = correct_orientation_lens(processed)
    if suppress_glare:
        processed = suppress_specular_glare_lens(processed)
    if enhance_contrast:
        processed = enhance_contrast_clahe(processed)
    if denoise:
        processed = denoise_and_sharpen(processed)
    if normalize_height:
        processed = normalize_crop_dimensions(processed, target_height=target_height)
    return processed
