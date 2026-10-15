"""End-to-end OCR pipeline for general scene, sign, and document images.

Integrates text detection (PaddleOCR DBNet), reading-order reconstruction,
region cropping, text recognition (TrOCR / swappable recognizers), and
script detection into a structured, modular contract for downstream
modules (TTS, translation, LLM, or visual display).
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError

from smartvision_ocr.config import MODEL_NAME_OR_PATH
from smartvision_ocr.inference.detect_and_ocr import (
    DETECTOR_MODEL_NAME,
    TextRegion,
    load_text_detector,
    parse_detector_results,
)
from smartvision_ocr.inference.predict import (
    load_model,
    predict_image,
    select_device,
)

LOGGER = logging.getLogger(__name__)


def detect_scripts(text: str) -> list[str]:
    """Detect writing scripts (Telugu, Devanagari, Latin, etc.) present in text."""
    scripts: set[str] = set()
    for char in text:
        codepoint = ord(char)
        if 0x0C00 <= codepoint <= 0x0C7F:
            scripts.add("Telugu")
        elif 0x0900 <= codepoint <= 0x097F:
            scripts.add("Devanagari")
        elif (
            (0x0041 <= codepoint <= 0x005A)
            or (0x0061 <= codepoint <= 0x007A)
            or (0x00C0 <= codepoint <= 0x024F)
        ):
            scripts.add("Latin")
    return sorted(scripts)


@dataclass(frozen=True)
class RegionResult:
    """Individual recognized text region with geometry and metadata."""

    region_index: int
    reading_order: int
    polygon: tuple[tuple[float, float], ...]
    recognized_text: str
    detector_score: float | None = None
    scripts: list[str] = field(default_factory=list)
    status: str = "success"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_index": self.region_index,
            "reading_order": self.reading_order,
            "polygon": [list(point) for point in self.polygon],
            "recognized_text": self.recognized_text,
            "detector_score": self.detector_score,
            "scripts": list(self.scripts),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class OCRResult:
    """Structured end-to-end OCR result meeting the SmartVision contract."""

    text: str
    regions: list[RegionResult]
    image_path: str | None = None
    image_size: tuple[int, int] = (0, 0)
    detector_model: str = DETECTOR_MODEL_NAME
    recognizer_model: str = MODEL_NAME_OR_PATH
    device: str = "cpu"
    detected_scripts: list[str] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    postprocessed_text: str = ""
    domain_insights: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "postprocessed_text": self.postprocessed_text,
            "domain_insights": dict(self.domain_insights),
            "regions": [region.to_dict() for region in self.regions],
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "detector_model": self.detector_model,
            "recognizer_model": self.recognizer_model,
            "device": self.device,
            "detected_scripts": list(self.detected_scripts),
            "timing_ms": dict(self.timing_ms),
            "status": self.status,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def load_multilingual_recognizer(language: str = "telugu", device: str = "cpu") -> Any:
    """Load lightweight mobile PP-OCRv5 text recognizer for Telugu, Hindi, or English."""
    from paddleocr import TextRecognition

    model_map = {
        "telugu": "te_PP-OCRv5_mobile_rec",
        "te": "te_PP-OCRv5_mobile_rec",
        "hindi": "devanagari_PP-OCRv5_mobile_rec",
        "hi": "devanagari_PP-OCRv5_mobile_rec",
        "devanagari": "devanagari_PP-OCRv5_mobile_rec",
        "english": "en_PP-OCRv5_mobile_rec",
        "en": "en_PP-OCRv5_mobile_rec",
        "auto": "te_PP-OCRv5_mobile_rec",
    }
    model_name = model_map.get(language.lower(), "te_PP-OCRv5_mobile_rec")
    return TextRecognition(
        model_name=model_name,
        device=device,
        engine="paddle_static",
        enable_mkldnn=False,
    )


def predict_full_image(
    image: str | Path | Image.Image,
    *,
    detector: Any | None = None,
    detector_model: str = DETECTOR_MODEL_NAME,
    model: Any | None = None,
    processor: Any | None = None,
    recognizer: Any | None = None,
    recognizer_model: str = MODEL_NAME_OR_PATH,
    device: str | None = None,
    language: str = "auto",
    preprocess: bool = True,
    postprocess: bool = True,
) -> OCRResult:
    """Run full end-to-end OCR on an arbitrary scene, sign, or document image.

    Flow:
    1. Validate/convert input to RGB PIL Image.
    2. Detect text regions using PaddleOCR detector.
    3. Rectify and order text crops in natural reading order.
    4. Recognize each text region (via lightweight PP-OCRv5 or TrOCR).
    5. Reconstruct full readable text, detect scripts, and measure latency.
    """
    total_start = time.perf_counter()
    image_path_str: str | None = None

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        path = Path(image).expanduser()
        image_path_str = str(path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        try:
            with Image.open(path) as source:
                pil_image = source.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ValueError(f"Could not read image '{path}': {error}") from error

    selected_device = select_device(device)
    device_name = str(selected_device)
    timing: dict[str, float] = {}

    # 1. Load detector and recognizer if not provided
    if detector is None:
        detector = load_text_detector(detector_model)

    active_rec_name = recognizer_model
    if recognizer is None and (model is None and processor is None):
        if language.lower() in ("trocr", "legacy"):
            processor, model = load_model(recognizer_model)
            active_rec_name = recognizer_model
        else:
            try:
                recognizer = load_multilingual_recognizer(language, device="cpu")
                active_rec_name = getattr(recognizer, "model_name", f"{language}_PP-OCRv5_mobile_rec")
            except Exception:
                processor, model = load_model(recognizer_model)
                active_rec_name = recognizer_model
    elif model is not None and processor is None:
        raise ValueError("Provide both recognizer model and processor, or neither.")

    # 2. Text detection
    detection_start = time.perf_counter()
    try:
        # PaddleOCR detector accepts a file path or numpy array
        detector_input = image_path_str if image_path_str else pil_image
        # If detector expects a file path string or numpy array
        try:
            detected_output = detector.predict(detector_input)
        except (TypeError, ValueError, AttributeError):
            import numpy as np

            detected_output = detector.predict(np.array(pil_image))

        regions: list[TextRegion] = parse_detector_results(detected_output, pil_image)
    except Exception as error:
        timing["detection_ms"] = round((time.perf_counter() - detection_start) * 1000, 2)
        timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        LOGGER.exception("Detection failed: %s", error)
        return OCRResult(
            text="",
            regions=[],
            image_path=image_path_str,
            image_size=pil_image.size,
            detector_model=detector_model,
            recognizer_model=recognizer_model,
            device=device_name,
            timing_ms=timing,
            status="detection_failed",
            error=f"{type(error).__name__}: {error}",
        )

    timing["detection_ms"] = round((time.perf_counter() - detection_start) * 1000, 2)

    if not regions:
        timing["recognition_ms"] = 0.0
        timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        return OCRResult(
            text="",
            regions=[],
            image_path=image_path_str,
            image_size=pil_image.size,
            detector_model=detector_model,
            recognizer_model=recognizer_model,
            device=device_name,
            timing_ms=timing,
            status="no_text_regions",
        )

    # 3. Recognition for each detected region in reading order
    recognition_start = time.perf_counter()
    recognized_regions: list[RegionResult] = []
    lines_text: list[str] = []

    for index, region in enumerate(regions):
        region_text = ""
        region_status = "success"
        region_error = ""

        crop_to_rec = region.crop
        if preprocess:
            try:
                from smartvision_ocr.inference.preprocessing import preprocess_crop_for_ocr

                crop_to_rec = preprocess_crop_for_ocr(region.crop)
            except Exception as prep_err:
                LOGGER.debug("Preprocessing fallback for region %d: %s", index, prep_err)
                crop_to_rec = region.crop

        try:
            if hasattr(recognizer, "predict"):
                import numpy as np

                crop_array = np.array(crop_to_rec)
                preds = recognizer.predict(crop_array)
                if preds and len(preds) > 0 and isinstance(preds[0], dict):
                    region_text = str(preds[0].get("rec_text", "")).strip()
            elif model is not None and processor is not None:
                region_text = predict_image(
                    crop_to_rec,
                    model=model,
                    processor=processor,
                    device=device_name,
                    model_name_or_path=recognizer_model,
                ).strip()
            if not region_text:
                region_status = "empty_text"
        except Exception as error:
            region_status = "recognition_failed"
            region_error = f"{type(error).__name__}: {error}"
            LOGGER.warning("Recognition failed for region %d: %s", index, error)

        scripts = detect_scripts(region_text)
        if region_text:
            lines_text.append(region_text)

        recognized_regions.append(
            RegionResult(
                region_index=index,
                reading_order=index,
                polygon=region.polygon,
                recognized_text=region_text,
                detector_score=region.detector_score,
                scripts=scripts,
                status=region_status,
                error=region_error,
            )
        )

    timing["recognition_ms"] = round((time.perf_counter() - recognition_start) * 1000, 2)
    timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

    full_text = "\n".join(lines_text)
    overall_scripts = detect_scripts(full_text)

    postprocessed_text = full_text
    domain_insights: dict[str, Any] = {}
    if postprocess and full_text.strip():
        try:
            from smartvision_ocr.inference.postprocess import postprocess_ocr_text

            post_res = postprocess_ocr_text(full_text)
            postprocessed_text = post_res.corrected_text
            domain_insights = {
                "detected_medicines": post_res.detected_medicines,
                "detected_signage": post_res.detected_signage,
                "extracted_dosages": post_res.extracted_dosages,
                "corrections": post_res.corrections_applied,
            }
        except Exception as post_err:
            LOGGER.debug("Postprocessing fallback: %s", post_err)

    return OCRResult(
        text=full_text,
        regions=recognized_regions,
        image_path=image_path_str,
        image_size=pil_image.size,
        detector_model=detector_model,
        recognizer_model=active_rec_name,
        device=device_name,
        detected_scripts=overall_scripts,
        timing_ms=timing,
        postprocessed_text=postprocessed_text,
        domain_insights=domain_insights,
        status="success",
    )
