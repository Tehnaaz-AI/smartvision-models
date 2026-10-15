"""OCR inference entry points."""

__all__ = [
    "OCRResult",
    "RegionResult",
    "detect_scripts",
    "predict_full_image",
    "predict_image",
]


def __getattr__(name: str):
    if name in {"OCRResult", "RegionResult", "detect_scripts", "predict_full_image"}:
        from smartvision_ocr.inference import pipeline

        return getattr(pipeline, name)
    if name == "predict_image":
        from smartvision_ocr.inference import predict

        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

