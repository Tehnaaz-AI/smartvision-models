"""Single-image inference using pretrained or locally fine-tuned TrOCR."""

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from smartvision_ocr.config import MODEL_NAME_OR_PATH
from smartvision_ocr.utils.logging_utils import configure_logging


def select_device(requested: str | None = None) -> torch.device:
    """Select an explicitly requested device or CUDA when available, else CPU."""
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_name_or_path: str = MODEL_NAME_OR_PATH,
) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the processor and model, downloading them on first use if needed."""
    try:
        processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    except (
        HfHubHTTPError,
        OSError,
        RequestException,
        ValueError,
        RuntimeError,
    ) as error:
        raise RuntimeError(
            f"Could not load OCR model '{model_name_or_path}'. Check the model name, "
            "network connection, and Hugging Face access."
        ) from error
    required_tokens = (
        "decoder_start_token_id",
        "pad_token_id",
        "eos_token_id",
    )
    generation_config = getattr(model, "generation_config", None)
    missing_tokens = (
        required_tokens
        if generation_config is None
        else [
            name
            for name in required_tokens
            if getattr(generation_config, name, None) is None
        ]
    )
    if missing_tokens:
        raise ValueError(
            f"OCR model '{model_name_or_path}' has an invalid generation "
            f"configuration; missing: {', '.join(missing_tokens)}."
        )
    return processor, model


def predict_image(
    image_path: str | Path,
    model: Any | None = None,
    processor: Any | None = None,
    device: str | None = None,
    model_name_or_path: str = MODEL_NAME_OR_PATH,
) -> str:
    """Recognize printed text in one image and return its transcription."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Could not read image '{path}': {error}") from error

    if (model is None) != (processor is None):
        raise ValueError("Provide both model and processor, or neither.")
    if model is None or processor is None:
        processor, model = load_model(model_name_or_path)

    selected_device = select_device(device)
    model.to(selected_device)
    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(selected_device)
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize printed text in an image.")
    parser.add_argument("image", help="Path to an image containing one printed text line.")
    parser.add_argument(
        "--model",
        default=MODEL_NAME_OR_PATH,
        help="Hugging Face model ID or local fine-tuned model directory.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Inference device.")
    args = parser.parse_args()
    logger = configure_logging()
    try:
        selected_device = select_device(args.device)
        print(f"Model: {args.model}")
        print(f"Device: {selected_device}")
        print(f"Input image: {Path(args.image).expanduser()}")
        result = predict_image(
            args.image,
            device=str(selected_device),
            model_name_or_path=args.model,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error
    print(f"Predicted text: {result}")


if __name__ == "__main__":
    main()
