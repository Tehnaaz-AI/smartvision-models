"""Lightweight local HTTP demo server for SmartVision OCR.

Provides a web interface for testing full-image OCR on user uploads or
curated local samples, calling the actual PyTorch / PaddleOCR inference
pipeline with live latency breakdown and bounding-box overlays.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, UnidentifiedImageError

from smartvision_ocr.config import (
    DATA_DIR,
    MODEL_NAME_OR_PATH,
    PROJECT_ROOT,
)
from smartvision_ocr.inference.detect_and_ocr import (
    DETECTOR_MODEL_NAME,
    load_text_detector,
)
from smartvision_ocr.inference.pipeline import (
    OCRResult,
    predict_full_image,
)
from smartvision_ocr.inference.predict import (
    load_model,
    select_device,
)
from smartvision_ocr.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "template.html"

# Global pipeline cache to avoid reloading 1.3GB model weights on every request
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_CACHED_DETECTOR: Any | None = None
_CACHED_MODEL: Any | None = None
_CACHED_PROCESSOR: Any | None = None


def get_cached_models(device: str | None = None) -> tuple[Any, Any, Any]:
    """Retrieve or lazily initialize detector, recognizer, and processor."""
    global _CACHED_DETECTOR, _CACHED_MODEL, _CACHED_PROCESSOR
    with _MODEL_LOCK:
        if _CACHED_DETECTOR is None:
            LOGGER.info("Loading text detector (%s)...", DETECTOR_MODEL_NAME)
            _CACHED_DETECTOR = load_text_detector(DETECTOR_MODEL_NAME)

        if _CACHED_MODEL is None or _CACHED_PROCESSOR is None:
            LOGGER.info("Loading recognizer (%s)...", MODEL_NAME_OR_PATH)
            _CACHED_PROCESSOR, _CACHED_MODEL = load_model(MODEL_NAME_OR_PATH)
            selected_device = select_device(device)
            _CACHED_MODEL.to(selected_device)
            _CACHED_MODEL.eval()

        return _CACHED_DETECTOR, _CACHED_MODEL, _CACHED_PROCESSOR


class OCRDemoRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving UI and OCR API endpoints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Forward access logs through standard logger."""
        LOGGER.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)

    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Helper to send a JSON response safely."""
        try:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(payload)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as err:
            LOGGER.warning("Client disconnected before JSON response could be sent: %s", err)
        except Exception as err:
            LOGGER.error("Failed to write JSON response: %s", err)

    def send_api_error(self, message: str, status: int = 400) -> None:
        """Helper to send structured JSON error for all API routes."""
        self.send_json({"success": False, "error": message}, status=status)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Serve HTML page, static images, and status APIs."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.serve_template()
            return

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if path == "/api/status":
            self.serve_status()
            return

        if path == "/api/samples":
            self.serve_samples()
            return

        if path == "/api/image":
            self.serve_image(parse_qs(parsed.query))
            return

        if path.startswith("/api/"):
            self.send_api_error(f"Endpoint not found: {path}", status=HTTPStatus.NOT_FOUND)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        """Handle text extraction requests."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/extract":
            self.handle_extract()
            return

        if parsed.path.startswith("/api/"):
            self.send_api_error(f"Endpoint not found: {parsed.path}", status=HTTPStatus.NOT_FOUND)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_template(self) -> None:
        """Serve the UI HTML template."""
        if not TEMPLATE_PATH.is_file():
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Template file not found")
            return

        html_bytes = TEMPLATE_PATH.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def serve_status(self) -> None:
        """Return engine metadata and language support limits."""
        status_data = {
            "engine": "SmartVision Modular OCR",
            "detector": DETECTOR_MODEL_NAME,
            "recognizer": MODEL_NAME_OR_PATH,
            "device": "cpu",
            "supported_languages": ["English (Printed)"],
            "roadmap_languages": [
                "Telugu (Primary Priority - in development)",
                "Hindi (Secondary Priority - in development)",
            ],
            "models_loaded": _CACHED_MODEL is not None and _CACHED_DETECTOR is not None,
            "disclaimer": (
                "The current recognizer checkpoint (microsoft/trocr-base-printed) "
                "is trained strictly on English printed text. Indic scripts (Telugu/Hindi) "
                "are not supported by this checkpoint."
            ),
        }
        self.send_json(status_data)

    def serve_samples(self) -> None:
        """Return a list of curated local sample images available for 1-click test."""
        samples = [
            {
                "id": "paracetamol_crop",
                "name": "Paracetamol Label Crop (English Printed)",
                "path": "data/cropped/ocr_test.png",
                "category": "Medicine packaging crop",
            },
            {
                "id": "telugu_wikisource_0",
                "name": "Telugu Classical Text Line 0000",
                "path": "data/telugu_wikisource/images/te_line_0000.png",
                "category": "Telugu printed line crop",
            },
            {
                "id": "telugu_wikisource_32",
                "name": "Telugu Book Crop Line 0032",
                "path": "data/telugu_wikisource/images/te_line_0032.png",
                "category": "Telugu printed line crop",
            },
            {
                "id": "line_001",
                "name": "Medicine Packaging Crop 001",
                "path": "data/evaluation/images/line_001.png",
                "category": "Medicine packaging crop",
            },
            {
                "id": "bottle_02",
                "name": "Medicine Bottle Photo 02",
                "path": "data/Medicine_data/images02.jpg",
                "category": "Full product scene",
            },
            {
                "id": "box_03",
                "name": "Medicine Package Box Photo 03",
                "path": "data/Medicine_data/images03.jpg",
                "category": "Full product scene",
            },
        ]
        # Filter to only existing files
        valid_samples = [s for s in samples if (PROJECT_ROOT / s["path"]).is_file()]
        self.send_json({"samples": valid_samples})

    def serve_image(self, query: dict[str, list[str]]) -> None:
        """Safely serve an image from the project data directory."""
        path_param = query.get("path", [""])[0]
        if not path_param:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing 'path' parameter")
            return

        target_path = (PROJECT_ROOT / path_param).resolve()
        # Security: verify path stays within PROJECT_ROOT
        if not target_path.is_relative_to(PROJECT_ROOT.resolve()):
            self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
            return

        if not target_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"Image not found: {path_param}")
            return

        mime_type, _ = mimetypes.guess_type(target_path)
        if not mime_type:
            mime_type = "image/png" if target_path.suffix.lower() == ".png" else "image/jpeg"

        image_bytes = target_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.end_headers()
        self.wfile.write(image_bytes)

    def handle_extract(self) -> None:
        """Execute full OCR extraction pipeline on requested image."""
        if not _INFERENCE_LOCK.acquire(blocking=False):
            self.send_api_error(
                "Another OCR inference task is currently in progress. Please wait for it to complete.",
                status=HTTPStatus.TOO_MANY_REQUESTS,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_api_error("Empty request body", status=400)
                return

            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_api_error("Malformed JSON in request payload", status=400)
                return

            sample_path_str = data.get("sample_path")
            image_data_str = data.get("image_data")

            image_input: Image.Image | str

            if sample_path_str:
                resolved_sample = (PROJECT_ROOT / sample_path_str).resolve()
                if not resolved_sample.is_relative_to(PROJECT_ROOT.resolve()) or not resolved_sample.is_file():
                    self.send_api_error(f"Sample image file not found: {sample_path_str}", status=404)
                    return
                image_input = str(resolved_sample)
            elif image_data_str:
                # Format: data:image/png;base64,... or pure base64
                if "," in image_data_str:
                    _, b64_data = image_data_str.split(",", 1)
                else:
                    b64_data = image_data_str

                try:
                    raw_bytes = base64.b64decode(b64_data)
                except Exception as b64_err:
                    self.send_api_error(f"Invalid base64 encoding: {b64_err}", status=400)
                    return

                try:
                    with Image.open(io.BytesIO(raw_bytes)) as source:
                        image_input = source.convert("RGB")
                except (OSError, UnidentifiedImageError) as err:
                    self.send_api_error(f"Uploaded file is not a valid image: {err}", status=400)
                    return
            else:
                self.send_api_error("Provide either 'image_data' (base64) or 'sample_path'.", status=400)
                return

            language = str(data.get("language", "auto")).strip().lower()

            # Retrieve cached models
            detector, model, processor = get_cached_models()

            # Run actual inference pipeline
            ocr_result: OCRResult = predict_full_image(
                image_input,
                detector=detector,
                model=model,
                processor=processor,
                recognizer_model=MODEL_NAME_OR_PATH,
                detector_model=DETECTOR_MODEL_NAME,
                language=language,
            )

            if "trocr" in ocr_result.recognizer_model.lower():
                disclaimer = (
                    "Recognized using microsoft/trocr-base-printed (English printed-text only). "
                    "Indic text (Telugu/Hindi) is not supported by this checkpoint."
                )
            else:
                disclaimer = (
                    f"Recognized using {ocr_result.recognizer_model} (Engine: {language}). "
                    "Supports Telugu, Hindi, English, and general scene text."
                )
            response_payload = {
                "success": True,
                "status": ocr_result.status,
                "text": ocr_result.text,
                "regions": [r.to_dict() for r in ocr_result.regions],
                "timing_ms": ocr_result.timing_ms,
                "detected_scripts": ocr_result.detected_scripts,
                "detector_model": ocr_result.detector_model,
                "recognizer_model": ocr_result.recognizer_model,
                "device": ocr_result.device,
                "error": ocr_result.error,
                "disclaimer": disclaimer,
            }
            self.send_json(response_payload)

        except Exception as error:
            LOGGER.exception("OCR Extraction failed unexpectedly: %s", error)
            self.send_api_error(
                "Internal OCR inference pipeline error occurred. Please verify input image and retry.",
                status=500,
            )
        finally:
            _INFERENCE_LOCK.release()


def run_demo_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    preload: bool = False,
    open_browser: bool = True,
) -> None:
    """Launch local HTTP demo server."""
    configure_logging()
    if preload:
        LOGGER.info("Preloading models into RAM...")
        get_cached_models()

    server = ThreadingHTTPServer((host, port), OCRDemoRequestHandler)
    url = f"http://{host}:{port}/"
    print("\n" + "=" * 60)
    print("  SmartVision OCR — Interactive Local Demo Server")
    print(f"  URL: {url}")
    print("  Engine: Microsoft TrOCR Base Printed (English Printed-Text)")
    print("  Detector: PaddleOCR Mobile DBNet (PP-OCRv5)")
    print("  Status: Offline Local Execution (Press Ctrl+C to stop)")
    print("=" * 60 + "\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SmartVision demo server...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch local interactive demo interface for SmartVision OCR."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload PaddleOCR and TrOCR models into RAM on server startup.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser.",
    )
    args = parser.parse_args()

    run_demo_server(
        host=args.host,
        port=args.port,
        preload=args.preload,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
