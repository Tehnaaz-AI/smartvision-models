import io
import json
import threading
from http.server import HTTPServer
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from smartvision_ocr.demo.server import OCRDemoRequestHandler
from smartvision_ocr.inference.pipeline import OCRResult, RegionResult


@pytest.fixture(scope="module")
def live_server():
    """Start test server on localhost with an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), OCRDemoRequestHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def test_serve_html_ui(live_server):
    with urlopen(f"{live_server}/") as response:
        assert response.status == 200
        content = response.read().decode("utf-8")
        assert "SmartVision OCR Explorer" in content
        assert "microsoft/trocr-base-printed" in content
        assert "Telugu" in content


def test_api_status(live_server):
    with urlopen(f"{live_server}/api/status") as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "English (Printed)" in data["supported_languages"]
        assert "Telugu" in data["roadmap_languages"][0]
        assert "disclaimer" in data


def test_api_samples(live_server):
    with urlopen(f"{live_server}/api/samples") as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "samples" in data
        assert isinstance(data["samples"], list)


def test_api_image_forbidden_outside_root(live_server):
    from urllib.error import HTTPError

    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{live_server}/api/image?path=../../outside.png")
    assert exc_info.value.code in (403, 404)


def test_api_extract_empty_request(live_server):
    req = Request(
        f"{live_server}/api/extract",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    from urllib.error import HTTPError

    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 400


def test_api_extract_mocked_pipeline(live_server, monkeypatch):
    # Mock get_cached_models and predict_full_image to test API contract quickly
    fake_result = OCRResult(
        text="PARACETAMOL 500 MG",
        regions=[
            RegionResult(
                region_index=0,
                reading_order=0,
                polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
                recognized_text="PARACETAMOL 500 MG",
                detector_score=0.98,
                scripts=["Latin"],
            )
        ],
        timing_ms={"detection_ms": 15.0, "recognition_ms": 30.0, "total_ms": 45.0},
        detected_scripts=["Latin"],
    )

    monkeypatch.setattr(
        "smartvision_ocr.demo.server.get_cached_models",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(
        "smartvision_ocr.demo.server.predict_full_image",
        lambda *args, **kwargs: fake_result,
    )

    req = Request(
        f"{live_server}/api/extract",
        data=json.dumps({"sample_path": "data/cropped/ocr_test.png"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["success"] is True
        assert data["text"] == "PARACETAMOL 500 MG"
        assert len(data["regions"]) == 1
        assert data["regions"][0]["detector_score"] == 0.98
        assert data["timing_ms"]["total_ms"] == 45.0
        assert "English printed-text only" in data["disclaimer"]


def test_favicon_returns_no_content(live_server):
    with urlopen(f"{live_server}/favicon.ico") as response:
        assert response.status == 204


def test_api_unknown_endpoint_returns_json_error(live_server):
    from urllib.error import HTTPError

    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{live_server}/api/non_existent_route")
    assert exc_info.value.code == 404
    data = json.loads(exc_info.value.read().decode("utf-8"))
    assert data["success"] is False
    assert "Endpoint not found" in data["error"]


def test_api_extract_malformed_json(live_server):
    from urllib.error import HTTPError

    req = Request(
        f"{live_server}/api/extract",
        data=b"{not valid json...",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 400
    data = json.loads(exc_info.value.read().decode("utf-8"))
    assert data["success"] is False
    assert "Malformed JSON" in data["error"]


def test_api_extract_concurrent_lock_rejection(live_server, monkeypatch):
    from urllib.error import HTTPError
    import smartvision_ocr.demo.server as server_mod

    # Acquire lock artificially to simulate a running inference task
    acquired = server_mod._INFERENCE_LOCK.acquire(blocking=False)
    assert acquired is True
    try:
        req = Request(
            f"{live_server}/api/extract",
            data=json.dumps({"sample_path": "data/cropped/ocr_test.png"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req)
        assert exc_info.value.code == 429
        data = json.loads(exc_info.value.read().decode("utf-8"))
        assert data["success"] is False
        assert "in progress" in data["error"]
    finally:
        server_mod._INFERENCE_LOCK.release()
