"""Unit Tests for FastAPI REST Endpoints and Caregiver Portal Routes."""

import pytest
from fastapi.testclient import TestClient
from smartvision_backend.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SmartVision" in data["service"]


def test_reminders_endpoint(client):
    response = client.get("/api/reminders?time=08:00 AM")
    assert response.status_code == 200
    data = response.json()
    assert "medications" in data


def test_verify_endpoint_safe(client):
    response = client.post(
        "/api/verify",
        json={
            "detected_text": "DOLO 500 TABLETS",
            "detected_strength": "500 mg",
            "scheduled_medicine_name": "Paracetamol",
            "ocr_confidence": 0.95,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_safe_to_take"] is True
    assert data["status"] == "VERIFIED_MATCH"
    assert "Paracetamol" in data["matched_medicine"]


def test_verify_endpoint_unsafe_triggers_alert(client):
    response = client.post(
        "/api/verify",
        json={
            "detected_text": "UNPRESCRIBED DRUG XYZ 500MG",
            "scheduled_medicine_name": "Dolo",
            "ocr_confidence": 0.90,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_safe_to_take"] is False

    # Check alert was created
    alerts_resp = client.get("/api/caregiver/alerts")
    assert alerts_resp.status_code == 200
    alerts_data = alerts_resp.json()["alerts"]
    assert len(alerts_data) >= 1
    assert any("UNPRESCRIBED DRUG XYZ" in a["message"] for a in alerts_data)


def test_caregiver_dashboard_endpoint(client):
    response = client.get("/api/caregiver/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "patient" in data
    assert "device_telemetry" in data
    assert data["device_telemetry"]["status"] == "ONLINE"


def test_caregiver_add_prescription_endpoint(client):
    new_med = {
        "name": "Vitamin C",
        "strength": "500 mg",
        "dosage": "1 Chewable",
        "schedule_time": "10:00 AM",
        "meal_relation": "AFTER_FOOD",
        "instructions": "Chew one tablet daily",
        "instructions_te": "రోజూ ఒక టాబ్లెట్ నమలండి",
    }
    response = client.post("/api/caregiver/prescriptions", json=new_med)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_portal_html_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SmartVision" in response.text


def test_location_current_endpoint(client):
    response = client.get("/api/location/current")
    assert response.status_code == 200
    data = response.json()
    assert "coordinates" in data
    assert "google_maps_pin" in data
    assert "https://www.google.com/maps?q=" in data["google_maps_pin"]


def test_location_update_and_wandering_alert(client):
    # Post location outside safe-zone (e.g. 500m away)
    response = client.post(
        "/api/location/update",
        json={"latitude": 17.4450, "longitude": 78.4530, "speed_kmh": 4.2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["location_status"]["is_inside_safe_zone"] is False


def test_nearby_pharmacies_endpoint(client):
    response = client.get("/api/location/nearby-pharmacies?language=te")
    assert response.status_code == 200
    data = response.json()
    assert "closest_pharmacy" in data
    assert "spoken_te" in data
    assert len(data["pharmacies"]) >= 1


def test_safe_zone_config_endpoint(client):
    response = client.post(
        "/api/location/safe-zone",
        json={
            "center_latitude": 17.4399,
            "center_longitude": 78.4482,
            "radius_meters": 250.0,
            "home_address": "Home Villa, Banjara Hills",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_v1_verify_multipart_blurry(client):
    import cv2
    import numpy as np
    import io

    # Blank image (will be flagged as blurry/unusable by VideoModel)
    blank_img = np.zeros((120, 160, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".jpg", blank_img)

    files = {"image": ("frame.jpg", io.BytesIO(img_bytes.tobytes()), "image/jpeg")}
    data = {
        "latitude": "17.4399",
        "longitude": "78.4482",
        "language": "te",
    }
    response = client.post("/api/v1/verify", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "BLURRY_CAPTURE"
    assert payload["is_safe_to_take"] is False
    assert payload["gps_status"] is not None
    assert payload["gps_status"]["is_inside_safe_zone"] is True


def test_v1_verify_invalid_image(client):
    import io
    files = {"image": ("bad.jpg", io.BytesIO(b"not_an_image_data"), "image/jpeg")}
    response = client.post("/api/v1/verify", files=files)
    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


def test_v1_audio_stream_endpoint(client, tmp_path):
    from pathlib import Path
    import wave
    import struct

    # Test 404
    resp_404 = client.get("/api/v1/audio/stream/non_existent_file.wav")
    assert resp_404.status_code == 404

    # Create dummy WAV in tts_output
    tts_dir = Path.cwd() / "tts_output"
    tts_dir.mkdir(parents=True, exist_ok=True)
    test_wav = tts_dir / "test_stream_sample.wav"

    with wave.open(str(test_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 160)

    try:
        resp_200 = client.get("/api/v1/audio/stream/test_stream_sample.wav")
        assert resp_200.status_code == 200
        assert "audio/wav" in resp_200.headers["content-type"]
        assert len(resp_200.content) > 0
    finally:
        test_wav.unlink(missing_ok=True)


def test_v1_ask_conversational_endpoint(client):
    # Standard health question
    resp = client.post("/api/v1/ask", json={"question": "What is Dolo 650 used for?", "language": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Dolo 650" in data["answer"] or "paracetamol" in data["answer"].lower()
    assert data["is_safe"] is True
    assert data["guardrail_triggered"] is False


def test_v1_ask_emergency_interceptor_triggers_alert(client):
    resp = client.post("/api/v1/ask", json={"question": "I have severe chest pain and left arm numbness", "language": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_emergency"] is True
    assert data["guardrail_triggered"] is True
    assert "emergency" in data["answer"].lower()

    # Verify high priority alert was recorded
    alerts_resp = client.get("/api/caregiver/alerts")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()["alerts"]
    assert any("chest pain" in a["message"].lower() for a in alerts)

