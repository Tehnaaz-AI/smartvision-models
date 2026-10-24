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
