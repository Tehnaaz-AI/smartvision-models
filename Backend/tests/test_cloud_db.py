"""Unit Tests for SmartVision Cloud Database Client & Offline SQLite Cache."""

import pytest
from pathlib import Path

from smartvision_backend.cloud_db import CloudDatabaseClient
from smartvision_backend.models import VerificationResult, VerificationStatus


def test_cloud_db_cache_initialization(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    client = CloudDatabaseClient(cache_db_path=str(db_file))
    assert db_file.exists()


def test_sync_patient_and_retrieve(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    client = CloudDatabaseClient(cache_db_path=str(db_file))

    patient_payload = {
        "patient_id": "P-999",
        "patient_name": "Sita Devi",
        "preferred_language": "te",
        "caregiver_contact": "+91-9999988888",
        "prescriptions": [
            {
                "prescription_id": "RX-CLOUD-01",
                "doctor_name": "Dr. Varma",
                "date_issued": "2026-10-24",
                "medications": [
                    {
                        "medicine_name": "Dolo",
                        "strength": "650 mg",
                        "dosage_form": "tablet",
                        "quantity_per_dose": "1 tablet",
                        "schedule_time": "09:00 AM",
                        "instructions": "Take after breakfast",
                        "instructions_te": "అల్పాహారం తర్వాత తీసుకోండి",
                    }
                ],
            }
        ],
    }

    client.sync_patient_to_cache(patient_payload)
    patient = client.get_patient("P-999")

    assert patient is not None
    assert patient.patient_name == "Sita Devi"
    assert patient.preferred_language == "te"
    assert len(patient.prescriptions) == 1
    assert patient.prescriptions[0].medications[0].medicine_name == "Dolo"


def test_log_verification_audit(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    client = CloudDatabaseClient(cache_db_path=str(db_file))

    res = VerificationResult(
        status=VerificationStatus.VERIFIED_MATCH,
        is_safe_to_take=True,
        matched_medication=None,
        scheduled_medication=None,
        spoken_message="Verified Paracetamol 500 mg.",
    )

    log_id = client.log_verification("P-999", res)
    assert log_id > 0

    pending = client.get_pending_sync_logs()
    assert len(pending) == 1
    assert pending[0]["patient_id"] == "P-999"
    assert pending[0]["status"] == "VERIFIED_MATCH"
    assert pending[0]["is_safe"] is True
