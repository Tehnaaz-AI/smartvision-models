"""Unit Tests for SmartVision Medication Verifier and Scheduler."""

import pytest

from smartvision_backend.models import (
    Medication,
    Patient,
    Prescription,
    VerificationStatus,
)
from smartvision_backend.verifier import MedicationVerifier, get_verifier
from smartvision_backend.scheduler import MedicationScheduler, get_scheduler


@pytest.fixture
def mock_patient():
    med_paracetamol = Medication(
        medicine_name="Paracetamol",
        strength="500 mg",
        dosage_form="tablet",
        quantity_per_dose="1 tablet",
        schedule_time="08:00 AM",
        instructions="Take 1 tablet after food",
        instructions_te="భోజనం తర్వాత 1 టాబ్లెట్ వేసుకోండి",
        aliases=["Dolo", "Crocin"],
    )
    med_metformin = Medication(
        medicine_name="Metformin",
        strength="500 mg",
        dosage_form="tablet",
        quantity_per_dose="1 tablet",
        schedule_time="08:00 PM",
        instructions="Take with dinner",
        instructions_te="రాత్రి భోజనంతో తీసుకోండి",
        aliases=["Glycomet"],
    )
    rx = Prescription(
        prescription_id="RX-TEST-1",
        doctor_name="Dr. Rao",
        date_issued="2026-10-01",
        medications=[med_paracetamol, med_metformin],
    )
    return Patient(
        patient_id="P-TEST",
        patient_name="Test Patient",
        preferred_language="te",
        prescriptions=[rx],
    )


def test_verified_match_direct_name(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    result = verifier.verify_capture(
        detected_text="PARACETAMOL 500 MG",
        detected_strength="500 mg",
        scheduled_medicine_name="Paracetamol",
    )
    assert result.status == VerificationStatus.VERIFIED_MATCH
    assert result.is_safe_to_take is True
    assert result.action_required == "PROCEED"


def test_verified_match_alias(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    result = verifier.verify_capture(
        detected_text="Dolo 500 Tablets",
        detected_strength="500 mg",
        scheduled_medicine_name="Paracetamol",
    )
    assert result.status == VerificationStatus.VERIFIED_MATCH
    assert result.is_safe_to_take is True


def test_name_match_wrong_strength(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    result = verifier.verify_capture(
        detected_text="Paracetamol",
        detected_strength="650 mg",  # Prescribed is 500 mg!
        scheduled_medicine_name="Paracetamol",
    )
    assert result.status == VerificationStatus.NAME_MATCH_WRONG_STRENGTH
    assert result.is_safe_to_take is False
    assert result.action_required == "CALL_CAREGIVER"


def test_wrong_medication_scheduled(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    # Holding Metformin, but morning scheduled dose is Paracetamol!
    result = verifier.verify_capture(
        detected_text="Metformin 500 mg",
        detected_strength="500 mg",
        scheduled_medicine_name="Paracetamol",
    )
    assert result.status == VerificationStatus.WRONG_MEDICATION
    assert result.is_safe_to_take is False
    assert result.action_required == "ABORT"
    assert "Stop" in result.spoken_message


def test_not_in_prescription(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    result = verifier.verify_capture(
        detected_text="Amoxicillin 250mg",
        scheduled_medicine_name="Paracetamol",
    )
    assert result.status == VerificationStatus.NOT_IN_PRESCRIPTION
    assert result.is_safe_to_take is False
    assert "Warning" in result.spoken_message


def test_uncertain_capture(mock_patient):
    verifier = MedicationVerifier(patient=mock_patient)
    result = verifier.verify_capture(
        detected_text="blurred text ??",
        ocr_confidence=0.25,
    )
    assert result.status == VerificationStatus.UNCERTAIN_CAPTURE
    assert result.is_safe_to_take is False
    assert result.action_required == "RETRY"


def test_scheduler_due_medications(mock_patient):
    scheduler = MedicationScheduler(patient=mock_patient)
    morning_meds = scheduler.get_due_medications("08:00 AM")
    assert len(morning_meds) == 1
    assert morning_meds[0].medicine_name == "Paracetamol"

    evening_meds = scheduler.get_due_medications("08:00 PM")
    assert len(evening_meds) == 1
    assert evening_meds[0].medicine_name == "Metformin"

    afternoon_meds = scheduler.get_due_medications("02:00 PM")
    assert len(afternoon_meds) == 0
