"""Data Models and Schema Definitions for SmartVision Backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class VerificationStatus(str, Enum):
    """Deterministic status resulting from prescription cross-examination."""

    VERIFIED_MATCH = "VERIFIED_MATCH"          # Name & strength match current scheduled dose
    NAME_MATCH_WRONG_STRENGTH = "NAME_MATCH_WRONG_STRENGTH"  # Correct drug, incorrect dosage strength
    WRONG_MEDICATION = "WRONG_MEDICATION"      # Holding different drug than what is scheduled
    NOT_IN_PRESCRIPTION = "NOT_IN_PRESCRIPTION"# Holding medicine not prescribed at all
    UNCERTAIN_CAPTURE = "UNCERTAIN_CAPTURE"    # OCR text confidence too low to verify safely


@dataclass
class Medication:
    """Represents a prescribed medicine item."""

    medicine_name: str
    strength: str
    dosage_form: str
    quantity_per_dose: str
    schedule_time: str
    instructions: str
    instructions_te: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class Prescription:
    """Represents a validated doctor's prescription."""

    prescription_id: str
    doctor_name: str
    date_issued: str
    medications: List[Medication] = field(default_factory=list)


@dataclass
class Patient:
    """Patient profile and active prescriptions."""

    patient_id: str
    patient_name: str
    preferred_language: str = "te"
    caregiver_contact: Optional[str] = None
    prescriptions: List[Prescription] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Outcome of medication verification against patient prescription."""

    status: VerificationStatus
    is_safe_to_take: bool
    matched_medication: Optional[Medication]
    scheduled_medication: Optional[Medication]
    spoken_message: str
    spoken_message_te: Optional[str] = None
    confidence: float = 1.0
    action_required: str = "PROCEED"  # PROCEED, ABORT, RETRY, CALL_CAREGIVER
