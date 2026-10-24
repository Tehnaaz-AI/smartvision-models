"""SmartVision Backend & Deterministic Verification Module.

Provides deterministic medication verification, patient prescription matching,
schedule reminders, and REST API service for smart glasses.
"""

from .models import (
    Patient,
    Medication,
    Prescription,
    VerificationStatus,
    VerificationResult,
)
from .verifier import MedicationVerifier, get_verifier
from .scheduler import MedicationScheduler, get_scheduler
from .cloud_db import CloudDatabaseClient, get_cloud_db_client

__all__ = [
    "Patient",
    "Medication",
    "Prescription",
    "VerificationStatus",
    "VerificationResult",
    "MedicationVerifier",
    "get_verifier",
    "MedicationScheduler",
    "get_scheduler",
    "CloudDatabaseClient",
    "get_cloud_db_client",
]
