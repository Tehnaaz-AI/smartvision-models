"""FastAPI REST Service for SmartVision Smart Glasses Backend."""

from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel

from .models import VerificationStatus
from .verifier import get_verifier
from .scheduler import get_scheduler

app = FastAPI(
    title="SmartVision Backend Service",
    description="Deterministic medication verification and scheduling API for assistive smart glasses",
    version="1.0.0",
)

scheduler = get_scheduler()
verifier = get_verifier(patient=scheduler.patient)


class VerificationRequest(BaseModel):
    detected_text: str
    detected_strength: Optional[str] = None
    scheduled_medicine_name: Optional[str] = None
    ocr_confidence: float = 0.90


class VerificationResponse(BaseModel):
    status: str
    is_safe_to_take: bool
    matched_medicine: Optional[str] = None
    prescribed_strength: Optional[str] = None
    spoken_message: str
    spoken_message_te: Optional[str] = None
    action_required: str


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SmartVision Backend",
        "patient": scheduler.patient.patient_name,
    }


@app.get("/api/reminders")
def get_reminders(time: str = Query("08:00 AM", description="Schedule time, e.g. '08:00 AM'")):
    meds = scheduler.get_due_medications(time)
    return {
        "time": time,
        "due_count": len(meds),
        "medications": [
            {
                "name": m.medicine_name,
                "strength": m.strength,
                "dosage": m.quantity_per_dose,
                "instructions": m.instructions,
                "instructions_te": m.instructions_te,
            }
            for m in meds
        ],
    }


@app.post("/api/verify", response_model=VerificationResponse)
def verify_medication(req: VerificationRequest):
    result = verifier.verify_capture(
        detected_text=req.detected_text,
        detected_strength=req.detected_strength,
        scheduled_medicine_name=req.scheduled_medicine_name,
        ocr_confidence=req.ocr_confidence,
    )
    return VerificationResponse(
        status=result.status.value,
        is_safe_to_take=result.is_safe_to_take,
        matched_medicine=result.matched_medication.medicine_name if result.matched_medication else None,
        prescribed_strength=result.matched_medication.strength if result.matched_medication else None,
        spoken_message=result.spoken_message,
        spoken_message_te=result.spoken_message_te,
        action_required=result.action_required,
    )
