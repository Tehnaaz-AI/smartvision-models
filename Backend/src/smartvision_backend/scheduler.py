"""Medication Scheduler and Reminder Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .models import Medication, Patient, Prescription


class MedicationScheduler:
    """Manages scheduled daily reminders for elderly patients."""

    def __init__(self, patient: Optional[Patient] = None):
        self.patient = patient or self._load_default_patient()

    def get_due_medications(self, time_str: str) -> List[Medication]:
        """Returns medications scheduled at or near the specified time (e.g. '08:00 AM')."""
        due = []
        target = time_str.strip().upper()
        for rx in self.patient.prescriptions:
            for med in rx.medications:
                if med.schedule_time.strip().upper() == target:
                    due.append(med)
        return due

    def _load_default_patient(self) -> Patient:
        """Loads default sample patient from data/sample_prescriptions.json."""
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_prescriptions.json"
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if records:
                    first = records[0]
                    prescriptions = []
                    for rx in first.get("prescriptions", []):
                        meds = []
                        for m in rx.get("medications", []):
                            meds.append(
                                Medication(
                                    medicine_name=m["medicine_name"],
                                    strength=m["strength"],
                                    dosage_form=m["dosage_form"],
                                    quantity_per_dose=m["quantity_per_dose"],
                                    schedule_time=m["schedule_time"],
                                    instructions=m["instructions"],
                                    instructions_te=m.get("instructions_te"),
                                    aliases=m.get("aliases", []),
                                )
                            )
                        prescriptions.append(
                            Prescription(
                                prescription_id=rx["prescription_id"],
                                doctor_name=rx["doctor_name"],
                                date_issued=rx["date_issued"],
                                medications=meds,
                            )
                        )
                    return Patient(
                        patient_id=first["patient_id"],
                        patient_name=first["patient_name"],
                        preferred_language=first.get("preferred_language", "te"),
                        caregiver_contact=first.get("caregiver_contact"),
                        prescriptions=prescriptions,
                    )

        # Fallback default
        default_med = Medication(
            medicine_name="Paracetamol",
            strength="500 mg",
            dosage_form="tablet",
            quantity_per_dose="1 tablet",
            schedule_time="08:00 AM",
            instructions="Take 1 tablet after food",
            instructions_te="భోజనం తర్వాత 1 టాబ్లెట్ వేసుకోండి",
            aliases=["Dolo", "Crocin"],
        )
        return Patient(
            patient_id="P-DEFAULT",
            patient_name="Patient",
            preferred_language="te",
            prescriptions=[
                Prescription(
                    prescription_id="RX-DEFAULT",
                    doctor_name="Doctor",
                    date_issued="2026-10-01",
                    medications=[default_med],
                )
            ],
        )


def get_scheduler(patient: Optional[Patient] = None) -> MedicationScheduler:
    """Factory helper to obtain a MedicationScheduler instance."""
    return MedicationScheduler(patient=patient)
