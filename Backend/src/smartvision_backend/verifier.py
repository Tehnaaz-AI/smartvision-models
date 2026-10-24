"""Deterministic Medication Verifier.

Executes strict rule-based validation between OCR extracted text
and the patient's validated doctor's prescription.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import List, Optional, Tuple

from .models import (
    Medication,
    Patient,
    Prescription,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class MedicationVerifier:
    """Verifies captured medicine labels against patient prescription records."""

    def __init__(self, patient: Optional[Patient] = None):
        self.patient = patient

    def verify_capture(
        self,
        detected_text: str,
        detected_strength: Optional[str] = None,
        scheduled_medicine_name: Optional[str] = None,
        ocr_confidence: float = 0.90,
    ) -> VerificationResult:
        """Compares extracted OCR details with scheduled medication."""
        # 1. Check OCR certainty threshold
        if ocr_confidence < 0.40 or not detected_text or not detected_text.strip():
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN_CAPTURE,
                is_safe_to_take=False,
                matched_medication=None,
                scheduled_medication=None,
                spoken_message="The medicine label could not be read clearly. Please hold steady and try again.",
                spoken_message_te="మందు లేబుల్ స్పష్టంగా చదవలేకపోయాము. దయచేసి కదలకుండా మళ్లీ చూపించండి.",
                confidence=ocr_confidence,
                action_required="RETRY",
            )

        clean_text = detected_text.strip()

        # Find target scheduled medication in prescription
        all_prescribed = self._get_all_prescribed_meds()
        scheduled_med = self._find_prescribed_med(scheduled_medicine_name, all_prescribed) if scheduled_medicine_name else None

        # If no specific scheduled med provided, default to first prescription med
        if scheduled_med is None and all_prescribed:
            scheduled_med = all_prescribed[0]

        # 2. Match detected text against all prescribed medications
        matched_med, match_sim = self._match_medicine(clean_text, all_prescribed)

        if matched_med is None or match_sim < 0.60:
            return VerificationResult(
                status=VerificationStatus.NOT_IN_PRESCRIPTION,
                is_safe_to_take=False,
                matched_medication=None,
                scheduled_medication=scheduled_med,
                spoken_message="Warning: This medicine does not match any medication on your active prescription. Do not take it.",
                spoken_message_te="హెచ్చరిక: ఈ మందు మీ ప్రిస్క్రిప్షన్‌లో లేదు. దయచేసి ఇది వేసుకోకండి.",
                confidence=round(match_sim, 2),
                action_required="ABORT",
            )

        # 3. Check if the matched med is the one SCHEDULED right now
        if scheduled_med and matched_med.medicine_name.lower() != scheduled_med.medicine_name.lower():
            return VerificationResult(
                status=VerificationStatus.WRONG_MEDICATION,
                is_safe_to_take=False,
                matched_medication=matched_med,
                scheduled_medication=scheduled_med,
                spoken_message=(
                    f"Stop! You are holding {matched_med.medicine_name}, but your current scheduled "
                    f"dose is {scheduled_med.medicine_name}. Please take {scheduled_med.medicine_name} instead."
                ),
                spoken_message_te=(
                    f"ఆగండి! మీరు {matched_med.medicine_name} పట్టుకున్నారు. కానీ ఇప్పుడు మీరు వేసుకోవాల్సినది "
                    f"{scheduled_med.medicine_name}. దయచేసి {scheduled_med.medicine_name} తీసుకోండి."
                ),
                confidence=round(match_sim, 2),
                action_required="ABORT",
            )

        # 4. Check Dosage Strength (e.g. 500 mg vs 650 mg)
        if detected_strength:
            prescribed_strength_clean = re.sub(r"\s+", "", matched_med.strength.lower())
            detected_strength_clean = re.sub(r"\s+", "", detected_strength.lower())

            if prescribed_strength_clean and detected_strength_clean and prescribed_strength_clean != detected_strength_clean:
                return VerificationResult(
                    status=VerificationStatus.NAME_MATCH_WRONG_STRENGTH,
                    is_safe_to_take=False,
                    matched_medication=matched_med,
                    scheduled_medication=scheduled_med,
                    spoken_message=(
                        f"Strength mismatch! You are holding {matched_med.medicine_name} {detected_strength}, "
                        f"but your prescription specifies {matched_med.strength}. Please consult your caregiver."
                    ),
                    spoken_message_te=(
                        f"మోతాదు తేడా ఉంది! మీరు {matched_med.medicine_name} {detected_strength} పట్టుకున్నారు, "
                        f"కానీ ప్రిస్క్రిప్షన్ ప్రకారం {matched_med.strength} తీసుకోవాలి. కేర్‌గివర్‌ని సంప్రదించండి."
                    ),
                    confidence=round(match_sim, 2),
                    action_required="CALL_CAREGIVER",
                )

        # 5. All checks passed: Safe to take!
        return VerificationResult(
            status=VerificationStatus.VERIFIED_MATCH,
            is_safe_to_take=True,
            matched_medication=matched_med,
            scheduled_medication=scheduled_med,
            spoken_message=(
                f"Verified. This is your correct medicine: {matched_med.medicine_name} {matched_med.strength}. "
                f"{matched_med.instructions}."
            ),
            spoken_message_te=(
                f"ధృవీకరించబడింది. ఇది మీ సరైన మందు: {matched_med.medicine_name} {matched_med.strength}. "
                f"{matched_med.instructions_te or matched_med.instructions}."
            ),
            confidence=round(match_sim, 2),
            action_required="PROCEED",
        )

    def _get_all_prescribed_meds(self) -> List[Medication]:
        if not self.patient:
            return []
        meds = []
        for rx in self.patient.prescriptions:
            meds.extend(rx.medications)
        return meds

    def _find_prescribed_med(self, name: str, meds: List[Medication]) -> Optional[Medication]:
        name_lower = name.lower()
        for m in meds:
            if m.medicine_name.lower() == name_lower or any(a.lower() == name_lower for a in m.aliases):
                return m
        return None

    def _match_medicine(self, text: str, meds: List[Medication]) -> Tuple[Optional[Medication], float]:
        best_med = None
        best_score = 0.0
        text_lower = text.lower()

        for med in meds:
            names_to_check = [med.medicine_name] + med.aliases
            for target_name in names_to_check:
                tgt_lower = target_name.lower()
                # Direct substring check
                if tgt_lower in text_lower:
                    return med, 0.98

                # SequenceMatcher similarity
                sim = difflib.SequenceMatcher(None, tgt_lower, text_lower).ratio()
                if sim > best_score:
                    best_score = sim
                    best_med = med

        return best_med, best_score


def get_verifier(patient: Optional[Patient] = None) -> MedicationVerifier:
    """Factory helper to obtain a MedicationVerifier instance."""
    return MedicationVerifier(patient=patient)
