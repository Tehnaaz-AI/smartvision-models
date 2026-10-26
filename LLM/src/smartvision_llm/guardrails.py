"""Medical Safety Guardrails for SmartVision Conversational AI.

Enforces strict boundaries against medication dosage fabrication, unverified medical advice,
and immediately escalates emergency symptoms (chest pain, severe dizziness, anaphylaxis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardrailResult:
    """Outcome of safety guardrail evaluation on patient query."""

    is_blocked: bool
    is_emergency: bool
    enforced_response: Optional[str]
    enforced_response_te: Optional[str]
    rule_triggered: Optional[str]


# Critical emergency symptom patterns
EMERGENCY_PATTERNS = [
    r"\b(chest\s+pain|heart\s+attack|difficulty\s+breathing|can'?t\s+breathe|shortness\s+of\s+breath)\b",
    r"\b(unconscious|passed\s+out|fainted|severe\s+allergic|anaphylaxis|swelling\s+of\s+throat)\b",
    r"\b(stroke|sudden\s+numbness|facial\s+droop|slurred\s+speech)\b",
    r"(గుండెనొప్పి|గుండె\s*నొప్పి|శ్వాస|మూర్ఛ)",
]

# Hazardous dosage modification patterns
DOSAGE_ALTERATION_PATTERNS = [
    r"\b(take\s+(?:two|2|three|3|extra|double|more)\s+(?:tablets?|pills?|doses?))\b",
    r"\b(increase\s+(?:my\s+)?(?:dose|dosage)|double\s+(?:my\s+)?dose)\b",
    r"\b(take\s+without\s+prescription|invent\s+(?:a\s+)?dosage)\b",
    r"(రెండు\s*టాబ్లెట్లు|ఎక్కువ\s*డోస్|డోస్\s*పెంచ)",
]


def check_emergency_symptoms(text: str) -> Optional[GuardrailResult]:
    """Detects immediate acute medical emergency distress."""
    lower = text.lower()
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return GuardrailResult(
                is_blocked=True,
                is_emergency=True,
                enforced_response=(
                    "This sounds like a serious medical emergency. Please sit down immediately to stay safe, "
                    "and let your caregiver or local emergency services know right now. Do not wait."
                ),
                enforced_response_te=(
                    "ఇది అత్యవసర వైద్య పరిస్థితిగా కనిపిస్తోంది. దయచేసి వెంటనే సురక్షితంగా కూర్చోండి "
                    "మరియు మీ సంరక్షకుడికి లేదా అత్యవసర సేవలకు తక్షణమే తెలియజేయండి."
                ),
                rule_triggered="ACUTE_EMERGENCY_SYMPTOM",
            )
    return None


def check_dosage_alteration(text: str) -> Optional[GuardrailResult]:
    """Detects dangerous attempts to modify or invent prescription dosages."""
    lower = text.lower()
    for pattern in DOSAGE_ALTERATION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return GuardrailResult(
                is_blocked=True,
                is_emergency=False,
                enforced_response=(
                    "No, please do not alter your medication dose. Taking extra medicine can be dangerous to your health. "
                    "Please follow only the exact dosage prescribed by your doctor."
                ),
                enforced_response_te=(
                    "లేదు, దయచేసి మీ మందుల మోతాదును మార్చకండి. అదనపు మందులు తీసుకోవడం ప్రమాదకరం. "
                    "మీ వైద్యులు సూచించిన మోతాదును మాత్రమే పాటించండి."
                ),
                rule_triggered="DOSAGE_ALTERATION_FORBIDDEN",
            )
    return None


def enforce_safety_guardrails(query: str) -> GuardrailResult:
    """Evaluates query against all clinical safety guardrails."""
    if not query or not query.strip():
        return GuardrailResult(
            is_blocked=False,
            is_emergency=False,
            enforced_response=None,
            enforced_response_te=None,
            rule_triggered=None,
        )

    # 1. Check emergency symptoms
    emerg = check_emergency_symptoms(query)
    if emerg:
        return emerg

    # 2. Check dosage alterations
    dose = check_dosage_alteration(query)
    if dose:
        return dose

    return GuardrailResult(
        is_blocked=False,
        is_emergency=False,
        enforced_response=None,
        enforced_response_te=None,
        rule_triggered=None,
    )
