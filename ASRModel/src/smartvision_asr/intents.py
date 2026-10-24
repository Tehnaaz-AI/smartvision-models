"""Trilingual Voice Intent Classifier for SmartVision."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class UserIntent(str, Enum):
    """Structured intent categories for smart-glasses voice commands."""

    IDENTIFY_MEDICINE = "IDENTIFY_MEDICINE"
    CHECK_DOSAGE = "CHECK_DOSAGE"
    CHECK_SCHEDULE = "CHECK_SCHEDULE"
    READ_INSTRUCTIONS = "READ_INSTRUCTIONS"
    EMERGENCY_HELP = "EMERGENCY_HELP"
    GENERAL_QUERY = "GENERAL_QUERY"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentMatch:
    """Encapsulates the detected intent and extracted keywords."""

    intent: UserIntent
    confidence: float
    language: str
    raw_query: str
    target_medicine: Optional[str] = None


class IntentClassifier:
    """Classifies spoken queries across Telugu, Hindi, and English into actionable intents."""

    PATTERNS = {
        UserIntent.IDENTIFY_MEDICINE: {
            "en": [r"what is this medicine", r"which medicine is this", r"identify this", r"what tablet", r"name of medicine"],
            "te": [r"ఈ మందు ఏమిటి", r"ఈ టాబ్లెట్ ఏమిటి", r"మందు పేరు ఏమిటి", r"గుర్తించండి", r"ఏ మందు"],
            "hi": [r"यह कौन सी दवा है", r"यह क्या दवा है", r"दवा का नाम क्या है", r"पहचानो", r"कौन सी गोली"],
        },
        UserIntent.CHECK_DOSAGE: {
            "en": [r"how many tablets", r"what is the dosage", r"how much dose", r"how much should i take", r"how many pills"],
            "te": [r"ఎన్ని టాబ్లెట్లు", r"ఎంత మోతాదు", r"ఎంత వేసుకోవాలి", r"డోసేజ్ ఎంత"],
            "hi": [r"कितनी गोली", r"कितनी खुराक", r"कितना लेना है", r"डोज़ क्या है"],
        },
        UserIntent.CHECK_SCHEDULE: {
            "en": [r"when should i take", r"what time to take", r"before or after food", r"morning or night", r"schedule"],
            "te": [r"ఎప్పుడు తీసుకోవాలి", r"ఏ సమయంలో", r"భోజనం తర్వాత", r"ఉదయమా రాత్రా"],
            "hi": [r"कब लेना है", r"किस समय लेना है", r"खाने से पहले", r"खाने के बाद", r"सुबह या शाम"],
        },
        UserIntent.READ_INSTRUCTIONS: {
            "en": [r"read instructions", r"read the label", r"read out text", r"read packaging", r"read this"],
            "te": [r"సూచనలు చదవండి", r"లేబుల్ చదవండి", r"చదివి వినిపించండి", r"రాసి ఉన్నది చదవండి"],
            "hi": [r"निर्देश पढ़ें", r"लेबल पढ़कर सुनाओ", r"पढ़कर बताओ", r"क्या लिखा है"],
        },
        UserIntent.EMERGENCY_HELP: {
            "en": [r"help me", r"emergency", r"call doctor", r"feeling sick", r"urgent help"],
            "te": [r"సహాయం కావాలి", r"డాక్టర్ ని పిలవండి", r"అత్యవసరం", r"బాధగా ఉంది"],
            "hi": [r"मदद चाहिए", r"डॉक्टर को बुलाओ", r"इमरजेंसी", r"तबीयत खराब है"],
        },
    }

    def classify(self, text: str) -> IntentMatch:
        """Determines user intent from input transcription."""
        if not text or not text.strip():
            return IntentMatch(
                intent=UserIntent.UNKNOWN,
                confidence=0.0,
                language="unknown",
                raw_query="",
            )

        query = text.strip().lower()
        detected_lang = self._detect_language(query)

        # Check regex intent rules
        for intent, lang_dict in self.PATTERNS.items():
            patterns = lang_dict.get(detected_lang, []) + lang_dict.get("en", [])
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return IntentMatch(
                        intent=intent,
                        confidence=0.95,
                        language=detected_lang,
                        raw_query=text,
                    )

        # Fallback to general conversational query (for LLM handling)
        return IntentMatch(
            intent=UserIntent.GENERAL_QUERY,
            confidence=0.60,
            language=detected_lang,
            raw_query=text,
        )

    def _detect_language(self, text: str) -> str:
        """Lightweight script detector for Telugu, Hindi, and Latin/English."""
        for char in text:
            code = ord(char)
            if 0x0C00 <= code <= 0x0C7F:
                return "te"
            elif 0x0900 <= code <= 0x097F:
                return "hi"
        return "en"
