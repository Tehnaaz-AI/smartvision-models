"""Domain-specific post-processing and fuzzy lexicon matching for SmartVision.

Provides:
- Medical dictionary matching for common generic drugs, formulations, and dosages.
- Assistive signage matching for navigation and safety signs.
- Fuzzy Levenshtein correction to resolve character-level OCR slips (dropping WER).
- Structured dosage and frequency extraction (e.g. 500mg, twice daily).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

# Curated bilingual medical & assistive vocabulary (canonical terms)
MEDICINE_TERMS = {
    # English -> Telugu transliteration / common packaging variant
    "paracetamol": {"canonical": "Paracetamol", "te": "పారాసిటమాల్", "hi": "पैरासिटामोल", "category": "analgesic/antipyretic"},
    "dolo": {"canonical": "Dolo", "te": "డోలో", "hi": "डोलो", "category": "analgesic/antipyretic"},
    "cetirizine": {"canonical": "Cetirizine", "te": "సిట్రిజిన్", "hi": "सिट्रिजिन", "category": "antihistamine"},
    "amoxicillin": {"canonical": "Amoxicillin", "te": "అమోక్సిసిలిన్", "hi": "अमोक्सिसिलिन", "category": "antibiotic"},
    "metformin": {"canonical": "Metformin", "te": "మెట్‌ఫార్మిన్", "hi": "मेटफॉर्मिन", "category": "antidiabetic"},
    "aspirin": {"canonical": "Aspirin", "te": "ఆస్పిరిన్", "hi": "एस्पिरिन", "category": "blood thinner/analgesic"},
    "azithromycin": {"canonical": "Azithromycin", "te": "అజిత్రోమైసిన్", "hi": "एज़िथ्रोमाइसिन", "category": "antibiotic"},
    "omeprazole": {"canonical": "Omeprazole", "te": "ఒమెప్రజోల్", "hi": "ओमेप्राजोल", "category": "antacid"},
    "pantoprazole": {"canonical": "Pantoprazole", "te": "పాంటోప్రజోల్", "hi": "पैंटोप्रजोल", "category": "antacid"},
    "ibuprofen": {"canonical": "Ibuprofen", "te": "ఐబుప్రోఫెన్", "hi": "इबुप्रोफेन", "category": "NSAID"},
    "tablet": {"canonical": "Tablets", "te": "మాత్రలు", "hi": "गोलियाँ", "category": "formulation"},
    "capsule": {"canonical": "Capsules", "te": "క్యాప్సూల్స్", "hi": "कैप्सूल", "category": "formulation"},
    "syrup": {"canonical": "Syrup", "te": "సిరప్", "hi": "सिरप", "category": "formulation"},
}

SIGNAGE_TERMS = {
    "emergency exit": {"canonical": "Emergency Exit", "te": "అత్యవసర నిష్క్రమణ", "hi": "आपातकालीन निकास"},
    "entrance": {"canonical": "Entrance", "te": "ప్రవేశం", "hi": "प्रवेश"},
    "exit": {"canonical": "Exit", "te": "నిష్క్రమణ", "hi": "निकास"},
    "pharmacy": {"canonical": "Pharmacy", "te": "మందుల దుకాణం", "hi": "दवाई की दुकान"},
    "hospital": {"canonical": "Hospital", "te": "ఆసుపత్రి", "hi": "अस्पताल"},
    "restroom": {"canonical": "Restroom", "te": "శౌచాలయం", "hi": "शौचालय"},
    "danger": {"canonical": "Danger", "te": "ప్రమాదం", "hi": "खतरा"},
    "caution": {"canonical": "Caution", "te": "జాగ్రత్త", "hi": "सावधान"},
}

# Dosage regex pattern: e.g. "500 mg", "650mg", "10 ml", "500 మి.గ్రా"
DOSAGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mg|g|ml|mcg|మి\.?గ్రా|గ్రా|మిల్లీ|मि\.?ग्रा|ग्रा)",
    re.IGNORECASE,
)


@dataclass
class PostprocessResult:
    """Structured post-processing result."""

    original_text: str
    corrected_text: str
    detected_medicines: list[dict[str, Any]] = field(default_factory=list)
    detected_signage: list[dict[str, Any]] = field(default_factory=list)
    extracted_dosages: list[str] = field(default_factory=list)
    corrections_applied: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "detected_medicines": self.detected_medicines,
            "detected_signage": self.detected_signage,
            "extracted_dosages": self.extracted_dosages,
            "corrections_applied": self.corrections_applied,
        }


def _find_best_fuzzy_match(
    word: str,
    target_dict: dict[str, dict[str, str]],
    threshold: float = 0.80,
) -> tuple[str, float, dict[str, str]] | None:
    """Find closest canonical term using fuzzy sequence matcher."""
    word_norm = word.strip().lower()
    if len(word_norm) < 3:
        return None

    best_key = None
    best_ratio = 0.0

    # Match against English keys, Telugu translations, and Hindi translations
    for key, data in target_dict.items():
        # Check primary key
        ratio_key = difflib.SequenceMatcher(None, word_norm, key).ratio()
        # Check Telugu form
        te_form = data.get("te", "").strip()
        ratio_te = difflib.SequenceMatcher(None, word, te_form).ratio() if te_form else 0.0
        # Check Hindi form
        hi_form = data.get("hi", "").strip()
        ratio_hi = difflib.SequenceMatcher(None, word, hi_form).ratio() if hi_form else 0.0

        max_ratio = max(ratio_key, ratio_te, ratio_hi)
        if max_ratio > best_ratio:
            best_ratio = max_ratio
            best_key = key

    if best_ratio >= threshold and best_key is not None:
        return best_key, best_ratio, target_dict[best_key]

    return None


def postprocess_ocr_text(
    text: str,
    fuzzy_threshold: float = 0.80,
) -> PostprocessResult:
    """Enhance raw OCR string with domain lexicon matching and dosage parsing."""
    if not text or not text.strip():
        return PostprocessResult(original_text=text, corrected_text=text)

    # 1. Extract dosages
    dosages = [match.group(0) for match in DOSAGE_PATTERN.finditer(text)]

    words = text.split()
    corrected_words = []
    corrections = []
    detected_meds = []
    detected_signs = []

    for word in words:
        # Strip edge punctuation without removing Unicode Indic non-spacing marks (matras/viramas)
        clean_word = word.strip(".,!?:;\"'()[]{}/*-_#")
        match_med = _find_best_fuzzy_match(clean_word, MEDICINE_TERMS, threshold=fuzzy_threshold)

        if match_med:
            key, ratio, data = match_med
            canonical = data["canonical"]
            detected_meds.append({
                "term": canonical,
                "category": data.get("category", ""),
                "matched_from": word,
                "confidence": round(ratio, 3),
            })
            if ratio < 1.0 and ratio >= fuzzy_threshold:
                # If script was Indic, use appropriate Indic script; otherwise canonical English
                is_telugu = any(0x0C00 <= ord(c) <= 0x0C7F for c in word)
                replacement = data.get("te", canonical) if is_telugu else canonical
                corrections.append({"from": word, "to": replacement, "similarity": round(ratio, 2)})
                corrected_words.append(replacement)
            else:
                corrected_words.append(word)
            continue

        match_sign = _find_best_fuzzy_match(clean_word, SIGNAGE_TERMS, threshold=fuzzy_threshold)
        if match_sign:
            key, ratio, data = match_sign
            canonical = data["canonical"]
            detected_signs.append({
                "sign": canonical,
                "matched_from": word,
                "confidence": round(ratio, 3),
            })
            if ratio < 1.0 and ratio >= fuzzy_threshold:
                is_telugu = any(0x0C00 <= ord(c) <= 0x0C7F for c in word)
                replacement = data.get("te", canonical) if is_telugu else canonical
                corrections.append({"from": word, "to": replacement, "similarity": round(ratio, 2)})
                corrected_words.append(replacement)
            else:
                corrected_words.append(word)
            continue

        corrected_words.append(word)

    corrected_text = " ".join(corrected_words)

    return PostprocessResult(
        original_text=text,
        corrected_text=corrected_text,
        detected_medicines=detected_meds,
        detected_signage=detected_signs,
        extracted_dosages=dosages,
        corrections_applied=corrections,
    )
