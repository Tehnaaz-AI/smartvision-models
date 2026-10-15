"""Unit tests for domain post-processing and fuzzy lexicon matching."""

from smartvision_ocr.inference.postprocess import (
    _find_best_fuzzy_match,
    postprocess_ocr_text,
    MEDICINE_TERMS,
    SIGNAGE_TERMS,
)


def test_fuzzy_match_exact_and_near():
    # Exact match
    match = _find_best_fuzzy_match("paracetamol", MEDICINE_TERMS, threshold=0.8)
    assert match is not None
    assert match[0] == "paracetamol"
    assert match[1] == 1.0

    # Near match with 1 typo
    match_typo = _find_best_fuzzy_match("Paracetmol", MEDICINE_TERMS, threshold=0.8)
    assert match_typo is not None
    assert match_typo[0] == "paracetamol"
    assert match_typo[1] > 0.9


def test_postprocess_medicine_and_dosage_extraction():
    text = "Dolo650 Paracetmol 500mg"
    result = postprocess_ocr_text(text, fuzzy_threshold=0.75)
    assert "Paracetamol" in result.corrected_text
    assert len(result.extracted_dosages) >= 1
    assert any("500" in d for d in result.extracted_dosages)
    assert len(result.detected_medicines) >= 1
    assert result.detected_medicines[0]["term"] == "Paracetamol"


def test_postprocess_telugu_medicine_correction():
    text = "పరనిటమల్ 500 మి.గ్రా"
    result = postprocess_ocr_text(text, fuzzy_threshold=0.70)
    assert "పారాసిటమాల్" in result.corrected_text
    assert len(result.detected_medicines) == 1
    assert result.detected_medicines[0]["term"] == "Paracetamol"
    assert len(result.extracted_dosages) == 1


def test_postprocess_signage_matching():
    text = "Emergency Ext -> Hospital"
    result = postprocess_ocr_text(text, fuzzy_threshold=0.80)
    assert len(result.detected_signage) >= 1
    assert any(s["sign"] in ("Hospital", "Emergency Exit") for s in result.detected_signage)


def test_postprocess_empty_string():
    result = postprocess_ocr_text("")
    assert result.original_text == ""
    assert result.corrected_text == ""
