"""Unit Tests for SmartVision TranslationModel."""

import pytest

from smartvision_translation.translator import MedicalTranslator, get_translator
from smartvision_translation.dictionary import PHRASE_DICTIONARY, PHARMACEUTICAL_NAMES


def test_translation_english_to_telugu():
    translator = get_translator()
    res = translator.translate("Take 1 tablet twice daily after food", source_lang="en", target_lang="te")

    assert res.success is True
    assert "1 టాబ్లెట్" in res.translated_text
    assert "రోజుకు రెండుసార్లు" in res.translated_text
    assert "భోజనం తర్వాత" in res.translated_text


def test_translation_english_to_hindi():
    translator = get_translator()
    res = translator.translate("Take 1 tablet after meals", source_lang="en", target_lang="hi")

    assert res.success is True
    assert "1 गोली" in res.translated_text
    assert "भोजन के बाद" in res.translated_text


def test_pharmaceutical_preservation_dolo():
    translator = get_translator()
    res = translator.translate("Dolo 650 take 1 tablet", source_lang="en", target_lang="te")

    assert res.success is True
    assert "డోలో" in res.translated_text
    assert "Dolo" in res.preserved_terms or "డోలో" in res.preserved_terms


def test_pharmaceutical_preservation_paracetamol():
    translator = get_translator()
    res = translator.translate("Paracetamol for fever", source_lang="en", target_lang="hi")

    assert res.success is True
    assert "पैरासिटामोल" in res.translated_text
    assert "बुखार के लिए" in res.translated_text


def test_same_language_identity():
    translator = get_translator()
    text = "Take 1 tablet"
    res = translator.translate(text, source_lang="en", target_lang="en")
    assert res.success is True
    assert res.translated_text == text


def test_unsupported_language():
    translator = get_translator()
    res = translator.translate("Test", source_lang="fr", target_lang="te")
    assert res.success is False
    assert "Unsupported source" in res.error_message


def test_empty_input():
    translator = get_translator()
    res = translator.translate("   ")
    assert res.success is False
    assert "cannot be empty" in res.error_message
