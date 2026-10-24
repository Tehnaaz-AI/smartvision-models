"""Unit Tests for SmartVision ASR Module."""

import pytest
import wave
import struct
from pathlib import Path

from smartvision_asr.engine import ASREngine, get_asr_engine
from smartvision_asr.intents import IntentClassifier, UserIntent
from smartvision_asr.audio_utils import validate_wav


def test_intent_classification_english():
    classifier = IntentClassifier()

    m1 = classifier.classify("What is this medicine?")
    assert m1.intent == UserIntent.IDENTIFY_MEDICINE
    assert m1.language == "en"

    m2 = classifier.classify("How many tablets should I take?")
    assert m2.intent == UserIntent.CHECK_DOSAGE

    m3 = classifier.classify("When should I take this, before or after food?")
    assert m3.intent == UserIntent.CHECK_SCHEDULE

    m4 = classifier.classify("Please read instructions on the label")
    assert m4.intent == UserIntent.READ_INSTRUCTIONS

    m5 = classifier.classify("Help me emergency!")
    assert m5.intent == UserIntent.EMERGENCY_HELP


def test_intent_classification_telugu():
    classifier = IntentClassifier()

    m1 = classifier.classify("ఈ మందు ఏమిటి?")
    assert m1.intent == UserIntent.IDENTIFY_MEDICINE
    assert m1.language == "te"

    m2 = classifier.classify("ఎన్ని టాబ్లెట్లు వేసుకోవాలి?")
    assert m2.intent == UserIntent.CHECK_DOSAGE
    assert m2.language == "te"

    m3 = classifier.classify("ఇది ఎప్పుడు తీసుకోవాలి?")
    assert m3.intent == UserIntent.CHECK_SCHEDULE
    assert m3.language == "te"


def test_intent_classification_hindi():
    classifier = IntentClassifier()

    m1 = classifier.classify("यह कौन सी दवा है?")
    assert m1.intent == UserIntent.IDENTIFY_MEDICINE
    assert m1.language == "hi"

    m2 = classifier.classify("कितनी खुराक लेनी है?")
    assert m2.intent == UserIntent.CHECK_DOSAGE
    assert m2.language == "hi"


def test_empty_query():
    classifier = IntentClassifier()
    match = classifier.classify("   ")
    assert match.intent == UserIntent.UNKNOWN


def test_audio_utils_valid_wav(tmp_path):
    wav_file = tmp_path / "test_audio.wav"
    with wave.open(str(wav_file), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 0.5s silence
        for _ in range(8000):
            wf.writeframes(struct.pack("<h", 0))

    val = validate_wav(str(wav_file))
    assert val["valid"] is True
    assert val["channels"] == 1
    assert val["framerate_hz"] == 16000
    assert val["duration_seconds"] == 0.5
    assert val["is_mono"] is True
    assert val["is_16khz"] is True


def test_audio_utils_missing_file():
    val = validate_wav("non_existent_audio_sample_xyz.wav")
    assert val["valid"] is False
    assert "does not exist" in val["error"]


def test_asr_engine_text_processing():
    engine = get_asr_engine()
    res = engine.process_text_command("What tablet is this?")
    assert res.success is True
    assert res.intent == UserIntent.IDENTIFY_MEDICINE
