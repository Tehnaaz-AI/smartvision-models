"""Unit Tests for SmartVision Text-to-Speech Engine."""

import os
import wave
import pytest
from pathlib import Path

from smartvision_tts.engine import TTSEngine, TTSResult, get_tts_engine
from smartvision_tts.audio_player import AudioPlayer


def test_tts_engine_initialization(tmp_path):
    engine = TTSEngine(language="en", speech_rate=130, volume=0.9, output_dir=str(tmp_path))
    assert engine.language == "en"
    assert engine.speech_rate == 130
    assert engine.volume == 0.9


def test_tts_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        TTSEngine(language="fr")


def test_tts_synthesize_empty_text(tmp_path):
    engine = TTSEngine(output_dir=str(tmp_path))
    result = engine.synthesize("")
    assert not result.success
    assert "cannot be empty" in result.error_message


def test_tts_synthesize_english(tmp_path):
    engine = TTSEngine(language="en", output_dir=str(tmp_path))
    text = "Please take your medication on time."
    result = engine.synthesize(text)

    assert result.success
    assert result.audio_path is not None
    assert Path(result.audio_path).exists()
    assert result.duration_seconds > 0

    # Verify that the generated file is a valid readable WAV
    with wave.open(result.audio_path, "r") as wav_file:
        assert wav_file.getnchannels() in (1, 2)
        assert wav_file.getframerate() > 0


def test_tts_synthesize_telugu(tmp_path):
    engine = TTSEngine(language="te", output_dir=str(tmp_path))
    telugu_text = "దయచేసి మీ మందును సకాలంలో తీసుకోండి."
    result = engine.synthesize(telugu_text)

    assert result.success
    assert result.audio_path is not None
    assert Path(result.audio_path).exists()


def test_tts_synthesize_hindi(tmp_path):
    engine = TTSEngine(language="hi", output_dir=str(tmp_path))
    hindi_text = "कृपया अपनी दवा समय पर लें।"
    result = engine.synthesize(hindi_text)

    assert result.success
    assert result.audio_path is not None
    assert Path(result.audio_path).exists()


def test_audio_player_non_existent():
    # Playing a non-existent file should safely return False without crashing
    assert not AudioPlayer.play("non_existent_audio_file_123.wav")
