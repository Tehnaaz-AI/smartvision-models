"""SmartVision Text-to-Speech (TTS) Module.

Synthesizes voice responses for smart-glasses audio delivery
supporting Telugu, Hindi, and English.
"""

from .engine import TTSEngine, TTSResult, get_tts_engine
from .audio_player import AudioPlayer

__all__ = ["TTSEngine", "TTSResult", "get_tts_engine", "AudioPlayer"]
