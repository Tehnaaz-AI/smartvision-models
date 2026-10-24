"""SmartVision Automatic Speech Recognition (ASR) Module.

Processes spoken user commands from the smart-glasses microphone
with trilingual intent spotting (Telugu, Hindi, English).
"""

from .engine import ASREngine, ASRResult, get_asr_engine
from .intents import IntentClassifier, UserIntent

__all__ = ["ASREngine", "ASRResult", "get_asr_engine", "IntentClassifier", "UserIntent"]
