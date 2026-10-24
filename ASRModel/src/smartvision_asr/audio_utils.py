"""Audio verification and signal preprocessing utilities for SmartVision ASR."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, Any


def validate_wav(path: str) -> Dict[str, Any]:
    """Inspects a WAV audio recording for microphone input specifications."""
    file_path = Path(path)
    if not file_path.exists():
        return {"valid": False, "error": f"File does not exist: {path}"}

    try:
        with wave.open(str(file_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            duration = nframes / float(framerate) if framerate > 0 else 0.0

            return {
                "valid": True,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "framerate_hz": framerate,
                "duration_seconds": round(duration, 2),
                "is_mono": channels == 1,
                "is_16khz": framerate == 16000,
            }
    except wave.Error as err:
        return {"valid": False, "error": f"Invalid WAV file format: {err}"}
