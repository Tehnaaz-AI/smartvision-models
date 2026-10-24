"""Cross-platform audio playback utility for SmartVision smart-glasses hardware."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Handles audio playback through connected speakers or bone conduction."""

    @staticmethod
    def play(audio_path: str) -> bool:
        """Plays the given audio file using system players."""
        path = Path(audio_path)
        if not path.exists():
            logger.error(f"Audio file does not exist: {audio_path}")
            return False

        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return True
            elif sys.platform == "darwin":
                subprocess.Popen(["afplay", str(path)])
                return True
            else:
                # Linux / Raspberry Pi smart-glasses client
                for player in ["aplay", "paplay", "ffplay", "mplayer"]:
                    if subprocess.call(["which", player], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                        subprocess.Popen([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
        except Exception as err:
            logger.warning(f"Audio playback error: {err}")

        return False
