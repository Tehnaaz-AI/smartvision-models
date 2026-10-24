"""Unit Tests for SmartVision HardwareBridge."""

import pytest
import numpy as np
from pathlib import Path

from smartvision_hardware.controller import SmartGlassesController, SystemEvent, GlassesResponse


def test_system_event_enum():
    assert SystemEvent.BUTTON_PRESS == "BUTTON_PRESS"
    assert SystemEvent.SCHEDULED_ALARM == "SCHEDULED_ALARM"
    assert SystemEvent.VOICE_COMMAND == "VOICE_COMMAND"


def test_controller_blurry_rejection():
    # Provide pitch black / heavily blurred frames that VideoModel rejects
    blurry_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    controller = SmartGlassesController(patient_language="te")

    res = controller.process_medication_check(blurry_frames)
    assert res.is_safe is False
    assert res.status == "BLURRY_CAPTURE"
    assert "blurry" in res.spoken_text.lower() or "అస్పష్టంగా" in res.spoken_text_te
