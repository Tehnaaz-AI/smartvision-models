"""Unit Tests for On-Lens HUD Driver and Smart Glasses Power Manager."""

import numpy as np
import pytest

from smartvision_hardware.hud import SmartGlassesHUD, HUDCard, HUDTheme
from smartvision_hardware.power_manager import PowerManager, PowerState


def test_hud_safe_verification():
    hud = SmartGlassesHUD(width=128, height=64)
    card = hud.display_verification_result(
        is_safe=True,
        medicine_name="Dolo 650",
        dosage="1 Tablet",
    )
    assert card.status == "MATCHED"
    assert "SAFE" in card.header
    assert "Dolo 650" in card.main_text

    # Test ASCII preview rendering
    ascii_out = hud.render_ascii_preview()
    assert "[OK - SAFE]" in ascii_out
    assert "Dolo 650" in ascii_out


def test_hud_alert_verification():
    hud = SmartGlassesHUD(width=128, height=64)
    card = hud.display_verification_result(
        is_safe=False,
        medicine_name="Amlodipine",
        dosage="5 mg",
        warning_message="Unprescribed medicine!",
    )
    assert card.status == "MISMATCH"
    assert "DO NOT TAKE" in card.header

    ascii_out = hud.render_ascii_preview()
    assert "[!! ALERT !!]" in ascii_out


def test_hud_rgb_and_spi_framebuffer():
    hud = SmartGlassesHUD(width=128, height=64)
    card = hud.display_verification_result(is_safe=True, medicine_name="Dolo", dosage="1 Tab")

    fb = hud.render_rgb_framebuffer(card)
    assert isinstance(fb, np.ndarray)
    assert fb.shape == (64, 128, 3)

    # SPI 1-bit monochrome export (128 cols * 8 pages = 1024 bytes)
    spi_bytes = hud.export_spi_bitmap_1bit(card)
    assert isinstance(spi_bytes, bytes)
    assert len(spi_bytes) == (128 * 64) // 8


def test_power_manager_burst_and_duty_cycle():
    pm = PowerManager(initial_battery_pct=100)
    metrics = pm.get_metrics()
    assert metrics.battery_percent == 100
    assert metrics.state == PowerState.STANDBY
    assert metrics.voltage >= 4.0

    # Request camera burst
    allowed = pm.request_camera_burst()
    assert allowed is True
    assert pm.state == PowerState.BURST_CAPTURE
    assert pm.battery_pct == 99
    assert pm.temple_temp_c > 28.0

    pm.notify_processing_complete()
    assert pm.state == PowerState.STANDBY


def test_power_manager_thermal_throttling():
    pm = PowerManager(initial_battery_pct=80)
    pm.temple_temp_c = 43.0  # Above 42.0C limit

    allowed = pm.request_camera_burst()
    assert allowed is False
    assert pm.state == PowerState.THERMAL_THROTTLED


def test_power_manager_critical_battery_cutoff():
    pm = PowerManager(initial_battery_pct=3)  # Below 5% cutoff
    allowed = pm.request_camera_burst()
    assert allowed is False
