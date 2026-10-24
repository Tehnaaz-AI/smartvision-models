"""Power and Thermal Management Daemon for SmartVision Smart Glasses.

Manages battery state of charge (SoC), operational duty cycles, and thermal throttling
to prevent battery depletion and frame overheating on elderly wearable glasses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("smartvision.power")


class PowerState(str, Enum):
    DEEP_SLEEP = "DEEP_SLEEP"        # Sub-milliamp sleep, waits for hardware interrupt
    STANDBY = "STANDBY"              # Low-power idle, microphone VAD listening
    BURST_CAPTURE = "BURST_CAPTURE"  # Camera sensor & lighting active (1-2s)
    PROCESSING = "PROCESSING"        # Neural inference active on edge / cloud uplink
    THERMAL_THROTTLED = "THROTTLED"  # Temple temperature > 42C, cooling pause


@dataclass
class PowerMetrics:
    battery_percent: int
    voltage: float
    estimated_runtime_minutes: int
    temple_temperature_c: float
    state: PowerState
    is_charging: bool = False


class PowerManager:
    """Monitors battery SoC, throttles processing bursts, and manages deep-sleep duty cycles."""

    MAX_SAFE_TEMPLE_TEMP_C = 42.0   # Wearable safety comfort ceiling
    LOW_BATTERY_THRESHOLD = 15     # Trigger audio + HUD warning
    CRITICAL_BATTERY_THRESHOLD = 5  # Emergency shutdown

    def __init__(self, initial_battery_pct: int = 100):
        self.battery_pct = max(0, min(100, initial_battery_pct))
        self.state = PowerState.STANDBY
        self.temple_temp_c = 28.0   # Ambient starting temperature
        self._last_burst_time = time.time()
        self._active_burst_count = 0

    def get_metrics(self) -> PowerMetrics:
        """Returns the current real-time power and thermal telemetry."""
        # Estimate battery voltage (LiPo curve: ~3.3V empty to ~4.2V full)
        voltage = 3.3 + (self.battery_pct / 100.0) * 0.9
        # Estimate runtime: 400mAh battery / ~80mA average duty cycle ~ 5 hours
        runtime_min = int((self.battery_pct / 100.0) * 300)

        return PowerMetrics(
            battery_percent=self.battery_pct,
            voltage=round(voltage, 2),
            estimated_runtime_minutes=runtime_min,
            temple_temperature_c=round(self.temple_temp_c, 1),
            state=self.state,
        )

    def request_camera_burst(self) -> bool:
        """Requests authorization to power up camera and capture frames.

        Returns True if power and thermal conditions are safe.
        """
        if self.battery_pct <= self.CRITICAL_BATTERY_THRESHOLD:
            logger.error("Capture denied: Battery critical (%d%%)", self.battery_pct)
            return False

        if self.temple_temp_c >= self.MAX_SAFE_TEMPLE_TEMP_C:
            self.state = PowerState.THERMAL_THROTTLED
            logger.warning("Capture throttled: Temple temp too high (%.1f C)", self.temple_temp_c)
            return False

        self.state = PowerState.BURST_CAPTURE
        self._active_burst_count += 1
        # Each burst expends ~0.1% battery and increases temple temperature slightly
        self.battery_pct = max(0, self.battery_pct - 1)
        self.temple_temp_c = min(45.0, self.temple_temp_c + 0.3)
        return True

    def notify_processing_complete(self) -> None:
        """Returns glasses back to low-power standby after an identification cycle."""
        self.state = PowerState.STANDBY
        # Passive cooling when idle
        self.temple_temp_c = max(26.0, self.temple_temp_c - 0.2)

    def enter_deep_sleep(self) -> None:
        """Powers down camera, Wi-Fi radio, and display until physical tap or button."""
        self.state = PowerState.DEEP_SLEEP
        logger.info("Smart glasses entered DEEP_SLEEP (current drain < 1mA)")
