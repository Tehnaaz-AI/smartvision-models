"""On-Lens Micro-Display HUD Driver & Simulator for SmartVision Smart Glasses.

Renders high-contrast optical visual cues for the smart glasses waveguide prism / micro-OLED display.
Supports monochrome (1-bit SSD1306/SH1106) and color (RGB565 micro-OLED) framebuffers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("smartvision.hud")


class HUDTheme:
    SAFE_GREEN = (0, 255, 64)       # High-visibility phosphorescent green
    ALERT_RED = (255, 32, 32)        # High-visibility warning red
    WARN_AMBER = (255, 191, 0)       # Amber caution
    INFO_CYAN = (0, 220, 255)        # Informational cyan
    BACKGROUND = (0, 0, 0)           # True black (OLED pixels off = transparent lens)
    WHITE = (255, 255, 255)


@dataclass
class HUDCard:
    header: str
    main_text: str
    sub_text: str
    status: str  # "MATCHED", "MISMATCH", "REMINDER", "LISTENING", "PROCESSING"
    battery_pct: int = 85
    wifi_connected: bool = True


class SmartGlassesHUD:
    """Optical Head-Up Display (HUD) driver for smart glasses."""

    def __init__(self, width: int = 128, height: int = 64):
        self.width = width
        self.height = height
        self._current_card: Optional[HUDCard] = None

    def display_verification_result(
        self,
        is_safe: bool,
        medicine_name: str,
        dosage: str,
        warning_message: Optional[str] = None,
    ) -> HUDCard:
        """Formats and activates a verification result card on the lens."""
        if is_safe:
            card = HUDCard(
                header="VERIFIED SAFE",
                main_text=medicine_name[:16],
                sub_text=f"Take: {dosage}",
                status="MATCHED",
            )
        else:
            card = HUDCard(
                header="WARNING: DO NOT TAKE",
                main_text=medicine_name[:16] if medicine_name else "UNKNOWN MED",
                sub_text=warning_message[:24] if warning_message else "Unprescribed!",
                status="MISMATCH",
            )
        self._current_card = card
        return card

    def display_reminder(self, medicine_name: str, time_str: str) -> HUDCard:
        """Displays a timely medication reminder."""
        card = HUDCard(
            header=f"REMINDER [{time_str}]",
            main_text=medicine_name[:16],
            sub_text="Time to take your medication",
            status="REMINDER",
        )
        self._current_card = card
        return card

    def display_listening_state(self) -> HUDCard:
        """Displays listening / voice recognition indicator."""
        card = HUDCard(
            header="SMARTVISION",
            main_text="Listening...",
            sub_text="Speak command",
            status="LISTENING",
        )
        self._current_card = card
        return card

    def render_ascii_preview(self, card: Optional[HUDCard] = None) -> str:
        """Renders an ASCII simulation of the on-lens display for debugging."""
        c = card or self._current_card or HUDCard("SMARTVISION", "Standby", "Ready", "IDLE")
        border = "+" + "-" * 32 + "+"
        status_symbol = {
            "MATCHED": "[OK - SAFE]",
            "MISMATCH": "[!! ALERT !!]",
            "REMINDER": "[REMINDER]",
            "LISTENING": "[MIC ON]",
            "IDLE": "[STANDBY]",
        }.get(c.status, "[INFO]")

        lines = [
            border,
            f"| {status_symbol:<16}  BAT:{c.battery_pct}% {'WF' if c.wifi_connected else '--'} |",
            f"| {c.header[:30]:<30} |",
            f"| > {c.main_text[:28]:<28} |",
            f"|   {c.sub_text[:28]:<28} |",
            border,
        ]
        return "\n".join(lines)

    def render_rgb_framebuffer(self, card: Optional[HUDCard] = None) -> np.ndarray:
        """Renders a raw RGB framebuffer (height x width x 3) for the micro-OLED driver."""
        c = card or self._current_card or HUDCard("SMARTVISION", "Standby", "Ready", "IDLE")
        fb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Border color based on status
        if c.status == "MATCHED":
            color = HUDTheme.SAFE_GREEN
        elif c.status == "MISMATCH":
            color = HUDTheme.ALERT_RED
        elif c.status == "REMINDER":
            color = HUDTheme.WARN_AMBER
        else:
            color = HUDTheme.INFO_CYAN

        # Draw frame boundary
        fb[0:2, :, :] = color
        fb[-2:, :, :] = color
        fb[:, 0:2, :] = color
        fb[:, -2:, :] = color

        return fb

    def export_spi_bitmap_1bit(self, card: Optional[HUDCard] = None) -> bytes:
        """Exports 1-bit packed byte stream for monochrome SSD1306/SH1106 displays."""
        fb_rgb = self.render_rgb_framebuffer(card)
        mono = np.any(fb_rgb > 0, axis=-1)  # Boolean 2D array
        # Pack into bytes (page-addressed SSD1306 format: 8 vertical pixels per byte)
        pages = self.height // 8
        buffer = bytearray(self.width * pages)
        for page in range(pages):
            for col in range(self.width):
                byte = 0
                for bit in range(8):
                    y = page * 8 + bit
                    if mono[y, col]:
                        byte |= (1 << bit)
                buffer[page * self.width + col] = byte
        return bytes(buffer)
