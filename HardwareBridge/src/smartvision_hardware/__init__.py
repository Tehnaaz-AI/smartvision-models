"""SmartVision Hardware Bridge Module.

Interfaces smart-glasses physical peripherals (camera, microphone, speaker)
with the modular AI processing pipeline.
"""

from .controller import SmartGlassesController, SystemEvent
from .hud import SmartGlassesHUD, HUDCard, HUDTheme
from .power_manager import PowerManager, PowerState, PowerMetrics

__all__ = [
    "SmartGlassesController",
    "SystemEvent",
    "SmartGlassesHUD",
    "HUDCard",
    "HUDTheme",
    "PowerManager",
    "PowerState",
    "PowerMetrics",
]
