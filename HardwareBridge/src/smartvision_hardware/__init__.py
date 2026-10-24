"""SmartVision Hardware Bridge Module.

Interfaces smart-glasses physical peripherals (camera, microphone, speaker)
with the modular AI processing pipeline.
"""

from .controller import SmartGlassesController, SystemEvent

__all__ = ["SmartGlassesController", "SystemEvent"]
