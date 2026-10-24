"""SmartVision Machine Translation Module.

Translates medication instructions and assistance responses between
Telugu, Hindi, and English while safeguarding pharmaceutical terminology.
"""

from .translator import MedicalTranslator, TranslationResult, get_translator

__all__ = ["MedicalTranslator", "TranslationResult", "get_translator"]
