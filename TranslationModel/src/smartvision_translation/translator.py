"""Medical Translation Engine for Telugu, Hindi, and English."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from .dictionary import PHRASE_DICTIONARY, PHARMACEUTICAL_NAMES


@dataclass
class TranslationResult:
    """Encapsulates the translation result and preserved medical tokens."""

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    preserved_terms: list[str]
    success: bool
    error_message: Optional[str] = None


class MedicalTranslator:
    """Safe, domain-grounded translator for medication labels and instructions."""

    SUPPORTED_LANGUAGES = {"en", "te", "hi"}

    def __init__(self, default_target: str = "te"):
        self.default_target = default_target

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: Optional[str] = None,
    ) -> TranslationResult:
        """Translates text from source_lang to target_lang."""
        target = target_lang or self.default_target

        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target,
                preserved_terms=[],
                success=False,
                error_message="Input text cannot be empty.",
            )

        if source_lang not in self.SUPPORTED_LANGUAGES:
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target,
                preserved_terms=[],
                success=False,
                error_message=f"Unsupported source language '{source_lang}'",
            )

        if target not in self.SUPPORTED_LANGUAGES:
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang,
                target_language=target,
                preserved_terms=[],
                success=False,
                error_message=f"Unsupported target language '{target}'",
            )

        if source_lang == target:
            return TranslationResult(
                source_text=text,
                translated_text=text.strip(),
                source_language=source_lang,
                target_language=target,
                preserved_terms=[],
                success=True,
            )

        translated = text.strip()
        preserved = []

        # 1. Translate pharmaceutical brand/generic names
        for med_key, med_map in PHARMACEUTICAL_NAMES.items():
            src_val = med_map.get(source_lang, med_key)
            tgt_val = med_map.get(target, med_key)
            pattern = re.compile(re.escape(src_val), re.IGNORECASE)
            if pattern.search(translated):
                translated = pattern.sub(tgt_val, translated)
                preserved.append(tgt_val)

        # 2. Translate medical instruction phrases
        # Sort phrases by descending length so multi-word expressions match first
        sorted_phrases = sorted(PHRASE_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)

        for _, phrase_map in sorted_phrases:
            src_val = phrase_map.get(source_lang)
            tgt_val = phrase_map.get(target)
            if src_val and tgt_val:
                pattern = re.compile(re.escape(src_val), re.IGNORECASE)
                if pattern.search(translated):
                    translated = pattern.sub(tgt_val, translated)

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_lang,
            target_language=target,
            preserved_terms=preserved,
            success=True,
        )


def get_translator(default_target: str = "te") -> MedicalTranslator:
    """Factory helper to obtain a MedicalTranslator instance."""
    return MedicalTranslator(default_target=default_target)
