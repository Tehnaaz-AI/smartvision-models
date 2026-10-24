"""Smart Glasses Hardware Controller and Edge Pipeline Orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SystemEvent(str, Enum):
    """Triggers received from glasses physical sensors."""

    BUTTON_PRESS = "BUTTON_PRESS"
    SCHEDULED_ALARM = "SCHEDULED_ALARM"
    VOICE_COMMAND = "VOICE_COMMAND"


@dataclass
class GlassesResponse:
    """End-to-end response delivered back to smart glasses hardware."""

    spoken_text: str
    spoken_text_te: Optional[str]
    is_safe: bool
    status: str
    audio_file: Optional[str] = None
    display_summary: Optional[str] = None


class SmartGlassesController:
    """Coordinates hardware peripherals with the SmartVision AI modules."""

    def __init__(self, patient_language: str = "te", mock_hardware: bool = False):
        self.patient_language = patient_language
        self.mock_hardware = mock_hardware

    def process_medication_check(
        self,
        frames: List[np.ndarray],
        scheduled_medicine: Optional[str] = None,
        voice_query: Optional[str] = None,
    ) -> GlassesResponse:
        """Executes the complete hardware-to-AI processing cycle.

        1. Frame Quality & Keyframe Selection (VideoModel)
        2. Packaging ROI Detection (VisionModel)
        3. Multilingual Text Extraction (OCRModel)
        4. Deterministic Prescription Verification (Backend)
        5. Speech Synthesis (TTSModel)
        """
        # Step 1: Select optimal keyframe from burst
        from smartvision_video import KeyframeSelector
        video_selector = KeyframeSelector()
        video_res = video_selector.select_best_frame(frames)

        if not video_res.success or video_res.best_frame is None:
            return GlassesResponse(
                spoken_text="The captured images were too blurry or poorly lit. Please hold your medicine package steady.",
                spoken_text_te="చిత్రాలు అస్పష్టంగా ఉన్నాయి. దయచేసి మందు ప్యాకెట్‌ను కదలకుండా చూపించండి.",
                is_safe=False,
                status="BLURRY_CAPTURE",
                display_summary="Blurry / Retry",
            )

        best_frame = video_res.best_frame

        # Step 2: Packaging Detection and Region Isolation (VisionModel)
        from smartvision_vision import get_package_detector
        detector = get_package_detector()
        vis_res = detector.detect(best_frame)
        package_crop = vis_res.cropped_image if vis_res.success and vis_res.cropped_image is not None else best_frame

        # Step 3: Multilingual OCR (OCRModel)
        from smartvision_ocr.inference.pipeline import get_pipeline
        pipeline = get_pipeline(recognition_engine="multilingual_ppocr")
        ocr_res = pipeline.predict_full_image(package_crop, language="te", preprocess=True, postprocess=True)

        extracted_text = ocr_res.postprocessed_text or ocr_res.full_text
        detected_strength = ocr_res.domain_insights.get("strength")

        # Step 4: Deterministic Prescription Verification (Backend)
        from smartvision_backend import get_verifier
        verifier = get_verifier()
        verif_res = verifier.verify_capture(
            detected_text=extracted_text,
            detected_strength=detected_strength,
            scheduled_medicine_name=scheduled_medicine,
            ocr_confidence=ocr_res.average_confidence,
        )

        # Step 5: Text-to-Speech (TTSModel)
        spoken_msg = (
            verif_res.spoken_message_te
            if self.patient_language == "te" and verif_res.spoken_message_te
            else verif_res.spoken_message
        )

        from smartvision_tts import get_tts_engine
        tts = get_tts_engine(language=self.patient_language)
        audio_res = tts.synthesize(spoken_msg)

        return GlassesResponse(
            spoken_text=verif_res.spoken_message,
            spoken_text_te=verif_res.spoken_message_te,
            is_safe=verif_res.is_safe_to_take,
            status=verif_res.status.value,
            audio_file=audio_res.audio_path if audio_res.success else None,
            display_summary=f"{verif_res.status.value}: {verif_res.action_required}",
        )
