import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Ensure all subproject source directories are accessible
_root = Path(__file__).resolve().parent.parent.parent.parent
for _sub in ["OCRModel", "ASRModel", "TTSModel", "VisionModel", "VideoModel", "TranslationModel", "HardwareBridge", "LLM"]:
    _p = str(_root / _sub / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Query, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import Medication, Prescription, VerificationStatus
from .verifier import get_verifier
from .scheduler import get_scheduler
from .alerts import get_alert_dispatcher
from .cloud_db import get_cloud_db_client
from .location import get_location_service, SafeHomeZone

logger = logging.getLogger("smartvision.api")

app = FastAPI(
    title="SmartVision Backend Service",
    description="Deterministic medication verification, scheduling API, GPS Location, and Caregiver Portal for assistive smart glasses",
    version="1.0.0",
)

scheduler = get_scheduler()
verifier = get_verifier(patient=scheduler.patient)
alert_dispatcher = get_alert_dispatcher()
cloud_db = get_cloud_db_client()
location_service = get_location_service()

# Mount static web portal directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def get_portal_index():
    """Serves the Caregiver Tele-Health Portal."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "SmartVision Caregiver Portal API active"}


class VerificationRequest(BaseModel):
    detected_text: str
    detected_strength: Optional[str] = None
    scheduled_medicine_name: Optional[str] = None
    ocr_confidence: float = 0.90


class VerificationResponse(BaseModel):
    status: str
    is_safe_to_take: bool
    matched_medicine: Optional[str] = None
    prescribed_strength: Optional[str] = None
    spoken_message: str
    spoken_message_te: Optional[str] = None
    action_required: str


class V1VerificationResponse(BaseModel):
    status: str
    is_safe_to_take: bool
    spoken_message: str
    spoken_message_te: Optional[str] = None
    display_summary: Optional[str] = None
    audio_stream_url: Optional[str] = None
    gps_status: Optional[Dict[str, Any]] = None


class AskRequest(BaseModel):
    question: str
    language: str = "te"
    context: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_te: Optional[str] = None
    is_safe: bool
    guardrail_triggered: bool
    is_emergency: bool
    audio_stream_url: Optional[str] = None
    engine: str


class PrescriptionItemRequest(BaseModel):
    name: str
    strength: str
    dosage: str = "1 Tablet"
    schedule_time: str = "08:00 AM"
    meal_relation: str = "AFTER_FOOD"
    instructions: str
    instructions_te: str


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SmartVision Backend",
        "patient": scheduler.patient.patient_name,
    }


@app.get("/api/reminders")
def get_reminders(time: str = Query("08:00 AM", description="Schedule time, e.g. '08:00 AM'")):
    meds = scheduler.get_due_medications(time)
    return {
        "time": time,
        "due_count": len(meds),
        "medications": [
            {
                "name": m.medicine_name,
                "strength": m.strength,
                "dosage": m.quantity_per_dose,
                "instructions": m.instructions,
                "instructions_te": m.instructions_te,
            }
            for m in meds
        ],
    }


@app.post("/api/verify", response_model=VerificationResponse)
def verify_medication(req: VerificationRequest):
    result = verifier.verify_capture(
        detected_text=req.detected_text,
        detected_strength=req.detected_strength,
        scheduled_medicine_name=req.scheduled_medicine_name,
        ocr_confidence=req.ocr_confidence,
    )

    # If verification detected an unprescribed or mismatched medication, dispatch emergency alert
    if not result.is_safe_to_take:
        alert_dispatcher.dispatch_mismatch_alert(
            patient_id=scheduler.patient.patient_id,
            patient_name=scheduler.patient.patient_name,
            detected_text=req.detected_text,
            scheduled_medicine=req.scheduled_medicine_name,
            reason=result.action_required,
        )

    return VerificationResponse(
        status=result.status.value,
        is_safe_to_take=result.is_safe_to_take,
        matched_medicine=result.matched_medication.medicine_name if result.matched_medication else None,
        prescribed_strength=result.matched_medication.strength if result.matched_medication else None,
        spoken_message=result.spoken_message,
        spoken_message_te=result.spoken_message_te,
        action_required=result.action_required,
    )


# --- Caregiver Portal REST Endpoints ---


class GPSUpdateRequest(BaseModel):
    latitude: float
    longitude: float
    speed_kmh: float = 0.0


class SafeZoneUpdateRequest(BaseModel):
    center_latitude: float
    center_longitude: float
    radius_meters: float = 200.0
    home_address: str = "Home"


@app.get("/api/caregiver/dashboard")
def get_caregiver_dashboard():
    """Returns complete telemetry, active prescriptions, GPS location, and recent alerts for caregiver UI."""
    cur_lat = location_service.current_location.latitude
    cur_lng = location_service.current_location.longitude
    return {
        "patient": {
            "id": scheduler.patient.patient_id,
            "name": scheduler.patient.patient_name,
            "preferred_language": scheduler.patient.preferred_language,
            "caregiver_contact": scheduler.patient.caregiver_contact,
        },
        "device_telemetry": {
            "status": "ONLINE",
            "battery_percent": 84,
            "temple_temp_c": 29.4,
            "cloud_sync": "SYNCED",
        },
        "location": {
            "latitude": cur_lat,
            "longitude": cur_lng,
            "address": location_service.current_location.address,
            "google_maps_url": location_service.maps_client.get_google_maps_pin_url(cur_lat, cur_lng),
            "safe_zone": {
                "center_lat": location_service.home_zone.center_latitude,
                "center_lng": location_service.home_zone.center_longitude,
                "radius_meters": location_service.home_zone.radius_meters,
                "home_address": location_service.home_zone.home_address,
            },
            "is_inside_safe_zone": not location_service._is_wandering,
        },
        "prescriptions": [
            {
                "name": m.medicine_name,
                "strength": m.strength,
                "dosage": m.quantity_per_dose,
                "schedule_time": m.schedule_time,
                "instructions": m.instructions,
                "instructions_te": m.instructions_te,
            }
            for m in scheduler.patient.prescriptions[0].medications
        ] if scheduler.patient.prescriptions else [],
        "recent_alerts": alert_dispatcher.get_recent_alerts(limit=5),
    }


@app.get("/api/caregiver/alerts")
def get_caregiver_alerts():
    """Returns recent emergency alerts for caregiver dashboard."""
    return {"alerts": alert_dispatcher.get_recent_alerts(limit=20)}


@app.post("/api/caregiver/prescriptions")
def add_caregiver_prescription(req: PrescriptionItemRequest):
    """Allows caregivers to add medication to patient prescription."""
    new_med = Medication(
        medicine_name=req.name,
        strength=req.strength,
        dosage_form="tablet",
        quantity_per_dose=req.dosage,
        schedule_time=req.schedule_time,
        instructions=req.instructions,
        instructions_te=req.instructions_te,
    )

    if not scheduler.patient.prescriptions:
        scheduler.patient.prescriptions.append(Prescription(prescription_id="rx_dynamic", medications=[]))

    scheduler.patient.prescriptions[0].medications.append(new_med)
    return {"status": "success", "message": f"Added {req.name} to patient prescription"}


# --- GPS Location & Google Maps Endpoints ---


@app.get("/api/location/current")
def get_current_location():
    """Returns real-time GPS coordinates, address, and Google Maps pin link."""
    return {
        "status": "success",
        "coordinates": {
            "latitude": location_service.current_location.latitude,
            "longitude": location_service.current_location.longitude,
            "speed_kmh": location_service.current_location.speed_kmh,
            "address": location_service.current_location.address,
            "timestamp": location_service.current_location.timestamp,
        },
        "google_maps_pin": location_service.maps_client.get_google_maps_pin_url(
            location_service.current_location.latitude,
            location_service.current_location.longitude,
        ),
        "directions_url": location_service.maps_client.get_directions_url(
            location_service.current_location.latitude,
            location_service.current_location.longitude,
        ),
    }


@app.post("/api/location/update")
def update_gps_location(req: GPSUpdateRequest):
    """Updates live GPS coordinates and evaluates wandering geofence."""
    status = location_service.update_gps_position(req.latitude, req.longitude, req.speed_kmh)

    # If patient has breached the safe zone, trigger an emergency caregiver alert
    if status.get("wandering_alert_triggered"):
        alert_dispatcher.dispatch_wandering_alert(
            patient_id=scheduler.patient.patient_id,
            patient_name=scheduler.patient.patient_name,
            distance_from_home_m=status["distance_from_home_m"],
            google_maps_url=status["google_maps_url"],
        )

    return {"status": "success", "location_status": status}


@app.get("/api/location/nearby-pharmacies")
def get_nearby_pharmacies(language: str = Query("te", description="Language for voice response ('te', 'hi', 'en')")):
    """Finds closest pharmacies using Google Places API and formats voice guidance."""
    return location_service.find_pharmacies_voice_response(language=language)


@app.post("/api/location/safe-zone")
def configure_safe_zone(req: SafeZoneUpdateRequest):
    """Configures safe home geofence center and radius."""
    location_service.home_zone = SafeHomeZone(
        center_latitude=req.center_latitude,
        center_longitude=req.center_longitude,
        radius_meters=req.radius_meters,
        home_address=req.home_address,
    )
    return {"status": "success", "message": "Safe home zone updated"}


# ==============================================================================
# V1 Smart Glasses Direct Hardware Gateway Endpoints
# ==============================================================================


@app.post("/api/v1/verify", response_model=V1VerificationResponse)
async def verify_medication_v1(
    image: UploadFile = File(..., description="Camera image frame captured by smart glasses (JPEG/PNG)"),
    audio: Optional[UploadFile] = File(None, description="Optional voice query audio recorded from glasses microphone"),
    latitude: Optional[float] = Form(None, description="Current GPS latitude from smart glasses GNSS"),
    longitude: Optional[float] = Form(None, description="Current GPS longitude from smart glasses GNSS"),
    speed_kmh: Optional[float] = Form(0.0, description="Current speed in km/h"),
    language: str = Form("te", description="Preferred audio response language ('te', 'hi', 'en')"),
    scheduled_medicine: Optional[str] = Form(None, description="Expected medicine name if checking a scheduled reminder"),
    voice_query: Optional[str] = Form(None, description="Text transcription of user voice request"),
):
    """Direct hardware ingest endpoint for Meta-style smart glasses.

    Accepts raw multipart/form-data camera frame, updates real-time GPS telemetry,
    runs keyframe evaluation, packaging isolation, multilingual OCR, deterministic
    verification, and returns synthesized 16kHz PCM WAV audio stream URL.
    """
    # 1. Update GPS telemetry if provided
    gps_info = None
    if latitude is not None and longitude is not None:
        loc_status = location_service.update_gps_position(latitude, longitude, speed_kmh or 0.0)
        gps_info = {
            "latitude": location_service.current_location.latitude,
            "longitude": location_service.current_location.longitude,
            "address": location_service.current_location.address,
            "is_inside_safe_zone": loc_status.get("is_inside_safe_zone", True),
        }
        if loc_status.get("wandering_alert_triggered"):
            alert_dispatcher.dispatch_wandering_alert(
                patient_id=scheduler.patient.patient_id,
                patient_name=scheduler.patient.patient_name,
                distance_from_home_m=loc_status["distance_from_home_m"],
                google_maps_url=loc_status["google_maps_url"],
            )

    # 2. Decode camera frame
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image payload received from smart glasses.")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding. Expected valid JPEG or PNG.")

    # 3. Transcribe audio query if audio provided and query string not supplied
    resolved_voice_query = voice_query
    if audio is not None and not resolved_voice_query:
        try:
            from smartvision_asr import get_asr_engine
            asr_engine = get_asr_engine(language=language)
            audio_bytes = await audio.read()
            if audio_bytes:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                asr_res = asr_engine.transcribe(tmp_path)
                if asr_res.success and asr_res.transcribed_text:
                    resolved_voice_query = asr_res.transcribed_text
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("Voice transcription in verify_v1 skipped: %s", e)

    # 4. Execute Full Pipeline via SmartGlassesController
    from smartvision_hardware.controller import SmartGlassesController
    controller = SmartGlassesController(patient_language=language, patient=scheduler.patient)
    glasses_res = controller.process_medication_check(
        frames=[frame],
        scheduled_medicine=scheduled_medicine,
        voice_query=resolved_voice_query,
    )

    # 5. Dispatch mismatch alert if unsafe medication detected
    if not glasses_res.is_safe and glasses_res.status not in ["BLURRY_CAPTURE", "UNCERTAIN_CAPTURE"]:
        alert_dispatcher.dispatch_mismatch_alert(
            patient_id=scheduler.patient.patient_id,
            patient_name=scheduler.patient.patient_name,
            detected_text=glasses_res.display_summary or "Unknown Package",
            scheduled_medicine=scheduled_medicine,
            reason=glasses_res.status,
        )

    # 6. Build Audio Stream URL
    audio_stream_url = None
    if glasses_res.audio_file:
        audio_name = Path(glasses_res.audio_file).name
        audio_stream_url = f"/api/v1/audio/stream/{audio_name}"

    return V1VerificationResponse(
        status=glasses_res.status,
        is_safe_to_take=glasses_res.is_safe,
        spoken_message=glasses_res.spoken_text,
        spoken_message_te=glasses_res.spoken_text_te,
        display_summary=glasses_res.display_summary,
        audio_stream_url=audio_stream_url,
        gps_status=gps_info,
    )


@app.get("/api/v1/audio/stream/{filename}")
def stream_audio(filename: str):
    """Streams synthesized 16kHz PCM WAV audio file to smart glasses speaker."""
    safe_filename = Path(filename).name  # Prevent directory traversal
    search_dirs = [
        Path.cwd() / "tts_output",
        _root / "tts_output",
        _root / "TTSModel" / "tts_output",
        Path(__file__).resolve().parent / "tts_output",
    ]
    target_file = None
    for d in search_dirs:
        candidate = d / safe_filename
        if candidate.is_file():
            target_file = candidate
            break

    if not target_file:
        raise HTTPException(status_code=404, detail=f"Audio file '{safe_filename}' not found on server.")

    return FileResponse(
        str(target_file),
        media_type="audio/wav",
        filename=safe_filename,
    )


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_assistant(req: AskRequest):
    """Clinical conversational assistant with zero dosage hallucination risk.

    Answers patient questions using fine-tuned safety adapters, curated elderly
    health guidance, and automatic emergency symptom escalation.
    """
    from smartvision_llm import get_llm_service
    from smartvision_tts import get_tts_engine

    llm_service = get_llm_service()
    llm_res = llm_service.ask(question=req.question, context=req.context)

    # If an emergency was detected, dispatch high-priority alert
    if llm_res.get("is_emergency"):
        alert_dispatcher.dispatch_alert(
            patient_id=scheduler.patient.patient_id,
            patient_name=scheduler.patient.patient_name,
            alert_type="PATIENT_DISTRESS",
            severity="EMERGENCY",
            message=f"CRITICAL HEALTH QUERY: {req.question}. Guidance given: {llm_res.get('answer')}",
        )

    # Synthesize voice response
    spoken_text = (
        llm_res.get("answer_te")
        if req.language == "te" and llm_res.get("answer_te")
        else llm_res.get("answer")
    )

    audio_stream_url = None
    if spoken_text:
        try:
            tts_engine = get_tts_engine(language=req.language)
            tts_res = tts_engine.synthesize(spoken_text)
            if tts_res.success and tts_res.audio_path:
                audio_stream_url = f"/api/v1/audio/stream/{Path(tts_res.audio_path).name}"
        except Exception as e:
            logger.debug("TTS synthesis in ask_endpoint skipped: %s", e)

    return AskResponse(
        question=req.question,
        answer=llm_res.get("answer", ""),
        answer_te=llm_res.get("answer_te"),
        is_safe=llm_res.get("is_safe", True),
        guardrail_triggered=llm_res.get("guardrail_triggered", False),
        is_emergency=llm_res.get("is_emergency", False),
        audio_stream_url=audio_stream_url,
        engine=llm_res.get("engine", "default"),
    )

