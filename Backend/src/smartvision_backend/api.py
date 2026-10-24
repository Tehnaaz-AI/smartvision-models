"""FastAPI REST Service for SmartVision Smart Glasses Backend."""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import Medication, Prescription, VerificationStatus
from .verifier import get_verifier
from .scheduler import get_scheduler
from .alerts import get_alert_dispatcher
from .cloud_db import get_cloud_db_client
from .location import get_location_service, SafeHomeZone

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

