"""Emergency Alert Dispatcher Service for SmartVision.

Dispatches instant alerts to family caregivers and healthcare providers when
critical medication mismatches, overdose risks, or user distress events occur.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("smartvision.alerts")


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType(str, Enum):
    MEDICATION_MISMATCH = "MEDICATION_MISMATCH"
    WRONG_DOSAGE = "WRONG_DOSAGE"
    UNSCHEDULED_TIME = "UNSCHEDULED_TIME"
    PATIENT_DISTRESS = "PATIENT_DISTRESS"
    LOW_BATTERY = "LOW_BATTERY"


@dataclass
class CaregiverContact:
    name: str
    phone: str
    email: Optional[str] = None
    relation: str = "Family Member"
    webhook_url: Optional[str] = None


@dataclass
class EmergencyAlert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    patient_id: str
    patient_name: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    medicine_name: Optional[str] = None
    detected_ocr_text: Optional[str] = None
    status: str = "PENDING"  # PENDING, DELIVERED, FAILED
    delivery_channels: List[str] = field(default_factory=list)


class AlertDispatcher:
    """Dispatches emergency alerts to registered caregivers with offline queuing."""

    def __init__(self, contacts: Optional[List[CaregiverContact]] = None):
        self.contacts: List[CaregiverContact] = contacts or [
            CaregiverContact(
                name="Primary Caregiver",
                phone="+91-9876543210",
                relation="Son/Daughter",
            )
        ]
        self.alert_history: List[EmergencyAlert] = []
        self._offline_queue: List[EmergencyAlert] = []

    def register_contact(self, contact: CaregiverContact) -> None:
        """Adds a caregiver contact."""
        self.contacts.append(contact)
        logger.info("Registered caregiver contact: %s (%s)", contact.name, contact.phone)

    def dispatch_mismatch_alert(
        self,
        patient_id: str,
        patient_name: str,
        detected_text: str,
        scheduled_medicine: Optional[str] = None,
        reason: str = "Unprescribed or mismatched medication detected",
    ) -> EmergencyAlert:
        """Dispatches an alert when patient is about to take the wrong medicine."""
        alert = EmergencyAlert(
            alert_id=f"alt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(self.alert_history)+1}",
            alert_type=AlertType.MEDICATION_MISMATCH,
            severity=AlertSeverity.CRITICAL,
            patient_id=patient_id,
            patient_name=patient_name,
            message=(
                f"ALERT for {patient_name}: Potentially unsafe medication scanned: '{detected_text}'. "
                f"Expected: '{scheduled_medicine or 'None'}'. Reason: {reason}."
            ),
            detected_ocr_text=detected_text,
            medicine_name=scheduled_medicine,
        )
        return self._send_alert(alert)

    def dispatch_distress_alert(
        self,
        patient_id: str,
        patient_name: str,
        transcript: str,
    ) -> EmergencyAlert:
        """Dispatches an emergency alert when the patient speaks a distress phrase."""
        alert = EmergencyAlert(
            alert_id=f"alt_distress_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            alert_type=AlertType.PATIENT_DISTRESS,
            severity=AlertSeverity.EMERGENCY,
            patient_id=patient_id,
            patient_name=patient_name,
            message=f"EMERGENCY: {patient_name} requested assistance via smart glasses. Heard: '{transcript}'.",
        )
        return self._send_alert(alert)

    def _send_alert(self, alert: EmergencyAlert) -> EmergencyAlert:
        """Sends the alert across all configured contact channels."""
        delivered_channels: List[str] = []

        for contact in self.contacts:
            # 1. SMS / WhatsApp notification simulation
            delivered_channels.append(f"SMS({contact.phone})")
            logger.warning(
                "[EMERGENCY ALERT TO %s (%s)] %s: %s",
                contact.name,
                contact.phone,
                alert.severity.value,
                alert.message,
            )

            # 2. Webhook notification (if configured)
            if contact.webhook_url:
                try:
                    payload = json.dumps({
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "type": alert.alert_type.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        contact.webhook_url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        if resp.status < 300:
                            delivered_channels.append(f"Webhook({contact.webhook_url})")
                except Exception as err:
                    logger.warning("Webhook dispatch failed: %s", err)
                    self._offline_queue.append(alert)

        alert.delivery_channels = delivered_channels
        alert.status = "DELIVERED"
        self.alert_history.append(alert)
        return alert

    def get_pending_alerts(self) -> List[EmergencyAlert]:
        """Returns pending or undelivered alerts in queue."""
        return list(self._offline_queue)

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent alert records for caregiver dashboard."""
        return [
            {
                "alert_id": a.alert_id,
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "patient_name": a.patient_name,
                "message": a.message,
                "timestamp": a.timestamp,
                "status": a.status,
                "channels": a.delivery_channels,
            }
            for a in reversed(self.alert_history[-limit:])
        ]


_dispatcher_instance: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = AlertDispatcher()
    return _dispatcher_instance
