"""Unit Tests for SmartVision Emergency Alert Dispatcher."""

import pytest
from smartvision_backend.alerts import (
    AlertDispatcher,
    AlertSeverity,
    AlertType,
    CaregiverContact,
)


@pytest.fixture
def alert_dispatcher():
    contact = CaregiverContact(
        name="Test Daughter",
        phone="+91-9999988888",
        relation="Daughter",
    )
    return AlertDispatcher(contacts=[contact])


def test_dispatch_mismatch_alert(alert_dispatcher):
    alert = alert_dispatcher.dispatch_mismatch_alert(
        patient_id="P-101",
        patient_name="Lakshmi Devi",
        detected_text="Amlodipine 5mg",
        scheduled_medicine="Dolo 650",
        reason="Not in prescription",
    )
    assert alert.alert_type == AlertType.MEDICATION_MISMATCH
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.status == "DELIVERED"
    assert "SMS(+91-9999988888)" in alert.delivery_channels
    assert "Lakshmi Devi" in alert.message
    assert len(alert_dispatcher.alert_history) == 1


def test_dispatch_distress_alert(alert_dispatcher):
    alert = alert_dispatcher.dispatch_distress_alert(
        patient_id="P-101",
        patient_name="Lakshmi Devi",
        transcript="Help I feel dizzy please call doctor",
    )
    assert alert.alert_type == AlertType.PATIENT_DISTRESS
    assert alert.severity == AlertSeverity.EMERGENCY
    assert "dizzy" in alert.message


def test_get_recent_alerts(alert_dispatcher):
    alert_dispatcher.dispatch_mismatch_alert(
        patient_id="P-1", patient_name="Pat", detected_text="MedA"
    )
    alert_dispatcher.dispatch_distress_alert(
        patient_id="P-1", patient_name="Pat", transcript="Help"
    )
    recent = alert_dispatcher.get_recent_alerts(limit=5)
    assert len(recent) == 2
    assert recent[0]["type"] == AlertType.PATIENT_DISTRESS.value
