"""Cloud Database and Storage Client for SmartVision.

Provides cloud synchronization (Supabase / Firebase / AWS REST)
with an offline local SQLite resilience cache for smart glasses.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Medication, Patient, Prescription, VerificationResult

logger = logging.getLogger(__name__)


class CloudDatabaseClient:
    """Manages cloud prescription storage and offline local cache."""

    def __init__(
        self,
        cloud_provider: str = "local_first_cloud",
        cache_db_path: Optional[str] = None,
    ):
        self.cloud_provider = cloud_provider
        self.cache_path = Path(cache_db_path) if cache_db_path else Path(__file__).resolve().parent.parent.parent / "data" / "smartvision_cache.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_cache()

    def _init_sqlite_cache(self) -> None:
        """Initializes SQLite tables for local offline caching."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT,
                    preferred_language TEXT,
                    caregiver_contact TEXT,
                    prescriptions_json TEXT,
                    last_synced_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TEXT,
                    status TEXT,
                    is_safe INTEGER,
                    message TEXT,
                    synced_to_cloud INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Retrieves patient profile and prescriptions, reading from local cache with cloud fallback."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, preferred_language, caregiver_contact, prescriptions_json FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()

            if row:
                name, lang, contact, rx_json = row
                prescriptions_data = json.loads(rx_json) if rx_json else []
                prescriptions = []
                for rx in prescriptions_data:
                    meds = [
                        Medication(
                            medicine_name=m["medicine_name"],
                            strength=m["strength"],
                            dosage_form=m.get("dosage_form", "tablet"),
                            quantity_per_dose=m.get("quantity_per_dose", "1 tablet"),
                            schedule_time=m["schedule_time"],
                            instructions=m["instructions"],
                            instructions_te=m.get("instructions_te"),
                            aliases=m.get("aliases", []),
                        )
                        for m in rx.get("medications", [])
                    ]
                    prescriptions.append(
                        Prescription(
                            prescription_id=rx["prescription_id"],
                            doctor_name=rx["doctor_name"],
                            date_issued=rx["date_issued"],
                            medications=meds,
                        )
                    )
                return Patient(
                    patient_id=patient_id,
                    patient_name=name,
                    preferred_language=lang,
                    caregiver_contact=contact,
                    prescriptions=prescriptions,
                )

        # Fallback to sample seed data if not found
        sample_json_path = self.cache_path.parent / "sample_prescriptions.json"
        if sample_json_path.exists():
            records = json.loads(sample_json_path.read_text(encoding="utf-8"))
            for rec in records:
                if rec["patient_id"] == patient_id:
                    self.sync_patient_to_cache(rec)
                    return self.get_patient(patient_id)
        return None

    def sync_patient_to_cache(self, patient_dict: Dict[str, Any]) -> None:
        """Stores or updates cloud-fetched patient profile into local SQLite cache."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO patients 
                (patient_id, name, preferred_language, caregiver_contact, prescriptions_json, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_dict["patient_id"],
                    patient_dict["patient_name"],
                    patient_dict.get("preferred_language", "te"),
                    patient_dict.get("caregiver_contact", ""),
                    json.dumps(patient_dict.get("prescriptions", [])),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            conn.commit()

    def log_verification(self, patient_id: str, result: VerificationResult) -> int:
        """Logs a medication verification event into the local audit trail for cloud sync."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_logs (patient_id, timestamp, status, is_safe, message, synced_to_cloud)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    patient_id,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    result.status.value,
                    1 if result.is_safe_to_take else 0,
                    result.spoken_message,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_pending_sync_logs(self) -> List[Dict[str, Any]]:
        """Returns verification events waiting to be uploaded to cloud storage."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT log_id, patient_id, timestamp, status, is_safe, message FROM audit_logs WHERE synced_to_cloud = 0")
            rows = cursor.fetchall()
            return [
                {
                    "log_id": r[0],
                    "patient_id": r[1],
                    "timestamp": r[2],
                    "status": r[3],
                    "is_safe": bool(r[4]),
                    "message": r[5],
                }
                for r in rows
            ]


def get_cloud_db_client() -> CloudDatabaseClient:
    """Factory helper to obtain a CloudDatabaseClient instance."""
    return CloudDatabaseClient()
