# SmartVision — Backend & Medication Verification Service

The **Backend** subproject provides the core deterministic verification, prescription matching, schedule reminder engine, and REST API service for the SmartVision assistive smart glasses.

---

## 1. Why Deterministic Logic?

In assistive healthcare for elderly users, **safety is paramount**. Generative AI models (LLMs) can hallucinate doses or fail under pressure. 
SmartVision enforces a strict architectural boundary:
- **Computer Vision & OCR**: Detects and extracts text from medicine labels.
- **Backend Verifier (Deterministic Software)**: Compares the extracted medicine name, alias, and strength against the patient's verified doctor's prescription.
- **Rule Verification**: If a drug or strength does not match, the system issues a hard alert (`ABORT` / `CALL_CAREGIVER`) and forbids the patient from proceeding.

---

## 2. Directory Structure

```text
Backend/
├── src/smartvision_backend/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data schemas (Medication, Prescription, Patient, VerificationResult)
│   ├── verifier.py           # Deterministic prescription cross-checker
│   ├── scheduler.py          # Daily medication reminder engine
│   └── api.py                # FastAPI REST endpoints (/api/verify, /api/reminders)
├── data/
│   └── sample_prescriptions.json # Sample patient prescription records
├── scripts/
│   ├── run_server.py         # Launch FastAPI server on port 8000
│   └── test_verification_cli.py # CLI verification demonstrator
├── tests/
│   └── test_verifier.py      # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd Backend
pip install -r requirements.txt
```

### CLI Verification Demo
```bash
# Correct drug matching scheduled dose
python scripts/test_verification_cli.py -t "Dolo 500" -s "500 mg" --scheduled Paracetamol

# Wrong medication held (Metformin instead of Paracetamol)
python scripts/test_verification_cli.py -t "Metformin 500 mg" -s "500 mg" --scheduled Paracetamol

# Strength mismatch (Holding 650 mg instead of prescribed 500 mg)
python scripts/test_verification_cli.py -t "Paracetamol" -s "650 mg" --scheduled Paracetamol
```

### Starting the REST API Service
```bash
python scripts/run_server.py
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.
