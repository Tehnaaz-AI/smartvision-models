# SmartVision — Assistive AI Smart-Glasses System

SmartVision is a modular, assistive AI system for smart glasses designed to empower elderly individuals and visually impaired users with medication identification, deterministic prescription verification, timely schedule reminders, and multilingual voice interaction.

Developed with support from the college incubation centre, SmartVision is organized as a specialized modular architecture where each component is isolated into its own subproject with independent dependencies, workflows, and test suites.

---

## 1. Modular Subproject Architecture

| Subproject | Category | Role & Core Functionality | Languages Supported |
| :--- | :--- | :--- | :--- |
| **[OCRModel](OCRModel/README.md)** | Vision / OCR | Multilingual text detection and recognition on medicine labels, packaging, and signage. Integrates CLAHE/unsharp image preprocessing, PP-OCRv5 Mobile engines, and a pharmaceutical fuzzy-matching postprocessor. | Telugu (`te`), Hindi (`hi`), English (`en`) |
| **[LLM](LLM/README.md)** | Generative AI | Conversational reasoning assistant based on `Qwen/Qwen3-4B-Instruct-2507`. Provides compassionate, elderly-friendly explanations with strict safety grounding against dosage hallucination. | English (`en`), Telugu (`te`), Hindi (`hi`) |
| **[TTSModel](TTSModel/README.md)** | Audio / Voice | Text-to-speech voice synthesizer delivering natural spoken instructions through smart-glasses bone conduction or micro-speakers at an elderly-friendly pace (140 WPM). | Telugu (`te`), Hindi (`hi`), English (`en`) |
| **[ASRModel](ASRModel/README.md)** | Speech / Audio | Automatic speech recognition and trilingual voice-command intent spotter (`IDENTIFY_MEDICINE`, `CHECK_DOSAGE`, `CHECK_SCHEDULE`, `EMERGENCY_HELP`). | Telugu (`te`), Hindi (`hi`), English (`en`) |
| **[VisionModel](VisionModel/README.md)** | Computer Vision | Medicine package detection, form-factor classification (`TABLET_STRIP`, `SYRUP_BOTTLE`, `CARTON_BOX`, `OINTMENT_TUBE`), and region-of-interest cropping. | Agnostic |
| **[VideoModel](VideoModel/README.md)** | Video Processing | Real-time camera stream analysis, Laplacian motion blur filtering, glare rejection, and optimal keyframe selection from frame bursts. | Agnostic |
| **[TranslationModel](TranslationModel/README.md)** | NLP / MT | Domain-grounded machine translation preserving pharmaceutical trade/generic names while translating dosage instructions and warnings. | Telugu (`te`) ⟷ Hindi (`hi`) ⟷ English (`en`) |
| **[Backend](Backend/README.md)** | Deterministic Core | Deterministic medication verification cross-checking OCR output against doctor's prescriptions, schedule reminder scheduler, and FastAPI REST service. | Multi-language rules |
| **[HardwareBridge](HardwareBridge/README.md)** | Embedded / Edge | Hardware controller connecting physical glasses sensors (camera, mic, speaker, tactile buttons) with the AI processing pipeline. | System Orchestration |

---

## 2. End-to-End Smart Glasses Workflow

```text
               ┌──────────────────────────────────────────────────────────────┐
               │                SMART GLASSES HARDWARE FRAME                  │
               │  Camera (Burst)  │  MEMS Microphone  │  Speaker  │  Button   │
               └──────────┬───────────────────┬──────────────▲────────────────┘
                          │                   │              │
                    Frames Burst         Voice Audio    Spoken Audio
                          │                   │              │
                          ▼                   ▼              │
                  ┌───────────────┐   ┌───────────────┐      │
                  │  VideoModel   │   │   ASRModel    │      │
                  │ Blur Filtering│   │Intent Spotting│      │
                  └───────┬───────┘   └───────┬───────┘      │
                     Best Frame               │              │
                          │                   ▼              │
                          ▼           User Intent / Query    │
                  ┌───────────────┐           │              │
                  │  VisionModel  │           │              │
                  │ Packaging ROI │           │              │
                  └───────┬───────┘           │              │
                    Isolated Crop             │              │
                          │                   │              │
                          ▼                   │              │
                  ┌───────────────┐           │              │
                  │   OCRModel    │           │              │
                  │Multilingual te│           │              │
                  └───────┬───────┘           │              │
                 Extracted Drug Details       │              │
                          │                   │              │
                          ▼                   ▼              │
                  ┌───────────────────────────────────┐      │
                  │              Backend              │      │
                  │   Deterministic Prescription      │      │
                  │   Verification & Safety Rules     │      │
                  └─────────────────┬─────────────────┘      │
                                    │                        │
                      Verified Verdict / Alert               │
                                    │                        │
                     ┌──────────────┴──────────────┐         │
                     │                             │         │
                     ▼                             ▼         │
            ┌─────────────────┐           ┌─────────────────┐│
            │       LLM       │           │TranslationModel ││
            │Elderly Reasoning│           │Medical Phrasing ││
            └────────┬────────┘           └────────┬────────┘│
                     │                             │         │
                     └──────────────┬──────────────┘         │
                                    │                        │
                               Text Script                   │
                                    │                        │
                                    ▼                        │
                            ┌───────────────┐                │
                            │   TTSModel    │────────────────┘
                            │Voice Synthesis│
                            └───────────────┘
```

---

## 3. Repository Directory Structure

```text
smartvision-models/
├── OCRModel/               # Multilingual OCR pipeline (PP-OCRv5 Mobile, Preprocessor, Fuzzy Formulary)
├── LLM/                    # Conversational reasoning module (Qwen3-4B with elderly safety prompting)
├── TTSModel/               # Text-to-speech voice synthesis in Telugu, Hindi, and English
├── ASRModel/               # Speech recognition and trilingual voice-command intent classifier
├── VisionModel/            # Packaging detection and form-factor classification
├── VideoModel/             # Motion blur filtering, glare detection, and optimal keyframe selection
├── TranslationModel/       # Trilingual medical instruction translation with brand-name preservation
├── Backend/                # Deterministic prescription verifier, scheduler, and FastAPI service
├── HardwareBridge/         # Smart-glasses peripheral controller and end-to-end hardware loop
├── README.md               # Master monorepo architecture overview
├── .gitignore              # Unified ignore patterns for binaries, datasets, and virtualenvs
└── SmartVision.docx        # Project proposal and background documentation
```

---

## 4. Running Demonstrations & Tests

Each subproject contains its own test suite and CLI demonstration scripts:

### 1. Test All Subprojects
```bash
# OCRModel (125 tests)
python -m pytest OCRModel/tests -q

# TTSModel (7 tests)
python -m pytest TTSModel/tests -q

# ASRModel (7 tests)
python -m pytest ASRModel/tests -q

# VisionModel (4 tests)
python -m pytest VisionModel/tests -q

# VideoModel (4 tests)
python -m pytest VideoModel/tests -q

# TranslationModel (7 tests)
python -m pytest TranslationModel/tests -q

# Backend (7 tests)
python -m pytest Backend/tests -q

# HardwareBridge (2 tests)
python -m pytest HardwareBridge/tests -q
```

### 2. Live Hardware Loop Simulation
```bash
python HardwareBridge/scripts/simulate_hardware_loop.py --language te --scheduled Paracetamol
```

### 3. Interactive OCR Web Demo
```bash
python OCRModel/scripts/launch_demo.py --port 5000
```
Open `http://localhost:5000` to test multilingual detection and recognition in your browser.
