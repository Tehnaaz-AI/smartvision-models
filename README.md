# SmartVision

SmartVision is a modular AI project for smart glasses, developed with support
from the college incubation centre. Each model will live in its own subproject
so that its code, dependencies, datasets, and training workflow can be managed
independently.

## Model Subprojects

- **[OCRModel](OCRModel/README.md)** — Multilingual printed-text recognition for assistive smart glasses, targeting medicine packaging, instructions, and everyday signage in Telugu, Hindi, and English. Features lightweight PP-OCRv5 Mobile engines, CLAHE/unsharp image preprocessing, and a pharmaceutical fuzzy-matching postprocessor.
- **[LLM](LLM/README.md)** — Conversational reasoning module based on `Qwen/Qwen3-4B-Instruct-2507`. Acts as an empathetic, safety-grounded voice assistant for elderly users, synthesizing recognized medicine details into clear, spoken explanations without fabricating dosages.
- **TTSModel** — *(Planned)* Text-to-speech synthesis module for voice delivery.
- **ASRModel** — *(Planned)* Automatic speech recognition for voice commands.
- **VisionModel** — *(Planned)* Scene understanding and object detection.

## System Architecture

```text
[ Smart Glasses Camera ]
           │
           ▼
[ OCRModel Pipeline ]
  ├── Image Preprocessing (CLAHE, Bilateral Filter, Unsharp Mask)
  ├── Text Detection (PP-OCRv5 Mobile Det)
  ├── Multilingual Text Recognition (Telugu, Hindi, English PP-OCRv5)
  └── Pharmaceutical Lexicon & Fuzzy Matcher (Dosage & Medicine Validation)
           │
           ▼ (Extracted Text & Domain Insights)
[ LLM Reasoning (Qwen3-4B) ]
  ├── Safety Screening & Elderly-Friendly Explanation
  └── Grounded Medical Disclaimer
           │
           ▼
[ Audio Feedback / TTS to User ]
```

## Repository Structure

```text
smartvision-models/
├── OCRModel/               # Multilingual OCR pipeline, datasets, demo, tests
│   ├── src/smartvision_ocr/ # Core library (preprocessing, pipeline, postprocess)
│   ├── data/               # Evaluation splits, synthetic domain, benchmarks
│   ├── scripts/            # Demo launcher, dataset generation, evaluators
│   ├── tests/              # 125-test comprehensive verification suite
│   └── README.md           # OCRModel setup and API reference
├── LLM/                    # Conversational assistant reasoning module
│   ├── run_model.py        # Interactive assistant shell with safety system prompt
│   ├── test_accuracy.py    # 8-scenario elderly safety evaluation suite
│   └── README.md           # LLM configuration and usage guide
├── README.md               # Monorepo architecture overview
└── .gitignore              # Unified ignore rules across subprojects
```

## Quick Start

### 1. Running the Multilingual OCR Interactive Demo
```bash
cd OCRModel
.venv\Scripts\activate
python scripts\launch_demo.py --port 5000
```
Open `http://localhost:5000` to interactively detect and recognize Telugu, Hindi, and English text with real-time preprocessing and pharmaceutical fuzzy post-processing.

### 2. Running the Conversational Assistant (LLM)
```bash
cd LLM
python run_model.py
```
Type queries or provide OCR-extracted text to receive elderly-friendly, safety-constrained responses.

