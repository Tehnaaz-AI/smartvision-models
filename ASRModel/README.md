# SmartVision — Automatic Speech Recognition (ASR) Module

The **ASRModel** subproject handles voice-command ingestion and intent classification for the SmartVision assistive smart-glasses system. It allows elderly wearers to interact hands-free using voice in Telugu, Hindi, or English.

---

## 1. Capabilities & Workflow

```text
[ Glasses Microphone (16kHz PCM Audio) ]
                  │
                  ▼
         [ ASREngine ]
   (Audio Validation & Speech-to-Text)
                  │
                  ▼
       [ IntentClassifier ]
 (Telugu / Hindi / English Intent Spotting)
                  │
                  ▼
[ Structured Intent (IDENTIFY_MEDICINE, CHECK_DOSAGE, etc.) ]
                  │
                  ▼
     [ Sent to Backend & LLM ]
```

---

## 2. Directory Structure

```text
ASRModel/
├── src/smartvision_asr/
│   ├── __init__.py           # Package exports
│   ├── engine.py             # ASR engine interface & pipeline
│   ├── intents.py            # Trilingual regex & keyword intent parser
│   └── audio_utils.py        # 16kHz mono audio validation & preprocessing
├── scripts/
│   └── listen_and_parse.py   # CLI demo for intent extraction
├── tests/
│   └── test_asr_engine.py    # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd ASRModel
pip install -r requirements.txt
```

### CLI Demo
```bash
python scripts/listen_and_parse.py
# Or with a custom query:
python scripts/listen_and_parse.py --query "ఈ మందు పేరు ఏమిటి?"
```

### Python API
```python
from smartvision_asr import get_asr_engine, UserIntent

engine = get_asr_engine()
result = engine.process_text_command("When should I take this tablet?")
print(f"Intent: {result.intent} (Language: {result.language})")
```
