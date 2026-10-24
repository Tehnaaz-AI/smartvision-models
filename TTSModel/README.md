# SmartVision — Text-to-Speech (TTS) Module

The **TTSModel** subproject provides voice synthesis for the SmartVision assistive smart-glasses system. It converts structured text, medication instructions, and LLM reasoning into spoken audio delivered through the smart-glasses speaker or bone-conduction headset.

---

## 1. Objectives & Capabilities

- **Trilingual Speech Delivery**: Supports Telugu (`te`), Hindi (`hi`), and English (`en`).
- **Elderly-Friendly Speech Rate**: Default 140 words per minute (WPM), delivering measured and clearly enunciated speech.
- **Offline / Low-Latency Deployment**: Generates lightweight WAV audio suitable for playback on wearable client devices (e.g. Raspberry Pi Zero 2W / ESP32 with audio DAC).
- **Graceful Fallback**: Native platform synthesis (SAPI5 on Windows, NSSpeech on macOS, espeak on Linux) with synthetic audio generation fallback.

---

## 2. Directory Structure

```text
TTSModel/
├── src/smartvision_tts/
│   ├── __init__.py           # Package exports (TTSEngine, AudioPlayer)
│   ├── engine.py             # Unified synthesis engine
│   └── audio_player.py       # Cross-platform audio playback
├── scripts/
│   └── synthesize_reminder.py# CLI tool for generating medication reminders
├── tests/
│   └── test_tts_engine.py    # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Installation
```bash
cd TTSModel
pip install -r requirements.txt
```

### CLI Synthesis
```bash
# English reminder
python scripts/synthesize_reminder.py -l en

# Telugu medication reminder
python scripts/synthesize_reminder.py -l te

# Hindi medication reminder
python scripts/synthesize_reminder.py -l hi
```

### Python API Usage
```python
from smartvision_tts import get_tts_engine

engine = get_tts_engine(language="te", speech_rate=135)
result = engine.synthesize("సమయానికి మందులు వేసుకోండి.")

if result.success:
    print(f"Audio ready: {result.audio_path} ({result.duration_seconds}s)")
```

---

## 4. Hardware Integration on Smart Glasses

1. **Audio Output Interface**: Connect bone-conduction transducers or a 3.5mm micro-speaker to the glasses frame controller (e.g. I2S MAX98357A DAC on Raspberry Pi).
2. **Buffering**: The module exports 16-bit 16kHz mono WAV streams to ensure minimal memory overhead on embedded storage.
