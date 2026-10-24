# SmartVision — Hardware Bridge (HardwareBridge) Module

The **HardwareBridge** subproject connects the physical smart-glasses hardware peripherals (camera, microphone, speaker, tactile buttons) with the modular SmartVision AI and verification software stack.

---

## 1. Physical Hardware Flow

```text
[ Glasses Frame Hardware ]
  ├── Micro-Camera (OV5647 / IMX219)
  ├── I2S MEMS Microphone (INMP441)
  ├── Bone-Conduction / Micro-Speaker (MAX98357A DAC)
  └── Tactile Push Button / Touch Pad
           │
           ▼
[ HardwareBridge Controller ]
  ├── 1. Frame Selection (VideoModel)
  ├── 2. Packaging ROI Crop (VisionModel)
  ├── 3. Text Extraction (OCRModel)
  ├── 4. Deterministic Verification (Backend)
  └── 5. Voice Generation (TTSModel)
           │
           ▼
[ Spoken Voice to Wearer + Visual Display HUD ]
```

---

## 2. Directory Structure

```text
HardwareBridge/
├── src/smartvision_hardware/
│   ├── __init__.py           # Package exports
│   └── controller.py         # Central hardware-to-AI orchestrator
├── scripts/
│   └── simulate_hardware_loop.py # End-to-end hardware loop simulation
├── tests/
│   └── test_hardware_bridge.py   # Unit test suite
├── requirements.txt
└── README.md
```

---

## 3. Quick Start

### Simulation
```bash
python HardwareBridge/scripts/simulate_hardware_loop.py --language te --scheduled Paracetamol
```
This triggers a simulated smart-glasses button press, records a 5-frame camera burst, extracts the keyframe, localizes the blister package, performs OCR, verifies against the active prescription, and speaks the verdict in Telugu!
