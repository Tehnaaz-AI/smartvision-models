# SmartVision — Smart Glasses Hardware & Firmware Integration Guide

This guide details the physical hardware wiring, board configurations, and flashing steps to assemble an assistive smart-glasses prototype that connects to the SmartVision AI Gateway.

---

## 1. Supported Hardware Platforms

| Component | Recommended Model | Alternative | Function |
|---|---|---|---|
| **Microcontroller** | AI-Thinker ESP32-CAM | Seeed XIAO ESP32S3 Sense | Edge acquisition, Wi-Fi stream, I2S DAC output |
| **Camera Sensor** | OmniVision OV2640 (2MP) | OmniVision OV5640 (5MP) | High-resolution macro image capture of medicine packs |
| **Audio DAC / Amp** | Adafruit MAX98357A | UDA1334A I2S DAC | 16-bit 16kHz PCM audio decoding to speaker |
| **Speaker** | 1W 8Ω Bone Conduction / Open-Ear | 0.5W Micro Dynamic Speaker | Non-intrusive voice guidance for elderly wearer |
| **Capture Button** | 6mm Tactile Push Button | TTP223 Capacitive Touch | Temple-mounted check trigger |
| **Status Indicator** | 3mm Soft White / Green LED | Integrated Flash LED (GPIO 4) | User feedback for capture, processing, and alerts |
| **Battery & Power** | 3.7V 500mAh LiPo + TP4056 + AMS1117-3.3 | 5V Powerbank via Micro-USB/USB-C | Glasses frame power delivery |

---

## 2. Wiring & Pinout Diagram

### ESP32-CAM (AI-Thinker) Pin Mapping

```text
       +---------------------------------------------+
       |             AI-Thinker ESP32-CAM            |
       |                                             |
       |  GPIO 13 (INPUT_PULLUP) <--- [ Push Button ] ---> GND
       |                                             |
       |  GPIO 4  (OUTPUT)       ---> [ Flash/Status LED ]
       |                                             |
       |  GPIO 26 (I2S BCLK)     ---> MAX98357A BCLK |
       |  GPIO 25 (I2S LRC/WS)   ---> MAX98357A LRC  |
       |  GPIO 22 (I2S DOUT/DIN) ---> MAX98357A DIN  |
       |  GND                    ---> MAX98357A GND  |
       |  5V / VCC               ---> MAX98357A VIN  |
       |                                             |
       |  MAX98357A SPK+ / SPK-  ---> [ Open-Ear Speaker ]
       +---------------------------------------------+
```

> [!NOTE]
> On the standard AI-Thinker ESP32-CAM, GPIO 0 is used for flashing mode (ground during boot to enter flashing mode).
> In normal operation, keep GPIO 0 disconnected.

---

## 3. Firmware Configuration

Open [`esp32_smart_glasses.ino`](file:///D:/Projects/SmartVision/HardwareBridge/firmware/esp32_smart_glasses/esp32_smart_glasses.ino) in Arduino IDE or PlatformIO.

Update your network settings:
```cpp
const char* WIFI_SSID       = "Your_WiFi_Network";
const char* WIFI_PASSWORD   = "Your_WiFi_Password";
const char* SERVER_HOST     = "192.168.1.100";  // IP address of SmartVision backend host
const uint16_t SERVER_PORT  = 8000;
const char* PATIENT_LANG    = "te";             // "te" (Telugu), "hi" (Hindi), "en" (English)
```

---

## 4. Software Dependencies (Arduino Library Manager)

Install the following libraries in Arduino IDE (`Sketch -> Include Library -> Manage Libraries`):
1. **ArduinoJson** by Benoît Blanchon (Version 6.21.x or 7.x)
2. **esp32-camera** (Built into the official ESP32 Arduino Core by Espressif)

---

## 5. Board Settings in Arduino IDE

- **Board**: `AI Thinker ESP32-CAM`
- **CPU Frequency**: `240MHz (WiFi/BT)`
- **Flash Frequency**: `80MHz`
- **Flash Mode**: `QIO`
- **Partition Scheme**: `Huge APP (3MB No OTA/1MB SPIFFS)`
- **PSRAM**: `Enabled` (Crucial for SVGA image capture and frame buffers)

---

## 6. End-to-End Operational Lifecycle

1. **Boot**: The glasses connect to the Wi-Fi network and initialize the OV2640 camera in SVGA mode (800x600) and the I2S DAC in 16kHz 16-bit mono mode.
2. **Trigger**: When the wearer points their glasses at a medication package and taps the temple button (GPIO 13):
   - The status LED blinks once to confirm capture.
   - The captured JPEG image is streamed over HTTP multipart to `POST /api/v1/verify`.
3. **AI Pipeline**: The SmartVision backend runs packaging isolation, multilingual OCR, deterministic prescription verification, and synthesizes 16kHz WAV speech in the wearer's language.
4. **Playback**: The glasses stream the WAV audio from `GET /api/v1/audio/stream/{filename}` directly into the MAX98357A DAC, clearly speaking instructions (e.g. in Telugu: *"ఇది మీరు వేసుకోవలసిన పారాసిటమాల్ మందు. సురక్షితంగా వేసుకోవచ్చు."*).
