/**
 * ============================================================================
 * SmartVision AI — Smart Glasses Embedded Controller Firmware
 * ============================================================================
 * Target Hardware:
 *   - Microcontroller: ESP32-CAM (AI-Thinker) / ESP32-S3 Sense
 *   - Camera Sensor:   OmniVision OV2640 / OV5640
 *   - Audio DAC:       MAX98357A I2S 3.2W Class D Amplifier + Open-Ear Speaker
 *   - User Interface:  Tactile Temple Button (GPIO 13) + Notification LED
 *   - Network:         Wi-Fi 802.11 b/g/n -> SmartVision Backend Gateway
 *
 * Operational Flow:
 *   1. Hardware initialisation (OV2640, I2S DAC, Wi-Fi, temple trigger button).
 *   2. User taps temple button -> capture highest-clarity SVGA JPEG frame.
 *   3. Stream multipart HTTP POST to /api/v1/verify on SmartVision Backend.
 *   4. Parse JSON response -> extract status and audio_stream_url.
 *   5. Stream synthesized 16kHz PCM audio via HTTP GET and stream into I2S DAC.
 * ============================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "driver/i2s.h"

// ----------------------------------------------------------------------------
// Configuration Settings
// ----------------------------------------------------------------------------
const char* WIFI_SSID       = "SmartVision_AP";
const char* WIFI_PASSWORD   = "SmartVision2026";

// Host IP and port of the SmartVision Backend gateway
const char* SERVER_HOST     = "192.168.1.100";
const uint16_t SERVER_PORT  = 8000;

// Wearer preferences
const char* PATIENT_LANG    = "te";         // "te" (Telugu), "hi" (Hindi), "en" (English)
const char* GPS_LAT         = "17.4399";    // Live or GNSS module coordinates
const char* GPS_LNG         = "78.4482";

// Hardware Pin Definitions
#define TRIGGER_BUTTON_PIN  13              // Temple capture button (Active LOW)
#define STATUS_LED_PIN      4               // High-brightness flash/status LED
#define I2S_BCLK_PIN        26              // MAX98357A Bit Clock (BCLK)
#define I2S_LRC_PIN         25              // MAX98357A Word Select (LRC/WS)
#define I2S_DOUT_PIN        22              // MAX98357A Serial Data (DIN)

// Camera Pin Definition for AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Audio Configuration
#define I2S_NUM           I2S_NUM_0
#define AUDIO_SAMPLE_RATE 16000
#define AUDIO_BUF_SIZE    1024

// ----------------------------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------------------------
bool init_camera();
bool init_i2s();
void connect_wifi();
void handle_medication_check();
bool play_audio_stream(const String& stream_url);

// ----------------------------------------------------------------------------
// Setup
// ----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[SmartVision Glasses] Initializing firmware v1.0.0...");

    pinMode(TRIGGER_BUTTON_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    // Initialize Camera
    if (!init_camera()) {
        Serial.println("[ERROR] Camera initialization failed!");
        while (1) { delay(1000); }
    }
    Serial.println("[OK] OV2640 camera ready.");

    // Initialize I2S DAC Audio
    if (!init_i2s()) {
        Serial.println("[ERROR] I2S audio initialization failed!");
    } else {
        Serial.println("[OK] MAX98357A I2S DAC ready.");
    }

    // Connect to Wi-Fi
    connect_wifi();
    Serial.println("[OK] SmartVision Glasses Ready. Tap temple button to check medicine.");
}

// ----------------------------------------------------------------------------
// Main Loop
// ----------------------------------------------------------------------------
void loop() {
    // Detect temple button tap with debounce
    if (digitalRead(TRIGGER_BUTTON_PIN) == LOW) {
        delay(50); // Debounce
        if (digitalRead(TRIGGER_BUTTON_PIN) == LOW) {
            Serial.println("\n[EVENT] Temple button pressed. Initiating medication check...");
            handle_medication_check();
            // Wait until button is released
            while (digitalRead(TRIGGER_BUTTON_PIN) == LOW) {
                delay(20);
            }
        }
    }
    delay(20);
}

// ----------------------------------------------------------------------------
// Camera Initialization
// ----------------------------------------------------------------------------
bool init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Use SVGA (800x600) for sharp OCR text capture
    if (psramFound()) {
        config.frame_size   = FRAMESIZE_SVGA;
        config.jpeg_quality = 10; // 0-63, lower means higher quality
        config.fb_count     = 2;
    } else {
        config.frame_size   = FRAMESIZE_VGA;
        config.jpeg_quality = 12;
        config.fb_count     = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    return (err == ESP_OK);
}

// ----------------------------------------------------------------------------
// I2S DAC Initialization (16-bit 16kHz PCM Mono)
// ----------------------------------------------------------------------------
bool init_i2s() {
    i2s_config_t i2s_config = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate          = AUDIO_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 64,
        .use_apll             = false,
        .tx_desc_auto_clear   = true
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_BCLK_PIN,
        .ws_io_num    = I2S_LRC_PIN,
        .data_out_num = I2S_DOUT_PIN,
        .data_in_num  = I2S_PIN_NO_CHANGE
    };

    if (i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL) != ESP_OK) {
        return false;
    }
    if (i2s_set_pin(I2S_NUM, &pin_config) != ESP_OK) {
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Wi-Fi Connection
// ----------------------------------------------------------------------------
void connect_wifi() {
    Serial.printf("[WIFI] Connecting to %s...", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WIFI] Failed to connect. Check credentials.");
    }
}

// ----------------------------------------------------------------------------
// Capture Image & Send Multipart Request to Backend Gateway
// ----------------------------------------------------------------------------
void handle_medication_check() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[ERROR] Wi-Fi disconnected. Cannot send frame.");
        connect_wifi();
        return;
    }

    // Visual feedback: LED ON during capture
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(100);

    camera_fb_t* fb = esp_camera_fb_get();
    digitalWrite(STATUS_LED_PIN, LOW);

    if (!fb) {
        Serial.println("[ERROR] Failed to capture frame from OV2640.");
        return;
    }
    Serial.printf("[CAMERA] Frame captured. Size: %u bytes.\n", fb->len);

    // Build Multipart HTTP Request
    WiFiClient client;
    if (!client.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.println("[ERROR] Connection to SmartVision server failed.");
        esp_camera_fb_return(fb);
        return;
    }

    String boundary = "----SmartVisionBoundary7MA4YWxkTrZu0gW";
    String head = "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
                  String(PATIENT_LANG) + "\r\n" +
                  "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"latitude\"\r\n\r\n" +
                  String(GPS_LAT) + "\r\n" +
                  "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"longitude\"\r\n\r\n" +
                  String(GPS_LNG) + "\r\n" +
                  "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n" +
                  "Content-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--" + boundary + "--\r\n";

    uint32_t total_len = head.length() + fb->len + tail.length();

    // Send HTTP Headers
    client.println("POST /api/v1/verify HTTP/1.1");
    client.printf("Host: %s:%d\r\n", SERVER_HOST, SERVER_PORT);
    client.println("User-Agent: SmartVisionGlasses/1.0");
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    client.printf("Content-Length: %u\r\n", total_len);
    client.println("Connection: close");
    client.println();

    // Send Body
    client.print(head);
    client.write(fb->buf, fb->len);
    client.print(tail);

    esp_camera_fb_return(fb);
    Serial.println("[HTTP] Frame uploaded. Awaiting verification verdict...");

    // Read HTTP Response
    unsigned long timeout = millis() + 15000;
    while (client.connected() && millis() < timeout && !client.available()) {
        delay(10);
    }

    // Skip HTTP Headers
    while (client.available()) {
        String line = client.readStringUntil('\n');
        if (line == "\r") break;
    }

    // Read Response Body
    String response_body = client.readString();
    client.stop();

    Serial.println("[RESPONSE] " + response_body);

    // Parse JSON
    StaticJsonDocument<1024> doc;
    DeserializationError err = deserializeJson(doc, response_body);
    if (err) {
        Serial.printf("[JSON] Deserialization error: %s\n", err.c_str());
        return;
    }

    const char* status = doc["status"];
    bool is_safe = doc["is_safe_to_take"];
    const char* audio_url = doc["audio_stream_url"];

    Serial.printf("[VERDICT] Status: %s | Safe: %s\n", status, is_safe ? "YES" : "NO");

    // Play Voice Output over Smart Glasses Speaker
    if (audio_url && strlen(audio_url) > 0) {
        Serial.printf("[AUDIO] Streaming speech: %s\n", audio_url);
        play_audio_stream(String(audio_url));
    }
}

// ----------------------------------------------------------------------------
// Stream 16kHz PCM Audio to I2S DAC
// ----------------------------------------------------------------------------
bool play_audio_stream(const String& stream_url) {
    HTTPClient http;
    String full_url = "http://" + String(SERVER_HOST) + ":" + String(SERVER_PORT) + stream_url;
    
    if (!http.begin(full_url)) {
        Serial.println("[AUDIO] HTTP begin failed.");
        return false;
    }

    int httpCode = http.GET();
    if (httpCode != HTTP_CODE_OK) {
        Serial.printf("[AUDIO] HTTP GET failed: %d\n", httpCode);
        http.end();
        return false;
    }

    WiFiClient* stream = http.getStreamPtr();
    
    // Skip 44-byte WAV header
    uint8_t header[44];
    stream->readBytes(header, 44);

    uint8_t audio_buf[AUDIO_BUF_SIZE];
    size_t bytes_written = 0;

    while (http.connected() && stream->available()) {
        size_t bytes_read = stream->readBytes(audio_buf, AUDIO_BUF_SIZE);
        if (bytes_read > 0) {
            i2s_write(I2S_NUM, audio_buf, bytes_read, &bytes_written, portMAX_DELAY);
        }
    }

    // Zero-fill buffer to flush DAC output cleanly
    memset(audio_buf, 0, sizeof(audio_buf));
    i2s_write(I2S_NUM, audio_buf, sizeof(audio_buf), &bytes_written, portMAX_DELAY);

    http.end();
    Serial.println("[AUDIO] Playback completed.");
    return true;
}
