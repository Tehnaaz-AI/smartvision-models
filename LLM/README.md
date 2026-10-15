# AI Smart Glasses for Elderly People — LLM Testing

An AI-powered assistant concept designed to support elderly people through natural voice conversations, visual assistance, reminders, and emergency support.

This repository contains the initial **LLM experimentation and evaluation module** using [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from Hugging Face. It provides a foundation for integrating conversational AI into wearable smart glasses.

> **Current stage:** Text-based LLM prototype. Hardware integration, microphone input, speaker output, camera-based vision, and automated reminders are planned extensions.

## Features

- **Conversational AI:** Answers everyday questions using natural language.
- **Elderly-friendly responses:** Uses simple, clear explanations.
- **Safety-aware prompting:** Instructs the model not to invent medication dosages and to acknowledge uncertainty.
- **Hugging Face integration:** Loads a pretrained Qwen3 model using Transformers.
- **Interactive terminal chat:** Test the assistant with custom questions.
- **Basic evaluation suite:** Tests predefined scenarios and calculates a keyword-based screening score.
- **Extensible architecture:** Provides a starting point for future voice, vision, and smart-glasses integration.

## AI Model

| Property | Details |
|---|---|
| Model | Qwen3-4B-Instruct-2507 |
| Provider | Qwen |
| Platform | Hugging Face |
| Model type | Instruction-tuned language model |
| Framework | PyTorch and Transformers |
| Input | Text prompts |
| Output | Generated text responses |
| Deployment | Local machine or compatible server |

Model repository: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

## Project Architecture

```text
                 User Question
                       |
                       v
               Text Input Interface
                       |
                       v
                Qwen3-4B Model
                       |
                       v
               Generated Response
                       |
             +---------+---------+
             |                   |
             v                   v
        User Display       Evaluation Tests
```

### Planned Smart Glasses Architecture

```text
Microphone ----> Speech Recognition
                         |
                         v
Camera --------> Vision Model
                         |
                         v
                  AI Assistant
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          Speaker     OLED/HUD    Reminders
                                    |
                                    v
                              Family Support
```

The planned architecture is not yet implemented in this repository.

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Hub
- Accelerate
- Safetensors

## Requirements

- Python 3.10 or a compatible version supported by the installed PyTorch and Transformers releases
- Internet connection for the initial model download
- Several GB of free storage for model files and dependencies
- Sufficient system RAM or GPU memory for the selected model precision

A GPU is recommended for faster inference. Actual performance depends on hardware, model precision, and available memory.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TBPA1896/LLM-Baseline.git
cd LLM-Baseline
```

### 2. Create a virtual environment

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install torch transformers accelerate safetensors
```

For NVIDIA GPU acceleration, install the appropriate PyTorch build for your CUDA and driver configuration. Refer to the official guide: https://pytorch.org/get-started/locally/

## Running the Model

Start the interactive assistant:

```bash
python run_model.py
```

On the first run, Hugging Face downloads the model files. This may take time depending on your internet speed.

### Example Interaction

```text
You: Explain how to use a smartphone to an elderly person.

AI: A smartphone is a mobile phone with a touchscreen.
You can tap an app to open it, use the phone app to call
someone, and adjust the text size in the settings.
```

Example responses are illustrative; actual outputs depend on the model.

Enter `exit` or `quit` to stop the program.

## Testing and Evaluation

Run the predefined test suite:

```bash
python test_accuracy.py
```

The tests cover scenarios such as:

- Simple explanations for elderly users
- Forgotten medication
- Chest pain and breathing difficulties
- Unknown medication dosage
- Reminder requests
- Requests to identify a person without a camera input
- Basic everyday assistance
- Questions requiring real-time information

### Understanding the Results

The evaluation script checks whether generated responses contain selected keywords.

For example:

```text
Keyword checks passed: 6/8
Screening score: 75.0%
```

The numbers above are illustrative, not measured results.

**Important:** This screening score is not the model's actual accuracy. Keyword matching can incorrectly mark a response as correct or incorrect. Medical and emergency responses must be reviewed manually, and serious safety failures should be tracked separately.

## Use Cases

### 1. Voice Assistance

Allow elderly users to ask questions naturally and receive simple responses.

### 2. Medication Support

Provide reminders based on verified schedules and stored instructions. The LLM must not independently determine medication dosages or replace professional medical advice.

### 3. Reading Assistance

Future camera and OCR integration could read medicine labels, documents, bills, and signs aloud.

### 4. Family Communication

Future integration could enable voice-controlled calls to approved contacts, with appropriate confirmation and permissions.

### 5. Emergency Assistance

A dedicated physical button could initiate a predefined emergency workflow. This requires reliable connectivity, tested software, and appropriate fallback mechanisms.

## Roadmap

- [x] Load Qwen3-4B-Instruct-2507
- [x] Build an interactive text-chat prototype
- [x] Create a basic scenario-based evaluation script
- [ ] Measure response latency and memory usage
- [ ] Improve evaluation using manually reviewed test cases
- [ ] Integrate speech-to-text
- [ ] Integrate text-to-speech
- [ ] Add camera input and OCR
- [ ] Implement reliable medication reminders
- [ ] Add approved family contacts and emergency workflows
- [ ] Develop the spectacle-mounted display and hardware
- [ ] Optimize inference for affordable hardware

## Safety and Privacy

This project is an experimental assistive-technology prototype, not a certified medical device.

- Do not rely on generated responses for medical diagnosis or medication dosing.
- Do not treat object recognition or obstacle descriptions as guaranteed safety information.
- Protect personal conversations, camera images, contact details, and health-related information.
- Obtain appropriate consent before recording or transmitting audio and images.
- Provide an accessible fallback when connectivity or AI services are unavailable.

## Limitations

- The current prototype accepts text input only.
- The model does not have direct access to the user's camera, microphone, clock, contacts, or reminder database.
- The assistant cannot execute calls or reminders without additional application code and authorized integrations.
- Model responses may contain factual errors or unsafe suggestions.
- Performance depends on hardware resources and configuration.

## License

Review the model's current license and usage conditions on the official [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) before redistribution or commercial deployment.

Add an appropriate license to your own repository after deciding how you want your project code to be licensed.

## Acknowledgements

- [Qwen — Hugging Face](https://huggingface.co/Qwen)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch](https://pytorch.org/)

---

**Project goal:** To develop affordable, accessible, voice-first AI smart glasses that help elderly people interact with technology more independently.
