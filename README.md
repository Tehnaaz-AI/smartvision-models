# SmartVision OCR

This subproject recognizes one cropped line of English printed text per image.
The initial model is Microsoft's
[TrOCR Base Printed](https://huggingface.co/microsoft/trocr-base-printed).
The initial target is text on medicine packaging, including medicine names,
strengths, dosage instructions, and expiry dates. This is an OCR research
component, not a medical device: always verify recognized medicine details
against the physical package and a qualified professional.

Training is never started automatically. Pretrained weights are downloaded
from Hugging Face only when inference, training, or evaluation is explicitly
run for the first time.

## Requirements and setup

- Windows 10/11
- Python 3.11 (64-bit)
- VS Code with the Microsoft Python extension
- Internet access for dependency/model downloads
- CPU is supported; a compatible CUDA device is used if PyTorch detects one

Open a PowerShell terminal and run:

```powershell
Set-Location D:\Projects\SmartVision\OCRModel
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The versions are pinned in `requirements.txt`. The PyTorch CPU index installs
the CPU wheel when CUDA is not needed; no CUDA installation is required. To use
an NVIDIA GPU, install a PyTorch build compatible with the
installed CUDA runtime from the official PyTorch instructions, then install
the remaining pinned requirements.

In VS Code, select `OCRModel\.venv\Scripts\python.exe` using **Python: Select
Interpreter**. Check the runtime and device:

```powershell
python scripts\check_environment.py
```

`.env.example` lists optional environment settings, but the package does not
load `.env` files automatically. Set variables in the PowerShell session, for
example:

```powershell
$env:SMARTVISION_OCR_DATA_DIR = "D:\datasets\medicine-ocr"
```

## Dataset preparation

Images should be cropped to contain a **single printed text line**. Keep
original photographs in `data\raw`, line crops in `data\cropped`, and optional
cleaned crops in `data\processed`. Put CSV manifests in `data\labels` (or
another location) and images below the configured dataset directory. The CSV
format is UTF-8 with exactly the required columns `image,text`; for example:

```csv
image,text
cropped\package_line_001.png,Paracetamol 500 mg
cropped\package_line_002.png,Take one tablet every  six hours
```

The example above illustrates the format only. Do not use its sample rows as
training data. Transcribe each crop carefully from the original package,
preserving spelling, numbers, units, punctuation, and meaningful spacing.
Have a second person verify labels when possible. Do not infer obscured text,
copy labels from another package, or generate synthetic/fabricated labels.
Correct the source image or exclude it when the text cannot be read reliably.
The loader rejects missing or unreadable images and empty labels.

Make three manifests: `train.csv`, `validation.csv`, and `test.csv`. Each
manifest has exactly the `image,text` columns. Add only verified, genuine
image/transcription rows; the header is:

```csv
image,text
```

Split by original package/photo before making crops: crops from one source image
or package must not appear in different splits. The scripts also reject
identical image paths or byte-for-byte duplicate image files across splits.
Near-duplicate packages and crops from the same physical package still require
careful manual grouping.

## Inference

From `OCRModel`, set the source path and call the CLI:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m smartvision_ocr.inference.predict "data\cropped\ocr_test.png"
```

The default model is `microsoft/trocr-base-printed`. The command selects CUDA
when PyTorch reports it available and otherwise runs on CPU; pass `--device cpu`
to explicitly use CPU. It prints the model ID, selected device, input image, and
recognized text. For a clean synthetic smoke test, the project includes the
locally generated `data\cropped\ocr_test.png` (ignored by Git); it is not
included in this repository or a real medicine-label accuracy test.

Or use the Python function:

```python
from smartvision_ocr.inference.predict import predict_image

text = predict_image(r"data\cropped\your_line.png")
print(text)
```

The first call downloads the pretrained model and processor into the Hugging
Face cache. Later calls reuse the cached files. A fine-tuned model can be
selected with `SMARTVISION_OCR_MODEL` or the function's `model_name_or_path`
argument. Invalid image paths/formats and model download/load failures produce
clear errors.

If model loading fails, verify internet access, the model ID, and Hugging Face
availability. If an image cannot be opened, check that the supplied path exists
and that the file is a supported, readable image. TrOCR expects a cropped
single printed text line; it does not detect text regions in a full image.

## Training

Training is explicit; merely installing dependencies or running tests does
not train or download a model. From `OCRModel`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m smartvision_ocr.training.train `
  --train-csv "data\labels\train.csv" `
  --validation-csv "data\labels\validation.csv" `
  --test-csv "data\labels\test.csv" `
  --data-dir "data" `
  --learning-rate 5e-5 `
  --batch-size 2 `
  --epochs 3 `
  --seed 42 `
  --checkpoint-dir "models\checkpoints" `
  --output-dir "models\final"
```

Only start this after genuine, verified labels and a leakage-safe split are
ready. The scripts validate all three manifests before training. Learning rate,
batch size, epoch count, seed, and checkpoint/output paths are configurable.
Training runs on CPU if CUDA is unavailable; batch size 2 is a conservative
starting point, not a performance guarantee. Checkpoints include the model,
processor, optimizer state, and completed epoch. Resume with:

```powershell
python -m smartvision_ocr.training.train `
  --train-csv "data\labels\train.csv" `
  --validation-csv "data\labels\validation.csv" `
  --test-csv "data\labels\test.csv" `
  --data-dir "data" `
  --resume-from-checkpoint "models\checkpoints\epoch-0001"
```

## Evaluation

Real medicine-label evaluation is separate from the synthetic smoke-test image
at `data\cropped\ocr_test.png`. Place only images you have permission to use
under `data\evaluation\images`. Add one cropped printed line per image and
manually verified transcription to `data\evaluation\references.csv`; use paths
relative to `data\evaluation`, for example `images\permitted_crop_001.png`.
The tracked template contains only the `image,text` header. Do not use
synthetic text or guessed labels in the real evaluation.

Run evaluation from `OCRModel` after adding real, correctly labelled rows:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m smartvision_ocr.training.evaluate `
  --reference-csv "data\evaluation\references.csv" `
  --data-dir "data\evaluation" `
  --output-csv "data\evaluation\evaluation_results.csv" `
  --report-json "data\evaluation\evaluation_report.json" `
  --device cpu
```

The script loads the existing `microsoft/trocr-base-printed` model once, uses
CPU, and records every reference row. Missing or unreadable images and invalid
rows appear in the results CSV with a status and error; they are not treated as
correct or omitted. The JSON report includes row/evaluation/failure counts and
aggregate metrics calculated only from successful predictions. If there are
no reference rows, evaluation stops before loading the model or downloading
anything.

The per-image CSV reports CER, WER, case-sensitive exact match, and
case-insensitive exact match. Case-insensitive comparison applies Unicode
`casefold()` only: numbers, dosage units, punctuation, and whitespace are not
removed. WER tokenizes on whitespace while retaining punctuation in each word.
CER compares the original character sequences. Empty references use a
denominator of one so insertion-only errors remain defined. Aggregate exact
match rates use successfully evaluated images as their denominator; failure
counts are reported separately. Near-duplicate packages or crops still
require careful manual review.

Troubleshooting: ensure the image paths in the CSV are relative to the
evaluation directory, each transcription is manually checked, and images are
single-line crops. If model loading fails, check network access and Hugging
Face availability. If no real images are present, supply permitted images and
verified labels before running evaluation. Results from the synthetic smoke
test are not medicine-label accuracy measurements.

## Tests

Tests exercise CSV/image validation and metrics without loading a model, using
a GPU, or accessing Hugging Face. From `OCRModel`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

## Limitations

TrOCR is a general printed-text model, not a medicine-specific verifier.
Recognition may be wrong on small, curved, reflective, damaged, low-contrast,
or unusually formatted packaging. Cropped single lines do not capture layout
or relationships between fields. CER/WER do not measure clinical safety.
Validate on a representative, independently held-out dataset and require
human confirmation for medicine names, strengths, dosing, and expiry dates.
