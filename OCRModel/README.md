# SmartVision OCR

This subproject is the OCR component of SmartVision, intended to recognize
everyday text in signs, packaging, food labels, books, documents, and other
scenes. Its initial recognizer recognizes one cropped line of printed text per
image with Microsoft's
[TrOCR Base Printed](https://huggingface.co/microsoft/trocr-base-printed).
An optional text-detection stage composes that recognizer with a pretrained
PaddleOCR detector for full images. Medicine packaging is one use case, not
the only supported domain. The English checkpoint has not been shown to
recognize Telugu or Hindi adequately; those scripts require separate
verification. This is an OCR research component, not a medical device: verify
important recognized information against its source.

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

For new multi-source annotations, use the canonical JSONL schema documented in
[docs/canonical-manifest.md](docs/canonical-manifest.md). The validator checks
image readability and containment, required fields, geometry, duplicate IDs
and image/transcription pairs, approved split names, image hashes, and
source-group leakage. It does not change original annotations. For example,
from the OCRModel directory in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m smartvision_ocr.data.prepare validate `
  --manifest data\normalized\records.jsonl `
  --image-root data
```

Export one split to a new destination. Only verified, license-approved,
non-pseudo line labels with explicit valid regions are eligible. Images with
multiple text regions require one record and geometry per transcription;
whole-image transcriptions are reported as excluded, not guessed into line
labels.

```powershell
.\.venv\Scripts\python.exe -m smartvision_ocr.data.prepare export `
  --manifest data\normalized\records.jsonl `
  --image-root data `
  --output-dir data\derived\first-export `
  --split train
```

The exporter writes `train.csv`, lossless crops in `crops\`, and
`provenance.jsonl`. Use the output directory as `--data-dir` for the existing
training command only after independently reviewing the exported rows and
split policy. Existing output files are protected; pass `--overwrite` only
when replacement is intentional. These commands prepare data only and do not
load a model, download weights, train, or fine-tune. The existing simpler
`image,text` CSV guidance below remains available for already-prepared
single-line crops.

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

### Batch exploratory run on DrugSticker photos

The batch command below uses only `data\DrugSticker_data\` and defaults to a
deterministic sample of 10 images from its held-out `test` split. It passes each
complete package photo to TrOCR only as an explicitly exploratory probe: the
model returns one sequence for the image and does **not** locate text regions.
This is not full-image OCR, text detection, or an accuracy evaluation. No
ground-truth labels are present for this dataset.

From `OCRModel` in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m smartvision_ocr.inference.batch `
  --sample-size 10 `
  --seed 42 `
  --split test `
  --device cpu
```

The first run may download the existing TrOCR model and processor into the
Hugging Face cache; subsequent runs reuse the cache. Predictions and explicit
skip/failure records are written to the ignored local file
`data\predictions\drugsticker_test_predictions.csv`. Each row records its
source path, split, exploratory input mode, model, device, prediction, status,
and confidence availability. The current `generate()` inference path exposes
no calibrated confidence, so confidence remains blank. Existing outputs are
not overwritten unless `--overwrite` is passed. Use `--sample-size`, `--seed`,
`--split train`, `--model`, `--device`, or `--output-csv` to configure a run;
use only one split per run. Do not use test-split predictions to tune or select
model settings.

The COCO boxes in the training split identify the exported classes `entete`
and `vin`, not individual text lines, and do not contain transcriptions. No
text detector is included in the base OCR requirements. For a real
package-to-text pipeline, install the optional CPU detector requirements and
use the detector command below. The dataset's license metadata is `Unknown`;
check the Kaggle competition rules and upstream terms before further use, and
do not redistribute the dataset.

### Pretrained text detection followed by TrOCR

The separate `detect_and_ocr` command uses PaddleOCR's detection-only
`TextDetection` API with the `PP-OCRv5_mobile_det` model, rectifies its
validated text polygons, sorts them top-to-bottom and left-to-right, and
passes each crop to the existing TrOCR model. Four-corner regions are
perspective-rectified; if clipping a partly out-of-bounds region creates more
than four corners, its in-bounds bounding rectangle is used. Invalid, degenerate,
and fully out-of-bounds polygons are skipped without discarding other valid
regions from the same image. Existing `predict` and exploratory `batch`
commands are unchanged.

The main modules are:

- `src\smartvision_ocr\inference\detect_and_ocr.py`: loads the detector,
  validates/clips and orders detector polygons, creates region crops, calls
  TrOCR, and writes per-region or per-image status rows.
- `src\smartvision_ocr\inference\predict.py`: loads the TrOCR processor/model
  and recognizes a line crop (including an in-memory crop from the detector).
- `src\smartvision_ocr\inference\batch.py`: the separate whole-photo
  exploratory TrOCR command; it does not detect or crop text.
- `src\smartvision_ocr\config.py`: shared model and project/data paths.
- `tests\test_detect_and_ocr.py` and `tests\test_inference.py`: detector,
  polygon/crop, failure-handling, CSV, and recognizer tests.

This detector uses PaddlePaddle's CPU inference engine. The optional
`requirements-detector.txt` pins `paddleocr==3.7.0` and
`paddlepaddle==3.3.1`; for the tested environment (64-bit Windows and Python
3.11.8), pip resolves a compatible `cp311-win_amd64` wheel. The dry-run
resolved 36 packages total, including supporting runtime packages. Install
only when this detector is needed:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
.\.venv\Scripts\python.exe -m pip install -r requirements-detector.txt
```

This remains an optional dependency group: do not add it to the base
`requirements.txt` unless all OCR environments should carry Paddle's separate
runtime. PaddleOCR downloads its detector model on first model initialization
(by default from Hugging Face); TrOCR likewise downloads from Hugging Face if
its weights are not cached. Neither download occurs for `--help` or unit
tests. The current command explicitly runs the detector on CPU; `--device`
selects the existing TrOCR device and does not change the detector backend.

Run the first deterministic 10-image sample from the held-out test split:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m smartvision_ocr.inference.detect_and_ocr `
  --sample-size 10 `
  --seed 42 `
  --split test `
  --device cpu `
  --output-csv data\predictions\drugsticker_test_detected_text_demo.csv
```

The separate demo output path avoids replacing prior results. The command
refuses to overwrite an existing output unless `--overwrite` is explicitly
given. The ignored local experiment file
`data\predictions\drugsticker_test_detected_text_clipped.csv` contains the
latest recorded 10-image run (seed 42, test split, CPU): 876 CSV rows total,
with 212 successful non-empty region predictions, 4 duplicate-image skips,
and 660 rows outside the selected sample. All 10 selected images produced
region predictions; there were no image-level detection failures. These are
pipeline execution counts, **not** OCR accuracy results. The separate
`data\predictions\drugsticker_test_detected_text.csv` is an earlier run and
still contains the two polygon-clipping failures that motivated the fix.

The detected-text CSV has these columns:

| Column | Meaning |
| --- | --- |
| `source_image` | Source path relative to the project root. |
| `split` | Original `train` or `test` split; this command does not move or alter images. |
| `region_index` | One-based reading-order index; blank for image-level rows. |
| `polygon_xy` | JSON-encoded detector polygon in source-image pixel coordinates. |
| `detector_score` | Detector-provided region score, when available; not TrOCR confidence. |
| `detector_model` | Detector model identifier. |
| `recognizer_model` | TrOCR model identifier. |
| `device` | Device selected for the TrOCR recognizer. |
| `recognized_text` | Unverified model output for that detected crop. |
| `prediction_verified` | Always `false`; no human verification is performed. |
| `status` | Region/image outcome, such as `success`, `empty_text`, `no_text_regions`, or a skip/failure status. |
| `error` | Error detail when a region or image could not be processed. |

The 10-image CSV has been checked for reading-order monotonicity and exact
duplicate polygons. An overlay review found two substantially overlapping
regions in one image; they appear to cover different portions of the rotated
sticker, so both are retained pending more careful visual review. The detector
score is not a TrOCR confidence score. No OCR accuracy metric is calculated
and no prediction is a verified medicine label.

The output preserves the source image and split. The default test sample is
for qualitative inspection only; do not use it to tune the model. The
`--split train` option is available for separate exploratory runs, but no
training is implemented by this command. This does not make the detector
medical-grade: small, curved, reflective, blurred, or stylized text may be
missed, merged, split, or misrecognized. The competition dataset's permitted
use remains subject to its unknown license metadata and competition terms.

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

For a guided manual workflow, review `data\evaluation\candidates.csv` and run
`.\.venv\Scripts\python.exe scripts\select_evaluation_crops.py` from the
project root. The crop selector only saves user-selected image regions; it
records their original image source without changing it, and does not run OCR
or generate labels. The Vignette dataset license metadata is `Unknown`;
confirm the applicable terms permit evaluation before using its candidates.
Follow
[`data\evaluation\README.md`](data/evaluation/README.md) to verify and record
each transcription before evaluating.

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
