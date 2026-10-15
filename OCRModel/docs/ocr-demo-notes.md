# SmartVision OCR demo notes

## Objective and architecture

Demonstrate a CPU-capable OCR research pipeline on a deterministic sample of
medicine-package photos from `data\DrugSticker_data\test`:

1. PaddleOCR `PP-OCRv5_mobile_det` proposes text-region polygons.
2. The pipeline validates/clips polygons, skips invalid regions, puts valid
   regions in top-to-bottom/left-to-right order, and makes a crop for each.
3. Microsoft's pretrained `microsoft/trocr-base-printed` recognizes each crop.
4. A CSV records source provenance, coordinates, model text, status, and errors.

The detector score is not a TrOCR confidence. Every prediction is explicitly
unverified. This demo reports pipeline outcomes only; there are no verified
transcriptions for this dataset and no accuracy metrics.

## Main files

- `src\smartvision_ocr\inference\detect_and_ocr.py`: detector initialization,
  polygon validation/clipping, region ordering and crop creation, per-image
  orchestration, CSV writing, and CLI.
- `src\smartvision_ocr\inference\predict.py`: TrOCR processor/model loading,
  device selection, and recognition of path-based or in-memory images.
- `src\smartvision_ocr\inference\batch.py`: separate exploratory full-image
  TrOCR probe; it does not detect or crop text.
- `src\smartvision_ocr\config.py`: shared model identifiers and project/data
  paths.
- `requirements-detector.txt`: optional pinned PaddleOCR/PaddlePaddle runtime.
- `tests\test_detect_and_ocr.py`: polygon, crop, ordering, error, and CSV tests.

## Setup and demo command

Run from PowerShell. The base OCR environment should already exist at
`.venv`; install the optional detector runtime once if it is not installed:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
.\.venv\Scripts\python.exe -m pip install -r requirements-detector.txt
```

The demo command uses a new output name so it does not overwrite earlier
experiment files. If that name already exists, choose another `--output-csv`
path or explicitly add `--overwrite`.

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

The first run may download detector and TrOCR weights to local caches. The
recorded experimental output is
`data\predictions\drugsticker_test_detected_text_clipped.csv`; use it for a
presentation if avoiding model startup and inference time is important. Review
the CSV without changing it:

```powershell
Import-Csv data\predictions\drugsticker_test_detected_text_clipped.csv |
  Where-Object { $_.status -eq "success" } |
  Select-Object -First 12 source_image,region_index,polygon_xy,detector_score,recognized_text
```

## Recorded results

The recorded run used 10 deterministic images from the held-out test split,
seed `42`, and CPU inference. It wrote 876 CSV rows:

- 212 `success` region rows with non-empty text;
- 4 `skipped_duplicate` image rows;
- 660 `skipped_not_sampled` image rows.

All 10 sampled images produced at least one non-empty region result, and no
sampled image had a `detection_failed` row in this output. These are
execution/status counts, not recognition accuracy or proof that the returned
text is correct. The CSV sets `prediction_verified=false` for every row.

Tests previously reported for the polygon clipping update: 15 focused
`test_detect_and_ocr.py` tests and 68 tests in the complete project suite
passed. Rerun the tests in this checkout with:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest tests\test_detect_and_ocr.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

## CSV fields

`source_image` is project-root relative; `split` is the unchanged dataset split;
`region_index` is one-based in detected reading order; `polygon_xy` is JSON
coordinates in the source image's pixel coordinate system; `detector_score`
is the detector's own score when present; model and device columns identify
the detector, recognizer, and TrOCR execution device; `recognized_text` is
unverified output; `prediction_verified` is false; `status` identifies the
image/region outcome; `error` provides details for failures.

## Two-minute presentation sequence

1. **0:00–0:20 — Goal:** Explain that TrOCR recognizes line crops and cannot
   locate text in a full photo by itself.
2. **0:20–0:45 — Architecture:** Show the detection → polygon crop/order →
   TrOCR → CSV flow. Distinguish the detector score from recognizer confidence.
3. **0:45–1:10 — Code:** Open `detect_and_ocr.py` and `predict.py`; point to the
   detector/orchestration and recognizer responsibilities.
4. **1:10–1:35 — Output:** Show the recorded CSV and a couple of regions. Mention
   the 10-image sample and its 212 non-empty region outputs as run counts only.
5. **1:35–2:00 — Limitations:** State that predictions are unverified, no CER/WER
   or accuracy is available, and text regions may be missed, merged, duplicated,
   or misrecognized. Describe human-verified references as the next evaluation
   prerequisite.

## Limitations and next steps

- No model training or fine-tuning was performed.
- The dataset has no human-verified text transcriptions; model output must not
  be used as ground truth.
- The detector can miss, merge, split, or overlap regions. In the recorded
  sample, regions 12 and 15 overlap substantially on one rotated sticker. An
  overlay shows that they cover different portions; both are retained because
  overlap by itself does not prove a duplicate.
- Partly out-of-bounds polygons clipped to more than four vertices use their
  in-bounds bounding rectangle, which can include background.
- Small, blurry, curved, reflective, oblique, or low-contrast text may be
  missed or misread. Human verification is necessary, especially for names,
  strengths, dosage, and expiry.
- DrugSticker dataset permission and redistribution rights remain unresolved.
  Do not redistribute it; check the upstream terms before further use.
- Next: visually review detections, obtain permission-cleared data, create
  manually verified references separate from training, then evaluate on a
  held-out sample. Do not tune on test outputs.
