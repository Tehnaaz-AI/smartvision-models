# Real-image OCR evaluation data

This manual-reference workflow is separate from the batch exploratory
inference command. The active batch experiment reads only
`data/DrugSticker_data/`; it does not consume crops or labels from this folder.
Existing local crops and `crop_sources.csv` are preserved as historical local
artifacts and must not be treated as part of the DrugSticker experiment unless
their source and permission are independently confirmed.

Place only medicine-label crops that you have permission to use in `images`.
Each image should contain one cropped line of printed text. Keep the synthetic
smoke-test image at `data/cropped/ocr_test.png` separate from this dataset.

## Select crops and transcribe manually

The initial candidate list is `candidates.csv`. It contains 20 image paths
from the DrugSticker dataset's `test` split only; it is a review queue, not a
set of OCR labels. That dataset's annotations contain object boxes but no text
transcriptions. Candidates may be unsuitable on full-size inspection. Check
that you have permission to use each image before selecting a crop. The
dataset inspection found its license metadata to be `Unknown`; do not use
these images for evaluation until the Kaggle competition and upstream terms
have been checked and permit this use.

From the project root in PowerShell, start the interactive crop selector:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
.\.venv\Scripts\python.exe scripts\select_evaluation_crops.py
```

The selector opens source images without changing them. Drag around one
printed line and choose **Save crop**. It writes a lossless PNG crop to
`data\evaluation\images\line_NNN.png`; make several crops from an image if
useful, but avoid repeated or near-duplicate lines. The display is resized
only for viewing; saved pixels come from the original image. The tool does
not run OCR and does not write labels. It records each crop's source image,
split, and source-pixel rectangle in the local, Git-ignored
`data\evaluation\crop_sources.csv`. Use **Next** to review candidates. If you
add candidates, keep a single held-out split in `candidates.csv`; do not mix
train and test splits.

Open the saved crops and source map for review with:

```powershell
Invoke-Item "data\evaluation\images"
Import-Csv "data\evaluation\crop_sources.csv" |
  Format-Table crop,source_image,split,left,top,right,bottom -AutoSize
```

Open each saved crop at full size and transcribe only what is visibly
readable. For this English TrOCR baseline, exclude non-English lines. Compare
every character against the original packaging:

- Preserve letter case, digits, decimal points, units, punctuation, and
  spacing as printed. Do not normalize or autocorrect.
- Do not infer a character from medicine knowledge, context, filenames, or
  model predictions. If any character in a line is uncertain or obscured,
  leave that crop out of the evaluation rather than guessing or using an
  invented placeholder.
- Each crop should contain one line only. If the source line wraps, crop each
  physical line separately and transcribe only that line; do not insert a
  line break into a single-line crop.
- If verified text contains a comma or quote, use standard CSV quoting when
  editing the row; do not remove printed punctuation to simplify the CSV.
- Keep only crops you have permission to use. Do not include private patient
  information.

After manually checking a crop, add one row to `references.csv`. Its columns
are exactly:

```csv
image,text
```

Use the crop path relative to `data\evaluation` (for example,
`images\line_001.png`) and enter the exact transcription you verified against
the source. Do not add sample labels or leave placeholders in the file. Keep
an independent record of which rows have been checked.
`references.csv` intentionally starts with only its header.

The candidate images come from a held-out competition split, not the
competition training split. Keep the resulting evaluation crops and
transcriptions out of all future training/validation data. If any are later
used for training, retire them from evaluation and obtain a separate held-out
set.

The DrugSticker dataset currently provides no transcriptions. Its training
COCO boxes are labeled `entete` and `vin`; they are object/category boxes, not
verified text-line annotations. The batch command described in the project
README is an exploratory whole-photo TrOCR probe only. It does not run a text
detector and does not produce reference labels or accuracy metrics.

## Run evaluation after labels are complete

From the project root, use the project's Python environment and source tree:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
$env:PYTHONPATH = "$PWD\src"
```

Alternatively, activate the environment before running project commands:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
```

Start the selector as shown above, save crops, then review them at full size
and manually edit `data\evaluation\references.csv`. Validate the completed
manifest without loading or downloading the model:

```powershell
.\.venv\Scripts\python.exe -m smartvision_ocr.training.evaluate `
  --reference-csv "data\evaluation\references.csv" `
  --data-dir "data\evaluation" `
  --validate-only
```

The validator checks the exact `image,text` schema, non-empty text, relative
paths, supported image files, readable images, duplicate image references,
and one-line reference text. Fix all reported rows before evaluation.

After validation succeeds, run the evaluator:

```powershell
.\.venv\Scripts\python.exe -m smartvision_ocr.training.evaluate `
  --reference-csv "data\evaluation\references.csv" `
  --data-dir "data\evaluation" `
  --output-csv "data\evaluation\evaluation_results.csv" `
  --report-json "data\evaluation\evaluation_report.json" `
  --device cpu
```

Inspect per-image predictions and errors with:

```powershell
Import-Csv "data\evaluation\evaluation_results.csv" |
  Format-Table image,status,reference_text,predicted_text,cer,wer,exact_match,error -Wrap
Get-Content "data\evaluation\evaluation_report.json"
```

The existing evaluator reports CER, WER, and case-sensitive and
case-insensitive exact-match rates, with per-image results. It loads the
pretrained model and may download its weights when explicitly run. No metric
is available until valid, manually checked references exist and the evaluator
has actually run. Review mistakes in medicine names, strengths, decimal
points, and units separately. These small-sample results are an experiment,
not evidence of medical reliability; OCR output must never independently
authorize a medicine or dosage decision.
