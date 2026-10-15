# OCR dataset audit

**Audit date:** 2026-10-12

**Scope:** `D:\Projects\SmartVision\OCRModel`; inspection and design only. No
datasets, annotations, predictions, inference code, or tests were modified.

## Executive summary

The repository has one usable OCR-model training interface, but no verified
real-image transcription dataset. The existing `OCRDataset` and training
command require a crop paired with its text in an `image,text` CSV. The largest
local collection, DrugSticker/Vignette, has 750 training and 674 test photos
with 1,509 COCO boxes for two broad categories, but no transcriptions. A second
directory called `vignette-ocr` is a byte-for-byte duplicate of those images
and annotations, not a second dataset. The separate `Medicine_data` folder
contains 437 other JPGs and 11 contact sheets, without manifests or
annotations. None of these files establish permission for training or
redistribution.

The checked-in implementation is an English TrOCR line recognizer, with an
optional PaddleOCR text-detector stage. It does not currently provide a
general-purpose multilingual training-data importer or canonical annotation
manifest. The printed-English TrOCR checkpoint is not evidence of Telugu or
Hindi recognition ability. The existing README introduction is still
medicine-packaging centered and does not yet describe the full surroundings,
documents, and language scope in the project brief.

## Method and reproducibility

Repository status was inspected with `git status --short` before the audit.
The worktree was already dirty, including modified README, inference/evaluation
code and tests, deleted empty placeholders, and untracked reports/scripts.
Those pre-existing changes were preserved.

Counts below were measured locally with the project `.venv` Python 3.11.8.
Pillow opened and verified every JPG in `DrugSticker_data`, `vignette-ocr`, and
`Medicine_data`; JSON and CSV counts were obtained by parsing their contents,
not by trusting filenames. Byte comparisons were made between matching
DrugSticker/Vignette files. The following PowerShell/Python commands reproduce
the principal image and COCO counts without downloading or changing data:

```powershell
Set-Location "D:\Projects\SmartVision\OCRModel"
@'
from pathlib import Path
import json
from collections import Counter
from PIL import Image

for name in ("DrugSticker_data", "vignette-ocr", "Medicine_data"):
    root = Path("data") / name
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    unreadable = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            unreadable.append((str(path), str(error)))
    print(name, "images=", len(images), "unreadable=", len(unreadable))

    annotation_file = root / "train" / "_annotations.coco.json"
    if annotation_file.is_file():
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {row["id"]: row["name"] for row in coco["categories"]}
        counts = Counter(row["category_id"] for row in coco["annotations"])
        print(
            " COCO images=", len(coco["images"]),
            "annotations=", len(coco["annotations"]),
            "category_counts=",
            {categories[key]: value for key, value in counts.items()},
            "empty_segmentations=",
            sum(not row.get("segmentation") for row in coco["annotations"]),
        )
'@ | .\.venv\Scripts\python.exe -
```

For matching JPG filenames under `DrugSticker_data` and `vignette-ocr`, the
comparison found 750/750 train files and 674/674 test files byte-identical.
Their three root CSVs and train COCO JSON are also byte-identical. Do not count
these directory copies twice.

## Local dataset inventory

| Location | Measured contents | Annotation/label facts | Suitability and limits |
| --- | --- | --- | --- |
| `data/DrugSticker_data/` | 1,424 readable JPGs: 750 `train`, 674 `test`; 67,491,062 bytes across 1,428 files. No validation split. | `train/_annotations.coco.json` has 750 images, 1,509 boxes (754 `entete`, 755 `vin`), and empty segmentation for all boxes. COCO `file_name` maps to all 750 train images. Every annotation image ID resolves. All boxes have positive dimensions and lie within declared image dimensions. `train.csv` has 750 `id` rows; appending `.jpg` maps all to train images. `test.csv` has 674 `id` rows mapping exactly to test images. `sample_submission.csv` has 674 blank `text` values. No text transcript, line/word box, polygon, or structured medicine-field label was found. | Not a TrOCR dataset or OCR evaluation set. The boxes can support only the exported detection categories after their meaning is confirmed; they do not label text lines. Useful for supplementary qualitative detector/OCR experiments only, subject to rights. |
| `data/vignette-ocr/` | Same 1,424 readable JPGs, same 67,491,062 bytes, same split structure. | Images, CSVs, and COCO JSON compared byte-for-byte against `DrugSticker_data`; they are duplicates. | Not an independent dataset. Use one location as canonical to avoid double-counting and accidental split leakage. |
| `data/Medicine_data/` | 448 readable JPGs, 5,587,926 bytes: 437 non-contact-sheet images plus 11 files named `dataset_contact_sheet_*.jpg`. No split directories. | No label manifest, transcription file, COCO/other annotation file, language metadata, or license record was found in this directory. | Not directly suitable for detector training, TrOCR fine-tuning, or quantitative evaluation. Individual images could become manually annotated examples only after source/permission review and provenance capture. Contact sheets are not independent examples. |
| `data/evaluation/` | Candidate CSV has 20 DrugSticker test-image paths. `images/` contains one crop, `line_001.png`. | `references.csv` contains only its `image,text` header. `crop_sources.csv` contains two rows from `data/Medicine_data/images02.jpg`; one refers to the existing `line_001.png`, and the second refers to absent `line_002.png`. No human-verified text labels are present. | Not yet an evaluation set. The local crop/provenance mismatch needs correction or explicit retirement before these records are used. DrugSticker candidates remain a review queue, not labeled data. |
| `data/cropped/ocr_test.png` | One locally generated synthetic English text-line smoke-test image (visually inspected). | It visibly contains `Paracetamol 500 mg`; this is generated test content, not a label from a real medicine package, and it is not in a reference manifest. | Useful for a code-path smoke test only. Keep out of real-world evaluation and do not mix into real training without explicitly separating synthetic examples. |
| `data/predictions/` | Three ignored local CSV artifacts. The latest detection run, `drugsticker_test_detected_text_clipped.csv`, has 876 rows: 212 successful region rows, 4 duplicate skips, and 660 outside-sample rows. | These files contain model predictions, not references. The detection CSV marks `prediction_verified=false`; the older clipped-failure output is historical. | Never use predictions as ground truth. Pseudo-labels need a separate type and human verification before eligibility for supervised training/evaluation. |

### Visual and mapping checks

The COCO sample `WIN_20200702_17_33_33_Pro_jpg.rf.a5b4a8afa0bca4fea3720991d8046d69.jpg`
was displayed with its recorded boxes overlaid. The broad `vin` box covers
most of the printed label; the `entete` box covers an overlapping narrow
side section. These are coarse regions, not word/line boxes. The categories'
intended semantics are not defined in the JSON, so their names should not be
silently reinterpreted as text classes.

Displayed samples from `DrugSticker_data` and `Medicine_data` look like
photographs of medicine packaging. One visible DrugSticker label contains
French text, while the synthetic smoke test is English. This is only a sample
observation: neither dataset supplies language/script labels, and the visual
samples do not prove the full collection's image-generation provenance or
language distribution. No Telugu or Hindi ground-truth examples were found in
the local manifests.

## Annotation, split, and rights status

- **Transcriptions:** none for either DrugSticker/Vignette split or
  `Medicine_data`; no suitable references in `data/evaluation/references.csv`.
- **Boxes:** only the 1,509 COCO `bbox=[x,y,width,height]` records for the two
  training-set categories. No polygon segmentations; no individual
  character/word/line boxes or alignment.
- **Image mapping:** COCO IDs and `file_name` map to all 750 DrugSticker
  training files. The CSV IDs map to train/test images with the filename
  conventions above. Test has no COCO text annotations or reference
  transcriptions.
- **Language/script:** not declared in any local dataset manifest. Sample
  visible strings include French and English; systematic language counts
  cannot be inferred.
- **Real/synthetic:** local packaging samples look photographed; the
  synthetic smoke-test image is clearly generated. Complete dataset
  provenance has not been established.
- **License:** DrugSticker COCO metadata reports license name `Unknown` and an
  empty URL; the Roboflow metadata points to
  [Vignette Dataset version 4](https://app.roboflow.com/datasets/vignette-dataset/4).
  `Medicine_data` has no local source/license record. Treat both as
  permission-unresolved; check original Kaggle/upstream terms and any
  applicable privacy/consent restrictions. Do not redistribute either.

## Current code and model fit

- `src/smartvision_ocr/data/validation.py` reads UTF-8/UTF-8-BOM CSVs with
  exactly `image,text`, validates image existence/readability and non-empty
  labels, and rejects duplicate image paths. `validate_disjoint_splits`
  rejects shared paths and byte-identical images across splits. The loader
  strips leading/trailing whitespace from labels; strict preservation of
  source annotation text therefore belongs in the proposed canonical manifest
  and requires an explicit export/whitespace policy.
- `src/smartvision_ocr/data/dataset.py` yields `(RGB image, text)` pairs.
  `src/smartvision_ocr/training/train.py` consumes those line-image/text pairs
  and requires explicit train/validation/test manifests. It does not accept
  full-image transcripts or region annotations.
- `src/smartvision_ocr/inference/detect_and_ocr.py` uses the optional
  PaddleOCR `TextDetection` API, crops/orders detector regions, then calls the
  recognizer and records coordinates/status. It does not change detector
  proposals into verified labels. `src/smartvision_ocr/inference/predict.py`
  loads the configured TrOCR model.
- Base dependencies are pinned in `requirements.txt` (including
  `torch==2.5.1`, `transformers==4.46.3`, `Pillow==11.0.0`, and
  `numpy==1.26.4`). Optional CPU detection pins are in
  `requirements-detector.txt` (`paddleocr==3.7.0`,
  `paddlepaddle==3.3.1`). The existing environment measured Python 3.11.8,
  PyTorch 2.5.1+cpu, Transformers 4.46.3, Pillow 11.0.0, and no CUDA.
- The Hugging Face
  [TrOCR Base Printed model card](https://huggingface.co/microsoft/trocr-base-printed)
  identifies the checkpoint as fine-tuned on SROIE and describes OCR for
  single text-line images. Its decoder/tokenizer is RoBERTa-derived. The
  cached tokenizer encoded and round-tripped sample English, Telugu
  (`ఔషధం`), and Hindi (`दवा`) strings without an unknown-token ID. That proves
  only that tokenization can represent those Unicode strings; it does **not**
  establish visual recognition quality or training support for Telugu/Hindi.
  There are no verified local Telugu/Hindi results. Keep this checkpoint as
  an English baseline, not as the assumed regional-language solution.
- The official
  [PaddleOCR PP-OCRv5 multilingual documentation](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)
  lists Telugu (`te_PP-OCRv5_mobile_rec`) and Hindi in the Devanagari model.
  This is a candidate for a separate recognizer comparison, not proof of
  quality for SmartVision's images. It adds model weights/downloads and would
  need its own crop-based comparison on verified language-specific test sets.
  The current pipeline loads Paddle's detection-only API, not these recognizers.

## Public and synthetic data strategy candidates

| Approach/source | Appropriate use | Important cautions |
| --- | --- | --- |
| Controlled synthetic line generation | Start with exact generated labels to test manifest, crop, tokenizer, and training plumbing; generate English, Telugu, and Hindi only with Unicode-capable fonts whose licenses permit use. | Synthetic fonts/backgrounds and clean rendering have domain gap from camera blur, glare, curved labels, varied type, and natural text. It cannot establish real-world accuracy. Track generator, seed, font/version, and text provenance. |
| [SynthText](https://github.com/ankush-me/SynthText) | Candidate scene-text detector pretraining; its repo describes synthetic text in natural-image backgrounds with geometry/labels. | Not a ready-made clean line-pair collection. The upstream repository warns that it does not own the background-image copyright; review dataset, code, font, and image rights before use. Legacy generator dependencies and non-Latin variants also require review. |
| Permission-cleared real images, collected or licensed | Most relevant target-domain supervision and held-out evaluation across packaging, signs/shops, public notices, books/documents, displays, and food labels. | Costly manual annotation; collect only with rights/consent and privacy review. Record original-image identity, exact transcript, region geometry, script/language, and independent verification. |
| [COCO-Text](https://arxiv.org/abs/1601.07140) and [TextOCR](https://textvqa.org/textocr/) | Candidates for natural-scene word-level text detection/recognition beyond medicine packaging. Inspect the release schemas and image/license terms before adapter work. | Word-level labels are not automatically line-level TrOCR pairs; assemble lines only with geometric ordering/alignment. Verify release/version, split policy, annotation quality, and rights. |
| [ICDAR 2019 MLT](https://rrc.cvc.uab.es/?ch=15) | Candidate for multilingual scene-text detection/recognition and signage-like imagery. | Check actual script/language subsets, annotation granularity, terms, and label availability for the release used; do not assume it contains Telugu. |
| [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) | Receipt/document OCR and structured information extraction; relevant to printed documents and the domain used to fine-tune this TrOCR checkpoint. | Receipts do not represent every document or packaging condition. Review access and license/competition terms; do not treat structured key/value labels as line transcriptions without alignment. |
| Packaging/product OCR | No separate public packaging dataset with verified line transcriptions was confirmed in the local materials or sources reviewed here. DrugSticker is a possible image source only after rights review, but its supplied labels are broad boxes without text. | Treat packaging as an explicit gap: search and verify candidate releases for exact transcripts, region geometry, image mapping, and permitted use before selecting one. General scene-text datasets may contain product text but are not automatically representative packaging corpora. |

No public dataset was downloaded for this audit. Public-source suitability and
licensing remain to be confirmed from the exact release terms before data are
used. Packaging-specific public OCR references should be sought only after
checking that they provide actual transcriptions and line/word geometry, not
just product/object categories.

## Missing capabilities and unresolved questions

1. No normalized multi-source manifest, source adapters, annotation
   provenance, language/script columns, region-coordinate contract, or
   verification/pseudo-label state exists.
2. No local genuine verified crop/transcription set is available for model
   selection or metrics.
3. Existing leakage validation only compares paths and exact file hashes; it
   does not group crops by source package/document/capture session or detect
   near-duplicates.
4. `data/raw` and `data/processed` contain no prepared OCR rows; `data/labels`
   contains only its placeholder. The current training input remains absent.
5. The data registry, original acquisition source, permission, privacy, and
   redistribution rights for `Medicine_data` and DrugSticker require human
   confirmation.
6. Intended COCO class meanings, exact text-vs-object target policy, and
   whether image-level or line-level language tags are authoritative are
   unresolved.
7. Telugu collection priorities, scripts/fonts, mixed-language handling,
   handwriting scope, and device/runtime latency targets need product-owner
   decisions.

## References

- [TrOCR Base Printed model card](https://huggingface.co/microsoft/trocr-base-printed)
- [PaddleOCR PP-OCRv5 multilingual recognition documentation](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)
- [ICDAR 2019 MLT challenge](https://rrc.cvc.uab.es/?ch=15)
- [ICDAR 2019 SROIE challenge](https://rrc.cvc.uab.es/?ch=13)
- [COCO-Text paper](https://arxiv.org/abs/1601.07140)
- [TextOCR project page](https://textvqa.org/textocr/)
- [SynthText repository](https://github.com/ankush-me/SynthText)
