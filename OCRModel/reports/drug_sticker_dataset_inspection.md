# Vignette OCR Dataset Inspection

## Summary

The requested destination contains a verified copy of the already-extracted
Vignette OCR dataset. No ZIP archive was found under `data`, so an archive
filename, compressed size, validity check, and archive extraction cannot be
reported. The original extracted source directory remains in place.

The dataset contains 1,424 readable JPG images and a COCO-style annotation
file for the training split. The COCO file provides bounding boxes for the
literal classes `entete` and `vin`, but it contains no text transcriptions.
Consequently, the provided files do not directly support TrOCR fine-tuning or
ground-truth OCR evaluation.

## Locations and integrity

- Existing extracted source: `data/vignette-ocr/`
- Requested inspection copy: `data/DrugSticker_data/`
- Archive search: no `.zip` files found recursively under `data/`
- Destination inventory: 1,428 files totaling **67,491,062 bytes**
- The source and destination each contain 1,428 files; SHA-256 comparison
  confirmed that every relative path and file content matches.
- No original image files were modified.
- The dataset destination is ignored by Git. Verified with
  `git check-ignore -v data/DrugSticker_data/train.csv`.

Because there is no archive in the project data directory, 67,491,062 bytes is
the measured total file size of the extracted copy, not the compressed archive
size or necessarily the disk space allocated by the filesystem.

## Files and split layout

| Extension | Count |
| --- | ---: |
| `.jpg` | 1,424 |
| `.csv` | 3 |
| `.json` | 1 |
| **Total** | **1,428** |

The extracted directory structure is:

```text
DrugSticker_data/
|-- train/
|   |-- 750 JPG images
|   `-- _annotations.coco.json
|-- test/
|   `-- 674 JPG images
|-- train.csv
|-- test.csv
`-- sample_submission.csv
```

There is no validation split. `train.csv` has 750 `id` rows and no text
column. `test.csv` has 674 `id` rows and no text column. The 674 rows in
`sample_submission.csv` have `id,text` columns, but every `text` value is
blank. The train and test IDs each match the corresponding image filenames.
There is no train/test filename overlap.

## Annotation contents

The training JSON has top-level `info`, `licenses`, `images`, `annotations`,
and `categories` entries. It describes 750 training images and 1,509
annotations.

The categories are:

| Category ID | Category name | Annotation count |
| ---: | --- | ---: |
| 0 | `Images` | 0 |
| 1 | `entete` | 754 |
| 2 | `vin` | 755 |

Each annotation has `id`, `image_id`, `category_id`, `bbox`, `area`,
`segmentation`, and `iscrowd` fields. A representative record is:

```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 2,
  "bbox": [19, 119, 336, 222.5],
  "area": 74760,
  "segmentation": [],
  "iscrowd": 0
}
```

The referenced training image is
`WIN_20200702_17_33_33_Pro_jpg.rf.a5b4a8afa0bca4fea3720991d8046d69.jpg`.
In that image, the `vin` box covers a broad printed region on the sticker and
the `entete` box covers an overlapping narrow side region. This was checked
by drawing the actual boxes over the image. The exported category names are
reported literally; the JSON does not define their intended semantic
definitions.

All 1,509 annotations have bounding boxes. All have empty `segmentation`
arrays, so no polygon outlines are supplied. The boxes are for the two
exported classes, not individual words or text lines.

### Label availability

- **Text transcriptions:** None found in the JSON or CSV files.
- **Image-to-transcription mappings:** None. COCO maps training image IDs to
  category boxes only; CSV IDs map to images only.
- **Bounding boxes:** Yes, for `entete` and `vin` regions in training images.
- **Polygons/masks:** None; segmentation arrays are empty.
- **Explicit expiry-date, medicine-name, strength, dosage, or other
  field labels:** None. Text that is visibly printed in an image is not
  represented as a transcription or structured field label.
- **Test ground truth:** None in the supplied files; sample submission text
  values are blank.

## Image quality and duplicates

- All 1,424 JPGs were opened and verified with Pillow; no corrupt or
  unreadable images were detected.
- SHA-256 comparison found **4 exact-duplicate groups**, comprising **8
  test-split files**. No duplicate group spans train and test.
- The provided sample images show photographed printed labels on medicine
  packages, with varied rotation, framing, and image scale. Some are
  handheld or photographed at an angle. The contact sheet is a sample, not
  a complete visual quality review.

## Suitability for the OCR project

| Use | Assessment |
| --- | --- |
| Detection training | **Potentially, for the two exported box classes only.** The training split has box annotations for `entete` and `vin`. Their intended class meanings should be confirmed before treating them as general sticker or text detections. They are not word- or line-level text boxes. |
| TrOCR fine-tuning | **Not directly suitable as supplied.** TrOCR requires image/transcription pairs; this dataset has no transcribed labels or line crops. Manually verified transcription labels and appropriate crops would need to be created first. |
| Real-image OCR evaluation | **Not suitable as supplied.** There are no reference transcriptions, including for the test split. OCR accuracy metrics cannot be calculated from these files alone. |

The COCO metadata lists the license name as `Unknown` and its URL as empty.
This does not establish permission to use or redistribute the images. Review
the Kaggle competition rules and the upstream dataset terms before training,
publishing, or sharing any derived data.

## Contact sheet

The generated contact sheet contains 20 representative images and is
1200 x 1350 pixels. It is saved as
[`drug_sticker_contact_sheet.jpg`](./drug_sticker_contact_sheet.jpg).

## Recommended next steps

1. Confirm the intended definitions of `entete` and `vin` and the applicable
   dataset license/competition terms.
2. If OCR is the goal, create human-verified image/transcription references
   for representative medicine labels. Keep the test references held out
   from training and validation.
3. If detector training is the goal, first decide whether the existing two
   categories match the desired detection targets; the supplied annotations
   are not general word- or text-line annotations.

No model was trained or fine-tuned, no OCR accuracy was measured, and no
existing OCR code, evaluation logic, configuration, or tests were changed.
