# Proposed OCR dataset-preparation plan

**Status:** initial manifest-validation and crop-export foundation implemented.
This document remains the broader design plan; format-specific adapters,
source-rights registry, grouped split assignment, and near-duplicate review
remain proposed work and are not implemented.

## Design goals

1. Support multiple image sources and domains without making the OCR system
   medicine-specific.
2. Preserve original files and source train/test boundaries; store derived
   crops separately with traceable provenance.
3. Make TrOCR inputs unambiguous image-crop/exact-transcription pairs.
4. Keep detector labels, OCR labels, structured fields, and pseudo-labels
   semantically distinct.
5. Make language/script and verification status first-class, with Telugu as
   the first regional-language target and Hindi as a subsequent target.
6. Fail closed for unresolved paths, invalid geometry, unverified labels, or
   unresolved data-use rights.

## Proposed organization

```text
OCRModel/
├── data/
│   ├── <existing-source-dataset>/        # read-only, never rewritten
│   ├── registry/                         # source + license/provenance records
│   ├── normalized/
│   │   ├── records.jsonl                 # canonical annotation records
│   │   └── schema-version.txt
│   ├── derived/
│   │   └── crops/<dataset>/<source-id>/  # generated line/word crops
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── src/smartvision_ocr/data/
│   ├── adapters/                         # read-only format-specific importers
│   ├── schema.py                         # canonical typed records
│   ├── validate.py                       # record/image/geometry/split checks
│   ├── crop.py                           # reproducible region-to-crop steps
│   ├── split.py                          # source-group-aware split builder
│   └── export.py                         # validated crop pairs -> image,text CSV
└── tests/
    ├── data/
    ├── adapters/
    └── fixtures/                         # tiny synthetic, rights-safe examples
```

This is a proposal, not a request to create these paths now. Original datasets
should remain at their existing locations and be treated as read-only. Registry
entries point to source roots instead of copying images. Keep generated crops,
manifests, caches, and model artifacts ignored by Git.

## Canonical record proposal

Use versioned UTF-8 JSON Lines (JSONL) as the lossless, normalized source of
truth. Export the existing trainer's simple `image,text` CSV only after
filtering to verified crop records.

Example (illustrative schema, **not** a real annotation):

```json
{
  "schema_version": "1",
  "record_id": "sha256:...",
  "source_dataset": "dataset-name",
  "source_image": "relative/path/to/original.jpg",
  "source_image_sha256": "...",
  "source_record_id": "annotation-id-or-null",
  "region": {
    "kind": "polygon",
    "coordinates": [[12.0, 18.0], [180.0, 18.0], [180.0, 52.0], [12.0, 52.0]],
    "coordinate_space": "source-image-pixels",
    "image_width": 640,
    "image_height": 480
  },
  "crop_path": "derived/crops/dataset-name/source-id/region-id.png",
  "transcription": "Visible text exactly as transcribed",
  "text_scope": "line",
  "language": "en",
  "script": "Latn",
  "split": "train",
  "annotation_provenance": {
    "kind": "human",
    "source": "annotation-session-id",
    "annotator_id": "pseudonymous-id",
    "verified_by": ["pseudonymous-reviewer-id"],
    "verified_at": "ISO-8601-or-null"
  },
  "label_status": "verified",
  "label_origin": "human",
  "license_status": "approved-for-project-use",
  "source_group_id": "package-or-document-or-capture-session-id"
}
```

The final schema should define optional/null values and enums precisely. Do
not copy this illustrative transcription into a manifest. Suggested semantics:

- `source_image` always identifies the unchanged original, while `crop_path`
  identifies the derived recognizer input.
- `region.kind` distinguishes `bbox_xyxy`, COCO `bbox_xywh`, and `polygon`.
  The adapter converts supported formats into a documented pixel-coordinate
  representation while retaining the original source annotation ID and format.
- Include source width/height and orientation policy. Coordinates refer to
  pixels in the explicitly named image frame; transformations from EXIF
  orientation or resized images must be recorded.
- `text_scope` says what the transcript labels (`line`, `word`, `paragraph`,
  or `field`). The text extent and crop must correspond to that scope.
- Language and script may be unknown; do not infer either from filenames or
  model output. Use a documented language tag and script code (for example
  `te`/`Telu`, `hi`/`Deva`, `en`/`Latn`) when verified.
- `annotation_provenance`, `label_status`, and `label_origin` are required.
  A model prediction has `label_origin="pseudo"` and
  `label_status="unverified"` until a human explicitly checks/corrects it.
- Rights are per-source metadata, not an assumption inherited from the
  project. Unknown rights block export into training/evaluation.
- `source_group_id` groups all crops from one source photo, physical package,
  document, or capture session, as applicable.

## Preparation workflow

### 1. Register and gate source datasets

Add one source-registry record per dataset/version: source URL/acquisition
date, terms/license URL and version, permitted project use, redistribution
status, image root, native split structure, annotation files, and any privacy
restrictions. Keep original files untouched.

No source with unknown/unreviewed terms should be imported into a training or
evaluation split. A future adapter must declare its expected schema and
explicitly map native image IDs to files; never infer a match solely from
similar-looking filenames.

### 2. Import native annotations without losing meaning

Implement one adapter per actual source format (CSV, COCO, ICDAR text files,
etc.). Preserve original annotation IDs, category names, text strings, and
split metadata alongside normalized records. Validate every reference against
its image.

Do not reinterpret `entete` or `vin` as text categories, and do not treat a
product/object box as a transcript. A whole-image caption or page-level text
string must not be paired to a line crop. Word annotations may produce word
crops with word text; line-level TrOCR examples require genuine line
transcriptions or an explicit, verified grouping/alignment of words to that
line.

### 3. Review and verify transcript labels

Allowed origins should be explicit: original dataset annotation, human
transcription, or pseudo-label. Keep imported dataset text marked with its
source and review state; do not silently upgrade it to verified. For
high-risk strings (names, digits, decimal points, strengths, expiry dates,
and units), use a second human check where practical. Store disputes and
corrections as audit events rather than overwriting source annotations.

OCR output may help prioritize review but is never a reference label by
default. A human must compare the entire crop to the physical pixels and
accept or correct every character before a pseudo-label can be promoted.
Preserve the original model output and verifier identity/date separately.

### 4. Generate deterministic recognizer crops

For each usable text region:

1. Validate region geometry in the source image's pixel frame.
2. Crop the exact visible region from an unchanged image. Perspective rectify
   only where the geometry supports it; log the transform. Do not sharpen,
   inpaint, or hallucinate missing characters.
3. Save derived crops using a stable source/region digest and refuse
   accidental overwrite. Record crop dimensions and a digest.
4. Attach the transcription that labels that exact crop and set
   `text_scope="line"` for the current TrOCR fine-tuning path. If a full line
   wraps, represent its physical lines independently unless the model/task is
   explicitly extended to multi-line output.
5. Review a rendered crop plus its source overlay. Exclude unreadable or
   ambiguous samples rather than guessing.

The export adapter should produce `image,text` rows under a crop root, since
that is the interface currently expected by `read_manifest` and `OCRDataset`.
Do not expand the training interface until the richer provenance is retained
outside that compatibility CSV.

### 5. Validate before export

Block a training/evaluation export when any required condition fails:

- The source image or crop path is missing, escapes its configured root, has
  unsupported format, cannot be decoded, or has a mismatched recorded digest.
- CSV/JSONL is malformed, encoding is invalid, required fields are absent,
  identifiers are duplicated, or an image/region is mapped ambiguously.
- A transcription is absent/empty, contains a forbidden line break for a
  single-line record, or is not a string. Preserve exact Unicode, case,
  digits, punctuation, units, and spaces. Do not transliterate. Any Unicode
  normalization policy must be documented/versioned; keep the original
  annotation text.
- A coordinate is non-numeric/non-finite, the shape is invalid, a box has
  non-positive width/height, a polygon has fewer than three distinct points
  or zero area, or geometry falls outside its declared source-image bounds
  without an explicit clipping record.
- Provenance, image/source group, split, label origin, verification state, or
  rights status is missing/contradictory.
- A pseudo-label is selected for supervised training/evaluation without
  verified human review.

Report failures with record IDs and source line numbers; do not silently skip,
guess values, or rewrite source files.

### 6. Assign leakage-safe splits

Preserve an upstream official split when using it. For newly assembled data,
assign splits at the highest available source group (physical package,
document/book, location, capture session, or original photo) **before**
deriving line/word crops. No group may occur in multiple splits.

Validate:

- exact source-image paths and SHA-256 do not cross splits;
- crops from a shared original/group do not cross splits;
- perceptual-hash or embedding near-duplicate candidates are reported for
  human review rather than silently auto-merged;
- language, script, domain, acquisition conditions, and source are
  represented in each split where sample counts permit;
- the test set stays frozen and is not used for data/model selection.

The current validator already rejects repeated image paths and byte-identical
image bytes between splits. Extend that behavior with group-level identities
and an explicit near-duplicate review report; do not weaken existing checks.

## Dataset strategy

Use all three approaches as complementary streams, not substitutes:

1. **Synthetic text lines first for plumbing and controlled script coverage.**
   Create deterministic crop/transcription pairs from synthetic text and
   licensed fonts. This gives exact labels and can exercise English, Telugu,
   and Hindi Unicode handling before collection is large. It is not a measure
   of real-world performance and should not dominate the held-out evaluation.
2. **Permission-cleared real examples for target performance.** Collect
   diverse text from packaging, food labels, signs, shop boards, room numbers,
   notices, books/forms, and displays. Capture both full images and verified
   line regions/transcriptions, preserve source grouping, and evaluate on
   independently held-out people/locations/products/documents.
3. **Public corpora for efficient transfer and coverage.** Assess TextOCR or
   COCO-Text for real scene words, ICDAR MLT for multilingual scene text, and
   SROIE for receipts/document text. Build adapters only after checking the
   exact release's language list, annotation level, image mapping, split,
   quality, and legal terms. They supplement rather than replace SmartVision
   target-domain data.

Medicine packaging remains one evaluation domain. Build domain-stratified
reports so good performance on receipts, synthetic lines, or one product
dataset cannot hide failures on signs, small expiry text, Telugu displays, or
documents.

## Recognizer strategy for English, Telugu, and Hindi

- Keep `microsoft/trocr-base-printed` as the English printed-line baseline.
  Its model card describes single-line OCR and SROIE fine-tuning, not
  multilingual Telugu/Hindi certification.
- Evaluate separate script-capable recognizers on the same verified,
  leakage-safe line crops. The official PP-OCRv5 multilingual documentation
  lists a Telugu `te` recognizer and a Devanagari recognizer covering Hindi.
  This is a candidate, not a chosen or locally validated model. The current
  PaddleOCR dependency is optional and its present code initializes only the
  text detector. The recognizer API, model configuration/download, CPU
  behavior, mixed English/regional strings, crop expectations, and licensing
  still need an isolated compatibility test.
- Compare by language/script and domain using CER, WER, exact match, failure
  rates, and human review of critical strings. Keep English, Telugu, and Hindi
  model selection independent if one checkpoint is not adequate across them.
- A tokenizer that can encode/round-trip a script is not evidence that the
  image model recognizes that script. Do not add/replace models until
  benchmark references, licensing, runtime, latency, and quality have been
  reviewed.

## Tests and acceptance gates for a future implementation

Before any training run is enabled, add unit/fixture coverage for:

- each dataset adapter's image-ID mapping, annotation conversion, split, and
  provenance preservation;
- UTF-8 and UTF-8-BOM import, malformed CSV/JSON, empty/non-string text, and
  exact Unicode handling for English/Telugu/Hindi examples;
- missing/corrupt images, duplicate record IDs, duplicate files, and
  deterministic crop regeneration without overwriting originals;
- bounding boxes and polygons at image boundaries, partly outside bounds,
  reversed/zero dimensions, duplicate vertices, zero-area/self-crossing
  regions, and source-to-crop coordinate lineage;
- rejection of full-image transcripts as line labels unless a valid region
  alignment exists;
- pseudo-label exclusion by default and promotion only through explicit human
  verification;
- group-level and byte-identical train/validation/test leakage, with
  near-duplicate candidates surfaced for review;
- a small end-to-end fixture from native annotation to generated crop and
  export manifest, without downloading models or touching originals.

Acceptance criteria: every exported recognition row points to one readable
line/word crop and its exact verified text; every crop traces to an original
image and annotation event; unknown rights and unverified pseudo-labels are
blocked; source groups are disjoint across splits; and the test set is frozen.
Only after those gates and plan review should a small training pilot be
considered. Training is not part of this task.

## Review decisions before implementation

1. Confirm whether the first release is English-only with a Telugu data
   collection phase, or whether Telugu must be a launch criterion.
2. Approve the dataset/license gate and whether current local datasets may be
   used at all.
3. Decide target annotation scope (line-only initially, or word annotations
   retained and grouped into lines where possible).
4. Confirm privacy/consent rules for public-scene and user-captured photos.
5. Choose candidate multilingual recognizers and the hardware/latency
   envelope to benchmark.
