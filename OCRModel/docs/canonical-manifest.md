# Canonical OCR manifest, version 1

The preparation commands consume UTF-8 JSON Lines (`.jsonl`), one record per
transcribed image region. Each row is a JSON object. Keep source images and
their native annotation files unchanged; `image_path` is resolved relative to
the explicit `--image-root`.

## Fields

Required fields:

| Field | Type and meaning |
| --- | --- |
| `schema_version` | String `"1"`. |
| `sample_id` | Stable, unique identifier for this exact annotated sample. |
| `source_dataset` | Dataset/source name and, where useful, version. |
| `image_path` | Root-relative path using `/` separators; points to the original image. |
| `transcription` | Exact, non-empty Unicode transcription. |
| `split` | One of `train`, `validation`, or `test`; preserve an official source split. |
| `source_group_id` | Stable ID for the original image/document/package/capture group; groups must not cross splits within a source dataset. |
| `annotation_provenance` | Non-empty JSON object describing source annotation or human annotation method/reference. |
| `verification_status` | `verified`, `unverified`, or `rejected`. |
| `label_origin` | `human`, `source_annotation`, or `pseudo`. |
| `license_status` | `approved`, `unknown`, or `not_permitted`; only `approved` rows may be exported. |
| `text_scope` | `line`, `word`, `full_image`, `paragraph`, or `field`. |

Optional fields are `region`, `language`, and `script`. `language` and `script`
may be null when unknown. Do not infer them from a filename or OCR output.
Other source-native details may be retained as additional JSON properties;
they do not change how this version is validated.

`region` is either null or:

```json
{"type":"bbox_xyxy","coordinates":[12,18,180,52]}
```

or:

```json
{"type":"polygon","coordinates":[[12,18],[180,18],[180,52],[12,52]]}
```

Coordinates are finite source-image pixel edge coordinates, in the unrotated
image's pixel frame, with origin at the top left. Boxes use `[x1,y1,x2,y2]`
with positive area; the upper/right edges may equal image width/height.
Polygons require at least three distinct, non-self-intersecting points and
non-zero area. Coordinates outside image bounds are errors; validation never
clips or repairs them. Polygon export currently crops the polygon's
axis-aligned bounding rectangle; perspective rectification is not implemented.

## Text and missing values

The transcription is retained exactly as supplied, including case, Unicode
code points, punctuation, and leading/trailing or repeated spaces. Validation
rejects only values that are empty or consist entirely of whitespace; it does
not strip, transliterate, or Unicode-normalize labels. A JSON `null` is allowed
for optional language/script and region, but not for required fields.
Full-image/page text can be represented with `text_scope: "full_image"` and a
null region, but is not a valid TrOCR line-crop example. Export requires a
verified `line` record with explicit geometry and rejects/excludes full-image
text rather than guessing a region.

## Verification and export eligibility

Model-generated labels use `label_origin: "pseudo"` and normally
`verification_status: "unverified"`. Version 1 exports no pseudo-labels, even
if their status was manually changed; a later reviewed workflow must encode
auditable human verification before allowing that policy to change. Only rows
with `verification_status: "verified"`, a non-pseudo origin,
`license_status: "approved"`, `text_scope: "line"`, and valid geometry are
exported. Other rows in the selected split are counted by exclusion reason.

The exporter writes `<split>.csv` with the existing `image,text` contract,
lossless PNG crops under `crops/`, and `provenance.jsonl` mapping each crop to
its sample, original relative image, SHA-256, coordinates, split, and
annotation provenance. It does not modify source images. Outputs are written
only to the explicit `--output-dir`; existing target files require
`--overwrite`.
