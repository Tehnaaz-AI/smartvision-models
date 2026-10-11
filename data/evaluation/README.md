# Real-image OCR evaluation data

Place only medicine-label crops that you have permission to use in `images`.
Each image should contain one cropped line of printed text. Keep the synthetic
smoke-test image at `data/cropped/ocr_test.png` separate from this dataset.

Add one row per image to `references.csv`, with the exact, manually verified
printed transcription. The CSV has exactly these columns:

```csv
image,text
```

Image paths are relative to this directory, for example
`images\permitted_crop_001.png`. Do not add example labels, infer obscured text,
or use fabricated transcriptions. Verify spelling, strengths, units,
punctuation, and spacing against the source packaging. Keep a record of image
permissions and remove any image you are not authorized to use.

The CSV currently contains only its header. Evaluation metrics remain pending
until permitted real images and checked transcriptions are supplied.
