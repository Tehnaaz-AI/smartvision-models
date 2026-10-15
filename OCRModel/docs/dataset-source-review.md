# OCR dataset source and licensing review

**Reviewed:** 2026-10-12
**Scope:** Source-page research only. No candidate dataset was downloaded, and no
application code, source data, or prediction files were changed.

## Decision summary

There is no candidate cleared by this review for project use. The best
language/domain match to investigate first is the **real IndicSTR12** release:
CVIT describes real scene-text word images with Unicode labels across Telugu,
Hindi, and ten other Indian languages. Its official page links downloads but
does not state a dataset license or project-use permission. Obtain written
permission or authoritative terms from CVIT before downloading or using it.

For a small Telugu recognition-only evaluation, the
[Telugu Wikisource line benchmark](https://huggingface.co/datasets/harsha-desaraju/telugu-line-ocr-bench)
is a promising *conditional* candidate: its uploader describes 1,044 human-
checked Telugu line crops and declares CC BY-SA 4.0. This is a third-party
release; the underlying page scans and text need source-level rights and
provenance review before treating the dataset-wide license claim as sufficient.
It is not a scene-text or packaging benchmark.

Do not treat public availability, paper publication, a repository license, or
a Hugging Face `license` tag as proof that the images and annotations are
cleared for SmartVision project use. The existing local corpus findings and
manifest gates remain as documented in
[dataset-audit.md](dataset-audit.md) and
[canonical-manifest.md](canonical-manifest.md).

## Candidate comparison

| Candidate and source | Languages/scripts and domain | Labels and geometry | Size and access | Rights and project-use assessment | Fit and recommendation |
|---|---|---|---|---|---|
| **IndicSTR12 real** — [official CVIT project page](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr), [paper](https://link.springer.com/chapter/10.1007/978-3-031-41498-5_17), [official code repository](https://github.com/harshlunia7/IndicSTR12_ICDAR) | Official page lists Assamese, Bengali, Odia, Marathi, Hindi, Kannada, Urdu, Telugu, Malayalam, Tamil, Gujarati, and Punjabi. Real images were collected from Google Images and include signboards, shop nameplates, stations, advertisements, buildings, and other everyday scene text. | The official page describes real **word images** with corresponding Unicode labels; it does not describe full-scene bounding-box/polygon files for this release. Suitable recognizer pairs need confirmation against the exact archive and annotations. | At least 1,000 real word images per listed language, as stated by CVIT. The page links direct `real.zip` and `synthetic.tar.gz` downloads. No registration requirement is stated on the page; actual access conditions were not independently tested. | **Unresolved.** The official page does not publish license terms or state commercial/project permissions. Direct download links do not establish permission. A user-uploaded mirror defers to CVIT for licensing. | **First source to seek permission for.** Strongest real-image Telugu/Hindi scene-word match. Word crops and language variety are useful for recognition, but it is not a substitute for full-scene detection data or document/packaging coverage. |
| **IndicSTR12 user-curated signboard subset** — [dataset card](https://huggingface.co/datasets/jai23023/indicstr12-signboards) | Hindi, Kannada, Marathi, Telugu signboards, according to the uploader. | Card describes signboard images and a space-joined transcription assembled from source per-word polygons. This is a full-image transcript format; do not use it as a TrOCR line label without valid word/line alignment. | The card reports 975 records (877 train, 98 validation), including 298 Telugu and 173 Hindi records. Hugging Face API metadata reports it as public and not gated. These are uploader/platform claims, not counts independently checked from the files. | Card metadata declares CC BY 4.0. **Not cleared.** This is a downstream user’s license claim for a subset derived from IndicSTR12; the original CVIT terms and authority to relicense the source images were not established. | Useful lead for locating a small subset, not an approved source. Avoid using its aggregated signboard transcription as a crop-paired label. Prefer original source permission and per-word annotations. |
| **Telugu Wikisource OCR line benchmark** — [dataset card](https://huggingface.co/datasets/harsha-desaraju/telugu-line-ocr-bench) | Telugu script, numerals, and punctuation; grayscale scanned book text rather than scene text, according to the uploader. | Uploader describes 1,044 single-line crops with `image,text`, human-checked labels, plus source/page/line provenance. These claims have not been independently checked against the data. | Card reports 1,044 examples in a test split; Hugging Face API metadata says public and not gated. File size is about 15 MB in API metadata; not downloaded. | Card declares CC BY-SA 4.0 and says text inherits Wikisource terms. The [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/) permits commercial sharing/adaptation subject to attribution and share-alike, but that is only useful if the uploader had authority for all included images and text. Per-book and scan provenance remain to be confirmed. | **Conditional Telugu evaluation candidate** for scanned-document line recognition only. Do not use as a general-purpose scene benchmark or training source until row-level source rights and label quality are verified. |
| **Telugu Wikisource page OCR** — [third-party dataset card](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr) | Telugu scanned books/documents, as represented by the card. | Card describes page/text pairs rather than verified line crops; page-level text does not directly satisfy TrOCR's line-image contract. | Uploader claims about 25,565 page/text pairs from 221 books and approximately 11 GB. Not independently verified and not downloaded. | Card claims CC BY-SA 4.0. **Unresolved at the source-work level.** Verify individual books, page scans, and redistribution terms rather than relying on the blanket downstream claim. | Not suitable for direct TrOCR export without a validated segmentation/alignment process. Possible future document OCR resource after rights and page-to-line alignment review. |
| **ICDAR 2019 MLT** — [official RRC overview](https://rrc.cvc.uab.es/?ch=15&com=introduction), [official downloads page](https://rrc.cvc.uab.es/?ch=15&com=downloads), [challenge paper](https://arxiv.org/abs/1907.00945) | Official RRC page describes multilingual text detection, recognition, and script identification in natural environments. The exact language/script inventory was not established from the official pages reviewed here; do not assume Telugu or Hindi is included. | Scene-text detection/recognition challenge with text-region geometry and recognition annotations described by the challenge. Exact annotation details should be checked against its paper and release before selecting tasks. | Official downloads page explicitly requires RRC portal registration to access downloads. The portal says registration expresses interest and does not imply participation. | **Unresolved.** Registration is required; the pages reviewed did not establish a dataset license or commercial/project-use permission. Competition access is not a license. | Potential multilingual scene-text source only after confirming Hindi/Telugu coverage, label formats, and terms. Not a Telugu/Hindi choice on current evidence. |
| **TextOCR** — [official project page](https://textvqa.org/textocr/), [paper](https://arxiv.org/abs/2105.05486) | Paper describes real scene images from TextVQA. It is a scene-text resource, but the pages reviewed did not establish support for Telugu or Hindi; do not use it to claim Indic coverage. | Paper describes arbitrary-shaped scene text with word-level transcriptions and region annotations. | Paper reports 900,000 annotated words. The project page is client-rendered; a dataset-specific download gate or terms were not confirmed in this review. | **Unresolved.** The related [TextVQA GitHub repository](https://github.com/facebookresearch/TextVQA) has a CC BY-NC 4.0 repository license, but that alone does not establish the license of TextOCR data or the underlying photographs. Do not treat that repository license as a dataset grant. | Candidate for English scene-text recognition/detection experiments after data terms and image rights are confirmed. No Telugu/Hindi evidence, so not the regional-language priority. |
| **COCO-Text / COCO-Text V2** — [official V2 project page](https://bgshih.github.io/cocotext/), [paper](https://arxiv.org/abs/1601.07140), [MS COCO terms](https://cocodataset.org/#termsofuse) | Natural images from MS COCO; the research paper discusses text occurrence, legibility, machine-printed/handwritten attributes, and script attributes. The resource should be treated as English-oriented unless its released annotations demonstrate otherwise; no Telugu/Hindi coverage was established. | Text-region boxes and transcriptions/legibility annotations are described in the paper. The official V2 webpage resolved to a page titled “COCO-Text V2.0” but did not expose detailed release or rights text in the page retrieval. | Exact V2 size and download/access requirements were not verified. Do not rely on the original paper's size as a verified V2 release count. | **Unresolved.** No dataset-specific license was confirmed from the official V2 page. Since the images derive from MS COCO, inspect per-image source/license metadata and the current COCO terms; do not assume a single blanket permission for all images or annotations. | Possible supplementary English natural-scene benchmark after confirming V2 contents and per-image rights. Does not close Telugu/Hindi gaps. |
| **SynthText** — [official research code repository](https://github.com/ankush-me/SynthText), [paper](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) | Synthetic English scene text produced by rendering text onto background photographs. The repository mentions community adaptations for other scripts; that does not establish Telugu/Hindi availability or quality in the original release. | Generated text labels and region/segmentation/depth information are described by the repository. | Repository reports approximately 800,000 generated images in a torrent and documents smaller renderer assets. Not downloaded. | The repository's `LICENSE` file is CC BY-NC 4.0. Its README explicitly says the authors do not own copyright to the background images. Thus commercial/project use is **not cleared**; the code license does not resolve background-image or generated-dataset rights. | Could support synthetic-only pipeline experiments after rights review, but not real-image evaluation. English-focused; not a substitute for manually verified Telugu/Hindi data. |

## TrOCR language suitability

The current recognizer is
[microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed),
documented as a pretrained printed-text OCR checkpoint. Its published task
and model card do not establish Telugu- or Hindi-script recognition quality.
The fact that a tokenizer can encode Unicode code points is not evidence that
the image encoder/decoder was trained to recognize those scripts. There is no
verified Telugu/Hindi benchmark result in this repository. Treat it as an
English baseline only until a held-out, human-verified Telugu and Hindi
evaluation demonstrates otherwise. Select or train a regional-script
recognizer only after checking the model's actual training scripts, tokenizer,
license, and measured performance against verified references.

## Verified facts, claims, and unresolved checks

### Verified from primary pages reviewed

- CVIT's IndicSTR12 page explicitly lists 12 languages including Telugu and
  Hindi, describes real word images with Unicode labels, provides a minimum
  per-language image count, and links real and synthetic archives.
- The IndicSTR12 page does not state a dataset license or project-use
  permission.
- ICDAR RRC's MLT downloads page requires portal registration to access
  downloads; its overview describes a multilingual natural-scene challenge.
- The TextOCR paper reports 900,000 annotated words on real TextVQA images.
- The COCO-Text paper describes text regions, transcriptions, and text
  attributes; the official V2 page was reachable but exposed only its title in
  the available page retrieval.
- SynthText's repository reports approximately 800,000 synthetic images and
  explicitly says the authors do not own background-image copyright. The
  repository license file contains CC BY-NC 4.0 terms.
- The Telugu line benchmark card describes 1,044 human-checked line crops and
  declares CC BY-SA 4.0; the indicated verification and inherited-license
  statements are uploader claims, not independently validated source facts.

### Not established in this review

- Whether CVIT permits use of IndicSTR12 for this college project, research
  demonstrations, or any later commercial deployment.
- Whether the downstream IndicSTR12 subset's CC BY 4.0 claim is authorized by
  the original rights holders.
- Whether all Telugu Wikisource benchmark images and texts are properly
  covered by the declared license, and whether source-specific attribution or
  share-alike obligations apply.
- The exact language list, release structure, and data-use terms of ICDAR 2019
  MLT from the released files.
- TextOCR-specific data licensing, access requirements, image rights, and
  Indic-script coverage.
- COCO-Text V2's exact release contents, language distribution, download
  procedure, and rights for every underlying COCO image.
- IndicSTR12's exact real/synthetic archive contents and archive-level
  redistribution terms.
- Any adequate Telugu/Hindi performance for the existing TrOCR checkpoint.

## Recommended next decisions

1. Ask CVIT for the official IndicSTR12 real-dataset terms and explicit
   permission boundaries (research use, project demonstrations, redistribution,
   and commercial deployment). Do not use the synthetic archive as a substitute
   for real evaluation data.
2. If evaluating document Telugu is a near-term priority, ask the Telugu line
   benchmark uploader for source-book/page provenance and rights evidence, then
   verify a small sample of image/transcription pairs and preserve attribution
   and share-alike obligations if applicable.
3. If an English scene-text baseline is needed, resolve COCO-Text V2's release
   and per-image COCO rights, or TextOCR's exact data terms, before selecting
   either. MLT additionally requires registration and language-list
   confirmation.
4. Record each approved source's exact release/version, license evidence,
   image-level terms where applicable, and permitted uses in the source
   registry before running the existing manifest validator/exporter. Keep
   unknown or pseudo-labeled records out of verified training/evaluation.
5. Before selecting a Telugu/Hindi model, define a small manually verified
   line-crop evaluation set with source permission, then compare candidate
   models on the same held-out examples. Do not infer script ability from
   tokenizer coverage.

## Sources checked

- [CVIT IndicSTR12 project and download links](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr)
- [IndicSTR12 paper](https://link.springer.com/chapter/10.1007/978-3-031-41498-5_17) and [official code repository](https://github.com/harshlunia7/IndicSTR12_ICDAR)
- [IndicSTR12 cropped-word Hugging Face mirror](https://huggingface.co/datasets/darknight054/indicstr12-crops); its card explicitly refers users to the original source for licensing
- [User-curated IndicSTR12 signboard subset](https://huggingface.co/datasets/jai23023/indicstr12-signboards)
- [Telugu Wikisource line benchmark](https://huggingface.co/datasets/harsha-desaraju/telugu-line-ocr-bench) and [Telugu page OCR card](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr)
- [ICDAR 2019 MLT overview](https://rrc.cvc.uab.es/?ch=15&com=introduction), [downloads](https://rrc.cvc.uab.es/?ch=15&com=downloads), and [challenge paper](https://arxiv.org/abs/1907.00945)
- [TextOCR project](https://textvqa.org/textocr/) and [paper](https://arxiv.org/abs/2105.05486)
- [COCO-Text V2 project](https://bgshih.github.io/cocotext/), [COCO-Text paper](https://arxiv.org/abs/1601.07140), and [MS COCO terms page](https://cocodataset.org/#termsofuse)
- [SynthText repository and README](https://github.com/ankush-me/SynthText), [repository license](https://github.com/ankush-me/SynthText/blob/master/LICENSE), and [project paper page](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- [TrOCR Base Printed model card](https://huggingface.co/microsoft/trocr-base-printed)
- [Creative Commons Attribution-ShareAlike 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/)
