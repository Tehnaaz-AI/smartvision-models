"""Text-recognition metrics without discarding meaningful text content."""

def _edit_distance(reference: list[str], prediction: list[str]) -> int:
    """Return Levenshtein distance using O(min(n, m)) auxiliary memory."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[prediction_index] + 1,
                    previous_row[prediction_index - 1]
                    + (reference_item != prediction_item),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _pair_metrics(reference: str, prediction: str) -> dict[str, float | bool]:
    character_errors = _edit_distance(list(reference), list(prediction))
    word_errors = _edit_distance(reference.split(), prediction.split())
    return {
        "cer": character_errors / max(len(reference), 1),
        "wer": word_errors / max(len(reference.split()), 1),
        "exact_match": reference == prediction,
        "exact_match_case_insensitive": reference.casefold() == prediction.casefold(),
    }


def compute_text_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    """Return corpus CER/WER and exact-match rates for paired transcriptions.

    CER compares original characters. WER splits on whitespace but retains
    punctuation and all token content. A zero-length reference uses a
    denominator of one so insertion-only errors, including empty-string pairs,
    remain defined rather than being discarded.
    """
    if not references:
        raise ValueError("At least one reference/prediction pair is required.")
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction count mismatch: {len(references)} != "
            f"{len(predictions)}."
        )
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0
    exact_matches = 0
    case_insensitive_matches = 0
    for reference, prediction in zip(references, predictions):
        character_errors += _edit_distance(list(reference), list(prediction))
        reference_characters += len(reference)
        word_errors += _edit_distance(reference.split(), prediction.split())
        reference_words += len(reference.split())
        exact_matches += int(reference == prediction)
        case_insensitive_matches += int(reference.casefold() == prediction.casefold())

    return {
        "cer": character_errors / max(reference_characters, 1),
        "wer": word_errors / max(reference_words, 1),
        "exact_match_rate": exact_matches / len(references),
        "exact_match_rate_case_insensitive": case_insensitive_matches / len(references),
    }


def compute_pair_metrics(reference: str, prediction: str) -> dict[str, float | bool]:
    """Return CER, WER, and exact-match indicators for one image."""
    return _pair_metrics(reference, prediction)
