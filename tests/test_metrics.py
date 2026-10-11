import pytest

from smartvision_ocr.utils.metrics import (
    compute_pair_metrics,
    compute_text_metrics,
)


def test_text_metrics_perfect_predictions():
    assert compute_text_metrics(["one line", "second line"], ["one line", "second line"]) == {
        "cer": 0.0,
        "wer": 0.0,
        "exact_match_rate": 1.0,
        "exact_match_rate_case_insensitive": 1.0,
    }


def test_text_metrics_measure_errors():
    metrics = compute_text_metrics(["cat"], ["cut"])
    assert metrics["cer"] == pytest.approx(1 / 3)
    assert metrics["wer"] == 1.0


def test_text_metrics_reject_mismatched_counts():
    with pytest.raises(ValueError, match="count mismatch"):
        compute_text_metrics(["reference"], [])


def test_text_metrics_reject_empty_collections():
    with pytest.raises(ValueError, match="At least one"):
        compute_text_metrics([], [])


def test_pair_metrics_exact_match_and_case_only_difference():
    assert compute_pair_metrics("500 mg!", "500 mg!") == {
        "cer": 0.0,
        "wer": 0.0,
        "exact_match": True,
        "exact_match_case_insensitive": True,
    }

    case_only = compute_pair_metrics("Paracetamol 500 mg", "PARACETAMOL 500 MG")
    assert case_only["exact_match"] is False
    assert case_only["exact_match_case_insensitive"] is True


@pytest.mark.parametrize(
    ("reference", "prediction", "cer", "wer"),
    [
        ("abc", "abXc", 1 / 3, 1.0),
        ("abc", "ac", 1 / 3, 1.0),
        ("abc", "axc", 1 / 3, 1.0),
        ("one tablet", "one extra tablet", 6 / 10, 1 / 2),
    ],
)
def test_pair_metrics_insertions_deletions_and_substitutions(
    reference, prediction, cer, wer
):
    metrics = compute_pair_metrics(reference, prediction)
    assert metrics["cer"] == pytest.approx(cer)
    assert metrics["wer"] == pytest.approx(wer)


@pytest.mark.parametrize(
    ("reference", "prediction", "cer", "wer", "exact"),
    [
        ("", "", 0.0, 0.0, True),
        ("", "mg 500", 6.0, 2.0, False),
        ("mg 500", "", 1.0, 1.0, False),
    ],
)
def test_pair_metrics_empty_strings(reference, prediction, cer, wer, exact):
    metrics = compute_pair_metrics(reference, prediction)
    assert metrics["cer"] == pytest.approx(cer)
    assert metrics["wer"] == pytest.approx(wer)
    assert metrics["exact_match"] is exact
    assert metrics["exact_match_case_insensitive"] is exact


def test_case_insensitive_exact_match_preserves_numbers_and_punctuation():
    assert compute_pair_metrics("Drug 500 mg!", "DRUG 500 mg!")[
        "exact_match_case_insensitive"
    ]
    assert not compute_pair_metrics("Drug 500 mg!", "DRUG 50 mg!")[
        "exact_match_case_insensitive"
    ]


def test_aggregate_metrics_include_exact_match_rates():
    metrics = compute_text_metrics(
        ["Drug 500 mg", "Tablet!"],
        ["DRUG 500 MG", "Tablet?"],
    )

    assert metrics["exact_match_rate"] == 0.0
    assert metrics["exact_match_rate_case_insensitive"] == 0.5
