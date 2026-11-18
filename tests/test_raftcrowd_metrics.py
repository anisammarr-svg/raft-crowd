"""Synthetic analytical checks, not experimental evidence for the paper."""

import numpy as np
import pytest

from raftcrowd.metrics import (
    eer,
    evaluate,
    fixed_metrics,
    roc_auc,
    roc_curve,
    select_threshold,
)


def test_known_auc_and_curve():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, thresholds = roc_curve(y, s)
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    np.testing.assert_allclose(thresholds[1:], [0.8, 0.4, 0.35, 0.1])
    assert np.isposinf(thresholds[0])
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert eer(y, s) == pytest.approx(0.5)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2], [2, 0, 3, 1]])
def test_score_ties_are_order_invariant(order):
    y = np.array([0, 1, 0, 1])[order]
    s = np.array([0.2, 0.2, 0.8, 0.8])[order]
    assert roc_auc(y, s) == pytest.approx(0.5)
    assert eer(y, s) == pytest.approx(0.5)
    assert len(roc_curve(y, s)[0]) == 3


@pytest.mark.parametrize(
    "scores,expected_auc,expected_eer",
    [([0, 1], 1, 0), ([1, 0], 0, 1), ([0.5, 0.5], 0.5, 0.5)],
)
def test_extreme_and_constant_predictions(scores, expected_auc, expected_eer):
    assert roc_auc([0, 1], scores) == pytest.approx(expected_auc)
    assert eer([0, 1], scores) == pytest.approx(expected_eer)


def test_eer_interpolates_between_curve_points():
    # Descending labels 1,0,1,1 gives a crossing between (FPR=0,TPR=1/3)
    # and (FPR=1,TPR=1/3), so EER is 2/3.
    assert eer([1, 0, 1, 1], [4, 3, 2, 1]) == pytest.approx(2 / 3)


def test_fixed_metrics_threshold_is_inclusive_and_counts_are_exact():
    result = fixed_metrics([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9], 0.5)
    assert result == {
        "precision": 2 / 3,
        "recall": 1,
        "f1": 0.8,
        "tp": 2,
        "fp": 1,
        "tn": 1,
        "fn": 0,
    }


def test_zero_denominators_are_defined():
    result = fixed_metrics([0, 0], [0.1, 0.2], 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["tn"] == 2


def test_validation_threshold_selection_and_high_tie_break():
    # Descending labels 1,0,0,1: F1=2/3 after the first and fourth rows.
    y = [1, 0, 0, 1]
    s = [0.9, 0.8, 0.7, 0.6]
    assert select_threshold(y, s) == 0.9
    assert select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.35


def test_evaluate_does_not_optimize_supplied_threshold():
    result = evaluate([0, 1], [0, 1], 2)
    assert result["roc_auc"] == 1
    assert result["f1"] == 0
    assert result["fn"] == 1


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),
        ([0, 1], [1]),
        ([[0, 1]], [[0.1, 0.2]]),
        ([0, 2], [0.1, 0.2]),
        ([0, np.nan], [0.1, 0.2]),
        ([0, 1], [0.1, np.inf]),
        ([0, 1], [np.nan, 0.2]),
        (["0", "1"], [0.1, 0.2]),
        ([0, 1], [0.1 + 1j, 0.2]),
    ],
)
def test_invalid_arrays_rejected_by_all_entry_points(labels, scores):
    for method in [roc_curve, roc_auc, eer, select_threshold]:
        with pytest.raises(ValueError):
            method(labels, scores)
    with pytest.raises(ValueError):
        fixed_metrics(labels, scores, 0.5)


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_curve_and_threshold_selection_require_both_classes(labels):
    for method in [roc_curve, roc_auc, eer, select_threshold]:
        with pytest.raises(ValueError, match="both normal"):
            method(labels, [0.1, 0.2])


@pytest.mark.parametrize("threshold", [np.inf, np.nan, [0.5], "0.5", 0.5 + 1j])
def test_nonfinite_or_nonscalar_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        fixed_metrics([0, 1], [0.1, 0.2], threshold)
