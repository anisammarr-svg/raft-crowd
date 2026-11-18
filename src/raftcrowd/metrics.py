"""Frame-level binary metrics with explicit, validation-only threshold selection.

Higher scores mean more anomalous. No test-set threshold optimization is performed
by :func:`evaluate`; its caller must supply a previously frozen threshold.
"""

from __future__ import annotations

import numpy as np


def _validated(labels, scores, *, require_both: bool = False):
    y = np.asarray(labels)
    s = np.asarray(scores)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or not y.size:
        raise ValueError("labels and scores must be matching nonempty 1-D arrays")
    if y.dtype.kind not in "biuf" or s.dtype.kind not in "biuf":
        raise ValueError("labels and scores must be real numeric arrays")
    if not np.all(np.isfinite(y)) or not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must contain only binary 0/1 values")
    s = s.astype(np.float64, copy=False)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    y = y.astype(np.int64, copy=False)
    if require_both and (not np.any(y == 0) or not np.any(y == 1)):
        raise ValueError("both normal (0) and anomalous (1) labels are required")
    return y, s


def _counts_at_distinct_scores(y: np.ndarray, s: np.ndarray):
    # Keeping tied scores together makes ROC/AUC independent of input row order.
    order = np.argsort(-s, kind="stable")
    ranked = s[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    fp = ends + 1 - tp
    return tp, fp, ranked[ends]


def roc_curve(labels, scores):
    """Return ``(fpr, tpr, thresholds)`` with ties grouped exactly.

    The initial threshold is +inf, representing no positive predictions. This
    sentinel is for the curve only, not an admissible operational threshold.
    Both label classes must be present.
    """
    y, s = _validated(labels, scores, require_both=True)
    tp, fp, thresholds = _counts_at_distinct_scores(y, s)
    tpr = np.r_[0.0, tp / np.count_nonzero(y == 1)]
    fpr = np.r_[0.0, fp / np.count_nonzero(y == 0)]
    return fpr, tpr, np.r_[np.inf, thresholds]


def roc_auc(labels, scores) -> float:
    """Trapezoidal ROC AUC; a tied positive/negative pair contributes 0.5."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))


def eer(labels, scores) -> float:
    """Equal-error rate from linear interpolation of the empirical ROC.

    Interpolates between neighboring curve points where FPR and FNR cross.
    This is a descriptive curve statistic, not a deployed alarm threshold.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    index = int(np.flatnonzero(difference >= 0)[0])
    if difference[index] == 0:
        return float(fpr[index])
    left = index - 1
    fraction = -difference[left] / (difference[index] - difference[left])
    return float(fpr[left] + fraction * (fpr[index] - fpr[left]))


def fixed_metrics(labels, scores, threshold) -> dict:
    """Precision/recall/F1 and counts at ``score >= threshold``.

    Zero-denominator precision, recall and F1 are defined to be zero. Single-class
    labels are allowed here, unlike for ROC AUC and EER.
    """
    y, s = _validated(labels, scores)
    value = np.asarray(threshold)
    if value.ndim != 0 or value.dtype.kind not in "biuf" or not np.isfinite(value):
        raise ValueError("threshold must be a finite real scalar")
    predicted = s >= float(value)
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def select_threshold(labels, scores) -> float:
    """Maximize validation F1, choosing the highest score threshold on ties.

    Call only with the designated validation set. Both classes are required.
    The function cannot determine a split's provenance: callers must enforce it.
    Candidate thresholds are the distinct observed scores. With at least one
    positive label, a threshold predicting nothing cannot improve maximum F1.
    """
    y, s = _validated(labels, scores, require_both=True)
    tp, fp, thresholds = _counts_at_distinct_scores(y, s)
    fn = int(np.count_nonzero(y == 1)) - tp
    f1 = 2.0 * tp / (2 * tp + fp + fn)
    # thresholds are descending; argmax's first occurrence resolves ties high.
    return float(thresholds[int(np.argmax(f1))])


def evaluate(labels, scores, threshold) -> dict:
    """Evaluate one previously chosen threshold plus ROC AUC and EER."""
    result = fixed_metrics(labels, scores, threshold)
    result.update(roc_auc=roc_auc(labels, scores), eer=eer(labels, scores))
    return result
