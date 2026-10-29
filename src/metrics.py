"""Binary detection and localization metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


def compute_metrics(labels, scores, threshold: float = 0.5, *, select_threshold: bool = False) -> dict:
    """AUPR means trapezoidal PR area; AP is also reported separately.

    Undefined rank metrics for single-class inputs are represented by ``None``.
    With ``select_threshold=True``, ``best_f1`` is F1 at the threshold maximizing
    TPR - FPR on these labels and scores. Equal values select the highest
    threshold; tied scores enter the ROC curve together.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and scores must be non-empty aligned vectors.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores.")
    if not ((scores >= 0) & (scores <= 1)).all() or not 0 <= threshold <= 1:
        raise ValueError("Probabilities and threshold must lie in [0, 1].")
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0)
    both_classes = np.unique(labels).size == 2
    if both_classes:
        p, r, _ = precision_recall_curve(labels, scores)
        aupr = float(auc(r, p))
        auroc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    else:
        aupr = auroc = ap = None
    metrics = {
        "n": int(len(labels)), "n_positive": int(labels.sum()),
        "aupr": aupr, "average_precision": ap, "auroc": auroc,
        "f1": float(f1), "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision), "recall": float(recall), "threshold": threshold,
    }
    if select_threshold:
        best_f1 = None
        if both_classes:
            fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
            cutoff = float(thresholds[np.argmax(tpr - fpr)])
            if not np.isfinite(cutoff):
                cutoff = float(scores.max()) + 1.0
            _, _, best_f1, _ = precision_recall_fscore_support(
                labels, scores >= cutoff, average="binary", zero_division=0)
            best_f1 = float(best_f1)
        metrics["best_f1"] = best_f1
    return metrics
