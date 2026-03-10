"""Evaluation metrics: AUROC and F1-max."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from typing import Optional


def compute_auroc(
    anomaly_scores: list[float],
    labels: list[int],
) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        anomaly_scores: Predicted anomaly scores (higher = more anomalous).
        labels:         Ground-truth labels (1 = anomaly, 0 = normal).

    Returns:
        AUROC in [0, 1].
    """
    scores = np.array(anomaly_scores)
    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def compute_f1_max(
    anomaly_scores: list[float],
    labels: list[int],
) -> float:
    """
    Compute F1-max: the maximum F1 score over all possible thresholds.

    This is the primary metric used in the paper alongside AUROC.

    Args:
        anomaly_scores: Predicted anomaly scores.
        labels:         Ground-truth binary labels.

    Returns:
        F1-max in [0, 1].
    """
    # scores = np.array(anomaly_scores)
    # y = np.array(labels)
    # precision, recall, thresholds = precision_recall_curve(y, scores)
    # f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    # return float(np.max(f1_scores))
    scores = np.array(anomaly_scores)
    y = np.array(labels)
    precision, recall, _ = precision_recall_curve(y, scores)
    denom = precision + recall
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return float(np.max(f1_scores))


def compute_binary_f1(
    predictions: list[bool],
    labels: list[int],
) -> float:
    """
    Compute F1 from binary predictions (used for threshold-free evaluation).

    Args:
        predictions: Boolean predictions (True = anomaly).
        labels:      Ground-truth binary labels (1 = anomaly).

    Returns:
        F1 score.
    """
    pred = [int(p) for p in predictions]
    return float(f1_score(labels, pred, zero_division=0))


def print_evaluation_summary(
    class_name: str,
    anomaly_scores: list[float],
    predictions: list[bool],
    labels: list[int],
    logger,
) -> dict:
    """
    Print and return a full evaluation summary for a class.

    Returns:
        Dict with auroc, f1_max, binary_f1.
    """
    auroc = compute_auroc(anomaly_scores, labels)
    f1_max = compute_f1_max(anomaly_scores, labels)
    bin_f1 = compute_binary_f1(predictions, labels)

    total = len(labels)
    n_anomaly = sum(labels)
    n_normal = total - n_anomaly

    text_to_print = (
        f"\n{'='*50}\n"
        f" Evaluation: {class_name}\n"
        f"{'='*50}\n"
        f"  Total:    {total} ({n_normal} normal, {n_anomaly} anomaly)\n"
        f"  AUROC:    {auroc:.4f}  (paper target: ~0.876)\n"
        f"  F1-max:   {f1_max:.4f}  (paper target: ~0.870)\n"
        f"  Bin-F1:   {bin_f1:.4f}\n"
        f"{'='*50}\n"
    )
    logger.log(text_to_print)
    print(text_to_print)

    return {"auroc": auroc, "f1_max": f1_max, "binary_f1": bin_f1}
