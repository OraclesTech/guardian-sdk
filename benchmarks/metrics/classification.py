"""
Classification metrics for Guardian SDK benchmark suite.

Computes: precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, accuracy.
All metrics follow sklearn conventions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Full classification metric report for one dataset × one runner."""

    dataset: str
    runner: str
    n_samples: int
    n_attacks: int
    n_benign: int

    # Core classification
    precision: float = 0.0
    recall: float = 0.0        # sensitivity / TPR
    f1: float = 0.0
    accuracy: float = 0.0

    # Threshold-agnostic
    roc_auc: float = 0.0
    pr_auc: float = 0.0        # precision-recall AUC (better for imbalanced data)

    # Operational (critical for production deployment decisions)
    fpr: float = 0.0           # false positive rate (1 - specificity)
    fnr: float = 0.0           # false negative rate (miss rate)
    tnr: float = 0.0           # true negative rate (specificity)

    # Raw confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # Calibration
    threshold_used: float = 0.5
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "runner": self.runner,
            "n_samples": self.n_samples,
            "n_attacks": self.n_attacks,
            "n_benign": self.n_benign,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "tnr": round(self.tnr, 4),
            # Flat access for Jinja2 template (ds.tp, ds.fp, etc.)
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            # Also nested for JSON consumers that prefer structure
            "confusion_matrix": {
                "tp": self.tp, "fp": self.fp,
                "tn": self.tn, "fn": self.fn,
            },
            "threshold_used": self.threshold_used,
            "warnings": self.warnings,
        }


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
    dataset: str = "",
    runner: str = "",
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """
    Compute classification metrics from predictions and ground truth.

    Args:
        y_true:   Ground-truth labels (1=attack, 0=benign).
        y_pred:   Binary predictions (1=detected threat, 0=clean).
        y_scores: Continuous threat scores [0, 1] — used for AUC.
                  If None, uses y_pred as scores.
        dataset:  Dataset name (for labeling the result).
        runner:   Runner name (for labeling the result).
        threshold: Threshold used for binary prediction.

    Returns:
        ClassificationMetrics with all computed values.
    """
    warnings = []
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    n_samples = len(y_true_arr)
    n_attacks = int(y_true_arr.sum())
    n_benign = n_samples - n_attacks

    if n_samples == 0:
        warnings.append("Empty dataset — no metrics computed")
        return ClassificationMetrics(
            dataset=dataset, runner=runner,
            n_samples=0, n_attacks=0, n_benign=0,
            warnings=warnings,
        )

    if n_attacks == 0:
        warnings.append("No positive (attack) samples — recall/FNR undefined")
    if n_benign == 0:
        warnings.append("No negative (benign) samples — FPR undefined")

    # Confusion matrix
    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall / TPR = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Accuracy
    accuracy = (tp + tn) / n_samples

    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FNR = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # TNR = TN / (TN + FP)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC metrics — need scikit-learn
    roc_auc = 0.0
    pr_auc = 0.0
    scores = y_scores if y_scores is not None else [float(p) for p in y_pred]

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: PLC0415

        if n_attacks > 0 and n_benign > 0:
            roc_auc = float(roc_auc_score(y_true_arr, scores))
            pr_auc = float(average_precision_score(y_true_arr, scores))
        else:
            warnings.append("AUC requires both positive and negative samples")
    except ImportError:
        warnings.append("scikit-learn not available — AUC metrics skipped")

    metrics = ClassificationMetrics(
        dataset=dataset,
        runner=runner,
        n_samples=n_samples,
        n_attacks=n_attacks,
        n_benign=n_benign,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr=fpr,
        fnr=fnr,
        tnr=tnr,
        tp=tp, fp=fp, tn=tn, fn=fn,
        threshold_used=threshold,
        warnings=warnings,
    )

    logger.info(
        "[%s / %s] P=%.3f R=%.3f F1=%.3f AUC=%.3f FPR=%.3f FNR=%.3f",
        dataset, runner, precision, recall, f1, roc_auc, fpr, fnr,
    )
    return metrics
