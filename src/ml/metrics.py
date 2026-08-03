"""
sEMG Prosthetic Gesture Classification
Module: ml.metrics

Computes standard scientific evaluation metrics for multi-class classification.
"""

from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute standard evaluation metrics for gesture classification.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        A dictionary containing computed metrics:
        - accuracy
        - balanced_accuracy
        - precision_macro
        - recall_macro
        - f1_macro
        - mcc (Matthews Correlation Coefficient)
        - cohen_kappa
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))
    }
