"""
sEMG Prosthetic Gesture Classification
Module: ml.fold_metrics

Computes and saves performance metrics for individual LOSO validation folds.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    inference_time: float,
    throughput: float
) -> Dict[str, float]:
    """
    Compute classification performance metrics for a single fold.

    Parameters
    ----------
    y_true : np.ndarray
        True gesture labels.
    y_pred : np.ndarray
        Predicted gesture labels.
    train_time : float
        Training duration in seconds.
    inference_time : float
        Inference duration in seconds.
    throughput : float
        Inference throughput in samples/second.

    Returns
    -------
    dict
        A mapping of metric names to calculated scores.
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Cohen Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Training Time (s)": float(train_time),
        "Inference Time (s)": float(inference_time),
        "Prediction Throughput (sps)": float(throughput)
    }

def save_fold_checkpoint(
    save_dir: Path,
    model_name: str,
    subject_id: int,
    metrics: Dict[str, float],
    predictions: np.ndarray,
    true_labels: np.ndarray
) -> None:
    """
    Save fold results (metrics, predictions, labels) to disk.

    Parameters
    ----------
    save_dir : Path
        Directory to store the fold checkpoints.
    model_name : str
        Name of the classifier model.
    subject_id : int
        Unseen subject ID for this fold.
    metrics : dict
        Fold-wise performance metrics.
    predictions : np.ndarray
        Predicted labels.
    true_labels : np.ndarray
        True labels.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = save_dir / f"metrics_{model_name}_sub{subject_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Save predictions array
    pred_path = save_dir / f"predictions_{model_name}_sub{subject_id}.npz"
    np.savez_compressed(pred_path, y_pred=predictions, y_true=true_labels)

def load_fold_checkpoint(
    save_dir: Path,
    model_name: str,
    subject_id: int
) -> Union[None, Dict[str, Any]]:
    """
    Load fold results from disk if they exist.

    Parameters
    ----------
    save_dir : Path
        Directory containing fold checkpoints.
    model_name : str
        Name of the classifier model.
    subject_id : int
        Unseen subject ID for this fold.

    Returns
    -------
    dict or None
        Fold data dictionary containing metrics and predictions, or None if not found.
    """
    metrics_path = save_dir / f"metrics_{model_name}_sub{subject_id}.json"
    pred_path = save_dir / f"predictions_{model_name}_sub{subject_id}.npz"
    
    if metrics_path.exists() and pred_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        data = np.load(pred_path)
        return {
            "metrics": metrics,
            "y_pred": data["y_pred"],
            "y_true": data["y_true"]
        }
    return None
