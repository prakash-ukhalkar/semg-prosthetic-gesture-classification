"""
sEMG Prosthetic Gesture Classification
Module: ml.evaluation

Evaluates models, tracks inference statistics, and saves predictions, metrics, and metadata.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
import sklearn

from src.ml.metrics import compute_metrics

logger = logging.getLogger("semg_prosthetic_classification")

def evaluate_classifier(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str,
    feature_set_name: str,
    save_dir: Path,
    train_time: float = 0.0,
    memory_used_mb: float = 0.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a trained model pipeline on validation and test sets.
    Computes performance metrics, inference speed, and saves evaluation outputs.

    Parameters
    ----------
    model : Pipeline
        The trained scikit-learn Pipeline.
    X_val : pd.DataFrame
        Validation features.
    y_val : np.ndarray
        Validation labels.
    X_test : pd.DataFrame
        Test features.
    y_test : np.ndarray
        Test labels.
    model_name : str
        The classifier identifier.
    feature_set_name : str
        The feature set identifier (e.g., 'top25', 'top50').
    save_dir : Path
        Directory to save results.
    train_time : float, default=0.0
        The recorded training duration in seconds.
    memory_used_mb : float, default=0.0
        The recorded memory usage in MB.
    random_state : int, default=42
        The random seed used.

    Returns
    -------
    dict
        Consolidated metrics report including val/test performance, timings, and throughput.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = save_dir / f"metrics_{feature_set_name}_{model_name}.json"
    predictions_path = save_dir / f"predictions_{feature_set_name}_{model_name}.parquet"
    metadata_path = save_dir / f"metadata_{feature_set_name}_{model_name}.json"
    
    # 1. Validation set inference
    logger.info(f"Running validation inference for '{model_name}' on '{feature_set_name}'...")
    start_time = time.perf_counter()
    y_pred_val = model.predict(X_val)
    val_inference_time = time.perf_counter() - start_time
    val_throughput = len(X_val) / val_inference_time if val_inference_time > 0 else 0.0
    val_metrics = compute_metrics(y_val, y_pred_val)
    
    # 2. Test set inference
    logger.info(f"Running test inference for '{model_name}' on '{feature_set_name}'...")
    start_time = time.perf_counter()
    y_pred_test = model.predict(X_test)
    test_inference_time = time.perf_counter() - start_time
    test_throughput = len(X_test) / test_inference_time if test_inference_time > 0 else 0.0
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    # 3. Consolidate results
    results = {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "train_time_sec": train_time,
        "memory_used_mb": memory_used_mb,
        "val": {
            **val_metrics,
            "inference_time_sec": val_inference_time,
            "throughput_samples_per_sec": val_throughput,
            "avg_latency_ms": (val_inference_time / len(X_val)) * 1000.0 if len(X_val) > 0 else 0.0
        },
        "test": {
            **test_metrics,
            "inference_time_sec": test_inference_time,
            "throughput_samples_per_sec": test_throughput,
            "avg_latency_ms": (test_inference_time / len(X_test)) * 1000.0 if len(X_test) > 0 else 0.0
        }
    }
    
    # 4. Save metrics to JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved metrics to: {metrics_path}")
    
    # 5. Save predictions to Parquet
    df_pred_val = pd.DataFrame({
        "split": "val",
        "y_true": y_val,
        "y_pred": y_pred_val
    })
    df_pred_test = pd.DataFrame({
        "split": "test",
        "y_true": y_test,
        "y_pred": y_pred_test
    })
    df_predictions = pd.concat([df_pred_val, df_pred_test], ignore_index=True)
    df_predictions.to_parquet(predictions_path, compression="snappy")
    logger.info(f"Saved predictions to: {predictions_path}")
    
    # 6. Save execution metadata
    # Gather package versions
    def get_version(package_name: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except Exception:
            try:
                import importlib
                mod = importlib.import_module(package_name)
                return getattr(mod, "__version__", "unknown")
            except Exception:
                return "not installed"
                
    metadata = {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "random_state": random_state,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "software_versions": {
            "scikit-learn": get_version("scikit-learn"),
            "xgboost": get_version("xgboost"),
            "lightgbm": get_version("lightgbm"),
            "catboost": get_version("catboost"),
            "pandas": get_version("pandas"),
            "numpy": get_version("numpy")
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return results
