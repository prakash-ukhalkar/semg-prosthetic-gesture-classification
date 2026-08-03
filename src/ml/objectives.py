"""
sEMG Prosthetic Gesture Classification
Module: ml.objectives

Implements the Optuna objective function with metric tracking and pruning callbacks.
"""

import time
import pickle
import logging
from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna

from src.ml.search_space import get_search_space
from src.ml.pipelines import create_pipeline
from src.ml.early_stopping import get_early_stopping_params
from src.ml.metrics import compute_metrics

logger = logging.getLogger("semg_prosthetic_classification")

def create_objective(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int = 42,
    patience: int = 15,
    max_train_samples: Optional[int] = 20000,
    model_n_jobs: Optional[int] = None
) -> Callable[[optuna.Trial], float]:
    """
    Objective function factory for Optuna study.

    Parameters
    ----------
    model_name : str
        The identifier of the classifier model.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : np.ndarray
        Validation labels.
    random_state : int, default=42
        Random seed for reproducibility.
    patience : int, default=15
        Early stopping patience.
    max_train_samples : int, optional, default=20000
        Maximum training samples for search trials (to reduce optimization time).

    Returns
    -------
    callable
        The objective function to be minimized/maximized by Optuna.
    """
    model_name = model_name.lower().strip()
    
    # Stratified downsampling for optimization trials to keep runtimes fast
    if max_train_samples is not None and len(X_train) > max_train_samples:
        logger.info(
            f"Downsampling training set for '{model_name}' search trials from {len(X_train)} to {max_train_samples} "
            f"using stratified sampling (random_state={random_state})."
        )
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train,
            train_size=max_train_samples,
            stratify=y_train,
            random_state=random_state
        )
    else:
        X_fit = X_train
        y_fit = y_train
        
    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest parameters from search space
        params = get_search_space(trial, model_name)
        
        # 2. Instantiate the classifier
        # When trials run concurrently (Optuna n_jobs > 1), each model's own
        # internal thread pool must be capped to model_n_jobs so parallel
        # trials do not oversubscribe the same CPU cores and slow each other
        # down. model_n_jobs=None preserves each library's own default
        # (use-all-cores) behaviour for the single-trial-at-a-time case.
        if model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            if model_n_jobs is not None:
                params = {**params, "n_jobs": model_n_jobs}
            clf = LGBMClassifier(random_state=random_state, **params)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            if model_n_jobs is not None:
                params = {**params, "n_jobs": model_n_jobs}
            clf = XGBClassifier(random_state=random_state, early_stopping_rounds=patience, **params)
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            if model_n_jobs is not None:
                params = {**params, "thread_count": model_n_jobs}
            clf = CatBoostClassifier(random_state=random_state, early_stopping_rounds=patience, **params)
        else:
            raise ValueError(f"Unsupported model name in objective: '{model_name}'")
            
        # 3. Create pipeline
        pipeline = create_pipeline(model_name, clf)
        
        # 4. Get early stopping parameters (eval set)
        fit_params = get_early_stopping_params(model_name, X_val, y_val, patience=patience)
        
        # 5. Fit the model and track training time
        logger.info(f"Trial {trial.number}: Training '{model_name}' on {len(X_fit)} samples...")
        start_train = time.perf_counter()
        pipeline.fit(X_fit, y_fit, **fit_params)
        train_time = time.perf_counter() - start_train
        
        # 6. Evaluate on validation set
        start_inf = time.perf_counter()
        y_pred_val = pipeline.predict(X_val)
        val_inference_time = time.perf_counter() - start_inf
        
        if len(y_pred_val.shape) > 1 and y_pred_val.shape[1] == 1:
            y_pred_val = y_pred_val.ravel()
            
        val_throughput = len(X_val) / val_inference_time if val_inference_time > 0 else 0.0
        val_metrics = compute_metrics(y_val, y_pred_val)
        
        # 7. Measure model size in memory
        try:
            model_size_bytes = len(pickle.dumps(pipeline))
            model_size_mb = model_size_bytes / (1024 * 1024)
        except Exception:
            model_size_mb = 0.0
            
        # 8. Log trial statistics as user attributes in Optuna
        trial.set_user_attr("accuracy", val_metrics["accuracy"])
        trial.set_user_attr("balanced_accuracy", val_metrics["balanced_accuracy"])
        trial.set_user_attr("mcc", val_metrics["mcc"])
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("inference_time", val_inference_time)
        trial.set_user_attr("throughput", val_throughput)
        trial.set_user_attr("model_size_mb", model_size_mb)
        
        val_f1 = val_metrics["f1_macro"]
        logger.info(f"Trial {trial.number} finished in {train_time:.2f}s. Validation Macro F1: {val_f1:.4f}")
        
        return val_f1
        
    return objective
