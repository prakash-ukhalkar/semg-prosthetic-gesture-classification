"""
sEMG Prosthetic Gesture Classification
Module: ml.loso

Core Leave-One-Subject-Out (LOSO) cross-validation engine.
Trains on N-1 subjects, tests on 1 held-out subject, repeats for all subjects.
Supports resumable execution via fold checkpoints.
"""

import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.ml.fold_metrics import compute_fold_metrics, save_fold_checkpoint, load_fold_checkpoint

logger = logging.getLogger("semg_prosthetic_classification")


def load_optimized_pipeline(workspace_dir: Path, model_name: str) -> Any:
    """
    Load the optimized pipeline pickle for a given model name.

    Parameters
    ----------
    workspace_dir : Path
        Project root directory.
    model_name : str
        Upper-case model name (e.g., 'CATBOOST').

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline object (used to extract hyperparameters).
    """
    models_dir = workspace_dir / "models" / "optimized"
    # Try common capitalizations
    for candidate in [f"{model_name}.pkl", f"{model_name.title()}.pkl",
                      f"{model_name.lower()}.pkl", f"{model_name.upper()}.pkl"]:
        pkl_path = models_dir / candidate
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                pipeline = pickle.load(f)
            logger.info(f"Loaded optimized pipeline from {pkl_path}")
            return pipeline

    # Glob fallback
    pkl_files = list(models_dir.glob("*.pkl"))
    for p in pkl_files:
        if model_name.lower() in p.stem.lower():
            with open(p, "rb") as f:
                pipeline = pickle.load(f)
            logger.info(f"Loaded optimized pipeline from {p}")
            return pipeline

    raise FileNotFoundError(
        f"No optimized pipeline found for '{model_name}' in {models_dir}"
    )


def extract_classifier_params(pipeline) -> Dict[str, Any]:
    """
    Extract the classifier hyperparameters from a fitted sklearn pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline with a 'classifier' step.

    Returns
    -------
    dict
        The classifier's `get_params()` dictionary.
    """
    clf = pipeline.named_steps["classifier"]
    return clf.get_params()


def run_loso_validation(
    df: pd.DataFrame,
    workspace_dir: Path,
    model_name: str,
    checkpoint_dir: Optional[Path] = None,
    force_rerun: bool = False
) -> Dict[str, Any]:
    """
    Execute full Leave-One-Subject-Out cross-validation.

    For each of the N subjects:
      - Train a fresh classifier (with optimized hyperparameters) on the
        remaining N-1 subjects.
      - Evaluate on the held-out subject.
      - Record metrics, predictions, and timing.

    Parameters
    ----------
    df : pd.DataFrame
        The full feature DataFrame (all subjects).
    workspace_dir : Path
        Project root directory.
    model_name : str
        Upper-case model name (e.g., 'CATBOOST').
    checkpoint_dir : Path, optional
        Directory for saving/loading fold checkpoints. Defaults to
        ``outputs/loso_checkpoints/<model_name>/``.
    force_rerun : bool, default=False
        If True, ignore existing checkpoints and retrain every fold.

    Returns
    -------
    dict
        Keys: 'fold_metrics', 'all_y_true', 'all_y_pred', 'subjects',
              'model_name', 'n_folds'.
    """
    workspace_dir = Path(workspace_dir)
    if checkpoint_dir is None:
        checkpoint_dir = workspace_dir / "outputs" / "loso_checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── identify subjects and feature columns ──
    unique_subjects = sorted(df["subject_id"].unique().tolist())
    n_folds = len(unique_subjects)
    logger.info(
        f"Starting LOSO validation for '{model_name}' across {n_folds} subjects."
    )

    meta_cols = {
        "subject_id", "exercise_id", "gesture_id", "window_id",
        "repetition_id", "start_sample", "end_sample",
        "window_size_samples", "sampling_frequency_hz",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # ── load optimized pipeline and extract hyper-params ──
    optimized_pipeline = load_optimized_pipeline(workspace_dir, model_name)
    optimized_params = extract_classifier_params(optimized_pipeline)

    # ── iterate over folds ──
    fold_metrics_list: List[Dict[str, Any]] = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, test_subject in enumerate(unique_subjects):
        logger.info(
            f"[Fold {fold_idx + 1}/{n_folds}] Held-out subject: {test_subject}"
        )

        # Check for a cached checkpoint
        if not force_rerun:
            checkpoint = load_fold_checkpoint(checkpoint_dir, model_name, test_subject)
            if checkpoint is not None:
                logger.info(
                    f"  -> Resuming from checkpoint for subject {test_subject}."
                )
                metrics = checkpoint["metrics"]
                metrics["Subject ID"] = int(test_subject)
                metrics["Model"] = model_name
                fold_metrics_list.append(metrics)
                all_y_true.append(checkpoint["y_true"])
                all_y_pred.append(checkpoint["y_pred"])
                continue

        # ── split ──
        df_train = df[df["subject_id"] != test_subject]
        df_test = df[df["subject_id"] == test_subject]

        X_train = df_train[feature_cols].values
        y_train = df_train["gesture_id"].values
        X_test = df_test[feature_cols].values
        y_test = df_test["gesture_id"].values

        # ── build a fresh pipeline with optimized hyper-params ──
        fresh_pipeline = clone(optimized_pipeline)

        # ── dynamically configure GPU acceleration ──
        model_upper = model_name.upper()
        gpu_override_params = {}

        if model_upper == "CATBOOST":
            gpu_override_params = {"classifier__task_type": "GPU"}
        elif model_upper == "XGBOOST":
            gpu_override_params = {
                "classifier__tree_method": "hist",
                "classifier__device": "cuda"
            }
        elif model_upper == "LIGHTGBM":
            gpu_override_params = {"classifier__device": "gpu"}

        if gpu_override_params:
            try:
                fresh_pipeline.set_params(**gpu_override_params)
                if fold_idx == 0:
                    logger.info(f"Applied GPU acceleration settings for {model_name}: {gpu_override_params}")
            except Exception as e:
                logger.warning(
                    f"Could not set GPU acceleration parameters on pipeline for {model_name}: {e}. "
                    f"Falling back to original parameters."
                )

        # ── train ──
        t_train_start = time.perf_counter()
        fresh_pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t_train_start

        # ── predict ──
        t_infer_start = time.perf_counter()
        y_pred = fresh_pipeline.predict(X_test)
        inference_time = time.perf_counter() - t_infer_start

        throughput = len(X_test) / inference_time if inference_time > 0 else 0.0

        # ── compute metrics ──
        metrics = compute_fold_metrics(
            y_test, y_pred, train_time, inference_time, throughput
        )
        metrics["Subject ID"] = int(test_subject)
        metrics["Model"] = model_name

        fold_metrics_list.append(metrics)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

        # ── checkpoint ──
        save_fold_checkpoint(
            checkpoint_dir, model_name, test_subject, metrics, y_pred, y_test
        )

        logger.info(
            f"  -> Fold {fold_idx + 1} complete: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"F1={metrics['Macro F1']:.4f}, "
            f"Train={train_time:.1f}s"
        )

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    logger.info(
        f"LOSO validation finished for '{model_name}'. "
        f"Total predictions: {len(all_y_true)}"
    )

    return {
        "fold_metrics": fold_metrics_list,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "subjects": unique_subjects,
        "model_name": model_name,
        "n_folds": n_folds,
    }