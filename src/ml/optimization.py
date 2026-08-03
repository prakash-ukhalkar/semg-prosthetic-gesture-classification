"""
sEMG Prosthetic Gesture Classification
Module: ml.optimization

Orchestrates Optuna studies, saves optimized models, and generates visualizations and comparison tables.
"""

import os
import json
import time
import pickle
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

from src.ml.early_stopping import get_pruner, get_sampler
from src.ml.objectives import create_objective
from src.ml.trials import save_trial_history
from src.ml.metrics import compute_metrics
from src.ml.pipelines import create_pipeline

logger = logging.getLogger("semg_prosthetic_classification")

def run_optuna_optimization(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_trials: int = 150,
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    db_dir: Path = Path("outputs"),
    models_dir: Path = Path("models/optimized"),
    figures_dir: Path = Path("outputs/figures"),
    tables_dir: Path = Path("outputs/tables"),
    max_train_samples: Optional[int] = 20000,
    random_state: int = 42,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for a classifier using Optuna.

    Parameters
    ----------
    model_name : str
        The classifier identifier.
    X_train, y_train : DataFrames/Arrays
        Training dataset.
    X_val, y_val : DataFrames/Arrays
        Validation dataset.
    X_test, y_test : DataFrames/Arrays
        Test dataset.
    n_trials : int, default=150
        Number of Optuna trials.
    sampler_name : str, default='tpe'
        Optuna sampler name.
    pruner_name : str, default='median'
        Optuna pruner name.
    db_dir : Path
        Directory to save the SQLite database.
    models_dir : Path
        Directory to save optimized models.
    figures_dir, tables_dir : Path
        Directories to save outputs.
    max_train_samples : int, optional, default=20000
        Maximum training samples for search trials.
    n_jobs : int, default=1
        Number of Optuna trials to run concurrently (threads). Each model's
        own internal thread count is automatically capped to
        max(1, os.cpu_count() // n_jobs) so parallel trials do not
        oversubscribe the same CPU cores. n_jobs=1 preserves the original
        sequential behaviour exactly (each model uses all cores per trial).
        n_jobs=-1 uses all available cores for trial-level parallelism.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict
        A summary of the optimization run and final evaluations.
    """
    model_name = model_name.lower().strip()
    db_dir = Path(db_dir)
    models_dir = Path(models_dir)
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    
    db_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Optuna Storage and Study
    db_path = db_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_name = f"semg_opt_{model_name}"
    
    sampler = get_sampler(sampler_name, random_state)
    pruner = get_pruner(pruner_name)
    
    logger.info(f"Creating/Loading study '{study_name}' on SQLite database...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )
    
    # 2. Define Objective function
    cpu_count = os.cpu_count() or 1
    model_n_jobs = None if n_jobs == 1 else max(1, cpu_count // max(1, n_jobs))
    objective = create_objective(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        patience=15,
        max_train_samples=max_train_samples,
        model_n_jobs=model_n_jobs
    )

    # 3. Run Optimization
    logger.info(
        f"Starting optimization for {model_name} ({n_trials} trials, "
        f"n_jobs={n_jobs}, per-trial threads={model_n_jobs or 'default'})..."
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # 4-7. Finalize: export logs, retrain best params on full data, save, evaluate, plot
    return finalize_optimized_model(
        study=study,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        db_dir=db_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        random_state=random_state
    )


def finalize_optimized_model(
    study: optuna.Study,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    db_dir: Path,
    models_dir: Path,
    figures_dir: Path,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Finalize a (possibly still-in-progress) Optuna study: export trial logs,
    retrain the best-so-far parameters on the full training split, save the
    model and best-parameter files, evaluate on the held-out test set, and
    generate Optuna diagnostic plots. This is the shared "finalize" step used
    both at the natural end of run_optuna_optimization and by
    finalize_from_existing_study for an early/manual finalization.
    """
    model_name = model_name.lower().strip()
    db_dir = Path(db_dir)
    models_dir = Path(models_dir)
    figures_dir = Path(figures_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Export trial logs
    csv_log = db_dir / f"trials_{model_name}.csv"
    json_log = db_dir / f"trials_{model_name}.json"
    save_trial_history(study, csv_log, json_log)

    best_params = study.best_params
    logger.info(f"Best Trial params for {model_name} (trial #{study.best_trial.number}): {best_params}")

    # Save parameters in JSON and YAML
    best_params_json_path = models_dir / f"best_params_{model_name}.json"
    best_params_yaml_path = models_dir / f"best_params_{model_name}.yaml"

    with open(best_params_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    with open(best_params_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    # Fit final model on the full Train dataset using best params
    logger.info(f"Training final optimized '{model_name}' on full training split...")
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        final_clf = LGBMClassifier(random_state=random_state, **best_params)
        model_filename = "LightGBM.pkl"
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        final_clf = XGBClassifier(random_state=random_state, **best_params)
        model_filename = "XGBoost.pkl"
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        final_clf = CatBoostClassifier(random_state=random_state, **best_params)
        model_filename = "CatBoost.pkl"
    else:
        raise ValueError(f"Unsupported model name in finalize: '{model_name}'")

    final_pipeline = create_pipeline(model_name, final_clf)

    # Train final model on FULL training data using validation set for early stopping
    from src.ml.early_stopping import get_early_stopping_params
    fit_params = get_early_stopping_params(model_name, X_val, y_val, patience=15)

    start_train = time.perf_counter()
    final_pipeline.fit(X_train, y_train, **fit_params)
    final_train_time = time.perf_counter() - start_train

    # Save optimized model to pkl
    model_save_path = models_dir / model_filename
    with open(model_save_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    logger.info(f"Saved optimized model pipeline to: {model_save_path}")

    # Evaluate optimized model on test set
    start_inf = time.perf_counter()
    y_pred_test = final_pipeline.predict(X_test)
    test_inference_time = time.perf_counter() - start_inf

    if len(y_pred_test.shape) > 1 and y_pred_test.shape[1] == 1:
        y_pred_test = y_pred_test.ravel()

    test_throughput = len(X_test) / test_inference_time if test_inference_time > 0 else 0.0
    test_latency = (test_inference_time / len(X_test)) * 1000.0 if len(X_test) > 0 else 0.0

    test_metrics = compute_metrics(y_test, y_pred_test)

    optimized_stats = {
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "mcc": test_metrics["mcc"],
        "train_time_sec": final_train_time,
        "inference_time_sec": test_inference_time,
        "throughput_samples_per_sec": test_throughput,
        "latency_ms": test_latency
    }

    # Generate Optuna plots
    save_optuna_plots(study, model_name, figures_dir)

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_at_finalization": len(study.trials),
        "optimized_stats": optimized_stats
    }


def finalize_from_existing_study(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    db_dir: Path = Path("outputs"),
    models_dir: Path = Path("models/optimized"),
    figures_dir: Path = Path("outputs/figures"),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Finalize a model EARLY, using whatever trials are already recorded in its
    Optuna study, without running any additional trials. Use this to save a
    real, usable model from the current best trial instead of waiting for a
    search to reach its full configured n_trials budget -- e.g. after
    observing the best value has plateaued for many trials.

    Parameters mirror run_optuna_optimization (minus the search-specific
    ones); db_dir must point at the directory containing the existing
    'optuna_study.db' for this model.
    """
    model_name = model_name.lower().strip()
    db_dir = Path(db_dir)
    db_path = db_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_name = f"semg_opt_{model_name}"

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    logger.info(
        f"Loaded existing study '{study_name}' with {len(study.trials)} trials "
        f"for early finalization (best trial #{study.best_trial.number}, "
        f"value={study.best_value:.4f})."
    )

    return finalize_optimized_model(
        study=study,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        db_dir=db_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        random_state=random_state
    )

def save_optuna_plots(study: optuna.Study, model_name: str, save_dir: Path):
    """
    Generate and save all 7 Optuna diagnostic plots in PNG, SVG, and PDF.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    import optuna.visualization.matplotlib as ovis
    
    # Helper to save active figure
    def save_fig(name):
        fig = plt.gcf()
        fig.savefig(save_dir / f"optuna_{name}_{model_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(save_dir / f"optuna_{name}_{model_name}.svg", bbox_inches="tight")
        fig.savefig(save_dir / f"optuna_{name}_{model_name}.pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot '{name}' for {model_name}")

    # 1. Optimization History
    try:
        ovis.plot_optimization_history(study)
        plt.title(f"Optimization History: {model_name.upper()}")
        save_fig("history")
    except Exception as e:
        logger.warning(f"Could not plot history for {model_name}: {e}")
        plt.close()

    # 2. Parameter Importance
    try:
        ovis.plot_param_importances(study)
        plt.title(f"Hyperparameter Importance: {model_name.upper()}")
        save_fig("importance")
    except Exception as e:
        logger.warning(f"Could not plot importance for {model_name}: {e}")
        plt.close()

    # 3. Slice Plot
    try:
        ovis.plot_slice(study)
        plt.suptitle(f"Slice Plot: {model_name.upper()}", y=1.02)
        save_fig("slice")
    except Exception as e:
        logger.warning(f"Could not plot slice for {model_name}: {e}")
        plt.close()

    # 4. Contour Plot
    try:
        params_to_plot = list(study.best_params.keys())[:3]
        ovis.plot_contour(study, params=params_to_plot)
        plt.suptitle(f"Contour Plot: {model_name.upper()}", y=1.02)
        save_fig("contour")
    except Exception as e:
        logger.warning(f"Could not plot contour for {model_name}: {e}")
        plt.close()

    # 5. Parallel Coordinate Plot
    try:
        ovis.plot_parallel_coordinate(study)
        plt.suptitle(f"Parallel Coordinate Plot: {model_name.upper()}", y=1.02)
        save_fig("parallel_coordinate")
    except Exception as e:
        logger.warning(f"Could not plot parallel coordinate for {model_name}: {e}")
        plt.close()

    # 6. EDF Plot
    try:
        ovis.plot_edf(study)
        plt.title(f"Empirical Distribution Function: {model_name.upper()}")
        save_fig("edf")
    except Exception as e:
        logger.warning(f"Could not plot EDF for {model_name}: {e}")
        plt.close()

    # 7. Intermediate Values
    try:
        ovis.plot_intermediate_values(study)
        plt.title(f"Intermediate Values (Pruning Trial History): {model_name.upper()}")
        save_fig("intermediate_values")
    except Exception as e:
        logger.info(f"No intermediate values plotted for {model_name} (expected for GBDTs): {e}")
        plt.close()
