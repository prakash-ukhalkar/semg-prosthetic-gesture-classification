"""
sEMG Prosthetic Gesture Classification
Module: ml.search_space

Defines Optuna hyperparameter search spaces for LightGBM, XGBoost, and CatBoost.
"""

import logging
from typing import Dict, Any
import optuna

logger = logging.getLogger("semg_prosthetic_classification")

def get_search_space(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a given model using the trial object.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.
    model_name : str
        The identifier of the model ('lightgbm', 'xgboost', or 'catboost').

    Returns
    -------
    dict
        A dictionary of suggested hyperparameter values.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "n_jobs": -1,
            "verbose": -1
        }
        
    elif model_name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "eta": trial.suggest_float("eta", 0.005, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "n_jobs": -1,
            "eval_metric": "mlogloss"
        }
        
    elif model_name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 100, 500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "verbose": 0,
            "thread_count": -1
        }
        
    else:
        raise ValueError(f"Unknown model name for search space suggestion: '{model_name}'")
