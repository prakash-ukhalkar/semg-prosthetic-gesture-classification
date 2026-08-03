"""
sEMG Prosthetic Gesture Classification
Module: ml.early_stopping

Provides Optuna pruners, samplers, and model-specific early stopping configuration.
"""

import logging
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
import optuna

logger = logging.getLogger("semg_prosthetic_classification")

def get_pruner(pruner_name: str) -> Optional[optuna.pruners.BasePruner]:
    """
    Get the configured Optuna pruner.

    Parameters
    ----------
    pruner_name : str
        Name of the pruner ('median', 'successive_halving', or 'none').

    Returns
    -------
    optuna.pruners.BasePruner or None
        The initialized Optuna pruner object.
    """
    pruner_name = pruner_name.lower().strip()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner_name == "successive_halving" or pruner_name == "halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4)
    elif pruner_name == "none":
        return None
    else:
        logger.warning(f"Unknown pruner '{pruner_name}'. Using MedianPruner by default.")
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

def get_sampler(sampler_name: str, random_state: int = 42) -> optuna.samplers.BaseSampler:
    """
    Get the configured Optuna sampler.

    Parameters
    ----------
    sampler_name : str
        Name of the sampler ('tpe' or 'random').
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    optuna.samplers.BaseSampler
        The initialized Optuna sampler object.
    """
    sampler_name = sampler_name.lower().strip()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=random_state)
    elif sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=random_state)
    else:
        logger.warning(f"Unknown sampler '{sampler_name}'. Using TPESampler by default.")
        return optuna.samplers.TPESampler(seed=random_state)

def get_early_stopping_params(
    model_name: str,
    X_val: Any,
    y_val: Any,
    patience: int = 15,
    max_val_samples: Optional[int] = 20000
) -> Dict[str, Any]:
    """
    Generate model-specific fit parameters for early stopping using validation data.

    Parameters
    ----------
    model_name : str
        The classifier identifier ('lightgbm', 'xgboost', or 'catboost').
    X_val : Any
        Validation features.
    y_val : Any
        Validation labels.
    patience : int, default=15
        Number of rounds with no improvement before stopping.
    max_val_samples : int, optional, default=20000
        Maximum validation samples to evaluate on during fitting.

    Returns
    -------
    dict
        Dictionary of fit parameters.
    """
    model_name = model_name.lower().strip()
    
    # Downsample validation set for early-stopping monitoring only
    if max_val_samples is not None and len(X_val) > max_val_samples:
        X_val_fit, _, y_val_fit, _ = train_test_split(
            X_val, y_val,
            train_size=max_val_samples,
            stratify=y_val,
            random_state=42
        )
    else:
        X_val_fit = X_val
        y_val_fit = y_val
        
    if model_name == "lightgbm":
        import lightgbm
        return {
            "classifier__eval_set": [(X_val_fit, y_val_fit)],
            "classifier__eval_metric": "multi_logloss",
            "classifier__callbacks": [lightgbm.early_stopping(stopping_rounds=patience, verbose=False)]
        }
    elif model_name == "xgboost":
        return {
            "classifier__eval_set": [(X_val_fit, y_val_fit)],
            "classifier__verbose": False
        }
    elif model_name == "catboost":
        return {
            "classifier__eval_set": (X_val_fit, y_val_fit),
            "classifier__early_stopping_rounds": patience,
            "classifier__verbose": False
        }
    else:
        return {}
