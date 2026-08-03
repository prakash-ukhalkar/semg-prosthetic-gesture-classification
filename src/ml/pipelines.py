"""
sEMG Prosthetic Gesture Classification
Module: ml.pipelines

Builds sklearn Pipeline objects integrating scaling and classifiers to prevent data leakage.
"""

import logging
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger("semg_prosthetic_classification")

def get_scaler(scaler_type: str) -> Any:
    """
    Retrieve the appropriate scikit-learn scaler object.

    Parameters
    ----------
    scaler_type : str
        Type of scaler ('standard', 'robust', 'minmax', or 'passthrough').

    Returns
    -------
    Any
        The scikit-learn scaler object (or None for 'passthrough').
    """
    scaler_type = scaler_type.lower().strip()
    if scaler_type == "standard" or scaler_type == "std":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "passthrough" or scaler_type == "none":
        return "passthrough"
    else:
        raise ValueError(f"Unknown scaler type: '{scaler_type}'")

def determine_scaler_type(model_name: str) -> str:
    """
    Automatically select the optimal scaler type for a given model.

    Parameters
    ----------
    model_name : str
        The classifier identifier.

    Returns
    -------
    str
        The scaler identifier ('standard', 'minmax', or 'passthrough').
    """
    model_name = model_name.lower().strip()
    
    # Linear, Probabilistic, and Margin-Based models require Standard Scaling
    standard_scalers = {
        "logistic_regression",
        "lda",
        "linear_discriminant_analysis",
        "qda",
        "quadratic_discriminant_analysis",
        "gaussian_nb",
        "naive_bayes",
        "linear_svm",
        "rbf_svm"
    }
    
    # Distance-based models prefer MinMaxScaler (for uniform distance ranges)
    minmax_scalers = {
        "knn",
        "k_nearest_neighbors"
    }
    
    # Tree and boosting models are scale-invariant
    passthrough_scalers = {
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost"
    }
    
    if model_name in standard_scalers:
        return "standard"
    elif model_name in minmax_scalers:
        return "minmax"
    elif model_name in passthrough_scalers:
        return "passthrough"
    else:
        logger.warning(f"Model '{model_name}' not categorized for scaling. Using standard scaling by default.")
        return "standard"

def create_pipeline(model_name: str, model_obj: Any, scaler_type: str = "auto") -> Pipeline:
    """
    Create a scikit-learn Pipeline incorporating scaling and the classifier.

    Parameters
    ----------
    model_name : str
        The classifier identifier.
    model_obj : Any
        The instantiated classifier object.
    scaler_type : str, default='auto'
        Explicit scaler type, or 'auto' to choose automatically.

    Returns
    -------
    Pipeline
        The assembled scikit-learn Pipeline object.
    """
    if scaler_type == "auto":
        scaler_key = determine_scaler_type(model_name)
    else:
        scaler_key = scaler_type
        
    scaler_obj = get_scaler(scaler_key)
    
    logger.info(f"Pipeline created for '{model_name}' with scaler: '{scaler_key}'.")
    return Pipeline([
        ("scaler", scaler_obj),
        ("classifier", model_obj)
    ])
