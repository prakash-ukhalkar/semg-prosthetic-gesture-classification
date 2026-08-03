"""
sEMG Prosthetic Gesture Classification
Module: models

Helper classes and functions to define, initialize, train, and save
machine learning models for gesture classification.
"""

from typing import Any, Dict, Optional
from src.ml.models import initialize_model as ml_initialize_model

def initialize_model(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Initialize a machine learning classifier model.

    Parameters
    ----------
    model_name : str
        The identifier of the classifier model (e.g., 'logistic_regression', 'random_forest', 'xgboost').
    hyperparameters : dict, optional
        A dictionary containing hyperparameter settings to initialize the model.

    Returns
    -------
    Any
        The initialized model object.
    """
    model = ml_initialize_model(model_name)
    if hyperparameters:
        model.set_params(**hyperparameters)
    return model
