"""
sEMG Prosthetic Gesture Classification
Module: ml.models

Declares and initializes machine learning classifiers with default parameters.
"""

import logging
from typing import Any, Dict, Optional

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import LinearSVC, SVC

# Boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger("semg_prosthetic_classification")

def initialize_model(model_name: str, random_state: int = 42) -> Any:
    """
    Initialize a machine learning classifier with literature-supported default hyperparameters.

    Parameters
    ----------
    model_name : str
        The identifier of the classifier model.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Any
        The initialized model object.
    """
    model_name = model_name.lower().strip()
    
    # 1. Linear Models
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
    elif model_name == "lda" or model_name == "linear_discriminant_analysis":
        return LinearDiscriminantAnalysis()
    elif model_name == "qda" or model_name == "quadratic_discriminant_analysis":
        return QuadraticDiscriminantAnalysis()
        
    # 2. Probabilistic Models
    elif model_name == "gaussian_nb" or model_name == "naive_bayes":
        return GaussianNB()
        
    # 3. Distance-Based Models
    elif model_name == "knn" or model_name == "k_nearest_neighbors":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
    # 4. Tree Models
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    elif model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        
    # 5. Boosting Models
    elif model_name == "adaboost":
        return AdaBoostClassifier(random_state=random_state)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    elif model_name == "xgboost":
        return XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric="mlogloss")
    elif model_name == "lightgbm":
        return LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
    elif model_name == "catboost":
        return CatBoostClassifier(random_state=random_state, verbose=0, thread_count=-1)
        
    # 6. Margin-Based Models
    elif model_name == "linear_svm":
        # dual='auto' handles the warning in newer scikit-learn versions
        return LinearSVC(random_state=random_state, max_iter=2000, dual='auto')
    elif model_name == "rbf_svm":
        return SVC(kernel="rbf", random_state=random_state)
        
    else:
        raise ValueError(f"Unknown model identifier: '{model_name}'")

def get_model_catalog(random_state: int = 42) -> Dict[str, Any]:
    """
    Retrieve the dictionary of all supported classifiers initialized with default parameters.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        A mapping of classifier keys to initialized scikit-learn compatible classifiers.
    """
    model_keys = [
        "logistic_regression",
        "lda",
        "qda",
        "gaussian_nb",
        "knn",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "linear_svm",
        "rbf_svm"
    ]
    return {k: initialize_model(k, random_state) for k in model_keys}
