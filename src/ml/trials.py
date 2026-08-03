"""
sEMG Prosthetic Gesture Classification
Module: ml.trials

Manages trial logging, formatting, and saving study results to CSV and JSON formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import optuna

logger = logging.getLogger("semg_prosthetic_classification")

def save_trial_history(study: optuna.Study, csv_path: Path, json_path: Path) -> pd.DataFrame:
    """
    Export the trial history from an Optuna study into clean CSV and JSON files.

    Parameters
    ----------
    study : optuna.Study
        The completed or in-progress Optuna study.
    csv_path : Path
        Path to save the CSV report.
    json_path : Path
        Path to save the JSON report.

    Returns
    -------
    pd.DataFrame
        The formatted trial history DataFrame.
    """
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert study trials to DataFrame
    df = study.trials_dataframe()
    
    if len(df) == 0:
        logger.warning("No trials found in the study to save.")
        return df
        
    # 2. Extract and rename columns
    col_mapping = {
        "number": "Trial Number",
        "value": "Macro F1",
        "duration": "Trial Duration (s)",
        "state": "State",
        "user_attrs_accuracy": "Accuracy",
        "user_attrs_balanced_accuracy": "Balanced Accuracy",
        "user_attrs_mcc": "MCC",
        "user_attrs_train_time": "Training Time (s)",
        "user_attrs_inference_time": "Inference Time (s)",
        "user_attrs_throughput": "Prediction Throughput (s/sec)",
        "user_attrs_model_size_mb": "Model Size (MB)"
    }
    
    # Check which columns exist in df
    cols_to_keep = [c for c in df.columns if c in col_mapping or c.startswith("params_")]
    df_clean = df[cols_to_keep].copy()
    
    # Rename standard columns
    rename_dict = {c: col_mapping[c] for c in df_clean.columns if c in col_mapping}
    df_clean = df_clean.rename(columns=rename_dict)
    
    # Clean parameter column names: remove 'params_' prefix
    param_rename = {c: c.replace("params_", "") for c in df_clean.columns if c.startswith("params_")}
    df_clean = df_clean.rename(columns=param_rename)
    
    # Sort by Trial Number
    df_clean = df_clean.sort_values("Trial Number").reset_index(drop=True)
    
    # Convert duration to seconds float
    if "Trial Duration (s)" in df_clean.columns:
        df_clean["Trial Duration (s)"] = df_clean["Trial Duration (s)"].dt.total_seconds()
        
    # 3. Save as CSV
    df_clean.to_csv(csv_path, index=False)
    logger.info(f"Saved trial history CSV to: {csv_path}")
    
    # 4. Save as JSON
    # Compile study metadata and best trial details
    best_trial = study.best_trial
    
    best_trial_info = {
        "trial_number": best_trial.number,
        "macro_f1": best_trial.value,
        "parameters": best_trial.params,
        "user_attrs": {k: float(v) for k, v in best_trial.user_attrs.items()}
    }
    
    study_summary = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "total_trials": len(df),
        "best_trial": best_trial_info,
        "trials": df_clean.to_dict(orient="records")
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=4)
    logger.info(f"Saved trial history JSON to: {json_path}")
    
    return df_clean
