"""
sEMG Prosthetic Gesture Classification
Module: ml.aggregate_results

Aggregates fold-wise metrics, computes descriptive statistics, 95% confidence intervals,
and formats publication tables.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

def compute_95_ci(data: np.ndarray) -> tuple[float, float]:
    """
    Calculate the 95% confidence interval for the mean using Student's t-distribution.

    Parameters
    ----------
    data : np.ndarray
        Array of metric scores across folds.

    Returns
    -------
    lower_ci : float
        Lower confidence boundary.
    upper_ci : float
        Upper confidence boundary.
    """
    n = len(data)
    if n < 2:
        return float(np.min(data)), float(np.max(data))
    
    mean = np.mean(data)
    sem = stats.sem(data)
    # df = n - 1
    h = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    return float(mean - h), float(mean + h)

def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    save_dir: Path,
    model_name: str
) -> pd.DataFrame:
    """
    Aggregate fold metrics and compute statistical summaries.

    Parameters
    ----------
    fold_metrics : list of dict
        List of metrics dictionaries from each fold.
    save_dir : Path
        Directory to save the generated tables.
    model_name : str
        Name of the classifier model.

    Returns
    -------
    pd.DataFrame
        DataFrame of aggregated metric statistics.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load into DataFrame
    df_folds = pd.DataFrame(fold_metrics)
    
    # Save raw fold metrics
    df_folds.to_csv(save_dir / f"raw_folds_{model_name}.csv", index=False)
    df_folds.to_csv(save_dir / "fold_results.csv", index=False)
    df_folds.to_markdown(save_dir / "fold_results.md", index=False)
    df_folds.to_latex(save_dir / "fold_results.tex", index=False, float_format="%.4f")
    
    # 2. Compute statistics for each column
    metrics_columns = [c for c in df_folds.columns if c not in ["Subject ID", "Model"]]
    
    summary_rows = []
    for col in metrics_columns:
        data = df_folds[col].values
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        std_val = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        lower_ci, upper_ci = compute_95_ci(data)
        
        summary_rows.append({
            "Metric": col,
            "Mean": mean_val,
            "Median": median_val,
            "Std Dev": std_val,
            "Min": min_val,
            "Max": max_val,
            "Lower 95% CI": lower_ci,
            "Upper 95% CI": upper_ci
        })
        
    df_summary = pd.DataFrame(summary_rows)
    
    # Save summary tables
    df_summary.to_csv(save_dir / "metric_summary.csv", index=False)
    df_summary.to_markdown(save_dir / "metric_summary.md", index=False)
    df_summary.to_latex(save_dir / "metric_summary.tex", index=False, float_format="%.4f")
    
    return df_summary
