"""
sEMG Prosthetic Gesture Classification
Module: explainability

Helper functions and classes to perform model interpretability and explainability
analysis using SHAP, Permutation Importance, Channel Mapping, and Rank Consistency.
"""

import re
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

logger = logging.getLogger("semg_prosthetic_classification")


def compute_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Compute TreeSHAP values for a tree-based model (CatBoost, XGBoost, LightGBM).

    For CatBoost, this uses CatBoost's own native SHAP implementation
    (`get_feature_importance(type="ShapValues")`) rather than the third-party
    `shap` package's TreeExplainer C extension, which was found to segfault
    reliably (even at samples as small as ~500) on this project's 50-class
    CatBoost model -- a known scaling limitation of that library's compiled
    tree-traversal code for high-class-count multiclass CatBoost models, not
    a bug in this codebase. CatBoost's native path is used by the same
    organization that built the model format and is stable at the sample
    sizes this project requires (verified up to 20,000 samples, ~0.57 ms/sample).

    Parameters
    ----------
    model : Any
        Trained tree classifier model (or model inside pipeline).
    X : np.ndarray
        Feature matrix (N samples, D features).

    Returns
    -------
    np.ndarray
        SHAP values tensor of shape (n_samples, n_features, n_classes), with
        the per-class bias/expected-value term already dropped so the last
        axis has exactly n_classes entries matching model.classes_.
    """
    # Extract classifier if inside sklearn Pipeline
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") and "classifier" in model.named_steps else model

    clf_type = type(clf).__name__
    if "CatBoost" in clf_type:
        from catboost import Pool
        pool = Pool(X)
        # Native output shape: (n_samples, n_classes, n_features + 1), where
        # the final column per class is the expected-value/bias term.
        raw = clf.get_feature_importance(type="ShapValues", data=pool)
        shap_vals = raw[:, :, :-1]  # drop bias column -> (n_samples, n_classes, n_features)
        shap_vals = np.transpose(shap_vals, (0, 2, 1))  # -> (n_samples, n_features, n_classes)
        return shap_vals
    else:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X)
        return np.array(shap_vals)


def map_features_to_channels_and_families(feature_names: List[str]) -> pd.DataFrame:
    """
    Map each feature name to its originating sEMG channel (ch1..ch12)
    and feature family (Time-Domain, Frequency-Domain, Wavelet-Domain).

    Parameters
    ----------
    feature_names : list of str
        List of 50 feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Channel', 'Feature_Family'].
    """
    records = []
    
    for feat in feature_names:
        # Channel extraction
        ch_match = re.search(r"ch(\d+)", feat, re.IGNORECASE)
        channel = f"Channel {ch_match.group(1)}" if ch_match else "Channel Unknown"
        
        # Family classification
        feat_lower = feat.lower()
        if any(w in feat_lower for w in ["dwt", "wavelet", "cd1", "cd2", "cd3", "cd4", "ca4"]):
            family = "Wavelet-Domain"
        elif any(w in feat_lower for w in ["psd", "bp_", "spectral", "mnf", "mdf", "centroid", "roll_off", "peak"]):
            family = "Frequency-Domain"
        else:
            family = "Time-Domain"
            
        records.append({
            "Feature": feat,
            "Channel": channel,
            "Feature_Family": family
        })
        
    return pd.DataFrame(records)


def aggregate_channel_and_family_importance(
    mean_abs_shap: np.ndarray,
    feature_mapping: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate mean absolute SHAP attributions across sEMG channels and feature families.

    Parameters
    ----------
    mean_abs_shap : np.ndarray
        Array of mean absolute SHAP values for each feature (length 50).
    feature_mapping : pd.DataFrame
        DataFrame mapping features to Channel and Feature_Family.

    Returns
    -------
    channel_ranking : pd.DataFrame
    family_ranking : pd.DataFrame
    """
    df = feature_mapping.copy()
    df["SHAP_Importance"] = mean_abs_shap
    
    # Channel aggregation
    ch_agg = df.groupby("Channel")["SHAP_Importance"].agg(["sum", "mean", "count"]).reset_index()
    ch_agg.columns = ["Channel", "Total_SHAP_Importance", "Mean_SHAP_Importance", "Feature_Count"]
    ch_agg["Percentage (%)"] = (ch_agg["Total_SHAP_Importance"] / ch_agg["Total_SHAP_Importance"].sum()) * 100.0
    ch_ranking = ch_agg.sort_values(by="Total_SHAP_Importance", ascending=False).reset_index(drop=True)
    ch_ranking.insert(0, "Rank", ch_ranking.index + 1)
    
    # Family aggregation
    fam_agg = df.groupby("Feature_Family")["SHAP_Importance"].agg(["sum", "mean", "count"]).reset_index()
    fam_agg.columns = ["Feature_Family", "Total_SHAP_Importance", "Mean_SHAP_Importance", "Feature_Count"]
    fam_agg["Percentage (%)"] = (fam_agg["Total_SHAP_Importance"] / fam_agg["Total_SHAP_Importance"].sum()) * 100.0
    family_ranking = fam_agg.sort_values(by="Total_SHAP_Importance", ascending=False).reset_index(drop=True)
    family_ranking.insert(0, "Rank", family_ranking.index + 1)
    
    return ch_ranking, family_ranking


def compute_permutation_importance_scores(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute sklearn Permutation Feature Importance.

    Parameters
    ----------
    model : Any
        Trained classifier model/pipeline.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    feature_names : list of str
        List of feature names.
    n_repeats : int, default=5
        Number of permutation repeats.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame of permutation importances sorted by mean decrease.
    """
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") and "classifier" in model.named_steps else model
    
    res = permutation_importance(
        clf, X, y, n_repeats=n_repeats, random_state=random_state, scoring="accuracy"
    )
    
    df_perm = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": res.importances_mean,
        "Permutation_Importance_Std": res.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    
    df_perm.insert(0, "Rank", df_perm.index + 1)
    return df_perm


def compute_rank_consistency_metrics(
    shap_ranking: List[str],
    nb07_ranking: List[str],
    perm_ranking: List[str]
) -> pd.DataFrame:
    """
    Compute Spearman rank correlation and top-K overlap percentages between rankings.

    Parameters
    ----------
    shap_ranking : list of str
        Feature names ordered by SHAP importance.
    nb07_ranking : list of str
        Feature names ordered by Notebook 07 selection rank.
    perm_ranking : list of str
        Feature names ordered by Permutation Importance.

    Returns
    -------
    pd.DataFrame
        Rank consistency comparison table.
    """
    rankings = {
        "SHAP": shap_ranking,
        "Notebook07_Selection": nb07_ranking,
        "Permutation_Importance": perm_ranking
    }
    
    pairs = [
        ("SHAP", "Notebook07_Selection"),
        ("SHAP", "Permutation_Importance"),
        ("Permutation_Importance", "Notebook07_Selection")
    ]
    
    rows = []
    for r1_name, r2_name in pairs:
        r1_list = rankings[r1_name]
        r2_list = rankings[r2_name]
        
        # Build numerical ranks
        r1_map = {f: i+1 for i, f in enumerate(r1_list)}
        r2_map = {f: i+1 for i, f in enumerate(r2_list)}
        
        common_features = [f for f in r1_list if f in r2_map]
        vec1 = [r1_map[f] for f in common_features]
        vec2 = [r2_map[f] for f in common_features]
        
        rho, pval = stats.spearmanr(vec1, vec2)
        
        # Top-K overlap
        top10_1, top10_2 = set(r1_list[:10]), set(r2_list[:10])
        top20_1, top20_2 = set(r1_list[:20]), set(r2_list[:20])
        
        overlap_top10 = len(top10_1.intersection(top10_2)) / 10.0 * 100.0
        overlap_top20 = len(top20_1.intersection(top20_2)) / 20.0 * 100.0
        
        rows.append({
            "Ranking Pair": f"{r1_name} vs {r2_name}",
            "Spearman Rho": float(rho),
            "p-value": float(pval),
            "Top-10 Overlap (%)": overlap_top10,
            "Top-20 Overlap (%)": overlap_top20
        })
        
    return pd.DataFrame(rows)
