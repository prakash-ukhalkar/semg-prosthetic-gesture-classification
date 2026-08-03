"""
sEMG Prosthetic Gesture Classification
Module: ml.ablation

Real feature/channel/domain ablation engine. Each ablation configuration
(a named subset of the top-50 selected features) is evaluated by cloning
the tuned CatBoost pipeline's hyperparameters, refitting from scratch on
the real subject-disjoint training split restricted to that feature
subset, and evaluating on the real held-out test split. Resumable via
per-configuration checkpoints, mirroring src/ml/loso.py.
"""

import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.ml.fold_metrics import compute_fold_metrics

logger = logging.getLogger("semg_prosthetic_classification")


def build_ablation_configs(
    feature_names: List[str],
    global_ranking: pd.DataFrame,
    channel_ranking: pd.DataFrame,
    feature_mapping: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Build the named feature-subset configurations for the ablation study,
    using the real SHAP-based rankings computed in Notebook 12.

    Parameters
    ----------
    feature_names : list of str
        All 50 selected feature column names (baseline).
    global_ranking : pd.DataFrame
        `global_feature_ranking.csv` -- columns ['Rank', 'Feature', 'Mean_Absolute_SHAP'],
        sorted most- to least-important.
    channel_ranking : pd.DataFrame
        `channel_ranking.csv` -- columns include ['Rank', 'Channel', ...],
        sorted most- to least-important channel.
    feature_mapping : pd.DataFrame
        Output of `map_features_to_channels_and_families` -- columns
        ['Feature', 'Channel', 'Feature_Family'].

    Returns
    -------
    dict
        Mapping of configuration name -> list of feature column names.
    """
    shap_order = global_ranking.sort_values("Rank")["Feature"].tolist()
    configs: Dict[str, List[str]] = {"baseline_all_50_features": list(feature_names)}

    # --- Feature removal (top / bottom SHAP features) ---
    for n in (5, 10, 15, 20):
        remove = set(shap_order[:n])
        configs[f"remove_top_{n}_shap_features"] = [f for f in feature_names if f not in remove]
    for n in (5, 10, 15, 20):
        remove = set(shap_order[-n:])
        configs[f"remove_bottom_{n}_shap_features"] = [f for f in feature_names if f not in remove]

    # --- Channel efficiency (top-K most important channels) ---
    channel_order = channel_ranking.sort_values("Rank")["Channel"].tolist()
    feat_by_channel = feature_mapping.groupby("Channel")["Feature"].apply(list).to_dict()
    for k in (8, 6, 4, 2, 1):
        top_channels = channel_order[:k]
        feats = [f for ch in top_channels for f in feat_by_channel.get(ch, [])]
        label = "single_best_channel" if k == 1 else f"top_{k}_channels"
        configs[label] = feats

    # --- Feature family (domain) combinations ---
    fam_feats = {
        fam: feature_mapping[feature_mapping["Feature_Family"] == fam]["Feature"].tolist()
        for fam in feature_mapping["Feature_Family"].unique()
    }
    configs["time_domain_only"] = fam_feats.get("Time-Domain", [])
    configs["frequency_domain_only"] = fam_feats.get("Frequency-Domain", [])
    configs["wavelet_domain_only"] = fam_feats.get("Wavelet-Domain", [])
    configs["time_plus_frequency_domains"] = fam_feats.get("Time-Domain", []) + fam_feats.get("Frequency-Domain", [])
    configs["time_plus_wavelet_domains"] = fam_feats.get("Time-Domain", []) + fam_feats.get("Wavelet-Domain", [])
    configs["frequency_plus_wavelet_domains"] = fam_feats.get("Frequency-Domain", []) + fam_feats.get("Wavelet-Domain", [])

    return configs


def save_ablation_checkpoint(save_dir: Path, config_name: str, result: Dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"ablation_{config_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_ablation_checkpoint(save_dir: Path, config_name: str) -> Optional[Dict[str, Any]]:
    path = save_dir / f"ablation_{config_name}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_ablation_studies(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    workspace_dir: Path,
    model_name: str = "CATBOOST",
    checkpoint_dir: Optional[Path] = None,
    force_rerun: bool = False,
    extra_classifier_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run every ablation configuration: clone the tuned pipeline, refit on
    the real training split restricted to the configuration's feature
    subset, evaluate on the real held-out test split. Resumable per
    configuration (safe to interrupt and re-run).

    Parameters
    ----------
    df_train : pd.DataFrame
        Real training split (subject-disjoint), full 50 features + metadata columns.
    df_test : pd.DataFrame
        Real held-out test split, same columns.
    workspace_dir : Path
        Project root (used to load the tuned pipeline and SHAP ranking tables).
    model_name : str
        Upper-case model name, e.g. 'CATBOOST'.
    checkpoint_dir : Path, optional
        Where to write/read per-configuration checkpoints.
    force_rerun : bool
        If True, ignore existing checkpoints.
    extra_classifier_params : dict, optional
        Extra params applied to the classifier step before cloning per-config,
        e.g. {"task_type": "GPU", "devices": "0"} to run on a Colab GPU runtime.

    Returns
    -------
    dict
        Keys: 'results' (list of per-config metric dicts), 'configs' (dict
        of config_name -> feature list used).
    """
    from src.ml.loso import load_optimized_pipeline
    from src.explainability import map_features_to_channels_and_families

    workspace_dir = Path(workspace_dir)
    if checkpoint_dir is None:
        checkpoint_dir = workspace_dir / "outputs" / "ablation_checkpoints_v2" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = workspace_dir / "outputs" / "tables"
    meta_cols = {
        "subject_id", "exercise_id", "gesture_id", "window_id",
        "repetition_id", "start_sample", "end_sample",
        "window_size_samples", "sampling_frequency_hz",
    }
    feature_names = [c for c in df_train.columns if c not in meta_cols]

    global_ranking = pd.read_csv(tables_dir / "global_feature_ranking.csv")
    channel_ranking = pd.read_csv(tables_dir / "channel_ranking.csv")
    feature_mapping = map_features_to_channels_and_families(feature_names)

    configs = build_ablation_configs(feature_names, global_ranking, channel_ranking, feature_mapping)

    optimized_pipeline = clone(load_optimized_pipeline(workspace_dir, model_name))
    if extra_classifier_params:
        prefixed = {f"classifier__{k}": v for k, v in extra_classifier_params.items()}
        optimized_pipeline.set_params(**prefixed)
        logger.info(f"Applied classifier param overrides: {extra_classifier_params}")

    y_train_full = df_train["gesture_id"].values
    y_test_full = df_test["gesture_id"].values

    results: List[Dict[str, Any]] = []
    for i, (config_name, feats) in enumerate(configs.items()):
        logger.info(f"[{i + 1}/{len(configs)}] Ablation config: {config_name} ({len(feats)} features)")

        if not force_rerun:
            cached = load_ablation_checkpoint(checkpoint_dir, config_name)
            if cached is not None:
                logger.info(f"  -> Resuming from checkpoint for '{config_name}'.")
                results.append(cached)
                continue

        if len(feats) == 0:
            logger.warning(f"  -> Skipping '{config_name}': empty feature subset.")
            continue

        X_train = df_train[feats].values
        X_test = df_test[feats].values

        fresh_pipeline = clone(optimized_pipeline)

        t0 = time.perf_counter()
        fresh_pipeline.fit(X_train, y_train_full)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = fresh_pipeline.predict(X_test)
        inference_time = time.perf_counter() - t0
        throughput = len(X_test) / inference_time if inference_time > 0 else 0.0

        metrics = compute_fold_metrics(y_test_full, y_pred, train_time, inference_time, throughput)
        metrics["Config"] = config_name
        metrics["Model"] = model_name
        metrics["Feature Count"] = len(feats)
        metrics["Features"] = feats

        save_ablation_checkpoint(checkpoint_dir, config_name, metrics)
        results.append(metrics)

        logger.info(
            f"  -> {config_name}: Acc={metrics['Accuracy']:.4f}, Macro F1={metrics['Macro F1']:.4f}, "
            f"Train={train_time:.1f}s"
        )

    return {"results": results, "configs": configs}
