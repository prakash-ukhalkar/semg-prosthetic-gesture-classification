"""
sEMG Prosthetic Gesture Classification
Module: ml.benchmark

Orchestrates the machine learning benchmarking workflow, running all experiments,
performing model screening and ranking, and saving baseline results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.ml.splits import verify_dataset_integrity, get_subject_splits
from src.ml.models import get_model_catalog
from src.ml.pipelines import create_pipeline
from src.ml.training import train_classifier
from src.ml.evaluation import evaluate_classifier

logger = logging.getLogger("semg_prosthetic_classification")

def run_benchmark(
    data_dir: Path,
    models_dir: Path,
    outputs_dir: Path,
    feature_sets: List[str] = ["top25", "top50"],
    force_retrain: bool = False,
    random_state: int = 42,
    max_train_samples_svm: int = 20000
) -> Dict[str, Any]:
    """
    Execute the baseline benchmarking framework.

    Parameters
    ----------
    data_dir : Path
        Directory containing the feature parquet files.
    models_dir : Path
        Directory to save trained model files.
    outputs_dir : Path
        Directory to save results, tables, and rankings.
    feature_sets : list of str, default=['top25', 'top50']
        List of feature sets to evaluate.
    force_retrain : bool, default=False
        If True, ignores cached models and fits them from scratch.
    random_state : int, default=42
        Random seed for splits and model initializations.
    max_train_samples_svm : int, default=20000
        Maximum training samples for computationally expensive models (RBF SVM).

    Returns
    -------
    dict
        A dictionary containing the compiled metrics and screening outputs.
    """
    data_dir = Path(data_dir)
    models_dir = Path(models_dir) / "baseline"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(outputs_dir)
    
    tables_dir = outputs_dir / "tables"
    reports_dir = outputs_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting sEMG Machine Learning Benchmarking Framework...")
    
    # Get all classifiers initialized
    classifiers = get_model_catalog(random_state)
    logger.info(f"Initialized {len(classifiers)} classifiers: {list(classifiers.keys())}")
    
    all_results = []
    validation_reports = {}
    split_metadata_all = {}
    
    # Loop over feature sets
    for fs_name in feature_sets:
        file_path = data_dir / f"selected_features_{fs_name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")
            
        logger.info(f"==================================================")
        logger.info(f"Processing Feature Set: {fs_name.upper()}")
        logger.info(f"==================================================")
        
        # 1. Load data
        df = pd.read_parquet(file_path)
        
        # Downcast columns to save RAM and prevent MemoryError
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == np.int64:
                df[col] = df[col].astype(np.int32)
        
        # 2. Data Validation
        val_report = verify_dataset_integrity(df)
        validation_reports[fs_name] = val_report
        
        # Save validation report to JSON
        val_report_path = reports_dir / f"validation_report_{fs_name}.json"
        with open(val_report_path, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=4)
            
        # 3. Subject-Disjoint splits
        df_train, df_val, df_test, split_meta = get_subject_splits(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=random_state
        )
        split_metadata_all[fs_name] = split_meta
        
        # Save split metadata to JSON
        split_meta_path = reports_dir / f"split_metadata_{fs_name}.json"
        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=4)
            
        # Extract features and targets
        # Metadata columns to exclude from training
        meta_cols = [
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        ]
        
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        X_train = df_train[feature_cols]
        y_train = df_train["gesture_id"].values
        X_val = df_val[feature_cols]
        y_val = df_val["gesture_id"].values
        X_test = df_test[feature_cols]
        y_test = df_test["gesture_id"].values
        
        # Run experiments on each classifier
        for model_name, classifier_obj in tqdm(classifiers.items(), desc=f"Models on {fs_name}"):
            experiment_key = f"{fs_name}_{model_name}"
            
            # Check if metrics and model are already saved to support resumable run
            model_path = models_dir / f"model_{fs_name}_{model_name}.joblib"
            metrics_path = models_dir / f"metrics_{fs_name}_{model_name}.json"
            if model_path.exists() and metrics_path.exists() and not force_retrain:
                logger.info(f"Resuming: Saved model and metrics found for '{model_name}' on '{fs_name}'. Skipping training/evaluation.")
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        experiment_results = json.load(f)
                    all_results.append(experiment_results)
                    continue
                except Exception as e:
                    logger.error(f"Error loading cached metrics from {metrics_path}: {e}. Running anyway.")
            
            # Setup pipeline (includes scaler selection)
            pipeline = create_pipeline(model_name, classifier_obj)
            
            # Downsampling policy for slow algorithms
            # We downsample RBF SVM and AdaBoost/Gradient Boosting to keep runtime manageable.
            # But AdaBoost/Gradient Boosting can be run on full data if preferred. Let's downsample SVMs and boosting if too slow.
            # Let's downsample RBF SVM, AdaBoost, Gradient Boosting to max_train_samples_svm if they are slow.
            max_samples = None
            if model_name in ["rbf_svm", "adaboost", "gradient_boosting", "catboost", "extra_trees"]:
                max_samples = max_train_samples_svm
                
            try:
                # Train
                trained_pipe, train_time, mem_used, loaded = train_classifier(
                    pipeline=pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    model_name=model_name,
                    feature_set_name=fs_name,
                    save_dir=models_dir,
                    max_train_samples=max_samples,
                    force_retrain=force_retrain,
                    random_state=random_state
                )
                
                # If loaded from disk but we didn't return early (e.g. metrics file was missing),
                # train_time will be 0. We'll evaluate anyway.
                
                # Evaluate
                experiment_results = evaluate_classifier(
                    model=trained_pipe,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name,
                    feature_set_name=fs_name,
                    save_dir=models_dir,
                    train_time=train_time,
                    memory_used_mb=mem_used,
                    random_state=random_state
                )
                
                all_results.append(experiment_results)
                
                # Eagerly delete pipeline and collect garbage to free RAM
                del trained_pipe
                import gc; gc.collect()
                
            except Exception as e:
                logger.error(f"Experiment failed for '{model_name}' on '{fs_name}': {e}")
                # Log failure, continue to next experiment
                continue
                
        # Delete large dataset variables and collect garbage before loading the next feature set
        del df, df_train, df_val, df_test, X_train, y_train, X_val, y_val, X_test, y_test
        import gc; gc.collect()
        
    # ==============================================================================
    # Model Screening and Ranking
    # ==============================================================================
    logger.info("Executing model screening and ranking...")
    
    # Flatten results into a DataFrame for ranking
    rows = []
    for res in all_results:
        row = {
            "model_name": res["model_name"],
            "feature_set": res["feature_set"],
            "train_time_sec": res["train_time_sec"],
            "memory_used_mb": res["memory_used_mb"],
            # Test metrics
            "test_accuracy": res["test"]["accuracy"],
            "test_balanced_accuracy": res["test"]["balanced_accuracy"],
            "test_precision_macro": res["test"]["precision_macro"],
            "test_recall_macro": res["test"]["recall_macro"],
            "test_f1_macro": res["test"]["f1_macro"],
            "test_mcc": res["test"]["mcc"],
            "test_cohen_kappa": res["test"]["cohen_kappa"],
            "test_inference_time_sec": res["test"]["inference_time_sec"],
            "test_throughput": res["test"]["throughput_samples_per_sec"],
            "test_latency_ms": res["test"]["avg_latency_ms"],
            # Val metrics
            "val_accuracy": res["val"]["accuracy"],
            "val_balanced_accuracy": res["val"]["balanced_accuracy"],
            "val_f1_macro": res["val"]["f1_macro"],
            "val_mcc": res["val"]["mcc"]
        }
        rows.append(row)
        
    df_ranking = pd.DataFrame(rows)
    
    # Save the raw ranking table
    ranking_csv_path = outputs_dir / "model_ranking.csv"
    df_ranking.to_csv(ranking_csv_path, index=False)
    logger.info(f"Saved complete ranking table to: {ranking_csv_path}")
    
    # Sort ranking: F1 macro (descending), Balanced Accuracy (descending), MCC (descending)
    df_ranking_sorted = df_ranking.sort_values(
        by=["test_f1_macro", "test_balanced_accuracy", "test_mcc"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Save sorted ranking
    sorted_ranking_csv_path = tables_dir / "model_ranking_sorted.csv"
    df_ranking_sorted.to_csv(sorted_ranking_csv_path, index=False)
    
    # Generate Top 5 Best Models
    # We group by model_name and select the model's best F1 macro score across feature sets,
    # or select the top models on the highest-performing dataset configuration (Top 50).
    # Let's find the top 5 models based on their performance on Top 50 features.
    df_top50_sorted = df_ranking_sorted[df_ranking_sorted["feature_set"] == "top50"].reset_index(drop=True)
    top_models_list = df_top50_sorted["model_name"].head(5).tolist()
    
    # If we don't have enough top 50 models, take overall top models
    if len(top_models_list) < 5:
        top_models_list = df_ranking_sorted["model_name"].unique()[:5].tolist()
        
    top_models_path = outputs_dir / "top_models.json"
    with open(top_models_path, "w", encoding="utf-8") as f:
        json.dump(top_models_list, f, indent=4)
    logger.info(f"Top 5 Models selected: {top_models_list}. Saved to: {top_models_path}")
    
    return {
        "results": all_results,
        "ranking_df": df_ranking_sorted,
        "top_models": top_models_list,
        "validation_reports": validation_reports,
        "split_metadata": split_metadata_all
    }
