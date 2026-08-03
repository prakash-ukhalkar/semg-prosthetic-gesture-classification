"""
sEMG Prosthetic Gesture Classification
Module: ml.splits

Handles subject-disjoint dataset splitting and dataset integrity validation.
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("semg_prosthetic_classification")

def verify_dataset_integrity(df: pd.DataFrame, expected_features: List[str] = None) -> Dict[str, Any]:
    """
    Perform automatic dataset validation including missing values, infinite values,
    duplicate samples, label consistency, subject consistency, and feature dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        The feature DataFrame containing features and metadata.
    expected_features : list of str, optional
        List of expected feature names to verify existence.

    Returns
    -------
    dict
        A dictionary containing the validation status and details of any checks.
    """
    logger.info("Running dataset integrity validation...")
    
    report = {}
    is_valid = True
    
    # 1. Row/Col shape
    report["shape"] = df.shape
    
    # 2. Missing values
    missing_count = int(df.isna().sum().sum())
    report["missing_values"] = missing_count
    if missing_count > 0:
        is_valid = False
        logger.warning(f"Validation: Found {missing_count} missing values in the dataset.")
        
    # 3. Infinite values
    # Filter numerical columns to check for infs
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(df[num_cols]).sum().sum())
    report["infinite_values"] = inf_count
    if inf_count > 0:
        is_valid = False
        logger.warning(f"Validation: Found {inf_count} infinite values in the dataset.")
        
    # 4. Duplicate samples
    duplicate_count = int(df.duplicated().sum())
    report["duplicate_samples"] = duplicate_count
    if duplicate_count > 0:
        logger.warning(f"Validation: Found {duplicate_count} duplicate samples.")
        
    # 5. Label consistency (gesture_id)
    if "gesture_id" in df.columns:
        unique_labels = sorted(df["gesture_id"].unique().tolist())
        report["unique_gestures"] = len(unique_labels)
        report["gesture_range"] = (min(unique_labels), max(unique_labels))
        # Ensure labels are non-negative
        if min(unique_labels) < 0:
            is_valid = False
            logger.error("Validation: Negative gesture_id found.")
    else:
        is_valid = False
        report["unique_gestures"] = 0
        logger.error("Validation: Column 'gesture_id' is missing.")
        
    # 6. Subject consistency (subject_id)
    if "subject_id" in df.columns:
        unique_subjects = sorted(df["subject_id"].unique().tolist())
        report["unique_subjects"] = len(unique_subjects)
        report["subject_range"] = (min(unique_subjects), max(unique_subjects))
    else:
        is_valid = False
        report["unique_subjects"] = 0
        logger.error("Validation: Column 'subject_id' is missing.")
        
    # 7. Feature dimensions
    if expected_features is not None:
        missing_features = [f for f in expected_features if f not in df.columns]
        report["missing_features"] = missing_features
        report["has_expected_features"] = len(missing_features) == 0
        if len(missing_features) > 0:
            is_valid = False
            logger.error(f"Validation: {len(missing_features)} expected features are missing.")
    else:
        # Just count features excluding metadata
        meta_cols = {
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        }
        feature_cols = [c for c in df.columns if c not in meta_cols]
        report["features_count"] = len(feature_cols)
        
    report["is_valid"] = is_valid
    logger.info(f"Dataset integrity check complete. Is valid: {is_valid}")
    return report

def get_subject_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform a subject-disjoint split of the dataset using GroupShuffleSplit.
    Ensures that windows from the same subject never appear in multiple splits.

    Parameters
    ----------
    df : pd.DataFrame
        The feature DataFrame.
    train_ratio : float, default=0.7
        Proportion of subjects for the training set.
    val_ratio : float, default=0.15
        Proportion of subjects for the validation set.
    test_ratio : float, default=0.15
        Proportion of subjects for the test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame
        Training set.
    val_df : pd.DataFrame
        Validation set.
    test_df : pd.DataFrame
        Test set.
    metadata : dict
        Details about the split (subjects in each partition, sample counts).
    """
    # Normalize ratios to sum to 1.0 just in case
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # Step 1: Split train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df["subject_id"]))
    
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()
    
    # Step 2: Split temp into val and test
    val_in_temp_ratio = val_ratio / temp_ratio
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_in_temp_ratio, random_state=random_state)
    val_idx_temp, test_idx_temp = next(gss2.split(df_temp, groups=df_temp["subject_id"]))
    
    df_val = df_temp.iloc[val_idx_temp].copy()
    df_test = df_temp.iloc[test_idx_temp].copy()
    
    # Extract subjects lists
    train_subjects = sorted(df_train["subject_id"].unique().tolist())
    val_subjects = sorted(df_val["subject_id"].unique().tolist())
    test_subjects = sorted(df_test["subject_id"].unique().tolist())
    
    # Check disjointness
    assert set(train_subjects).isdisjoint(set(val_subjects)), "Train and Val subjects overlap!"
    assert set(train_subjects).isdisjoint(set(test_subjects)), "Train and Test subjects overlap!"
    assert set(val_subjects).isdisjoint(set(test_subjects)), "Val and Test subjects overlap!"
    
    metadata = {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "train_percentage": len(df_train) / len(df) * 100,
        "val_percentage": len(df_val) / len(df) * 100,
        "test_percentage": len(df_test) / len(df) * 100,
    }
    
    logger.info(
        f"Subject-disjoint split complete.\n"
        f"Train subjects: {train_subjects} ({metadata['train_samples']} samples)\n"
        f"Val subjects: {val_subjects} ({metadata['val_samples']} samples)\n"
        f"Test subjects: {test_subjects} ({metadata['test_samples']} samples)"
    )
    
    return df_train, df_val, df_test, metadata
