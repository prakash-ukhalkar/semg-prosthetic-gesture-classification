"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.quality

Provides tools to validate the quality of the feature database.
Detects missing values (NaNs), infinities (Infs), constant features, 
duplicate columns, and outlier statistics in a memory-efficient manner.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class FeatureQualityValidator:
    """
    Validates features for NaNs, Infs, constant values, duplicates, and outliers.
    """
    def __init__(self, feature_path: Path, report_path: Path):
        self.feature_path = Path(feature_path)
        self.report_path = Path(report_path)
        
        self.metadata_cols = {
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        }
        
    def validate_features(self, chunk_subjects: bool = True) -> Dict[str, Any]:
        """
        Scan the feature database in chunks to validate quality.
        
        Parameters
        ----------
        chunk_subjects : bool
            If True, processes subject-by-subject to save memory.
            
        Returns
        -------
        Dict[str, Any]
            The validation report.
        """
        logger.info("Starting feature quality validation...")
        
        # Get schema and feature list
        schema = pq.read_schema(self.feature_path)
        feature_cols = [c for c in schema.names if c not in self.metadata_cols]
        num_features = len(feature_cols)
        
        logger.info(f"Analyzing {num_features} feature columns across dataset...")
        
        # We will do a subject-by-subject run to collect stats
        # To compute global mean and std, we accumulate: count, sum, sum of squares
        sums = np.zeros(num_features, dtype=np.float64)
        sum_sqs = np.zeros(num_features, dtype=np.float64)
        counts = np.zeros(num_features, dtype=np.int64)
        nan_counts = np.zeros(num_features, dtype=np.int64)
        inf_counts = np.zeros(num_features, dtype=np.int64)
        mins = np.ones(num_features, dtype=np.float64) * np.inf
        maxs = np.ones(num_features, dtype=np.float64) * -np.inf
        
        # Get list of subjects
        # We can read subject_id column
        meta_table = pq.read_table(self.feature_path, columns=["subject_id"])
        subjects = sorted(meta_table["subject_id"].unique().to_pylist())
        
        total_rows = len(meta_table)
        logger.info(f"Dataset has {total_rows} total rows and {len(subjects)} subjects.")
        
        # Pass 1: Accumulate counts, sums, sum of squares, NaNs, Infs, min, max
        logger.info("Pass 1: Accumulating statistics and detecting NaNs, Infs, and constants...")
        
        for sub_id in subjects:
            df_sub = pd.read_parquet(
                self.feature_path,
                columns=feature_cols,
                filters=[("subject_id", "==", sub_id)]
            )
            
            arr = df_sub.values.astype(np.float64)
            
            # Count NaNs and Infs
            nans = np.isnan(arr)
            infs = np.isinf(arr)
            
            nan_counts += np.sum(nans, axis=0)
            inf_counts += np.sum(infs, axis=0)
            
            # Clean array for sums (replace NaN/Inf with 0 temp)
            arr_clean = np.where(nans | infs, 0.0, arr)
            valid_mask = ~(nans | infs)
            
            sums += np.sum(arr_clean, axis=0)
            sum_sqs += np.sum(arr_clean**2, axis=0)
            counts += np.sum(valid_mask, axis=0)
            
            # Track mins/maxs
            # Use masked arrays to ignore NaNs/Infs
            masked_arr = np.ma.masked_array(arr, mask=(nans | infs))
            if len(df_sub) > 0:
                sub_mins = masked_arr.min(axis=0)
                sub_maxs = masked_arr.max(axis=0)
                # Fill masked values with inf/-inf so they don't affect global min/max
                mins = np.minimum(mins, sub_mins.filled(np.inf))
                maxs = np.maximum(maxs, sub_maxs.filled(-np.inf))
                
        # Calculate global mean and std
        means = np.zeros(num_features)
        stds = np.zeros(num_features)
        
        for i in range(num_features):
            c = counts[i]
            if c > 0:
                means[i] = sums[i] / c
                var = (sum_sqs[i] / c) - (means[i]**2)
                stds[i] = np.sqrt(max(0.0, var))
                
        # Identify constant columns (std < 1e-12 or std == 0)
        constant_indices = np.where(stds <= 1e-12)[0]
        constant_features = [feature_cols[i] for i in constant_indices]
        
        logger.info(f"Detected {len(constant_features)} constant features (zero/near-zero variance).")
        
        # Pass 2: Detect Outliers and Duplicates
        # To detect duplicates, we hash the columns using stats: mean, std, min, max, nan_count, inf_count
        # If columns have the exact same hash, we will check if they are identical.
        logger.info("Pass 2: Hashing and identifying duplicate columns, and counting outliers...")
        
        hashes = {}
        for i, col in enumerate(feature_cols):
            # Create a floating point tolerant hash key
            h_key = (
                round(means[i], 6),
                round(stds[i], 6),
                round(mins[i], 6),
                round(maxs[i], 6),
                int(nan_counts[i]),
                int(inf_counts[i])
            )
            hashes.setdefault(h_key, []).append((i, col))
            
        duplicate_groups = []
        possible_duplicate_cols = []
        for h_key, cols_in_group in hashes.items():
            if len(cols_in_group) > 1:
                # We have potential duplicates, verify by loading them
                cols_to_check = [item[1] for item in cols_in_group]
                possible_duplicate_cols.append(cols_to_check)
                
        # Verify duplicate columns by loading them fully or chunked
        # Since we only check a few columns at a time, we can load them for the whole dataset safely
        actual_duplicates = set()
        duplicate_mappings = {}
        
        for cols_to_check in possible_duplicate_cols:
            # Load only these columns for all rows
            df_check = pd.read_parquet(self.feature_path, columns=cols_to_check)
            # Find which ones are identical
            checked = set()
            for col1 in cols_to_check:
                if col1 in checked or col1 in actual_duplicates:
                    continue
                for col2 in cols_to_check:
                    if col1 == col2 or col2 in checked or col2 in actual_duplicates:
                        continue
                    if df_check[col1].equals(df_check[col2]):
                        actual_duplicates.add(col2)
                        duplicate_mappings[col2] = col1
                checked.add(col1)
                
        logger.info(f"Detected {len(actual_duplicates)} duplicate feature columns.")
        
        # Count outliers (Z-score outlier threshold = 5.0)
        # We count values where |x - mean| / std > 5.0
        outlier_counts = np.zeros(num_features, dtype=np.int64)
        
        for sub_id in subjects:
            df_sub = pd.read_parquet(
                self.feature_path,
                columns=feature_cols,
                filters=[("subject_id", "==", sub_id)]
            )
            arr = df_sub.values.astype(np.float64)
            
            for i in range(num_features):
                if stds[i] > 1e-12:
                    col_vals = arr[:, i]
                    # Filter out NaN/Inf
                    valid_mask = ~np.isnan(col_vals) & ~np.isinf(col_vals)
                    z_scores = np.abs(col_vals[valid_mask] - means[i]) / stds[i]
                    outlier_counts[i] += np.sum(z_scores > 5.0)
                    
        # Construct report
        report = {
            "total_rows": int(total_rows),
            "num_features": int(num_features),
            "nan_features_count": int(np.sum(nan_counts > 0)),
            "inf_features_count": int(np.sum(inf_counts > 0)),
            "constant_features_count": int(len(constant_features)),
            "duplicate_features_count": int(len(actual_duplicates)),
            "total_outliers_count": int(np.sum(outlier_counts)),
            "nan_details": {
                feature_cols[i]: int(nan_counts[i]) for i in range(num_features) if nan_counts[i] > 0
            },
            "inf_details": {
                feature_cols[i]: int(inf_counts[i]) for i in range(num_features) if inf_counts[i] > 0
            },
            "constant_features": constant_features,
            "duplicate_mappings": duplicate_mappings,
            "outlier_details": {
                feature_cols[i]: {
                    "outlier_count": int(outlier_counts[i]),
                    "pct": float(outlier_counts[i] / total_rows) * 100
                } for i in range(num_features) if outlier_counts[i] > 0
            }
        }
        
        # Save report
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Quality report successfully saved to {self.report_path}")
        return report
