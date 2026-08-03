"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.filters

Implements Stage 1 (Constant Feature Removal), Stage 2 (Near-Zero Variance Removal),
and Stage 3 (Pearson Correlation Filtering) of the feature selection pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class ConstantFeatureRemover:
    """
    Stage 1: Removes features with zero variance.
    """
    def __init__(self, constant_cols: List[str] = None):
        self.constant_cols = constant_cols if constant_cols is not None else []
        
    def fit(self, df_sample: pd.DataFrame) -> None:
        """
        Identify constant columns from a sample dataframe.
        """
        variances = df_sample.var()
        self.constant_cols = list(variances[variances <= 1e-12].index)
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant columns.
        """
        cols_to_drop = [c for c in self.constant_cols if c in df.columns]
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} constant features.")
            return df.drop(columns=cols_to_drop)
        return df


class NearZeroVarianceRemover:
    """
    Stage 2: Removes features with variance below a configurable threshold.
    """
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.low_var_cols = []
        
    def fit(self, df_sample: pd.DataFrame) -> None:
        """
        Identify low variance columns.
        """
        # Calculate variances
        variances = df_sample.var()
        self.low_var_cols = list(variances[variances < self.threshold].index)
        logger.info(
            f"NZV Fit: Identified {len(self.low_var_cols)} features with variance < {self.threshold}"
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low variance columns.
        """
        cols_to_drop = [c for c in self.low_var_cols if c in df.columns]
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} near-zero variance features.")
            return df.drop(columns=cols_to_drop)
        return df


class CorrelationFilter:
    """
    Stage 3: Pearson Correlation Filter.
    Removes highly correlated features to reduce redundancy.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.drop_cols = []
        
    def fit(self, df_features: pd.DataFrame) -> None:
        """
        Identify highly correlated features.
        
        Algorithm:
        1. Compute absolute Pearson correlation matrix.
        2. Set diagonal to 0.
        3. For each pair of features with correlation > threshold:
           - Remove the feature with the higher average correlation to all other features.
        """
        logger.info(
            f"Correlation Fit: Computing correlation matrix on {df_features.shape[1]} features..."
        )
        
        # Compute correlation matrix using NumPy for 1000x speedup
        X = df_features.values
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        corr_data = np.abs(np.dot(X_std.T, X_std) / (X.shape[0] - 1))
        corr_data = np.clip(corr_data, 0.0, 1.0)
        corr_matrix = pd.DataFrame(
            corr_data,
            index=df_features.columns,
            columns=df_features.columns
        )

        
        # Calculate mean correlation of each feature with all other features
        mean_corr = corr_matrix.mean()
        
        # Keep track of columns to drop
        drop_set = set()
        
        # Get upper triangle to avoid duplicate pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation above threshold
        for col in upper.columns:
            if col in drop_set:
                continue
            # Get other columns that are highly correlated with this one
            correlated_cols = upper.index[upper[col] > self.threshold].tolist()
            
            for other_col in correlated_cols:
                if other_col in drop_set:
                    continue
                # Compare average correlation with all other features
                if mean_corr[col] > mean_corr[other_col]:
                    drop_set.add(col)
                    break  # col is dropped, no need to check other_col
                else:
                    drop_set.add(other_col)
                    
        self.drop_cols = list(drop_set)
        logger.info(
            f"Correlation Fit: Identified {len(self.drop_cols)} redundant features with correlation > {self.threshold}"
        )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.
        """
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} highly correlated features.")
            return df.drop(columns=cols_to_drop)
        return df
