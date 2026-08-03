"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.rankers

Implements feature ranking algorithms: Mutual Information, mRMR,
Tree-based importances, Recursive Feature Elimination (RFE),
and Consensus Ranking.
"""

import logging
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb

from src.config import LOGGER_NAME, RANDOM_STATE

logger = logging.getLogger(LOGGER_NAME)

class MIEstimator:
    """
    Stage 4: Mutual Information Classifier Ranking.
    """
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.scores_ = {}
        self.ranked_features_ = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Compute Mutual Information scores between each feature and the target.
        """
        logger.info(f"MI Fit: Computing Mutual Information on {X.shape[1]} features...")
        mi_scores = mutual_info_classif(
            X.values, y,
            discrete_features=False,
            random_state=self.random_state
        )
        
        self.scores_ = {col: score for col, score in zip(X.columns, mi_scores)}
        # Sort features by score in descending order
        self.ranked_features_ = sorted(self.scores_.keys(), key=lambda k: self.scores_[k], reverse=True)
        logger.info("MI Fit: Completed.")


class mRMRRanker:
    """
    Stage 5: Minimum Redundancy Maximum Relevance (mRMR) Ranker.
    Uses pre-computed Mutual Information scores as relevance and absolute Pearson correlation as redundancy.
    """
    def __init__(self):
        self.ranked_features_ = []
        
    def fit(self, X: pd.DataFrame, relevance_scores: Dict[str, float]) -> None:
        """
        Run mRMR greedy search.
        """
        logger.info("mRMR Fit: Starting greedy selection...")
        feature_names = list(X.columns)
        n_features = len(feature_names)
        
        # Absolute Pearson correlation matrix
        corr = X.corr().abs().values
        
        # Relevance scores matching features
        relevance = np.array([relevance_scores.get(col, 0.0) for col in feature_names])
        
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # Choose first feature with highest relevance
        first_idx = int(np.argmax(relevance))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Keep a running sum of absolute correlations of each feature with selected features
        running_corr_sum = corr[:, first_idx]
        
        # Greedy loop
        while remaining_indices:
            rem_arr = np.array(remaining_indices)
            # score = Relevance - Redundancy
            # Redundancy is average correlation with already selected features
            scores = relevance[rem_arr] - (running_corr_sum[rem_arr] / len(selected_indices))
            
            best_rem_idx = np.argmax(scores)
            best_idx = remaining_indices[best_rem_idx]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Update running correlation sum
            running_corr_sum += corr[:, best_idx]
            
        self.ranked_features_ = [feature_names[i] for i in selected_indices]
        logger.info("mRMR Fit: Completed.")


class TreeImportanceRanker:
    """
    Stage 6: Tree-Based Importance Ranker.
    Trains Random Forest, Extra Trees, XGBoost, and LightGBM, and aggregates their feature importances.
    """
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.scores_ = {}
        self.ranked_features_ = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train tree-based models and aggregate their feature importances.
        """
        logger.info("Tree Fit: Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=30,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        rf_imp = rf.feature_importances_
        
        logger.info("Tree Fit: Training Extra Trees Classifier...")
        et = ExtraTreesClassifier(
            n_estimators=30,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        et.fit(X, y)
        et_imp = et.feature_importances_
        
        logger.info("Tree Fit: Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=20,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
        xgb_model.fit(X, y)
        xgb_imp = xgb_model.feature_importances_
        
        logger.info("Tree Fit: Training LightGBM Classifier...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=20,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_model.fit(X, y)
        lgb_imp = lgb_model.feature_importances_

        
        # Normalize each model's importances using MinMaxScaler to range [0, 1]
        scaler = MinMaxScaler()
        rf_imp_norm = scaler.fit_transform(rf_imp.reshape(-1, 1)).flatten()
        et_imp_norm = scaler.fit_transform(et_imp.reshape(-1, 1)).flatten()
        xgb_imp_norm = scaler.fit_transform(xgb_imp.reshape(-1, 1)).flatten()
        lgb_imp_norm = scaler.fit_transform(lgb_imp.reshape(-1, 1)).flatten()
        
        # Average the normalized importances
        avg_imp = (rf_imp_norm + et_imp_norm + xgb_imp_norm + lgb_imp_norm) / 4.0
        
        self.scores_ = {col: avg_imp[i] for i, col in enumerate(X.columns)}
        self.rf_scores_ = {col: rf_imp_norm[i] for i, col in enumerate(X.columns)}
        self.et_scores_ = {col: et_imp_norm[i] for i, col in enumerate(X.columns)}
        self.xgb_scores_ = {col: xgb_imp_norm[i] for i, col in enumerate(X.columns)}
        self.lgb_scores_ = {col: lgb_imp_norm[i] for i, col in enumerate(X.columns)}
        
        self.ranked_features_ = sorted(self.scores_.keys(), key=lambda k: self.scores_[k], reverse=True)
        logger.info("Tree Fit: Completed.")


class RFERanker:
    """
    Stage 7: Recursive Feature Elimination (RFE) Ranker.
    Uses a step-wise elimination strategy with a fast Random Forest classifier.
    """
    def __init__(self, n_features_to_select: int = 50, random_state: int = RANDOM_STATE):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.ranked_features_ = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Perform RFE to rank features.
        """
        logger.info("RFE Fit: Running Recursive Feature Elimination...")
        # To make it fast, we use a simple RandomForest classifier with fewer trees
        estimator = RandomForestClassifier(
            n_estimators=15,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # We step-wise eliminate 20% of remaining features at each iteration
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features_to_select,
            step=0.2
        )
        rfe.fit(X, y)
        
        # RFE ranking_: ranking_[i] is 1 for selected, and > 1 for eliminated (lower ranking_ means selected earlier)
        # So we sort columns by their ranking_ value
        ranks = rfe.ranking_
        self.ranked_features_ = sorted(X.columns, key=lambda col: ranks[X.columns.get_loc(col)])
        logger.info("RFE Fit: Completed.")


class ConsensusRanker:
    """
    Combines individual rankings from MI, mRMR, Tree-based importances, and RFE
    to produce a final consensus ranking based on Borda Count (average rank).
    """
    def __init__(self):
        self.ranked_features_ = []
        self.consensus_df_ = pd.DataFrame()
        
    def fit(self, feature_names: List[str], rankings: Dict[str, List[str]]) -> None:
        """
        Aggregate rankings.
        rankings is a dict mapping method name (e.g. 'mi', 'mrmr') to ordered list of feature names.
        """
        logger.info("Consensus Fit: Aggregating rankings using Borda Count...")
        
        rank_records = []
        for feat in feature_names:
            record = {"feature_name": feat}
            # For each method, get the rank (1-indexed) of this feature
            for method, ranked_list in rankings.items():
                try:
                    rank = ranked_list.index(feat) + 1
                except ValueError:
                    # Feature not ranked by this method (should not happen if all features are input)
                    rank = len(feature_names)
                record[f"{method}_rank"] = rank
            rank_records.append(record)
            
        df = pd.DataFrame(rank_records)
        
        # Compute mean rank across all columns ending with '_rank'
        rank_cols = [c for c in df.columns if c.endswith("_rank")]
        df["consensus_score"] = df[rank_cols].mean(axis=1)
        
        # Sort features by consensus score (lower score is better rank)
        df_sorted = df.sort_values(by="consensus_score").reset_index(drop=True)
        
        # Assign final rank
        df_sorted["consensus_rank"] = df_sorted.index + 1
        
        self.consensus_df_ = df_sorted
        self.ranked_features_ = df_sorted["feature_name"].tolist()
        logger.info("Consensus Fit: Completed.")
