"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.pipeline

Coordinates the entire feature fusion and selection pipeline, supporting
resumable execution by caching intermediate results on disk.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from src.config import LOGGER_NAME, RANDOM_STATE
from src.feature_selection.fusion import FeatureFuser
from src.feature_selection.quality import FeatureQualityValidator
from src.feature_selection.filters import (
    ConstantFeatureRemover,
    NearZeroVarianceRemover,
    CorrelationFilter
)
from src.feature_selection.rankers import (
    MIEstimator,
    mRMRRanker,
    TreeImportanceRanker,
    RFERanker,
    ConsensusRanker
)
from src.feature_selection.metadata import FeatureMetadataManager

logger = logging.getLogger(LOGGER_NAME)

class FeatureSelectionPipeline:
    """
    Orchestrates the feature selection pipeline from fusion to consensus output.
    """
    def __init__(
        self,
        time_path: Path,
        freq_path: Path,
        workspace_dir: Path,
        random_state: int = RANDOM_STATE
    ):
        self.time_path = Path(time_path)
        self.freq_path = Path(freq_path)
        self.workspace_dir = Path(workspace_dir)
        self.random_state = random_state
        
        # Directories
        self.interim_dir = self.workspace_dir / "data" / "interim"
        self.final_dir = self.workspace_dir / "data" / "final"
        self.reports_dir = self.workspace_dir / "outputs" / "reports"
        
        # Intermediate/Output files
        self.merged_path = self.workspace_dir / "data" / "processed" / "merged_features.parquet"
        self.inventory_path = self.workspace_dir / "data" / "processed" / "feature_inventory.csv"
        self.quality_path = self.reports_dir / "quality_report.json"
        
        self.stage3_features_path = self.interim_dir / "stage3_features.json"
        self.rankings_cache_path = self.interim_dir / "rankings_individual.json"
        
        self.final_rankings_path = self.final_dir / "feature_rankings.parquet"
        self.metadata_json_path = self.final_dir / "feature_metadata.json"
        
        # Ensure directories exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_cols = [
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        ]
        
    def run_pipeline(
        self,
        nzv_threshold: float = 1e-4,
        corr_threshold: float = 0.85,
        training_subjects: List[int] = None,
        mi_sample_size: int = 20000,
        rfe_sample_size: int = 15000,
        rfe_n_select: int = 50,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the feature selection pipeline.
        """
        logger.info("Initializing Feature Selection Pipeline...")
        
        # Default training subjects to a representative 25% of subjects to save RAM
        if training_subjects is None:
            training_subjects = [1, 5, 10, 15, 20, 25, 30, 35, 40]
            
        # 1. Database Fusion
        if force_recompute or not self.merged_path.exists():
            fuser = FeatureFuser(
                time_path=self.time_path,
                freq_path=self.freq_path,
                output_path=self.merged_path,
                inventory_path=self.inventory_path
            )
            fuser.fuse_databases()
        else:
            logger.info(f"Fusion: Merged features database found at {self.merged_path}. Skipping fusion.")
            
        # 2. Quality Validation
        if force_recompute or not self.quality_path.exists():
            validator = FeatureQualityValidator(
                feature_path=self.merged_path,
                report_path=self.quality_path
            )
            quality_report = validator.validate_features()
        else:
            logger.info(f"Quality: Quality report found at {self.quality_path}. Loading.")
            with open(self.quality_path, "r", encoding="utf-8") as f:
                quality_report = json.load(f)
                
        # 3. Stages 1-3: Filtering (Constant, NZV, Correlation)
        if not force_recompute and self.stage3_features_path.exists():
            logger.info(f"Filters: Filtered feature list found at {self.stage3_features_path}. Loading.")
            with open(self.stage3_features_path, "r", encoding="utf-8") as f:
                filtered_features = json.load(f)
        else:
            logger.info("Filters: Executing Stage 1-3 Filtering...")
            # We load the data for training subjects only to run filtering fit steps (saves memory)
            logger.info(f"Filters: Loading data for representative training subjects: {training_subjects}")
            df_train = pd.read_parquet(
                self.merged_path,
                filters=[("subject_id", "in", training_subjects)]
            )
            df_feats = df_train.drop(columns=[c for c in self.metadata_cols if c in df_train.columns])
            
            # Stage 1: Constant Feature Removal
            constant_remover = ConstantFeatureRemover(constant_cols=quality_report.get("constant_features", []))
            df_s1 = constant_remover.transform(df_feats)
            logger.info(f"Stage 1 Complete. Columns: {df_s1.shape[1]}")
            
            # Stage 2: NZV Removal
            nzv_remover = NearZeroVarianceRemover(threshold=nzv_threshold)
            nzv_remover.fit(df_s1)
            df_s2 = nzv_remover.transform(df_s1)
            logger.info(f"Stage 2 Complete. Columns: {df_s2.shape[1]}")
            
            # Stage 3: Correlation Filtering
            corr_filter = CorrelationFilter(threshold=corr_threshold)
            # To speed up, we can sample the row count
            if len(df_s2) > 40000:
                df_s2_sample = df_s2.sample(n=40000, random_state=self.random_state)
            else:
                df_s2_sample = df_s2
            corr_filter.fit(df_s2_sample)
            df_s3 = corr_filter.transform(df_s2)
            logger.info(f"Stage 3 Complete. Columns: {df_s3.shape[1]}")
            
            filtered_features = list(df_s3.columns)
            with open(self.stage3_features_path, "w", encoding="utf-8") as f:
                json.dump(filtered_features, f)
            logger.info(f"Filters: Stage 1-3 features cached to {self.stage3_features_path}")
            
        logger.info(f"Total features remaining after filtering: {len(filtered_features)}")
        
        # 4. Stages 4-7: Feature Ranking & Consensus
        if not force_recompute and self.rankings_cache_path.exists():
            logger.info(f"Rankings: Cached rankings found at {self.rankings_cache_path}. Loading.")
            with open(self.rankings_cache_path, "r", encoding="utf-8") as f:
                rankings = json.load(f)
        else:
            logger.info("Rankings: Loading data for representative training subjects to run ranking algorithms...")
            # Load only the filtered features and metadata
            cols_to_load = self.metadata_cols + filtered_features
            df_train = pd.read_parquet(
                self.merged_path,
                columns=[c for c in cols_to_load if c in pq.read_schema(self.merged_path).names],
                filters=[("subject_id", "in", training_subjects)]
            )
            
            X_train = df_train[filtered_features]
            y_train = df_train["gesture_id"].values
            
            # Stage 4: Mutual Information Ranking
            # Downsample for MI
            if len(X_train) > mi_sample_size:
                # Stratified sample
                idx_sample = df_train.groupby("gesture_id", group_keys=False).apply(
                    lambda x: x.sample(n=max(1, int(mi_sample_size / len(np.unique(y_train)))), random_state=self.random_state)
                ).index
                X_mi = X_train.loc[idx_sample]
                y_mi = y_train[df_train.index.isin(idx_sample)]
            else:
                X_mi = X_train
                y_mi = y_train
                
            mi_estimator = MIEstimator(random_state=self.random_state)
            mi_estimator.fit(X_mi, y_mi)
            mi_ranked = mi_estimator.ranked_features_
            
            # Stage 5: mRMR
            mrmr_ranker = mRMRRanker()
            # Run mRMR on the same MI sample for correlation speed
            mrmr_ranker.fit(X_mi, mi_estimator.scores_)
            mrmr_ranked = mrmr_ranker.ranked_features_
            
            # Stage 6: Tree-Based Importances
            # Tree-based algorithms handle more data easily, but let's use a sample of 40,000 for training speed
            tree_sample_size = min(40000, len(X_train))
            idx_tree = df_train.groupby("gesture_id", group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(tree_sample_size / len(np.unique(y_train)))), random_state=self.random_state)
            ).index
            X_tree = X_train.loc[idx_tree]
            y_tree = y_train[df_train.index.isin(idx_tree)]
            
            tree_ranker = TreeImportanceRanker(random_state=self.random_state)
            tree_ranker.fit(X_tree, y_tree)
            tree_ranked = tree_ranker.ranked_features_
            
            # Stage 7: Recursive Feature Elimination (RFE)
            # RFE is computationally expensive, downsample to rfe_sample_size
            if len(X_train) > rfe_sample_size:
                idx_rfe = df_train.groupby("gesture_id", group_keys=False).apply(
                    lambda x: x.sample(n=max(1, int(rfe_sample_size / len(np.unique(y_train)))), random_state=self.random_state)
                ).index
                X_rfe = X_train.loc[idx_rfe]
                y_rfe = y_train[df_train.index.isin(idx_rfe)]
            else:
                X_rfe = X_train
                y_rfe = y_train
                
            rfe_ranker = RFERanker(n_features_to_select=rfe_n_select, random_state=self.random_state)
            rfe_ranker.fit(X_rfe, y_rfe)
            rfe_ranked = rfe_ranker.ranked_features_
            
            rankings = {
                "mi": mi_ranked,
                "mrmr": mrmr_ranked,
                "tree": tree_ranked,
                "rfe": rfe_ranked
            }
            
            with open(self.rankings_cache_path, "w", encoding="utf-8") as f:
                json.dump(rankings, f, indent=4)
            logger.info(f"Rankings: Rankings cached to {self.rankings_cache_path}")
            
        # 5. Consensus Ranking
        consensus_ranker = ConsensusRanker()
        consensus_ranker.fit(filtered_features, rankings)
        consensus_df = consensus_ranker.consensus_df_
        consensus_ranked_features = consensus_ranker.ranked_features_
        
        # Save feature rankings Parquet
        consensus_df.to_parquet(self.final_rankings_path, index=False)
        logger.info(f"Consensus rankings saved to {self.final_rankings_path}")
        
        # 6. Generate and save Top-N feature subsets
        sizes = [25, 50, 75, 100, 150]
        logger.info(f"Generating Top-N subsets: {sizes}...")
        
        # Read the full dataset and slice by selected features subject-by-subject (saves RAM)
        schema = pq.read_schema(self.merged_path)
        meta_cols_present = [c for c in self.metadata_cols if c in schema.names]
        
        # We will write the subsets to parquet subject-by-subject
        writers = {}
        for sz in sizes:
            out_p = self.final_dir / f"selected_features_top{sz}.parquet"
            writers[sz] = {
                "path": out_p,
                "cols": meta_cols_present + consensus_ranked_features[:sz],
                "writer": None
            }
            
        # Get list of subjects
        meta_table = pq.read_table(self.merged_path, columns=["subject_id"])
        subjects = sorted(meta_table["subject_id"].unique().to_pylist())
        
        for sub_id in subjects:
            logger.info(f"Slicing and writing subsets for subject {sub_id}...")
            # We load the columns needed for this subject
            # Let's load the maximum number of features we need (150) + metadata
            max_cols = meta_cols_present + consensus_ranked_features[:max(sizes)]
            df_sub = pd.read_parquet(
                self.merged_path,
                columns=max_cols,
                filters=[("subject_id", "==", sub_id)]
            )
            
            for sz, info in writers.items():
                df_subset = df_sub[info["cols"]]
                table = pa.Table.from_pandas(df_subset)
                if info["writer"] is None:
                    info["writer"] = pq.ParquetWriter(info["path"], table.schema, compression="snappy")
                info["writer"].write_table(table)
                
        # Close all writers
        for sz, info in writers.items():
            if info["writer"] is not None:
                info["writer"].close()
                logger.info(f"Saved selected_features_top{sz}.parquet to {info['path']}")
                
        # 7. Generate Feature Metadata JSON for top 150 features
        meta_manager = FeatureMetadataManager(self.metadata_json_path)
        meta_manager.generate_metadata_json(
            selected_features=consensus_ranked_features[:150],
            consensus_df=consensus_df,
            rankings=rankings,
            top_n_limit=150
        )
        
        logger.info("Feature Selection Pipeline completed successfully!")
        
        return {
            "consensus_df": consensus_df,
            "rankings": rankings,
            "filtered_features_count": len(filtered_features),
            "consensus_ranked_features": consensus_ranked_features,
            "quality_report": quality_report
        }
