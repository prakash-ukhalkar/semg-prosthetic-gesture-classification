"""
Unit tests for the src/feature_selection package.
"""

import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

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

@pytest.fixture
def dummy_data():
    """
    Generate small dummy data for testing.
    """
    np.random.seed(42)
    n_samples = 200
    
    # Metadata
    df_meta = pd.DataFrame({
        "subject_id": [1] * (n_samples // 2) + [2] * (n_samples // 2),
        "exercise_id": [1] * n_samples,
        "gesture_id": np.random.randint(0, 5, size=n_samples),
        "window_id": list(range(n_samples // 2)) * 2,
        "repetition_id": [1] * n_samples
    })
    
    # Time domain features (10 features)
    time_feats = {}
    for i in range(1, 11):
        if i == 5:
            # Constant feature
            time_feats[f"feat_ch{i}"] = np.ones(n_samples)
        elif i == 6:
            # Near zero variance feature
            time_feats[f"feat_ch{i}"] = np.random.normal(0, 0.001, size=n_samples)
        else:
            time_feats[f"feat_ch{i}"] = np.random.normal(0, 1.0, size=n_samples)
            
    df_time = pd.concat([df_meta, pd.DataFrame(time_feats)], axis=1)
    
    # Frequency domain features (10 features)
    freq_feats = {}
    for i in range(1, 11):
        freq_feats[f"freq_ch{i}"] = np.random.normal(0, 1.0, size=n_samples)
        
    # Introduce correlation
    # freq_ch3 is highly correlated with feat_ch1
    freq_feats["freq_ch3"] = df_time["feat_ch1"] * 0.99 + np.random.normal(0, 0.01, size=n_samples)
    
    # Include clashing columns to test renaming
    freq_feats["entropy_ch1"] = np.random.normal(0, 1.0, size=n_samples)
    df_time["entropy_ch1"] = np.random.normal(0, 1.0, size=n_samples)
    
    # Freq-only metadata
    freq_meta = pd.DataFrame({
        "start_sample": [0] * n_samples,
        "end_sample": [400] * n_samples,
        "window_size_samples": [400] * n_samples,
        "sampling_frequency_hz": [2000] * n_samples
    })
    
    df_freq = pd.concat([df_meta, freq_meta, pd.DataFrame(freq_feats)], axis=1)
    
    return df_time, df_freq

def test_fusion_and_quality(dummy_data):
    df_time, df_freq = dummy_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        time_p = tmp_path / "time.parquet"
        freq_p = tmp_path / "freq.parquet"
        out_p = tmp_path / "merged.parquet"
        inv_p = tmp_path / "inventory.csv"
        rep_p = tmp_path / "report.json"
        
        # Save dummy data
        df_time.to_parquet(time_p, index=False)
        df_freq.to_parquet(freq_p, index=False)
        
        # Fuse
        fuser = FeatureFuser(time_p, freq_p, out_p, inv_p)
        fuser.fuse_databases(subjects=[1, 2])
        
        # Check merged database output
        assert out_p.exists()
        df_merged = pd.read_parquet(out_p)
        assert len(df_merged) == len(df_time)
        assert "spectral_entropy_ch1" in df_merged.columns
        assert "entropy_ch1" in df_merged.columns
        
        # Check feature inventory
        assert inv_p.exists()
        df_inv = pd.read_csv(inv_p)
        assert len(df_inv) > 0
        assert "category" in df_inv.columns
        
        # Quality Validation
        validator = FeatureQualityValidator(out_p, rep_p)
        report = validator.validate_features()
        assert rep_p.exists()
        assert "constant_features" in report
        assert "feat_ch5" in report["constant_features"]

def test_filters(dummy_data):
    df_time, df_freq = dummy_data
    # Merge them manually for simple filtering test
    df_freq_renamed = df_freq.rename(columns={"entropy_ch1": "spectral_entropy_ch1"})
    df = pd.concat([df_time, df_freq_renamed.drop(columns=["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id"])], axis=1)
    df_feats = df.drop(columns=["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id", "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"])
    
    # Stage 1: Constant feature removal
    remover = ConstantFeatureRemover()
    remover.fit(df_feats)
    assert "feat_ch5" in remover.constant_cols
    df_s1 = remover.transform(df_feats)
    assert "feat_ch5" not in df_s1.columns
    
    # Stage 2: NZV removal
    nzv = NearZeroVarianceRemover(threshold=0.01)
    nzv.fit(df_s1)
    assert "feat_ch6" in nzv.low_var_cols
    df_s2 = nzv.transform(df_s1)
    assert "feat_ch6" not in df_s2.columns
    
    # Stage 3: Correlation filtering
    corr = CorrelationFilter(threshold=0.90)
    corr.fit(df_s2)
    assert "freq_ch3" in corr.drop_cols or "feat_ch1" in corr.drop_cols
    df_s3 = corr.transform(df_s2)
    assert not ("freq_ch3" in df_s3.columns and "feat_ch1" in df_s3.columns)

def test_rankers(dummy_data):
    df_time, df_freq = dummy_data
    df_freq_renamed = df_freq.rename(columns={"entropy_ch1": "spectral_entropy_ch1"})
    df = pd.concat([df_time, df_freq_renamed.drop(columns=["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id"])], axis=1)
    
    # Select feature cols
    meta_cols = ["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id", "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feat_cols]
    y = df["gesture_id"].values
    
    # MI Ranker
    mi = MIEstimator(random_state=42)
    mi.fit(X, y)
    assert len(mi.ranked_features_) == len(feat_cols)
    assert len(mi.scores_) == len(feat_cols)
    
    # mRMR
    mrmr = mRMRRanker()
    mrmr.fit(X, mi.scores_)
    assert len(mrmr.ranked_features_) == len(feat_cols)
    
    # Trees
    trees = TreeImportanceRanker(random_state=42)
    trees.fit(X, y)
    assert len(trees.ranked_features_) == len(feat_cols)
    
    # RFE
    rfe = RFERanker(n_features_to_select=3, random_state=42)
    rfe.fit(X, y)
    assert len(rfe.ranked_features_) == len(feat_cols)
    
    # Consensus
    rankings = {
        "mi": mi.ranked_features_,
        "mrmr": mrmr.ranked_features_,
        "tree": trees.ranked_features_,
        "rfe": rfe.ranked_features_
    }
    consensus = ConsensusRanker()
    consensus.fit(feat_cols, rankings)
    assert len(consensus.ranked_features_) == len(feat_cols)
    assert consensus.consensus_df_.shape[0] == len(feat_cols)
    assert "consensus_rank" in consensus.consensus_df_.columns

def test_metadata_manager(dummy_data):
    df_time, df_freq = dummy_data
    df_freq_renamed = df_freq.rename(columns={"entropy_ch1": "spectral_entropy_ch1"})
    df = pd.concat([df_time, df_freq_renamed.drop(columns=["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id"])], axis=1)
    meta_cols = ["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id", "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    rankings = {
        "mi": feat_cols,
        "mrmr": feat_cols,
        "tree": feat_cols,
        "rfe": feat_cols
    }
    
    consensus = ConsensusRanker()
    consensus.fit(feat_cols, rankings)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        meta_p = tmp_path / "metadata.json"
        
        manager = FeatureMetadataManager(meta_p)
        manager.generate_metadata_json(
            selected_features=consensus.ranked_features_[:5],
            consensus_df=consensus.consensus_df_,
            rankings=rankings,
            top_n_limit=5
        )
        
        assert meta_p.exists()
        with open(meta_p, "r") as f:
            data = json.load(f)
            assert len(data) == 5
            # Test that fields exist
            first_feat = list(data.keys())[0]
            assert "mathematical_definition" in data[first_feat]
            assert "category" in data[first_feat]
            assert "channel" in data[first_feat]
