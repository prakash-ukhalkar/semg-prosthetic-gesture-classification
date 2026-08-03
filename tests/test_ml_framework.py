"""
sEMG Prosthetic Gesture Classification
Tests for the Machine Learning benchmarking framework.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ml.splits import verify_dataset_integrity, get_subject_splits
from src.ml.models import initialize_model, get_model_catalog
from src.ml.pipelines import create_pipeline, determine_scaler_type
from src.ml.metrics import compute_metrics

@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset with features and metadata for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # 5 subjects, 4 repetitions, 5 gestures
    subject_ids = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
    repetition_ids = np.random.choice([1, 2, 3, 4], size=n_samples)
    gesture_ids = np.random.choice(range(5), size=n_samples)
    
    # Create some dummy feature columns
    features = {f"feat_{i}": np.random.randn(n_samples) for i in range(10)}
    
    # Add metadata
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "exercise_id": 1,
        "gesture_id": gesture_ids,
        "window_id": range(n_samples),
        "repetition_id": repetition_ids,
        "start_sample": 0,
        "end_sample": 400,
        "window_size_samples": 400,
        "sampling_frequency_hz": 2000,
        **features
    })
    return df

def test_verify_dataset_integrity_valid(dummy_dataset):
    """Test integrity validation on a valid dataset."""
    report = verify_dataset_integrity(dummy_dataset)
    assert report["is_valid"] is True
    assert report["missing_values"] == 0
    assert report["infinite_values"] == 0
    assert report["unique_subjects"] == 5
    assert report["unique_gestures"] == 5
    assert report["features_count"] == 10

def test_verify_dataset_integrity_invalid(dummy_dataset):
    """Test that integrity validation catches missing values, infinite values, etc."""
    bad_df = dummy_dataset.copy()
    # Introduce NaN
    bad_df.iloc[0, bad_df.columns.get_loc("feat_0")] = np.nan
    # Introduce Inf
    bad_df.iloc[1, bad_df.columns.get_loc("feat_1")] = np.inf
    
    report = verify_dataset_integrity(bad_df)
    assert report["is_valid"] is False
    assert report["missing_values"] == 1
    assert report["infinite_values"] == 1

def test_get_subject_splits(dummy_dataset):
    """Test that subject splits are disjoint and partitioned in expected ratios."""
    # We have 5 subjects: [1, 2, 3, 4, 5]
    # Let's split 60% Train, 20% Val, 20% Test (since 70/15/15 requires at least 6+ subjects to get 15%)
    train_df, val_df, test_df, meta = get_subject_splits(
        dummy_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42
    )
    
    train_subs = set(meta["train_subjects"])
    val_subs = set(meta["val_subjects"])
    test_subs = set(meta["test_subjects"])
    
    # Verify mutual exclusion of subject sets
    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)
    
    # Verify all subjects are covered
    all_subs = set(dummy_dataset["subject_id"].unique())
    assert train_subs.union(val_subs).union(test_subs) == all_subs

def test_determine_scaler_type():
    """Test auto-scaler mapping based on model type."""
    assert determine_scaler_type("logistic_regression") == "standard"
    assert determine_scaler_type("knn") == "minmax"
    assert determine_scaler_type("random_forest") == "passthrough"
    assert determine_scaler_type("xgboost") == "passthrough"
    assert determine_scaler_type("rbf_svm") == "standard"

def test_create_pipeline(dummy_dataset):
    """Test that pipeline wraps model and scaler correctly."""
    model = initialize_model("random_forest")
    pipe = create_pipeline("random_forest", model)
    assert isinstance(pipe, Pipeline)
    assert pipe.steps[0][0] == "scaler"
    assert pipe.steps[0][1] == "passthrough"
    assert pipe.steps[1][0] == "classifier"
    
    model_lr = initialize_model("logistic_regression")
    pipe_lr = create_pipeline("logistic_regression", model_lr)
    assert pipe_lr.steps[0][1].__class__.__name__ == "StandardScaler"

def test_compute_metrics():
    """Test metrics computation against manually verified values."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 1, 0]
    
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "balanced_accuracy" in metrics
    assert "mcc" in metrics
    assert "cohen_kappa" in metrics
    
    # Accuracy should be 4/6 = 0.6666...
    assert abs(metrics["accuracy"] - 4.0/6.0) < 1e-4

def test_initialize_model():
    """Test model catalog initialization and instantiation."""
    catalog = get_model_catalog()
    assert len(catalog) == 15
    for key, model in catalog.items():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
