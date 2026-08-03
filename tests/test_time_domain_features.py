"""
Tests for time-domain feature extraction modules
"""

import pytest
import numpy as np
from src.features.time_domain import TimeDomainFeatureExtractor, FeatureValidator

def test_extractor_shape():
    """
    Verify that extractor correctly outputs the expected tensor shape.
    """
    extractor = TimeDomainFeatureExtractor()
    # 5 windows, 400 samples, 12 channels
    dummy_windows = np.random.normal(size=(5, 400, 12))
    
    feats = extractor.extract_features(dummy_windows)
    
    # Expected shape: (N, num_features, C) = (5, 25, 12)
    assert feats.shape == (5, 25, 12)

def test_extractor_math():
    """
    Verify mathematical correctness of specific feature values.
    """
    extractor = TimeDomainFeatureExtractor()
    
    # 1 window, 100 samples, 1 channel with value 2.0
    constant_window = np.ones((1, 100, 1)) * 2.0
    
    feats = extractor.extract_features(constant_window)
    
    # Mean should be 2.0
    assert pytest.approx(feats[0, 0, 0]) == 2.0
    # Median should be 2.0
    assert pytest.approx(feats[0, 1, 0]) == 2.0
    # Max should be 2.0
    assert pytest.approx(feats[0, 2, 0]) == 2.0
    # Min should be 2.0
    assert pytest.approx(feats[0, 3, 0]) == 2.0
    # P2P should be 0.0
    assert pytest.approx(feats[0, 4, 0]) == 0.0
    # MAV should be 2.0
    assert pytest.approx(feats[0, 5, 0]) == 2.0
    # RMS should be 2.0
    assert pytest.approx(feats[0, 6, 0]) == 2.0
    # IEMG should be 200.0 (100 * 2.0)
    assert pytest.approx(feats[0, 7, 0]) == 200.0
    # Variance should be 0.0
    assert pytest.approx(feats[0, 8, 0]) == 0.0
    # Waveform length should be 0.0
    assert pytest.approx(feats[0, 10, 0]) == 0.0

def test_validator():
    """
    Verify that validator detects invalid shapes.
    """
    validator = FeatureValidator()
    
    # 2D instead of 3D
    invalid_windows = np.random.normal(size=(400, 12))
    
    with pytest.raises(ValueError):
        validator.validate_input(invalid_windows)
