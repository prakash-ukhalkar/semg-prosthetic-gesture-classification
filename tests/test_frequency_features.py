"""
Tests for frequency-domain feature extraction modules
"""

import pytest
import numpy as np
from src.features.frequency_domain import FrequencyFeatureExtractor, FeatureValidator

def test_frequency_extractor_shape():
    """
    Verify that frequency extractor correctly outputs the expected tensor shape.
    """
    extractor = FrequencyFeatureExtractor()
    # 5 windows, 400 samples, 12 channels
    dummy_windows = np.random.normal(size=(5, 400, 12))
    
    feats = extractor.extract_features(dummy_windows)
    
    # Expected shape: (N, num_features, C)
    # 17 features: 16 base features + 3 band powers = 19 features
    assert feats.shape == (5, 19, 12)

def test_frequency_extractor_math():
    """
    Verify mathematical correctness of frequency feature extraction.
    """
    extractor = FrequencyFeatureExtractor()
    
    # 1 window, 100 samples, 1 channel with value 2.0 (DC only)
    constant_window = np.ones((1, 100, 1)) * 2.0
    
    feats = extractor.extract_features(constant_window)
    
    # Mean frequency (MNF) should be 0.0 (only DC component present)
    assert pytest.approx(feats[0, 0, 0]) == 0.0
    
    # Median frequency (MDF) should be 0.0
    assert pytest.approx(feats[0, 1, 0]) == 0.0
    
    # Peak frequency (PKF) should be 0.0
    assert pytest.approx(feats[0, 2, 0]) == 0.0

def test_frequency_validator():
    """
    Verify that validator detects invalid shapes.
    """
    validator = FeatureValidator()
    
    # 2D instead of 3D
    invalid_windows = np.random.normal(size=(400, 12))
    
    with pytest.raises(ValueError):
        validator.validate_input(invalid_windows)
