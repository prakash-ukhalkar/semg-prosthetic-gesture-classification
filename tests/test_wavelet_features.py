"""
Tests for wavelet-domain feature extraction modules
"""

import pytest
import numpy as np
from src.features.wavelet import WaveletFeatureExtractor

def test_wavelet_extractor_shape():
    """
    Verify that wavelet extractor correctly outputs the expected tensor shape.
    """
    extractor = WaveletFeatureExtractor(wavelet_name="db4", level=4)
    # 5 windows, 400 samples, 12 channels
    dummy_windows = np.random.normal(size=(5, 400, 12))
    
    feats = extractor.extract_features(dummy_windows)
    
    # Expected shape: (N, num_features, C)
    # (level + 1) * 11 features = 5 * 11 = 55 features
    assert feats.shape == (5, 55, 12)

def test_wavelet_extractor_math():
    """
    Verify wavelet feature extraction on simple inputs.
    """
    extractor = WaveletFeatureExtractor(wavelet_name="db4", level=4)
    
    # 2 windows, 400 samples, 2 channels
    constant_window = np.ones((2, 400, 2)) * 3.5
    
    feats = extractor.extract_features(constant_window)
    
    # Check that outputs are finite and do not contain NaNs or Infs
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
    
    # Relative energy sum of all sub-bands should sum to 1.0 (approximately) for each window/channel
    # The relative energy features are at indices: 9, 20, 31, 42, 53
    rel_energy_indices = [9, 20, 31, 42, 53]
    for w in range(2):
        for c in range(2):
            sum_rel_energy = sum(feats[w, idx, c] for idx in rel_energy_indices)
            assert pytest.approx(sum_rel_energy) == 1.0
