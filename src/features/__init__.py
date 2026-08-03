"""
sEMG Prosthetic Gesture Classification
Package: features

Exposes feature engineering extractor classes for time, frequency, and wavelet domains.
"""

from .time_domain import (
    TimeDomainFeatureExtractor,
    BatchFeatureExtractor as TimeDomainBatchFeatureExtractor,
    MetadataGenerator as TimeDomainMetadataGenerator
)
from .frequency_domain import (
    FrequencyFeatureExtractor,
    BatchFeatureExtractor as SpectralBatchFeatureExtractor,
    MetadataGenerator as SpectralMetadataGenerator,
    FeatureValidator as SpectralFeatureValidator
)
from .wavelet import (
    WaveletFeatureExtractor,
    WaveletMetadataGenerator
)
