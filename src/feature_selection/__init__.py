"""
sEMG Prosthetic Gesture Classification
Module: feature_selection

Initialize the feature selection package and expose key components.
"""

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
from src.feature_selection.pipeline import FeatureSelectionPipeline

__all__ = [
    "FeatureFuser",
    "FeatureQualityValidator",
    "ConstantFeatureRemover",
    "NearZeroVarianceRemover",
    "CorrelationFilter",
    "MIEstimator",
    "mRMRRanker",
    "TreeImportanceRanker",
    "RFERanker",
    "ConsensusRanker",
    "FeatureMetadataManager",
    "FeatureSelectionPipeline"
]
