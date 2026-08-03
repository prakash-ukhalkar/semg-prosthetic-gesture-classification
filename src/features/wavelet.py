"""
sEMG Prosthetic Gesture Classification
Module: features.wavelet

Provides a high-performance, vectorized wavelet-domain feature extraction library
capable of processing 3D window tensors.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pywt

from src.config import WAVELET_FAMILY, WAVELET_LEVEL


class WaveletFeatureExtractor:
    """
    Computes Discrete Wavelet Transform (DWT) features on sEMG window tensors.
    All calculations are vectorized over windows and channels for optimal performance.
    """
    def __init__(
        self,
        wavelet_name: str = WAVELET_FAMILY,
        level: int = WAVELET_LEVEL
    ):
        self.wavelet_name = wavelet_name
        self.level = level
        self.logger = logging.getLogger("semg_prosthetic_classification")

        # Generate feature name strings based on decomposition level
        self.subbands = [f"ca{level}"] + [f"cd{i}" for i in range(level, 0, -1)]
        self.subband_features = [
            "energy", "entropy", "mean", "var", "std",
            "rms", "max", "min", "ratio", "rel_energy", "log_energy"
        ]

        self.feature_names = []
        for sb in self.subbands:
            for feat in self.subband_features:
                self.feature_names.append(f"dwt_{sb}_{feat}")

    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Decomposes a 3D window tensor and extracts wavelet features.
        
        Args:
            windows: 3D float array of shape (num_windows, window_samples, channels).
            
        Returns:
            np.ndarray: 3D feature tensor of shape (num_windows, num_features, channels).
        """
        # Ensure float64 precision
        windows = windows.astype(np.float64)
        N, L, C = windows.shape
        num_feats = len(self.feature_names)

        # 1. Multi-level DWT decomposition along sample axis (axis=1)
        # Returns [cA_L, cD_L, cD_L-1, ..., cD_1]
        coeffs = pywt.wavedec(windows, self.wavelet_name, level=self.level, axis=1)

        # 2. Compute energies for all subbands first (needed for ratios)
        energies = []
        for coeff in coeffs:
            # coeff shape: (N, coeff_samples, C)
            energies.append(np.sum(coeff**2, axis=1))
        
        total_energy = np.sum(energies, axis=0)
        energy_approx = energies[0] # cA_L is the first element

        # 3. Compute all features for each subband
        feat_tensor = np.zeros((N, num_feats, C))
        feat_idx = 0

        for sb_idx, coeff in enumerate(coeffs):
            coeff_len = coeff.shape[1]
            
            # Energy (already calculated)
            energy = energies[sb_idx]
            feat_tensor[:, feat_idx, :] = energy
            
            # Entropy: Shannon entropy of normalized coefficient distribution
            energy_safe = np.where(energy == 0.0, 1e-12, energy)
            p = (coeff**2) / energy_safe[:, np.newaxis, :]
            entropy = -np.sum(p * np.log2(p + 1e-12), axis=1)
            feat_tensor[:, feat_idx + 1, :] = entropy
            
            # Mean
            mean = np.mean(coeff, axis=1)
            feat_tensor[:, feat_idx + 2, :] = mean
            
            # Variance and SD (with sample correction ddof=1)
            if coeff_len > 1:
                var = np.var(coeff, axis=1, ddof=1)
                std = np.std(coeff, axis=1, ddof=1)
            else:
                var = np.zeros((N, C))
                std = np.zeros((N, C))
            feat_tensor[:, feat_idx + 3, :] = var
            feat_tensor[:, feat_idx + 4, :] = std
            
            # Root Mean Square (RMS)
            rms = np.sqrt(np.mean(coeff**2, axis=1))
            feat_tensor[:, feat_idx + 5, :] = rms
            
            # Max & Min
            feat_tensor[:, feat_idx + 6, :] = np.max(coeff, axis=1)
            feat_tensor[:, feat_idx + 7, :] = np.min(coeff, axis=1)
            
            # Coefficient Energy Ratio: subband energy relative to approximation subband
            approx_safe = np.where(energy_approx == 0.0, 1e-12, energy_approx)
            ratio = energy / approx_safe
            feat_tensor[:, feat_idx + 8, :] = ratio
            
            # Relative Energy: subband energy relative to total wavelet energy
            total_safe = np.where(total_energy == 0.0, 1e-12, total_energy)
            rel_energy = energy / total_safe
            feat_tensor[:, feat_idx + 9, :] = rel_energy
            
            # Log Energy
            log_energy = np.sum(np.log(coeff**2 + 1e-12), axis=1)
            feat_tensor[:, feat_idx + 10, :] = log_energy

            feat_idx += len(self.subband_features)

        return feat_tensor


class WaveletMetadataGenerator:
    """
    Generates reference metadata for extracted wavelet-domain features.
    """
    @staticmethod
    def get_definitions(wavelet_name: str = WAVELET_FAMILY, level: int = WAVELET_LEVEL) -> Dict[str, Any]:
        """
        Return descriptions, formulas, and references for all wavelet features.
        """
        subbands = [f"ca{level}"] + [f"cd{i}" for i in range(level, 0, -1)]
        definitions = {}
        
        for sb in subbands:
            sb_type = "approximation" if "ca" in sb else "detail"
            level_num = sb.replace("ca", "").replace("cd", "")
            
            definitions[f"dwt_{sb}_energy"] = {
                "name": f"DWT {sb.upper()} Energy",
                "formula": f"sum(c_{{{sb},i}}^2)",
                "description": f"Total energy of the wavelet {sb_type} coefficients at level {level_num}, reflecting signal power in this frequency band.",
                "units": "Normalized amplitude^2",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_entropy"] = {
                "name": f"DWT {sb.upper()} Shannon Entropy",
                "formula": f"-sum(p_i * log2(p_i)) where p_i = c_i^2 / energy",
                "description": f"Shannon entropy of normalized coefficient energy at level {level_num}, measuring signal complexity in this sub-band.",
                "units": "Bits",
                "reference": "Rosso et al., 2001"
            }
            definitions[f"dwt_{sb}_mean"] = {
                "name": f"DWT {sb.upper()} Mean",
                "formula": f"1/N * sum(c_i)",
                "description": f"Average value of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_var"] = {
                "name": f"DWT {sb.upper()} Variance",
                "formula": f"1/(N-1) * sum((c_i - mean)^2)",
                "description": f"Variance of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude^2",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_std"] = {
                "name": f"DWT {sb.upper()} Standard Deviation",
                "formula": f"sqrt(VAR)",
                "description": f"Standard deviation of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_rms"] = {
                "name": f"DWT {sb.upper()} Root Mean Square",
                "formula": f"sqrt(1/N * sum(c_i^2))",
                "description": f"Root mean square value of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude",
                "reference": "Englehart & Hudgins, 2003"
            }
            definitions[f"dwt_{sb}_max"] = {
                "name": f"DWT {sb.upper()} Maximum",
                "formula": f"max(c_i)",
                "description": f"Maximum amplitude of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_min"] = {
                "name": f"DWT {sb.upper()} Minimum",
                "formula": f"min(c_i)",
                "description": f"Minimum amplitude of the wavelet {sb_type} coefficients at level {level_num}.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            }
            definitions[f"dwt_{sb}_ratio"] = {
                "name": f"DWT {sb.upper()} Coefficient Energy Ratio",
                "formula": f"energy_{sb} / energy_approx",
                "description": f"Ratio of {sb_type} coefficient energy at level {level_num} to the approximation (ca{level}) coefficient energy.",
                "units": "Dimensionless",
                "reference": "Englehart et al., 1999"
            }
            definitions[f"dwt_{sb}_rel_energy"] = {
                "name": f"DWT {sb.upper()} Relative Energy",
                "formula": f"energy_{sb} / energy_total",
                "description": f"Relative energy contribution of the {sb_type} sub-band at level {level_num} to the total DWT decomposition energy.",
                "units": "Dimensionless",
                "reference": "Rosso et al., 2001"
            }
            definitions[f"dwt_{sb}_log_energy"] = {
                "name": f"DWT {sb.upper()} Log Energy",
                "formula": f"sum(log(c_i^2))",
                "description": f"Logarithmic energy sum of the wavelet {sb_type} coefficients at level {level_num}, highlighting low-energy transients.",
                "units": "Dimensionless",
                "reference": "Phinyomark et al., 2012"
            }

        return definitions
