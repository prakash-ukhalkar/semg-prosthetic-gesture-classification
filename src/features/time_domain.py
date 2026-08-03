"""
sEMG Prosthetic Gesture Classification
Module: features.time_domain

Provides a high-performance, vectorized time-domain feature extraction library
capable of processing 3D window tensors.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    SAMPLING_RATE, ZC_THRESHOLD, SSC_THRESHOLD, WAMP_THRESHOLD, FEATURE_DATABASE_PATH
)

class TimeDomainFeatureExtractor:
    """
    Computes 25 publication-quality time-domain features on sEMG window tensors.
    All calculations are vectorized over the window samples axis (axis=1) for speed.
    """
    def __init__(
        self,
        zc_thresh: float = ZC_THRESHOLD,
        ssc_thresh: float = SSC_THRESHOLD,
        wamp_thresh: float = WAMP_THRESHOLD
    ):
        self.zc_thresh = zc_thresh
        self.ssc_thresh = ssc_thresh
        self.wamp_thresh = wamp_thresh
        
        # Feature names list in alphabetical/logical order
        self.feature_names = [
            "mean", "median", "max", "min", "p2p",
            "mav", "rms", "iemg", "var", "std",
            "wl", "mad", "zc", "ssc", "wamp",
            "ssi", "energy", "power", "ld", "aac",
            "dasdv", "skew", "kurt", "cov", "entropy"
        ]

    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from a 3D windows tensor of shape (num_windows, window_samples, channels).
        
        Returns:
            np.ndarray: 3D feature tensor of shape (num_windows, num_features, channels).
        """
        # Ensure input is 3D float
        windows = windows.astype(np.float64)
        N, L, C = windows.shape
        num_feats = len(self.feature_names)
        
        # Initialize output
        feat_tensor = np.zeros((N, num_feats, C))
        
        # 1. Mean
        mean = np.mean(windows, axis=1)
        feat_tensor[:, 0, :] = mean
        
        # 2. Median
        feat_tensor[:, 1, :] = np.median(windows, axis=1)
        
        # 3. Maximum
        max_val = np.max(windows, axis=1)
        feat_tensor[:, 2, :] = max_val
        
        # 4. Minimum
        min_val = np.min(windows, axis=1)
        feat_tensor[:, 3, :] = min_val
        
        # 5. Peak-to-Peak (P2P)
        feat_tensor[:, 4, :] = max_val - min_val
        
        # 6. Mean Absolute Value (MAV)
        abs_windows = np.abs(windows)
        mav = np.mean(abs_windows, axis=1)
        feat_tensor[:, 5, :] = mav
        
        # 7. Root Mean Square (RMS)
        rms = np.sqrt(np.mean(windows**2, axis=1))
        feat_tensor[:, 6, :] = rms
        
        # 8. Integrated EMG (IEMG)
        feat_tensor[:, 7, :] = np.sum(abs_windows, axis=1)
        
        # 9. Variance (VAR)
        # Use ddof=1 for sample variance; if L <= 1, variance is 0.
        var = np.var(windows, axis=1, ddof=1) if L > 1 else np.zeros((N, C))
        feat_tensor[:, 8, :] = var
        
        # 10. Standard Deviation (STD)
        std = np.std(windows, axis=1, ddof=1) if L > 1 else np.zeros((N, C))
        feat_tensor[:, 9, :] = std
        
        # 11. Waveform Length (WL)
        diff_windows = np.diff(windows, axis=1)
        abs_diff = np.abs(diff_windows)
        feat_tensor[:, 10, :] = np.sum(abs_diff, axis=1)
        
        # 12. Mean Absolute Deviation (MAD)
        mad = np.mean(np.abs(windows - mean[:, np.newaxis, :]), axis=1)
        feat_tensor[:, 11, :] = mad
        
        # 13. Zero Crossing (ZC)
        sign_change = np.diff(np.sign(windows), axis=1)
        # Count crossing only if sign changes (sign_change != 0) and amplitude difference exceeds zc_thresh
        zc_cond = (sign_change != 0) & (abs_diff >= self.zc_thresh)
        feat_tensor[:, 12, :] = np.sum(zc_cond, axis=1)
        
        # 14. Slope Sign Changes (SSC)
        # SSSC counts sign changes in slope d = x_i - x_{i-1}
        # SSSC if (x_i - x_{i-1}) * (x_i - x_{i+1}) > 0
        d1 = windows[:, 1:-1, :] - windows[:, :-2, :]
        d2 = windows[:, 1:-1, :] - windows[:, 2:, :]
        ssc_cond = (d1 * d2 > 0) & ((np.abs(d1) >= self.ssc_thresh) | (np.abs(d2) >= self.ssc_thresh))
        feat_tensor[:, 13, :] = np.sum(ssc_cond, axis=1)
        
        # 15. Willison Amplitude (WAMP)
        feat_tensor[:, 14, :] = np.sum(abs_diff > self.wamp_thresh, axis=1)
        
        # 16. Simple Square Integral (SSI)
        ssi = np.sum(windows**2, axis=1)
        feat_tensor[:, 15, :] = ssi
        
        # 17. Signal Energy
        feat_tensor[:, 16, :] = ssi
        
        # 18. Signal Power
        feat_tensor[:, 17, :] = rms**2
        
        # 19. Log Detector (LD)
        # Add epsilon to prevent log(0)
        feat_tensor[:, 18, :] = np.exp(np.mean(np.log(abs_windows + 1e-10), axis=1))
        
        # 20. Average Amplitude Change (AAC)
        feat_tensor[:, 19, :] = np.mean(abs_diff, axis=1)
        
        # 21. Difference Absolute Standard Deviation Value (DASDV)
        # DASDV = sqrt( 1/(L-1) * sum((x_i - x_{i-1})^2) )
        dasdv = np.sqrt(np.mean(diff_windows**2, axis=1))
        feat_tensor[:, 20, :] = dasdv
        
        # 22. Skewness
        # skew = mean((x - mean)^3) / std^3
        std_safe = np.where(std == 0.0, 1e-8, std)
        skew = np.mean((windows - mean[:, np.newaxis, :])**3, axis=1) / (std_safe**3)
        feat_tensor[:, 21, :] = skew
        
        # 23. Kurtosis
        # kurtosis = mean((x - mean)^4) / std^4 - 3 (excess kurtosis)
        kurt = np.mean((windows - mean[:, np.newaxis, :])**4, axis=1) / (std_safe**4) - 3.0
        feat_tensor[:, 22, :] = kurt
        
        # 24. Coefficient of Variation (CoV)
        # CoV = std / mean
        mean_safe = np.where(mean == 0.0, 1e-8, mean)
        feat_tensor[:, 23, :] = std / mean_safe
        
        # 25. Signal Entropy (Shannon formulation over normalized energy distribution)
        # p_i = x_i^2 / sum(x_k^2)
        ssi_safe = np.where(ssi == 0.0, 1e-8, ssi)
        p = (windows**2) / ssi_safe[:, np.newaxis, :]
        entropy = -np.sum(p * np.log2(p + 1e-12), axis=1)
        feat_tensor[:, 24, :] = entropy
        
        return feat_tensor


class FeatureValidator:
    """
    Validates feature dimensions, types, values, and quality flags.
    """
    def __init__(self):
        self.logger = logging.getLogger("semg_prosthetic_classification")

    def validate_input(self, windows: np.ndarray) -> None:
        """
        Verify input tensor shape and absence of invalid numbers.
        """
        if windows.ndim != 3:
            raise ValueError(f"Input windows must be 3D of shape (N, L, C), got ndim={windows.ndim}")
        if np.isnan(windows).any():
            self.logger.warning("Input windows contain NaN values! These will propagate to features.")
        if np.isinf(windows).any():
            self.logger.warning("Input windows contain Inf values! These will propagate to features.")

    def validate_output(self, features: np.ndarray, expected_shape: Tuple[int, int, int]) -> None:
        """
        Check that features array dimensions and finite counts are valid.
        """
        if features.shape != expected_shape:
            raise ValueError(f"Output features shape mismatch: expected {expected_shape}, got {features.shape}")
        
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        if nan_count > 0:
            self.logger.error(f"Features extraction produced {nan_count} NaN values!")
        if inf_count > 0:
            self.logger.error(f"Features extraction produced {inf_count} Inf values!")


class BatchFeatureExtractor:
    """
    Extracts features subject-by-subject and saves a consolidated Parquet file.
    """
    def __init__(self, extractor: TimeDomainFeatureExtractor):
        self.extractor = extractor
        self.validator = FeatureValidator()
        self.logger = logging.getLogger("semg_prosthetic_classification")

    def process_all_subjects(self, processed_dir: Path, output_parquet: Path) -> pd.DataFrame:
        """
        Run subject-by-subject batch extraction and return the consolidated DataFrame.
        """
        subject_dirs = sorted(list(processed_dir.glob("DB2_s*")))
        if not subject_dirs:
            raise FileNotFoundError(f"No processed subject directories found at: {processed_dir}")
            
        all_features_df_list = []
        
        self.logger.info(f"Starting batch feature extraction over {len(subject_dirs)} subjects...")
        start_time = time.time()
        
        for sub_dir in subject_dirs:
            npz_files = sorted(list(sub_dir.glob("*.npz")))
            self.logger.info(f"Processing subject {sub_dir.name} ({len(npz_files)} segmented runs)...")
            
            for npz_file in npz_files:
                # Load corresponding metadata CSV if exists
                file_stem = npz_file.name.replace("_segmented.npz", "")
                csv_path = sub_dir / f"{file_stem}_segmented_metadata.csv"
                
                if not csv_path.exists():
                    self.logger.warning(f"Metadata CSV missing for {npz_file.name}, skipping.")
                    continue
                    
                df_meta = pd.read_csv(csv_path)
                data = np.load(str(npz_file))
                windows = data["windows"]
                labels = data["labels"]
                reps = data["repetitions"]
                
                # Validation
                self.validator.validate_input(windows)
                
                # Extraction
                feat_tensor = self.extractor.extract_features(windows)
                
                # Validate output
                N, num_feats, C = feat_tensor.shape
                self.validator.validate_output(feat_tensor, (len(windows), len(self.extractor.feature_names), C))
                
                # Reshape to 2D: (N, num_features * channels)
                # Layout: f1_ch1, f1_ch2... f1_chC, f2_ch1...
                feat_2d = feat_tensor.reshape(N, num_feats * C)
                
                # Column naming
                col_names = []
                for feat in self.extractor.feature_names:
                    for ch in range(C):
                        col_names.append(f"{feat}_ch{ch+1}")
                        
                df_features = pd.DataFrame(feat_2d, columns=col_names)
                
                # Add metadata columns
                df_features["window_id"] = df_meta["window_id"]
                df_features["subject_id"] = df_meta["subject_id"]
                df_features["exercise_id"] = df_meta["exercise_id"]
                df_features["gesture_id"] = df_meta["gesture_id"]
                df_features["repetition_id"] = df_meta["repetition_id"]
                
                all_features_df_list.append(df_features)
                
        # Consolidate
        self.logger.info("Consolidating all feature tables...")
        df_global = pd.concat(all_features_df_list, ignore_index=True)
        
        # Save as Parquet
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_global.to_parquet(output_parquet, index=False)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Batch feature extraction completed in {elapsed:.2f} seconds. Saved to: {output_parquet}")
        return df_global


class MetadataGenerator:
    """
    Generates reference metadata for extracted time-domain features.
    """
    @staticmethod
    def get_definitions() -> Dict[str, Any]:
        """
        Return descriptions, formulas, and references for all 25 features.
        """
        return {
            "mean": {
                "name": "Mean Value",
                "formula": "1/N * sum(x_i)",
                "description": "Calculates the average offset of the sEMG signal, useful for tracking slow baseline changes.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "median": {
                "name": "Median Value",
                "formula": "median(x_i)",
                "description": "Calculates the median value of the window samples.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "max": {
                "name": "Maximum Amplitude",
                "formula": "max(x_i)",
                "description": "Peak positive deflection in the window.",
                "units": "Normalized amplitude",
                "reference": "Boostani & Moradi, 2003"
            },
            "min": {
                "name": "Minimum Amplitude",
                "formula": "min(x_i)",
                "description": "Peak negative deflection in the window.",
                "units": "Normalized amplitude",
                "reference": "Boostani & Moradi, 2003"
            },
            "p2p": {
                "name": "Peak-to-Peak Amplitude",
                "formula": "max(x_i) - min(x_i)",
                "description": "Range of the signal amplitude deflections.",
                "units": "Normalized amplitude",
                "reference": "Boostani & Moradi, 2003"
            },
            "mav": {
                "name": "Mean Absolute Value",
                "formula": "1/N * sum(|x_i|)",
                "description": "Estimate of muscle contraction amplitude, highly standard in myoelectric control.",
                "units": "Normalized amplitude",
                "reference": "Englehart & Hudgins, 2003"
            },
            "rms": {
                "name": "Root Mean Square",
                "formula": "sqrt(1/N * sum(x_i^2))",
                "description": "Relates to the constant power of the signal and motor unit recruitment.",
                "units": "Normalized amplitude",
                "reference": "Hogan et al., 1980"
            },
            "iemg": {
                "name": "Integrated EMG",
                "formula": "sum(|x_i|)",
                "description": "Represents the total area under the absolute curve, indicating effort.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "var": {
                "name": "Variance",
                "formula": "1/(N-1) * sum((x_i - mean)^2)",
                "description": "Measures the dispersion of the signal power around the mean.",
                "units": "Normalized amplitude^2",
                "reference": "Phinyomark et al., 2012"
            },
            "std": {
                "name": "Standard Deviation",
                "formula": "sqrt(VAR)",
                "description": "Standard deviation representing variation magnitude.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "wl": {
                "name": "Waveform Length",
                "formula": "sum(|x_i - x_{i-1}|)",
                "description": "Accumulated changes in sign, slope, and frequency, indicative of complexity.",
                "units": "Normalized amplitude",
                "reference": "Hudgins et al., 1993"
            },
            "mad": {
                "name": "Mean Absolute Deviation",
                "formula": "1/N * sum(|x_i - mean|)",
                "description": "Dispersion measure less sensitive to outliers than variance.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "zc": {
                "name": "Zero Crossings",
                "formula": "sum(1 if sign changes and diff >= thresh else 0)",
                "description": "Counts crossing points, capturing dominant frequency trends in time domain.",
                "units": "Count",
                "reference": "Hudgins et al., 1993"
            },
            "ssc": {
                "name": "Slope Sign Changes",
                "formula": "sum(1 if slope changes sign and deflection >= thresh else 0)",
                "description": "Represents another descriptor of frequency content in the time domain.",
                "units": "Count",
                "reference": "Hudgins et al., 1993"
            },
            "wamp": {
                "name": "Willison Amplitude",
                "formula": "sum(1 if |x_i - x_{i-1}| > thresh else 0)",
                "description": "Counts times signal deflection changes significantly, indicating motor unit firing.",
                "units": "Count",
                "reference": "Willison, 1964"
            },
            "ssi": {
                "name": "Simple Square Integral",
                "formula": "sum(x_i^2)",
                "description": "Sum of squared values, representing total signal energy.",
                "units": "Normalized amplitude^2",
                "reference": "Phinyomark et al., 2012"
            },
            "energy": {
                "name": "Signal Energy",
                "formula": "sum(x_i^2)",
                "description": "Identical to SSI, tracking energy magnitude.",
                "units": "Normalized amplitude^2",
                "reference": "Boostani & Moradi, 2003"
            },
            "power": {
                "name": "Signal Power",
                "formula": "1/N * sum(x_i^2)",
                "description": "Average energy per sample.",
                "units": "Normalized amplitude^2",
                "reference": "Boostani & Moradi, 2003"
            },
            "ld": {
                "name": "Log Detector",
                "formula": "exp(1/N * sum(log(|x_i| + eps)))",
                "description": "Provides non-linear scaling of muscle activity level.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "aac": {
                "name": "Average Amplitude Change",
                "formula": "1/N * sum(|x_i - x_{i-1}|)",
                "description": "Mean absolute derivative of the signal.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "dasdv": {
                "name": "Difference Absolute Standard Deviation Value",
                "formula": "sqrt(1/(N-1) * sum((x_i - x_{i-1})^2))",
                "description": "Standard deviation of consecutive sample differences.",
                "units": "Normalized amplitude",
                "reference": "Phinyomark et al., 2012"
            },
            "skew": {
                "name": "Skewness",
                "formula": "mean((x - mean)^3) / std^3",
                "description": "Measures asymmetry of the signal distribution.",
                "units": "Dimensionless",
                "reference": "Kandasamy et al., 2013"
            },
            "kurt": {
                "name": "Kurtosis",
                "formula": "mean((x - mean)^4) / std^4 - 3",
                "description": "Measures tailedness/peakedness of the signal distribution.",
                "units": "Dimensionless",
                "reference": "Kandasamy et al., 2013"
            },
            "cov": {
                "name": "Coefficient of Variation",
                "formula": "std / mean",
                "description": "Measures relative variability compared to the signal mean.",
                "units": "Dimensionless",
                "reference": "Phinyomark et al., 2012"
            },
            "entropy": {
                "name": "Time-Domain Shannon Entropy",
                "formula": "-sum(p_i * log2(p_i + eps)) where p_i = x_i^2 / energy",
                "description": "Measures complexity and uncertainty of muscle activation energy in the time domain.",
                "units": "Bits",
                "reference": "Shannon, 1948"
            }
        }
