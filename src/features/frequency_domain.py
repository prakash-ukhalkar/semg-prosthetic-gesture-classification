"""
sEMG Prosthetic Gesture Classification
Module: features.frequency_domain

Provides a high-performance, vectorized frequency-domain feature extraction library
capable of processing 3D window tensors, along with batch processing orchestrations.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    SAMPLING_RATE, SPECTRAL_BANDS, FR_SPLIT_FREQ, FREQ_WAVELET_DATABASE_PATH,
    SPECTRAL_BATCH_SIZE, SPECTRAL_PARALLEL_WORKERS, WAVELET_FAMILY, WAVELET_LEVEL
)
from src.features.wavelet import WaveletFeatureExtractor, WaveletMetadataGenerator


class FrequencyFeatureExtractor:
    """
    Computes 17 publication-quality frequency-domain features on sEMG window tensors.
    All calculations are vectorized over windows and channels for optimal performance.
    """
    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        bands: List[Tuple[float, float]] = SPECTRAL_BANDS,
        fr_split_freq: float = FR_SPLIT_FREQ,
        roll_off_thresh: float = 0.90
    ):
        self.sampling_rate = sampling_rate
        self.bands = bands
        self.fr_split_freq = fr_split_freq
        self.roll_off_thresh = roll_off_thresh
        self.logger = logging.getLogger("semg_prosthetic_classification")

        # Static list of feature name roots
        self.feature_names = [
            "mnf", "mdf", "pkf", "centroid", "entropy",
            "energy", "power", "variance", "skewness", "kurtosis",
            "psd_mean", "psd_peak", "roll_off", "flatness", "flux", "fr"
        ]
        # Append band powers dynamically
        for band in self.bands:
            self.feature_names.append(f"bp_{int(band[0])}_{int(band[1])}")

    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract frequency features from a 3D windows tensor of shape (num_windows, window_samples, channels).
        
        Returns:
            np.ndarray: 3D feature tensor of shape (num_windows, num_features, channels).
        """
        windows = windows.astype(np.float64)
        N, L, C = windows.shape
        num_feats = len(self.feature_names)

        # 1. Compute FFT Magnitude and Power Spectral Density (PSD)
        fft_vals = np.fft.rfft(windows, axis=1)
        freqs = np.fft.rfftfreq(L, d=1.0 / self.sampling_rate)
        M = len(freqs)
        X = np.abs(fft_vals)

        # PSD calculation (one-sided periodogram)
        P = (X**2) / (self.sampling_rate * L)
        if M > 2:
            P[:, 1:-1, :] *= 2.0  # Scale to conserve power in one-sided spectrum

        # Initialize feature tensor
        feat_tensor = np.zeros((N, num_feats, C))

        # 2. Vectorized Feature Extraction
        # Spectral Moments (needed for multiple features)
        # M_k = sum(f^k * P)
        M0 = np.sum(P, axis=1)  # Total Power (also SM0)
        M0_safe = np.where(M0 == 0.0, 1e-12, M0)

        # Mean Frequency (MNF)
        M1 = np.sum(P * freqs[np.newaxis, :, np.newaxis], axis=1)
        mnf = M1 / M0_safe
        feat_tensor[:, 0, :] = mnf

        # Median Frequency (MDF)
        cum_P = np.cumsum(P, axis=1)
        mdf_idx = np.argmax(cum_P >= 0.5 * M0[:, np.newaxis, :], axis=1)
        feat_tensor[:, 1, :] = freqs[mdf_idx]

        # Peak Frequency (PKF)
        pkf_idx = np.argmax(P, axis=1)
        feat_tensor[:, 2, :] = freqs[pkf_idx]

        # Spectral Centroid (calculated using magnitude spectrum)
        X_sum = np.sum(X, axis=1)
        X_sum_safe = np.where(X_sum == 0.0, 1e-12, X_sum)
        centroid = np.sum(X * freqs[np.newaxis, :, np.newaxis], axis=1) / X_sum_safe
        feat_tensor[:, 3, :] = centroid

        # Spectral Entropy
        p_norm = P / M0_safe[:, np.newaxis, :]
        entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12), axis=1)
        feat_tensor[:, 4, :] = entropy

        # Spectral Energy (sum of squared FFT magnitude)
        feat_tensor[:, 5, :] = np.sum(X**2, axis=1)

        # Spectral Power (Mean Power MNP)
        feat_tensor[:, 6, :] = np.mean(P, axis=1)

        # Spectral Variance, Skewness, Kurtosis
        diff = freqs[np.newaxis, :, np.newaxis] - mnf[:, np.newaxis, :]
        variance = np.sum(p_norm * (diff**2), axis=1)
        feat_tensor[:, 7, :] = variance

        std_safe = np.sqrt(np.where(variance == 0.0, 1e-12, variance))
        skewness = np.sum(p_norm * (diff / std_safe[:, np.newaxis, :])**3, axis=1)
        feat_tensor[:, 8, :] = skewness

        kurtosis = np.sum(p_norm * (diff / std_safe[:, np.newaxis, :])**4, axis=1) - 3.0
        feat_tensor[:, 9, :] = kurtosis

        # PSD Statistics (Mean and Peak)
        feat_tensor[:, 10, :] = np.mean(P, axis=1)
        feat_tensor[:, 11, :] = np.max(P, axis=1)

        # Spectral Roll-off
        roll_idx = np.argmax(cum_P >= self.roll_off_thresh * M0[:, np.newaxis, :], axis=1)
        feat_tensor[:, 12, :] = freqs[roll_idx]

        # Spectral Flatness
        geo_mean = np.exp(np.mean(np.log(P + 1e-12), axis=1))
        ari_mean = np.mean(P, axis=1)
        ari_mean_safe = np.where(ari_mean == 0.0, 1e-12, ari_mean)
        feat_tensor[:, 13, :] = geo_mean / ari_mean_safe

        # Spectral Flux (rate of change of magnitude spectrum across windows)
        X_sum_keep = np.sum(X, axis=1, keepdims=True)
        X_sum_keep_safe = np.where(X_sum_keep == 0.0, 1e-12, X_sum_keep)
        X_norm = X / X_sum_keep_safe
        
        flux = np.zeros((N, C))
        if N > 1:
            flux[1:] = np.sum(np.diff(X_norm, axis=0)**2, axis=1)
        feat_tensor[:, 14, :] = flux

        # Frequency Ratio (FR)
        idx_low = freqs <= self.fr_split_freq
        idx_high = freqs > self.fr_split_freq
        low_power = np.sum(P[:, idx_low, :], axis=1)
        high_power = np.sum(P[:, idx_high, :], axis=1)
        high_power_safe = np.where(high_power == 0.0, 1e-12, high_power)
        feat_tensor[:, 15, :] = low_power / high_power_safe

        # Band Powers
        feat_idx = 16
        for band in self.bands:
            idx_band = (freqs >= band[0]) & (freqs <= band[1])
            feat_tensor[:, feat_idx, :] = np.sum(P[:, idx_band, :], axis=1)
            feat_idx += 1

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


def _process_single_subject(
    sub_dir_str: str,
    temp_dir_str: str,
    sampling_rate: int,
    bands: List[Tuple[float, float]],
    fr_split_freq: float,
    wavelet_name: str,
    wavelet_level: int,
    batch_size: int
) -> str:
    """
    Top-level helper function to process a single subject.
    Must be defined at the module level for pickling on Windows.
    """
    sub_dir = Path(sub_dir_str)
    temp_dir = Path(temp_dir_str)
    subject_name = sub_dir.name
    out_parquet = temp_dir / f"{subject_name}_features.parquet"

    # Skip extraction if this subject is already processed (enabling resumability)
    if out_parquet.exists():
        return f"✔ Subject {subject_name} already processed (loaded cache)."

    # Initialize extractors & validator inside worker process
    freq_extractor = FrequencyFeatureExtractor(
        sampling_rate=sampling_rate,
        bands=bands,
        fr_split_freq=fr_split_freq
    )
    wavelet_extractor = WaveletFeatureExtractor(
        wavelet_name=wavelet_name,
        level=wavelet_level
    )
    validator = FeatureValidator()
    
    npz_files = sorted(list(sub_dir.glob("*.npz")))
    if not npz_files:
        return f"⚠ Subject {subject_name} has no segmented npz files."

    all_subject_dfs = []

    for npz_file in npz_files:
        file_stem = npz_file.name.replace("_segmented.npz", "")
        csv_path = sub_dir / f"{file_stem}_segmented_metadata.csv"
        
        if not csv_path.exists():
            continue
            
        df_meta = pd.read_csv(csv_path)
        data = np.load(str(npz_file))
        windows = data["windows"]
        
        validator.validate_input(windows)
        
        N, L, C = windows.shape
        
        # Chunked processing to minimize memory usage
        freq_chunks = []
        wavelet_chunks = []
        
        for idx in range(0, N, batch_size):
            chunk = windows[idx:idx+batch_size]
            
            # Extract frequency features
            feat_freq = freq_extractor.extract_features(chunk)
            validator.validate_output(feat_freq, (len(chunk), len(freq_extractor.feature_names), C))
            
            # Extract wavelet features
            feat_wav = wavelet_extractor.extract_features(chunk)
            validator.validate_output(feat_wav, (len(chunk), len(wavelet_extractor.feature_names), C))
            
            # Reshape features to 2D
            n_chunk = len(chunk)
            feat_freq_2d = feat_freq.reshape(n_chunk, len(freq_extractor.feature_names) * C)
            feat_wav_2d = feat_wav.reshape(n_chunk, len(wavelet_extractor.feature_names) * C)
            
            freq_chunks.append(feat_freq_2d)
            wavelet_chunks.append(feat_wav_2d)

        # Concatenate chunks
        subject_freq = np.vstack(freq_chunks)
        subject_wav = np.vstack(wavelet_chunks)

        # Generate wide column names
        freq_cols = []
        for feat in freq_extractor.feature_names:
            for ch in range(C):
                freq_cols.append(f"{feat}_ch{ch+1}")

        wav_cols = []
        for feat in wavelet_extractor.feature_names:
            for ch in range(C):
                wav_cols.append(f"{feat}_ch{ch+1}")

        df_freq = pd.DataFrame(subject_freq, columns=freq_cols)
        df_wav = pd.DataFrame(subject_wav, columns=wav_cols)

        # Combine features with metadata
        df_combined = pd.concat([df_meta, df_freq, df_wav], axis=1)
        all_subject_dfs.append(df_combined)

    if all_subject_dfs:
        df_subject_global = pd.concat(all_subject_dfs, ignore_index=True)
        # Ensure target directory exists and save
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_subject_global.to_parquet(out_parquet, index=False)
        return f"✔ Subject {subject_name} completed successfully."
    else:
        return f"⚠ Subject {subject_name} failed (no data processed)."


class BatchFeatureExtractor:
    """
    Orchestrates subject-by-subject batch extraction using parallel worker processes.
    Generates a unified feature database containing frequency and wavelet features.
    """
    def __init__(
        self,
        freq_extractor: FrequencyFeatureExtractor,
        wavelet_extractor: WaveletFeatureExtractor,
        n_jobs: int = SPECTRAL_PARALLEL_WORKERS,
        batch_size: int = SPECTRAL_BATCH_SIZE
    ):
        self.freq_extractor = freq_extractor
        self.wavelet_extractor = wavelet_extractor
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.logger = logging.getLogger("semg_prosthetic_classification")

    def process_all_subjects(self, processed_dir: Path, output_parquet: Path) -> pd.DataFrame:
        """
        Extract features for all subjects using ProcessPoolExecutor.
        """
        subject_dirs = sorted(list(processed_dir.glob("DB2_s*")))
        if not subject_dirs:
            raise FileNotFoundError(f"No processed subject directories found at: {processed_dir}")

        if output_parquet.exists():
            self.logger.info(f"✔ Consolidated feature database already exists at {output_parquet}. Loading directly...")
            return pd.read_parquet(output_parquet)

        temp_dir = output_parquet.parent / "temp_spectral"
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Starting parallel batch spectral feature extraction over {len(subject_dirs)} subjects "
            f"using {self.n_jobs} parallel workers..."
        )
        start_time = time.time()

        # Submit jobs to ProcessPoolExecutor
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _process_single_subject,
                    str(sub_dir),
                    str(temp_dir),
                    self.freq_extractor.sampling_rate,
                    self.freq_extractor.bands,
                    self.freq_extractor.fr_split_freq,
                    self.wavelet_extractor.wavelet_name,
                    self.wavelet_extractor.level,
                    self.batch_size
                ): sub_dir.name
                for sub_dir in subject_dirs
            }

            for future in as_completed(futures):
                sub_name = futures[future]
                try:
                    res_msg = future.result()
                    self.logger.info(res_msg)
                except Exception as e:
                    self.logger.error(f"❌ Subject {sub_name} failed: {e}")

        # Consolidate all subject parquets
        self.logger.info("Consolidating all subject feature tables...")
        subject_parquet_files = sorted(list(temp_dir.glob("*_features.parquet")))
        
        all_features_df_list = []
        for f in subject_parquet_files:
            all_features_df_list.append(pd.read_parquet(f))

        if not all_features_df_list:
            raise RuntimeError("No features were successfully extracted!")

        df_global = pd.concat(all_features_df_list, ignore_index=True)

        # Save unified parquet
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_global.to_parquet(output_parquet, index=False)

        # Cleanup intermediate subject files
        for f in subject_parquet_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        elapsed = time.time() - start_time
        self.logger.info(
            f"Parallel batch feature extraction completed in {elapsed:.2f} seconds. "
            f"Saved to: {output_parquet}"
        )
        return df_global


class MetadataGenerator:
    """
    Generates reference metadata for both frequency-domain and wavelet-domain features.
    """
    @staticmethod
    def get_definitions(
        sampling_rate: int = SAMPLING_RATE,
        bands: List[Tuple[float, float]] = SPECTRAL_BANDS,
        fr_split_freq: float = FR_SPLIT_FREQ,
        wavelet_name: str = WAVELET_FAMILY,
        wavelet_level: int = WAVELET_LEVEL
    ) -> Dict[str, Any]:
        """
        Merge frequency-domain and wavelet-domain definitions.
        """
        # Frequency-domain definitions
        freq_defs = {
            "mnf": {
                "name": "Mean Frequency",
                "formula": "sum(f_j * P_j) / sum(P_j)",
                "description": "The first spectral moment, indicating the centroid of the power spectral density.",
                "units": "Hz",
                "reference": "Phinyomark et al., 2012"
            },
            "mdf": {
                "name": "Median Frequency",
                "formula": "sum_{j=1}^{MDF} P_j = sum_{j=MDF}^{M} P_j = 0.5 * sum(P)",
                "description": "The frequency that divides the PSD into two regions of equal power, highly sensitive to muscle fatigue.",
                "units": "Hz",
                "reference": "Phinyomark et al., 2012"
            },
            "pkf": {
                "name": "Peak Frequency",
                "formula": "argmax_f(P)",
                "description": "The frequency corresponding to the maximum power spectral density value.",
                "units": "Hz",
                "reference": "Boostani & Moradi, 2003"
            },
            "centroid": {
                "name": "Spectral Centroid",
                "formula": "sum(f_j * X_j) / sum(X_j)",
                "description": "Centroid of the magnitude spectrum (rather than PSD), measuring spectral shape.",
                "units": "Hz",
                "reference": "Phinyomark et al., 2012"
            },
            "entropy": {
                "name": "Spectral Entropy",
                "formula": "-sum(p_norm_j * log2(p_norm_j)) where p_norm_j = P_j / sum(P)",
                "description": "Shannon entropy of the normalized PSD, indicating spectral complexity and flatness.",
                "units": "Bits",
                "reference": "Inouye et al., 1991"
            },
            "energy": {
                "name": "Spectral Energy",
                "formula": "sum(X_j^2)",
                "description": "Sum of squared FFT magnitude values, representing total signal energy in the frequency domain.",
                "units": "Normalized amplitude^2",
                "reference": "Phinyomark et al., 2012"
            },
            "power": {
                "name": "Spectral Power",
                "formula": "1/M * sum(P)",
                "description": "Average power spectral density across all bins (Mean Power MNP).",
                "units": "Normalized amplitude^2 / Hz",
                "reference": "Phinyomark et al., 2012"
            },
            "variance": {
                "name": "Spectral Variance",
                "formula": "sum(p_norm_j * (f_j - MNF)^2)",
                "description": "Second central moment of the PSD, describing the spread of spectral frequencies.",
                "units": "Hz^2",
                "reference": "Phinyomark et al., 2012"
            },
            "skewness": {
                "name": "Spectral Skewness",
                "formula": "sum(p_norm_j * ((f_j - MNF)/std)^3)",
                "description": "Third central moment of the PSD, indicating the asymmetry of the spectrum.",
                "units": "Dimensionless",
                "reference": "Phinyomark et al., 2012"
            },
            "kurtosis": {
                "name": "Spectral Kurtosis",
                "formula": "sum(p_norm_j * ((f_j - MNF)/std)^4) - 3",
                "description": "Fourth central moment of the PSD, indicating the peakedness or flatness of the spectrum.",
                "units": "Dimensionless",
                "reference": "Phinyomark et al., 2012"
            },
            "psd_mean": {
                "name": "PSD Mean Value",
                "formula": "1/M * sum(P)",
                "description": "Mean power density value across the spectrum, identical to average spectral power.",
                "units": "Normalized amplitude^2 / Hz",
                "reference": "Phinyomark et al., 2012"
            },
            "psd_peak": {
                "name": "PSD Peak Value",
                "formula": "max(P)",
                "description": "Maximum power spectral density value, highlighting the dominant firing power.",
                "units": "Normalized amplitude^2 / Hz",
                "reference": "Boostani & Moradi, 2003"
            },
            "roll_off": {
                "name": "Spectral Roll-off",
                "formula": "f_k where sum_{j=1}^k P_j >= thresh * sum(P)",
                "description": "The frequency below which 90% of the total spectral power resides.",
                "units": "Hz",
                "reference": "Scheirer & Slaney, 1997"
            },
            "flatness": {
                "name": "Spectral Flatness",
                "formula": "exp(1/M * sum(log(P))) / (1/M * sum(P))",
                "description": "Ratio of geometric mean to arithmetic mean of the PSD, indicating signal noisiness.",
                "units": "Dimensionless",
                "reference": "Johnston, 1988"
            },
            "flux": {
                "name": "Spectral Flux",
                "formula": "sum((X_t - X_{t-1})^2) where X is normalized magnitude",
                "description": "Rate of change of the normalized magnitude spectrum between consecutive windows.",
                "units": "Dimensionless",
                "reference": "Scheirer & Slaney, 1997"
            },
            "fr": {
                "name": "Frequency Ratio",
                "formula": "sum(P_{f <= split}) / sum(P_{f > split})",
                "description": "Ratio of power in low-frequency band (<= 150 Hz) to high-frequency band (> 150 Hz).",
                "units": "Dimensionless",
                "reference": "Phinyomark et al., 2012"
            }
        }

        # Add band powers
        for band in bands:
            low, high = int(band[0]), int(band[1])
            freq_defs[f"bp_{low}_{high}"] = {
                "name": f"Band Power {low}-{high} Hz",
                "formula": f"sum(P_j) for f_j in [{low}, {high}]",
                "description": f"Integrated power spectral density in the {low} to {high} Hz frequency band.",
                "units": "Normalized amplitude^2 / Hz",
                "reference": "Phinyomark et al., 2012"
            }

        # Wavelet-domain definitions
        wavelet_defs = WaveletMetadataGenerator.get_definitions(
            wavelet_name=wavelet_name,
            level=wavelet_level
        )

        # Merge them
        unified_defs = {}
        unified_defs.update(freq_defs)
        unified_defs.update(wavelet_defs)
        return unified_defs
