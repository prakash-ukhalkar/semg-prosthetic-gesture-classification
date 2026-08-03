"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.metadata

Generates metadata reports containing mathematical definitions, domain classifications,
channel tracking, and consensus scores for selected features.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


from src.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Mathematical definitions for core features
CORE_MATHEMATICAL_DEFINITIONS = {
    # Time domain
    "mean": "Mean: 1/N * sum(x_i)",
    "median": "Median: median(x)",
    "max": "Maximum: max(x)",
    "min": "Minimum: min(x)",
    "p2p": "Peak-to-Peak Amplitude: max(x) - min(x)",
    "mav": "Mean Absolute Value: 1/N * sum(|x_i|)",
    "rms": "Root Mean Square: sqrt(1/N * sum(x_i^2))",
    "iemg": "Integrated EMG: sum(|x_i|)",
    "var": "Variance: 1/(N-1) * sum((x_i - mean)^2)",
    "std": "Standard Deviation: sqrt(VAR)",
    "wl": "Waveform Length: sum(|x_i - x_{i-1}|)",
    "mad": "Mean Absolute Deviation: 1/N * sum(|x_i - mean|)",
    "zc": "Zero Crossing Rate: sum((x_i * x_{i+1} < 0) & (|x_i - x_{i+1}| > thresh))",
    "ssc": "Slope Sign Change Rate: sum(((x_i - x_{i-1})*(x_i - x_{i+1}) > 0) & threshold check)",
    "wamp": "Willison Amplitude: sum(|x_i - x_{i-1}| > thresh)",
    "ssi": "Simple Square Integral: sum(x_i^2)",
    "energy": "Energy: sum(x_i^2)",
    "power": "Power: 1/N * sum(x_i^2)",
    "ld": "Log Detector: exp(1/N * sum(log(|x_i|)))",
    "aac": "Average Amplitude Change: 1/(N-1) * sum(|x_i - x_{i-1}|)",
    "dasdv": "Difference Absolute Standard Deviation Value: sqrt(1/(N-1) * sum((x_i - x_{i-1})^2))",
    "skew": "Skewness (3rd Moment): E[(x - mean)^3] / std^3",
    "kurt": "Kurtosis (4th Moment): E[(x - mean)^4] / std^4",
    "cov": "Coefficient of Variation: std / mean",
    "entropy": "Shannon Entropy of standard distribution: -sum(p_i * log2(p_i))",
    
    # Frequency domain
    "mnf": "Mean Frequency: sum(f_i * P_i) / sum(P_i)",
    "mdf": "Median Frequency: frequency splitting power spectrum into equal halves",
    "pkf": "Peak Frequency: frequency with maximum power spectrum value",
    "centroid": "Spectral Centroid",
    "spectral_entropy": "Spectral Entropy: Shannon entropy of normalized power spectrum",
    "spectral_energy": "Spectral Energy: sum(P_i)",
    "spectral_power": "Spectral Power: mean power spectrum value",
    "spectral_variance": "Spectral Variance: variance of power spectrum",
    "spectral_skewness": "Spectral Skewness: skewness of power spectrum",
    "spectral_kurtosis": "Spectral Kurtosis: kurtosis of power spectrum",
    "psd_mean": "PSD Mean: average power spectral density",
    "psd_peak": "PSD Peak: peak of power spectral density",
    "roll_off": "Spectral Roll-off: frequency below which 85% of spectral power lies",
    "flatness": "Spectral Flatness: ratio of geometric mean to arithmetic mean of spectrum",
    "flux": "Spectral Flux: change in power spectrum between windows",
    "fr": "Frequency Ratio: ratio of low-frequency (20-150Hz) to high-frequency (150-500Hz) power",
    "bp_20_150": "Band Power (20-150Hz): low frequency range power",
    "bp_150_300": "Band Power (150-300Hz): mid frequency range power",
    "bp_300_500": "Band Power (300-500Hz): high frequency range power"
}

WAVELET_COEF_NAMES = {
    "ca4": "Approximation Level 4 (0-62.5 Hz)",
    "cd4": "Detail Level 4 (62.5-125 Hz)",
    "cd3": "Detail Level 3 (125-250 Hz)",
    "cd2": "Detail Level 2 (250-500 Hz)",
    "cd1": "Detail Level 1 (500-1000 Hz)"
}

WAVELET_STAT_NAMES = {
    "energy": "Wavelet energy of coefficient level: sum(c_i^2)",
    "entropy": "Wavelet Shannon entropy of coefficients",
    "mean": "Wavelet mean: mean(c_i)",
    "var": "Wavelet variance: var(c_i)",
    "std": "Wavelet standard deviation: std(c_i)",
    "rms": "Wavelet root mean square: sqrt(mean(c_i^2))",
    "max": "Wavelet maximum: max(c_i)",
    "min": "Wavelet minimum: min(c_i)",
    "ratio": "Wavelet ratio: energy(c_i) / total_wavelet_energy",
    "rel_energy": "Wavelet relative energy of coefficient level",
    "log_energy": "Wavelet log energy of coefficient level: sum(log(c_i^2))"
}

class FeatureMetadataManager:
    """
    Manages generation of feature metadata JSON describing mathematical and architectural traits.
    """
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        
    def generate_metadata_json(
        self,
        selected_features: List[str],
        consensus_df: pd.DataFrame,
        rankings: Dict[str, List[str]],
        top_n_limit: int = 150
    ) -> Path:
        """
        Build feature_metadata.json detailing categories, channels, math formulas, and scores.
        """
        logger.info(f"Generating feature metadata JSON for top {len(selected_features)} features...")
        
        metadata_dict = {}
        
        for idx, feat in enumerate(selected_features):
            # Parse channel number (e.g. mean_ch1 -> 1, dwt_ca4_energy_ch12 -> 12)
            ch_match = re.search(r"_ch(\d+)$", feat)
            channel = int(ch_match.group(1)) if ch_match else None
            
            # Determine base name and categories
            base_name = re.sub(r"_ch\d+$", "", feat)
            
            math_def = ""
            category = ""
            domain = ""
            
            if base_name in CORE_MATHEMATICAL_DEFINITIONS:
                math_def = CORE_MATHEMATICAL_DEFINITIONS[base_name]
                if feat.startswith("spectral_") or base_name in ["mnf", "mdf", "pkf", "centroid", "psd_mean", "psd_peak", "roll_off", "flatness", "flux", "fr"] or base_name.startswith("bp_"):
                    category = "Frequency-Domain"
                    domain = "Frequency"
                else:
                    category = "Time-Domain"
                    domain = "Time"
            elif base_name.startswith("dwt_"):
                category = "Time-Frequency (Wavelet)"
                domain = "Time-Frequency"
                # Parse wavelet coef and stat, e.g. dwt_ca4_energy
                parts = base_name.split("_")
                coef = parts[1]  # ca4, cd4, cd3, etc.
                stat = "_".join(parts[2:])  # energy, rel_energy, etc.
                
                coef_desc = WAVELET_COEF_NAMES.get(coef, f"Wavelet Coef {coef}")
                stat_desc = WAVELET_STAT_NAMES.get(stat, f"Wavelet Stat {stat}")
                
                math_def = f"Discrete Wavelet Transform (db4, Level 4) - {coef_desc}. Stat: {stat_desc}."
            else:
                category = "Unknown"
                domain = "Unknown"
                math_def = "Custom computed sEMG feature."
                
            # Get consensus ranking info
            row = consensus_df[consensus_df["feature_name"] == feat]
            consensus_rank = int(row["consensus_rank"].values[0]) if not row.empty else None
            consensus_score = float(row["consensus_score"].values[0]) if not row.empty else None
            
            # Determine which individual methods support this feature (were in top_n_limit)
            supporting_methods = []
            for method, ranked_list in rankings.items():
                if feat in ranked_list[:top_n_limit]:
                    supporting_methods.append(method)
                    
            metadata_dict[feat] = {
                "name": feat,
                "base_feature": base_name,
                "category": category,
                "domain": domain,
                "channel": channel,
                "mathematical_definition": math_def,
                "consensus_rank": consensus_rank,
                "consensus_score": consensus_score,
                "selection_methods_supporting": supporting_methods
            }
            
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)
            
        logger.info(f"Feature metadata JSON successfully saved to {self.output_path}")
        return self.output_path
