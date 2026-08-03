"""
sEMG Prosthetic Gesture Classification
Module: preprocessing

Defines the core PreprocessingPipeline class that orchestrates signal validation,
filtering (band-pass and notch), and amplitude normalization on MAT files.
"""

import time
import json
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import scipy.io

from src.config import (
    SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    NOTCH_FREQ, NOTCH_Q, NORMALIZATION_METHOD
)
from src.filtering import apply_filtering_pipeline, validate_signal
from src.normalization import normalize_signal

class PreprocessingPipeline:
    """
    Orchestrates the sEMG preprocessing workflow: Validation -> Filtering -> Normalization.
    """
    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        lowcut: float = BANDPASS_LOW,
        highcut: float = BANDPASS_HIGH,
        bp_order: int = BANDPASS_ORDER,
        notch_freq: float = NOTCH_FREQ,
        notch_q: float = NOTCH_Q,
        norm_method: str = NORMALIZATION_METHOD
    ):
        """
        Initialize the pipeline with configurable signal processing settings.
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.bp_order = bp_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.norm_method = norm_method
        self.logger = logging.getLogger("semg_prosthetic_classification")

    def process_signal(self, emg: np.ndarray) -> np.ndarray:
        """
        Process the raw sEMG signal through validation, filtering, and normalization.

        Parameters
        ----------
        emg : np.ndarray
            Raw signal array of shape (samples, channels).

        Returns
        -------
        np.ndarray
            Fully preprocessed signal array.
        """
        # 1. Validation of raw signal
        validate_signal(emg)
        
        # 2. Sequential filtering (Band-pass butterworth followed by IIR Notch)
        filtered = apply_filtering_pipeline(
            emg, self.fs, self.lowcut, self.highcut,
            self.bp_order, self.notch_freq, self.notch_q
        )
        
        # 3. Normalization (e.g., Z-score, Min-Max, Robust)
        normalized = normalize_signal(filtered, self.norm_method)
        
        # 4. Post-validation of processed signal
        validate_signal(normalized)
        
        return normalized

    def process_file(self, file_path: Path, output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Load a raw MAT file, preprocess its EMG signal, save the preprocessed file
        to the interim folder preserving other variables, and write metadata.

        Parameters
        ----------
        file_path : Path
            Path to the raw MAT file.
        output_dir : Path
            Path to the target interim data directory.

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            A tuple of (success_status, metadata_or_error_dict).
        """
        subject_folder = file_path.parent.name
        file_name = file_path.name
        target_folder = output_dir / subject_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = target_folder / file_name
        metadata_path = target_folder / f"{file_name.split('.')[0]}_metadata.json"
        
        metadata = {
            "file_name": file_name,
            "subject_folder": subject_folder,
            "filter_params": {
                "sampling_frequency_hz": self.fs,
                "bandpass_low_hz": self.lowcut,
                "bandpass_high_hz": self.highcut,
                "bandpass_order": self.bp_order,
                "notch_frequency_hz": self.notch_freq,
                "notch_q_factor": self.notch_q
            },
            "normalization_method": self.norm_method,
            "processing_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__
            }
        }
        
        try:
            # Load MAT file
            mat_data = scipy.io.loadmat(str(file_path))
            if 'emg' not in mat_data:
                raise ValueError(f"Variable 'emg' not found in {file_name}")
                
            raw_emg = mat_data['emg']
            # Execute processing
            processed_emg = self.process_signal(raw_emg)
            
            # Prepare clean save dict without matlab structural variables
            save_dict = {
                k: v for k, v in mat_data.items()
                if not k.startswith('__')
            }
            save_dict['emg'] = processed_emg
            
            # Save preprocessed arrays
            scipy.io.savemat(str(target_path), save_dict)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
                
            return True, metadata
        except Exception as e:
            self.logger.error(f"Failed to preprocess file {file_name}: {e}")
            return False, {"error": str(e), "file_name": file_name}
