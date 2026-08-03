"""
sEMG Prosthetic Gesture Classification
Module: segmentation

Provides utilities for partitioning continuous preprocessed sEMG signals
into sliding window analysis segments with label propagation and metadata tracking.
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.io

from src.config import (
    SAMPLING_RATE, WINDOW_SIZE, WINDOW_OVERLAP,
    SEGMENTATION_PADDING, LABEL_POLICY, PROCESSED_FORMAT
)

@dataclass
class WindowMetadata:
    """
    Dataclass for tracking metadata of individual analysis windows.
    """
    window_id: int
    subject_id: int
    exercise_id: int
    gesture_id: int
    repetition_id: int
    start_sample: int
    end_sample: int
    window_size_samples: int
    sampling_frequency_hz: float

class WindowGenerator:
    """
    Handles partitioning continuous signals into analysis windows.
    """
    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        window_size: float = WINDOW_SIZE,
        window_overlap: float = WINDOW_OVERLAP,
        padding: str = SEGMENTATION_PADDING,
        label_policy: str = LABEL_POLICY
    ):
        self.fs = fs
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.padding = padding.lower()
        self.label_policy = label_policy.lower()
        
        self.window_samples = int(self.window_size * self.fs)
        self.stride_samples = int((self.window_size - self.window_overlap) * self.fs)
        
        if self.stride_samples <= 0:
            raise ValueError("Window overlap must be strictly less than window size.")

    def _propagate_label(self, label_window: np.ndarray) -> Optional[int]:
        """
        Assign a single gesture label to the window based on the chosen policy.
        """
        if self.label_policy == "majority":
            # Majority voting: take the most frequent label in the window
            counts = np.bincount(label_window.squeeze().astype(np.int64))
            return int(np.argmax(counts))
        elif self.label_policy == "center":
            # Central sample labeling
            return int(label_window[self.window_samples // 2])
        elif self.label_policy == "strict":
            # Strict labeling: all samples in the window must have the same label
            unique = np.unique(label_window)
            if len(unique) == 1:
                return int(unique[0])
            return None  # Discard window
        else:
            raise ValueError(f"Unknown label policy: '{self.label_policy}'")

    def segment_signal(
        self,
        emg: np.ndarray,
        labels: np.ndarray,
        repetitions: np.ndarray,
        subject_id: int,
        exercise_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[WindowMetadata]]:
        """
        Segment raw arrays into overlapping window matrices.
        """
        total_samples = emg.shape[0]
        channels = emg.shape[1]
        
        windows = []
        window_labels = []
        window_reps = []
        metadata_list = []
        
        start_idx = 0
        window_id = 0
        
        while start_idx + self.window_samples <= total_samples:
            end_idx = start_idx + self.window_samples
            
            # Extract slices
            emg_slice = emg[start_idx:end_idx, :]
            label_slice = labels[start_idx:end_idx]
            rep_slice = repetitions[start_idx:end_idx]
            
            # Label propagation
            assigned_label = self._propagate_label(label_slice)
            
            # Repetition propagation (always majority voting for repetition)
            rep_counts = np.bincount(rep_slice.squeeze().astype(np.int64))
            assigned_rep = int(np.argmax(rep_counts))
            
            if assigned_label is not None:
                windows.append(emg_slice)
                window_labels.append(assigned_label)
                window_reps.append(assigned_rep)
                
                meta = WindowMetadata(
                    window_id=window_id,
                    subject_id=subject_id,
                    exercise_id=exercise_id,
                    gesture_id=assigned_label,
                    repetition_id=assigned_rep,
                    start_sample=start_idx,
                    end_sample=end_idx,
                    window_size_samples=self.window_samples,
                    sampling_frequency_hz=self.fs
                )
                metadata_list.append(meta)
                window_id += 1
                
            start_idx += self.stride_samples
            
        # Optional padding handling for final trailing window
        if self.padding == "pad" and start_idx < total_samples:
            remaining = total_samples - start_idx
            if remaining >= self.window_samples // 2: # Only pad if we have at least half a window left
                end_idx = total_samples
                emg_slice = np.zeros((self.window_samples, channels))
                emg_slice[:remaining, :] = emg[start_idx:, :]
                
                label_slice = np.zeros(self.window_samples)
                label_slice[:remaining] = labels[start_idx:]
                
                rep_slice = np.zeros(self.window_samples)
                rep_slice[:remaining] = repetitions[start_idx:]
                
                assigned_label = self._propagate_label(label_slice)
                rep_counts = np.bincount(rep_slice.astype(np.int64))
                assigned_rep = int(np.argmax(rep_counts))
                
                if assigned_label is not None:
                    windows.append(emg_slice)
                    window_labels.append(assigned_label)
                    window_reps.append(assigned_rep)
                    
                    meta = WindowMetadata(
                        window_id=window_id,
                        subject_id=subject_id,
                        exercise_id=exercise_id,
                        gesture_id=assigned_label,
                        repetition_id=assigned_rep,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        window_size_samples=self.window_samples,
                        sampling_frequency_hz=self.fs
                    )
                    metadata_list.append(meta)
                    
        if len(windows) == 0:
            return np.empty((0, self.window_samples, channels)), np.empty((0,)), np.empty((0,)), []
            
        return np.array(windows), np.array(window_labels), np.array(window_reps), metadata_list


class SegmentationPipeline:
    """
    Coordinates file loading, segmentation execution, and output saving.
    """
    def __init__(
        self,
        fs: float = SAMPLING_RATE,
        window_size: float = WINDOW_SIZE,
        window_overlap: float = WINDOW_OVERLAP,
        padding: str = SEGMENTATION_PADDING,
        label_policy: str = LABEL_POLICY,
        output_format: str = PROCESSED_FORMAT
    ):
        self.generator = WindowGenerator(fs, window_size, window_overlap, padding, label_policy)
        self.output_format = output_format.lower()
        self.logger = logging.getLogger("semg_prosthetic_classification")

    def process_file(self, file_path: Path, output_dir: Path) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Segment a preprocessed MAT file and save the resulting window tensors.
        """
        file_name = file_path.name
        subject_folder = file_path.parent.name
        
        # Parse subject ID and exercise ID from the filename (e.g. S1_E1_A1.mat)
        try:
            parts = file_name.split('_')
            subject_id = int(parts[0].replace('S', ''))
            exercise_id = int(parts[1].replace('E', ''))
        except Exception:
            self.logger.error(f"Cannot parse subject/exercise ID from filename: {file_name}")
            return False, []
            
        target_folder = output_dir / subject_folder
        target_folder.mkdir(parents=True, exist_ok=True)
        
        target_path = target_folder / f"{file_name.split('.')[0]}_segmented.{self.output_format}"
        metadata_csv_path = target_folder / f"{file_name.split('.')[0]}_segmented_metadata.csv"
        
        try:
            mat_data = scipy.io.loadmat(str(file_path))
            emg = mat_data['emg']
            stimulus = mat_data['stimulus'].squeeze()
            repetition = mat_data['repetition'].squeeze()
            
            # Segment continuous data into windows
            windows, labels, reps, metadata_list = self.generator.segment_signal(
                emg, stimulus, repetition, subject_id, exercise_id
            )
            
            # Save segmented window tensors
            if self.output_format == "npz":
                windows_f32 = np.array(windows, dtype=np.float32)
                labels_i16 = np.array(labels, dtype=np.int16)
                reps_i16 = np.array(reps, dtype=np.int16)
                
                np.savez(
                    str(target_path),
                    windows=windows_f32,
                    labels=labels_i16,
                    repetitions=reps_i16
                )
            else:
                raise ValueError(f"Unsupported output format: '{self.output_format}'")
                
            # Save metadata as CSV
            meta_dicts = [asdict(m) for m in metadata_list]
            if meta_dicts:
                import pandas as pd
                df_meta = pd.DataFrame(meta_dicts)
                df_meta.to_csv(metadata_csv_path, index=False)
                
            self.logger.info(f"Successfully segmented {file_name}: {len(windows)} windows generated.")
            return True, meta_dicts
        except Exception as e:
            self.logger.error(f"Failed to segment file {file_name}: {e}")
            return False, []

def validate_windows(windows: np.ndarray, window_size_samples: int, channels: int) -> None:
    """
    Verify that the window array dimensions match expectations.
    """
    if windows.ndim != 3:
         raise ValueError(f"Windows array must be 3D of shape (windows, samples, channels), got ndim={windows.ndim}")
    if windows.shape[1] != window_size_samples:
         raise ValueError(f"Window sample size mismatch: expected {window_size_samples}, got {windows.shape[1]}")
    if windows.shape[2] != channels:
         raise ValueError(f"Window channel count mismatch: expected {channels}, got {windows.shape[2]}")
