"""
sEMG Prosthetic Gesture Classification
Module: feature_selection.fusion

Provides tools to merge time-domain and frequency-domain feature databases
using a memory-efficient, subject-by-subject verification and concatenation process.
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class FeatureFuser:
    """
    Handles memory-efficient merging of time-domain and frequency-domain sEMG features.
    """
    def __init__(
        self,
        time_path: Path,
        freq_path: Path,
        output_path: Path,
        inventory_path: Path
    ):
        self.time_path = Path(time_path)
        self.freq_path = Path(freq_path)
        self.output_path = Path(output_path)
        self.inventory_path = Path(inventory_path)
        
        self.join_keys = ["subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id"]
        
    def fuse_databases(self, subjects: List[int] = None) -> Path:
        """
        Merge the time-domain and frequency/wavelet databases subject-by-subject
        and write the merged features into a unified Parquet file.
        
        Parameters
        ----------
        subjects : List[int], optional
            List of subject IDs to process. If None, processes subjects 1 to 40.
            
        Returns
        -------
        Path
            Path to the merged feature Parquet file.
        """
        if subjects is None:
            subjects = list(range(1, 41))
            
        logger.info(f"Starting feature fusion for {len(subjects)} subjects...")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine overlapping feature names to rename them (e.g. entropy, energy, power, variance)
        # Read the schema of both files to find columns
        time_schema = pq.read_schema(self.time_path)
        freq_schema = pq.read_schema(self.freq_path)
        
        time_cols = set(time_schema.names)
        freq_cols = set(freq_schema.names)
        
        overlap_cols = time_cols.intersection(freq_cols) - set(self.join_keys)
        if overlap_cols:
            logger.info(f"Detected {len(overlap_cols)} overlapping feature columns: {sorted(list(overlap_cols))}")
            logger.info("These columns in the frequency-domain database will be prefixed with 'spectral_'.")
            
        # Initialize pyarrow writer
        writer = None
        
        try:
            for sub_id in subjects:
                logger.info(f"Processing subject {sub_id}...")
                
                # Load time domain features for this subject
                df_time = pd.read_parquet(
                    self.time_path,
                    filters=[("subject_id", "==", sub_id)]
                )
                
                # Load frequency domain features for this subject
                df_freq = pd.read_parquet(
                    self.freq_path,
                    filters=[("subject_id", "==", sub_id)]
                )
                
                if df_time.empty:
                    logger.warning(f"No time-domain features found for subject {sub_id}. Skipping.")
                    continue
                if df_freq.empty:
                    logger.warning(f"No frequency-domain features found for subject {sub_id}. Skipping.")
                    continue
                    
                # Validate shapes match
                if len(df_time) != len(df_freq):
                    raise ValueError(
                        f"Row count mismatch for subject {sub_id}: "
                        f"Time domain has {len(df_time)} rows, Frequency has {len(df_freq)} rows."
                    )
                    
                # Sort both by keys to align rows
                df_time = df_time.sort_values(by=self.join_keys).reset_index(drop=True)
                df_freq = df_freq.sort_values(by=self.join_keys).reset_index(drop=True)
                
                # Verify keys match exactly
                for key in self.join_keys:
                    if not df_time[key].equals(df_freq[key]):
                        raise ValueError(
                            f"Metadata key mismatch for subject {sub_id} in column '{key}'."
                        )
                        
                # Rename duplicate columns in freq
                rename_dict = {}
                for col in overlap_cols:
                    if col in df_freq.columns:
                        rename_dict[col] = f"spectral_{col}"
                df_freq_renamed = df_freq.rename(columns=rename_dict)
                
                # Perform the merge (since they are sorted and aligned, we can inner merge or concat columns)
                # An inner merge on join_keys is safest and verifies keys again.
                df_merged = pd.merge(df_time, df_freq_renamed, on=self.join_keys, how="inner")
                
                # Verify no rows were lost or duplicated
                if len(df_merged) != len(df_time):
                    raise ValueError(
                        f"Merge failed for subject {sub_id}. "
                        f"Expected {len(df_time)} rows, got {len(df_merged)} rows."
                    )
                    
                # Check for duplicate columns in merged dataframe
                if df_merged.columns.duplicated().any():
                    duplicated_cols = df_merged.columns[df_merged.columns.duplicated()].tolist()
                    raise ValueError(f"Duplicate columns found in merged DataFrame: {duplicated_cols}")
                    
                # Write to parquet
                table = pa.Table.from_pandas(df_merged)
                if writer is None:
                    writer = pq.ParquetWriter(self.output_path, table.schema, compression="snappy")
                writer.write_table(table)
                
                logger.info(f"Subject {sub_id} successfully merged and written. Shape: {df_merged.shape}")
                
        finally:
            if writer is not None:
                writer.close()
                logger.info("ParquetWriter closed successfully.")
                
        # Generate feature inventory
        self.generate_feature_inventory()
        
        return self.output_path
        
    def generate_feature_inventory(self) -> None:
        """
        Scans the merged feature database columns and generates an inventory report.
        """
        logger.info("Generating feature inventory...")
        
        # Read just the schema of the merged file
        schema = pq.read_schema(self.output_path)
        all_cols = schema.names
        
        metadata_cols = {
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        }
        
        feature_cols = [c for c in all_cols if c not in metadata_cols]
        
        inventory_records = []
        
        # Identify feature categories
        # Time-domain: features in time_domain_features.parquet (which we know are mean, var, etc.)
        # Let's inspect which columns came from time-domain by checking the original time-domain schema
        time_schema = pq.read_schema(self.time_path)
        time_feature_set = set(time_schema.names) - set(self.join_keys)
        
        for col in feature_cols:
            # Parse channel number (e.g. mean_ch1 -> 1, dwt_ca4_energy_ch12 -> 12)
            import re
            ch_match = re.search(r"_ch(\d+)$", col)
            channel = int(ch_match.group(1)) if ch_match else None
            
            # Determine category, base name, domain
            if col in time_feature_set:
                category = "Time-Domain"
                domain = "Time"
                base_name = re.sub(r"_ch\d+$", "", col)
            elif col.startswith("dwt_"):
                category = "Time-Frequency (Wavelet)"
                domain = "Time-Frequency"
                base_name = re.sub(r"_ch\d+$", "", col)
            else:
                category = "Frequency-Domain"
                domain = "Frequency"
                base_name = re.sub(r"_ch\d+$", "", col)
                
            inventory_records.append({
                "feature_name": col,
                "base_feature": base_name,
                "channel": channel,
                "category": category,
                "domain": domain
            })
            
        df_inventory = pd.DataFrame(inventory_records)
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        df_inventory.to_csv(self.inventory_path, index=False)
        logger.info(f"Feature inventory saved to {self.inventory_path}. Total features: {len(df_inventory)}")
        
        # Log summary
        summary = df_inventory.groupby("category").size()
        for cat, val in summary.items():
            logger.info(f"Category '{cat}': {val} features")
