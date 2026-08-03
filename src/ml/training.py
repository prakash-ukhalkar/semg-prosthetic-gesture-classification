"""
sEMG Prosthetic Gesture Classification
Module: ml.training

Handles model training, resource tracking (time, memory), and resumable execution.
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Tuple, Optional
import psutil
import joblib
from sklearn.model_selection import train_test_split

logger = logging.getLogger("semg_prosthetic_classification")

def get_process_memory() -> float:
    """
    Get current process Resident Set Size (RSS) memory footprint in MB.

    Returns
    -------
    float
        Memory footprint in MB.
    """
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception as e:
        logger.warning(f"Failed to measure memory usage: {e}")
        return 0.0

def train_classifier(
    pipeline: Any,
    X_train: Any,
    y_train: Any,
    model_name: str,
    feature_set_name: str,
    save_dir: Path,
    max_train_samples: Optional[int] = None,
    force_retrain: bool = False,
    random_state: int = 42
) -> Tuple[Any, float, float, bool]:
    """
    Train a classifier pipeline on the provided training set, tracking time and memory.
    Supports downsampling for slow algorithms and resumable execution.

    Parameters
    ----------
    pipeline : scikit-learn Pipeline
        The machine learning pipeline to train.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    model_name : str
        The classifier identifier.
    feature_set_name : str
        The feature set identifier (e.g., 'top25', 'top50').
    save_dir : Path
        Directory to save/load the trained model.
    max_train_samples : int, optional
        Maximum number of samples to use for training (for stratified downsampling).
    force_retrain : bool, default=False
        If True, trains the model even if a saved version exists.
    random_state : int, default=42
        Random seed for stratified splitting.

    Returns
    -------
    model : Any
        The trained pipeline (loaded from disk or newly trained).
    train_time : float
        Training duration in seconds.
    memory_used_mb : float
        RSS memory difference during training in MB.
    loaded_from_disk : bool
        True if the model was loaded from a saved file, False otherwise.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"model_{feature_set_name}_{model_name}.joblib"
    
    # 1. Check for resumable execution
    if model_path.exists() and not force_retrain:
        logger.info(f"Resuming: Saved model found for '{model_name}' on '{feature_set_name}'. Loading from disk...")
        try:
            model = joblib.load(model_path)
            # Retrieve saved training time from cached metadata if possible,
            # or default to 0.0. High-level orchestrator will handle this.
            return model, 0.0, 0.0, True
        except Exception as e:
            logger.error(f"Failed to load saved model from {model_path}: {e}. Retraining...")
            
    # 2. Downsampling if requested
    X_fit = X_train
    y_fit = y_train
    is_downsampled = False
    
    if max_train_samples is not None and len(X_train) > max_train_samples:
        logger.info(
            f"Downsampling training set for '{model_name}' from {len(X_train)} to {max_train_samples} samples "
            f"using stratified sampling (random_state={random_state})."
        )
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train,
            train_size=max_train_samples,
            stratify=y_train,
            random_state=random_state
        )
        is_downsampled = True
        
    # 3. Train the model
    logger.info(f"Training '{model_name}' on '{feature_set_name}' ({len(X_fit)} samples)...")
    
    mem_start = get_process_memory()
    start_time = time.perf_counter()
    
    try:
        pipeline.fit(X_fit, y_fit)
        
        train_time = time.perf_counter() - start_time
        mem_end = get_process_memory()
        memory_used_mb = max(0.0, mem_end - mem_start)
        
        logger.info(f"Finished training '{model_name}' in {train_time:.2f} seconds. Memory used: {memory_used_mb:.2f} MB.")
        
        # Save model to disk with compression to save space
        joblib.dump(pipeline, model_path, compress=3)
        logger.info(f"Saved trained model (compressed) to: {model_path}")
        
        return pipeline, train_time, memory_used_mb, False
        
    except Exception as e:
        logger.error(f"Error training model '{model_name}' on '{feature_set_name}': {e}", exc_info=True)
        raise e
