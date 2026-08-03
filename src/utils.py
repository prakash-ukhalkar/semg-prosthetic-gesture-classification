"""
sEMG Prosthetic Gesture Classification
Module: utils

Provides utility functions for logging, environment checks, repository structure
validation, dataset verification, and seed management.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.metadata

import numpy as np

def setup_logging(log_file_name: str) -> logging.Logger:
    """
    Configure and initialize the project-wide logging system.

    Sets up a logger that outputs to both a file (saved in outputs/logs/)
    and the console. Sets standard formatting with timestamps and log levels.

    Parameters
    ----------
    log_file_name : str
        The name of the log file (e.g., 'notebook_01.log').

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    from src.config import LOGS_DIR, LOGGER_NAME

    # Ensure the logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / log_file_name

    logger = logging.getLogger(LOGGER_NAME)
    
    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file saved to: {log_file_path}")
    return logger

def verify_directory_structure(expected_dirs: List[Path]) -> Dict[str, bool]:
    """
    Verify that the required project directories exist, and create them if missing.

    Parameters
    ----------
    expected_dirs : List[Path]
        A list of Path objects representing the directories to verify/create.

    Returns
    -------
    Dict[str, bool]
        A dictionary mapping directory path strings to a boolean indicating
        whether the directory exists (or was successfully created).
    """
    status = {}
    for directory in expected_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            status[str(directory)] = directory.is_dir()
        except Exception as e:
            status[str(directory)] = False
            logging.getLogger().error(f"Failed to create directory {directory}: {e}")
    return status

def get_package_versions(packages: List[str]) -> Dict[str, str]:
    """
    Retrieve the version numbers for a list of Python packages.

    Parameters
    ----------
    packages : List[str]
        A list of package names to check (e.g., ['numpy', 'pandas']).

    Returns
    -------
    Dict[str, str]
        A dictionary mapping package names to version strings. If a package
        is not installed, its status is reported as 'Not Installed'.
    """
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "Not Installed"
    return versions

def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for standard Python and NumPy random generators to ensure
    reproducibility of scientific results.

    Note: Scikit-learn, XGBoost, and LightGBM do not have a global random seed
    state, and instead accept a `random_state` parameter during classifier 
    instantiation.

    Parameters
    ----------
    seed : int
        The seed value to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Set os-level environmental seed if needed
    os.environ['PYTHONHASHSEED'] = str(seed)

def verify_raw_dataset(raw_dir: Path) -> Dict[str, Any]:
    """
    Verify the raw dataset files in the specified directory.

    Scans the directory for files, counting the files, formats, and total size
    to verify integrity.

    Parameters
    ----------
    raw_dir : Path
        The Path to the raw data directory.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing statistics:
        - 'exists': bool
        - 'file_count': int
        - 'total_size_mb': float
        - 'file_formats': List[str]
        - 'files': List[Dict[str, Any]] (list of files with name and size)
    """
    result = {
        'exists': raw_dir.exists() and raw_dir.is_dir(),
        'file_count': 0,
        'total_size_mb': 0.0,
        'file_formats': [],
        'files': []
    }

    if not result['exists']:
        return result

    file_list = []
    total_bytes = 0
    formats = set()

    for file_path in raw_dir.rglob('*'):
        if file_path.is_file() and file_path.name != ".gitkeep":
            size_bytes = file_path.stat().st_size
            total_bytes += size_bytes
            formats.add(file_path.suffix)
            file_list.append({
                'name': file_path.name,
                'size_bytes': size_bytes,
                'path': str(file_path)
            })

    result['file_count'] = len(file_list)
    result['total_size_mb'] = round(total_bytes / (1024 * 1024), 2)
    result['file_formats'] = sorted(list(formats))
    result['files'] = file_list

    return result

def download_and_extract_subject(subject_id: int, target_dir: Path) -> bool:
    """
    Download and extract a single subject's sEMG recording from the NinaPro DB2 public repository.

    Parameters
    ----------
    subject_id : int
        The subject ID to download (1 to 40).
    target_dir : Path
        The target directory to save and extract the files.

    Returns
    -------
    bool
        True if the download and extraction completed successfully, False otherwise.
    """
    import urllib.request
    import zipfile

    url = f"https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{subject_id}.zip"
    zip_path = target_dir / f"DB2_s{subject_id}.zip"
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("semg_prosthetic_classification")
    logger.info(f"Downloading dataset from: {url}")
    try:
        # Download the file
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"Downloaded zip file to: {zip_path}")
        
        # Extract the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(f"Extracted zip files to: {target_dir}")
        
        # Remove zip file to save space
        zip_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to download and extract dataset for subject {subject_id}: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

