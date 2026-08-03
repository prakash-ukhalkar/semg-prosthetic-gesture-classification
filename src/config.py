"""
sEMG Prosthetic Gesture Classification
Module: config

Contains project-wide configurations including file directory paths,
signal processing parameters, and machine learning options to ensure
reproducibility and maintainability.
"""

from pathlib import Path

# ==============================================================================
# Path Configurations
# ==============================================================================

# Project root path (assumed to be parent directory of this module's package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory and its subfolders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Outputs directory and its subfolders
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ==============================================================================
# Reproducibility Settings
# ==============================================================================
RANDOM_STATE = 42

# ==============================================================================
# sEMG Signal Processing Parameters
# ==============================================================================
# Sampling frequency of the EMG hardware (in Hz)
# NinaPro Database 2 is recorded at 2000 Hz, while DB1 is recorded at 100 Hz.
SAMPLING_RATE = 2000

# Sliding window duration (in seconds)
# Typically 200 ms is selected to balance classification latency and feature representation.
WINDOW_SIZE = 0.200

# Sliding window overlap increment (in seconds)
# Overlap of 50 ms provides continuous prediction density.
WINDOW_OVERLAP = 0.050

# ==============================================================================
# sEMG Preprocessing Parameters
# ==============================================================================
# Band-pass filter cut-off frequencies (in Hz)
# 20 Hz removes low-frequency motion artefacts.
# 500 Hz removes high-frequency noise outside the physiological range of sEMG.
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 500.0

# Band-pass filter order (Butterworth filter)
BANDPASS_ORDER = 4

# Power-line notch frequency (in Hz)
# NinaPro DB2 was recorded in Europe, where the power line frequency is 50 Hz.
NOTCH_FREQ = 50.0

# Notch filter quality factor (Q-factor)
NOTCH_Q = 30.0

# Normalization method: 'zscore', 'minmax', 'robust', or 'unit_vector'
NORMALIZATION_METHOD = "zscore"

# ==============================================================================
# sEMG Segmentation Parameters
# ==============================================================================
# Segmentation padding policy: 'drop' (discard remaining samples) or 'pad' (zero-pad)
SEGMENTATION_PADDING = "drop"

# Label propagation policy: 'majority', 'center', or 'strict'
LABEL_POLICY = "majority"

# Processed data format: 'npz'
PROCESSED_FORMAT = "npz"

# ==============================================================================
# sEMG Feature Engineering Configuration
# ==============================================================================
# Noise threshold parameters (Z-score standardized units)
ZC_THRESHOLD = 0.01
SSC_THRESHOLD = 0.01
WAMP_THRESHOLD = 0.01

# Feature Database output path
FEATURE_DATABASE_PATH = PROCESSED_DATA_DIR / "time_domain_features.parquet"

# ==============================================================================
# sEMG Frequency & Wavelet Feature Engineering Configuration
# ==============================================================================
# FFT Parameters
FFT_N = None  # None means length of window (400 samples)
# Configurable frequency bands for Band Power feature (in Hz)
SPECTRAL_BANDS = [
    (20, 150),    # Low frequency band (dominant sEMG power)
    (150, 300),   # Mid frequency band
    (300, 500)    # High frequency band (near physiological limit)
]
# Frequency ratio split frequency (in Hz)
FR_SPLIT_FREQ = 150.0

# Wavelet Parameters
WAVELET_FAMILY = "db4"  # Daubechies 4 wavelet (highly recommended for sEMG)
WAVELET_LEVEL = 4       # Decomposition level

# Output paths
FREQ_WAVELET_DATABASE_PATH = PROCESSED_DATA_DIR / "frequency_wavelet_features.parquet"

# Batch Processing
SPECTRAL_BATCH_SIZE = 10000      # Chunk size for processing
SPECTRAL_PARALLEL_WORKERS = 4    # Number of parallel jobs

# ==============================================================================
# Logging Configuration
# ==============================================================================
LOGGER_NAME = "semg_prosthetic_classification"

# ==============================================================================
# Optuna Hyperparameter Optimization Configuration
# ==============================================================================
# Number of optimization trials per model
OPTUNA_TRIALS = 150


