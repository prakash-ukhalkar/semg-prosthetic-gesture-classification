"""
sEMG Prosthetic Gesture Classification
Module: normalization

Provides amplitude normalization methods for multi-channel sEMG signals,
including Z-score, Min-Max scaling, Robust scaling, and Unit Vector scaling.
"""

import numpy as np

def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """
    Standardize the signal to zero-mean and unit variance.

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (samples, channels).

    Returns
    -------
    np.ndarray
        Standardized signal array.
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    # Prevent division by zero for inactive channels
    stds[stds == 0.0] = 1.0
    return (data - means) / stds

def min_max_scale(data: np.ndarray) -> np.ndarray:
    """
    Scale the signal to the range [-1, 1].

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (samples, channels).

    Returns
    -------
    np.ndarray
        Scaled signal array.
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0.0] = 1.0
    # Map range [0, 1] to [-1, 1]
    scaled = (data - min_vals) / ranges
    return 2.0 * scaled - 1.0

def robust_scale(data: np.ndarray) -> np.ndarray:
    """
    Scale the signal using median and interquartile range (IQR) to reduce outlier bias.

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (samples, channels).

    Returns
    -------
    np.ndarray
        Scaled signal array.
    """
    medians = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqrs = q75 - q25
    iqrs[iqrs == 0.0] = 1.0
    return (data - medians) / iqrs

def unit_vector_scale(data: np.ndarray) -> np.ndarray:
    """
    Scale each signal sample vector (across channels) to unit L2 norm.

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (samples, channels).

    Returns
    -------
    np.ndarray
        Scaled signal array.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return data / norms

def normalize_signal(data: np.ndarray, method: str) -> np.ndarray:
    """
    Dispatch and apply the selected normalization method.

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (samples, channels).
    method : str
        Normalization method: 'zscore', 'minmax', 'robust', or 'unit_vector'.

    Returns
    -------
    np.ndarray
        Normalized signal array.

    Raises
    ------
    ValueError
        If the selected normalization method is not supported.
    """
    method_lower = method.lower().replace("_", "").replace("-", "")
    if method_lower == "zscore":
        return z_score_normalize(data)
    elif method_lower == "minmax":
        return min_max_scale(data)
    elif method_lower == "robust":
        return robust_scale(data)
    elif method_lower == "unitvector":
        return unit_vector_scale(data)
    else:
        raise ValueError(
            f"Unsupported normalization method: '{method}'. "
            "Choose from: 'zscore', 'minmax', 'robust', or 'unit_vector'."
        )
