"""
sEMG Prosthetic Gesture Classification
Module: filtering

Provides digital filters and signal validation utilities for surface EMG.
Uses zero-phase filtering (filtfilt) to avoid introducing phase delay,
which is critical for preserving real-time gesture transition alignment.
"""

import numpy as np
import scipy.signal
from typing import Tuple

def butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth band-pass filter and return coefficients (b, a).

    Parameters
    ----------
    lowcut : float
        Low cut-off frequency in Hz.
    highcut : float
        High cut-off frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Order of the filter (default is 4).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Filter coefficients (b, a).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter to a multi-channel signal.

    Parameters
    ----------
    data : np.ndarray
        Input signal of shape (samples, channels).
    lowcut : float
        Low cut-off frequency in Hz.
    highcut : float
        High cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Order of the filter.

    Returns
    -------
    np.ndarray
        Band-pass filtered signal of shape (samples, channels).
    """
    validate_signal(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    # Apply zero-phase filter along the sample axis (axis=0)
    filtered = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered

def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    q: float = 30.0
) -> np.ndarray:
    """
    Apply a zero-phase IIR notch filter to remove power-line interference.

    Parameters
    ----------
    data : np.ndarray
        Input signal of shape (samples, channels).
    notch_freq : float
        Frequency to remove (e.g., 50.0 Hz).
    fs : float
        Sampling rate of the signal in Hz.
    q : float
        Quality factor (Q) of the notch filter.

    Returns
    -------
    np.ndarray
        Notch filtered signal of shape (samples, channels).
    """
    validate_signal(data)
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = scipy.signal.iirnotch(w0, q)
    filtered = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered

def apply_filtering_pipeline(
    data: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    bp_order: int,
    notch_freq: float,
    notch_q: float
) -> np.ndarray:
    """
    Execute the consecutive filtering pipeline: Band-pass -> Notch.

    Parameters
    ----------
    data : np.ndarray
        Raw signal of shape (samples, channels).
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Low cut-off for bandpass in Hz.
    highcut : float
        High cut-off for bandpass in Hz.
    bp_order : int
        Butterworth bandpass filter order.
    notch_freq : float
        Notch cut frequency in Hz.
    notch_q : float
        Notch filter quality factor.

    Returns
    -------
    np.ndarray
        Filtered signal array.
    """
    # 1. Band-pass filtering to remove low-frequency drift and high-frequency noise
    bp_filtered = bandpass_filter(data, lowcut, highcut, fs, bp_order)
    # 2. Notch filtering to remove power-line interference (e.g., 50 Hz)
    filtered = notch_filter(bp_filtered, notch_freq, fs, notch_q)
    return filtered

def validate_signal(data: np.ndarray) -> None:
    """
    Validate that the signal array is well-formed.

    Parameters
    ----------
    data : np.ndarray
        Signal array to validate.

    Raises
    ------
    ValueError
        If the array contains NaNs, infinite values, or is not 2-dimensional.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy.ndarray.")
    if data.ndim != 2:
        raise ValueError(f"Input must be a 2D array of shape (samples, channels), got ndim={data.ndim}")
    if np.any(np.isnan(data)):
        raise ValueError("Signal contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Signal contains infinite values.")
