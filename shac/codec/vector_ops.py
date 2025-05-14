"""
Vectorized Audio Operations Module

This module provides highly optimized vectorized operations for spatial audio processing
using NumPy's SIMD capabilities to maximize performance on modern CPUs.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import numba


@numba.njit(fastmath=True, parallel=True)
def fast_encode_mono(audio: np.ndarray, sh_matrix: np.ndarray) -> np.ndarray:
    """
    Fast mono source encoding using pre-computed spherical harmonic coefficients.
    
    Args:
        audio: Mono audio signal of shape (n_samples,)
        sh_matrix: Pre-computed spherical harmonic coefficients of shape (n_channels,)
    
    Returns:
        Encoded ambisonic signals of shape (n_channels, n_samples)
    """
    n_channels = sh_matrix.shape[0]
    n_samples = audio.shape[0]
    result = np.zeros((n_channels, n_samples), dtype=np.float32)
    
    # Broadcasting the multiply operation for all channels at once
    for i in numba.prange(n_channels):
        result[i] = audio * sh_matrix[i]
    
    return result


@numba.njit(fastmath=True)
def fast_mix_sources(sources: List[np.ndarray]) -> np.ndarray:
    """
    Fast mixing of multiple ambisonic sources.
    
    Args:
        sources: List of ambisonic signals, each with shape (n_channels, n_samples)
    
    Returns:
        Mixed ambisonic signals of shape (n_channels, n_samples)
    """
    if not sources:
        return np.array([])
    
    # Get dimensions from the first source
    n_channels, n_samples = sources[0].shape
    result = np.zeros((n_channels, n_samples), dtype=np.float32)
    
    # Add all sources to the result
    for source in sources:
        result += source
    
    return result


@numba.njit(fastmath=True)
def fast_normalize(signal: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """
    Fast normalization of audio signals to prevent clipping.
    
    Args:
        signal: Audio signal of any shape
        threshold: Normalization threshold (0.0 to 1.0)
    
    Returns:
        Normalized audio signal of the same shape
    """
    max_val = np.max(np.abs(signal))
    if max_val > threshold:
        return signal * (threshold / max_val)
    return signal


@numba.njit(fastmath=True)
def apply_distance_attenuation(signal: np.ndarray, distance: float, 
                              reference_distance: float = 1.0,
                              rolloff_factor: float = 1.0) -> np.ndarray:
    """
    Apply distance-based attenuation to an audio signal using the inverse distance law.
    
    Args:
        signal: Audio signal of any shape
        distance: Distance in meters
        reference_distance: Reference distance in meters (gain is 1.0 at this distance)
        rolloff_factor: Rolloff factor (1.0 = inverse distance law)
    
    Returns:
        Attenuated audio signal of the same shape
    """
    if distance <= reference_distance:
        return signal
    
    # Inverse distance law: gain = reference_distance / distance
    gain = reference_distance / (reference_distance + rolloff_factor * (distance - reference_distance))
    return signal * gain


@numba.njit(fastmath=True)
def fast_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Compute a fast rotation matrix for first-order ambisonics.
    
    This is a simplified version that only works for first-order (4-channel) ambisonics.
    For higher orders, use the full rotation implementation.
    
    Args:
        yaw: Rotation around vertical axis in radians
        pitch: Rotation around side axis in radians
        roll: Rotation around front axis in radians
    
    Returns:
        Rotation matrix of shape (4, 4)
    """
    # First-order rotation matrix (simplified)
    c1 = np.cos(yaw)
    s1 = np.sin(yaw)
    c2 = np.cos(pitch)
    s2 = np.sin(pitch)
    
    # Create rotation matrix
    matrix = np.eye(4, dtype=np.float32)
    
    # ACN/SN3D ordering: W, Y, Z, X
    matrix[1, 1] = c1 * c2
    matrix[1, 3] = s1
    matrix[3, 1] = -s1
    matrix[3, 3] = c1
    
    return matrix


@numba.njit(fastmath=True)
def fast_apply_rotation_first_order(signal: np.ndarray, 
                                   yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Apply rotation to first-order ambisonic signals.
    
    This is an optimized version that only works for first-order (4-channel) ambisonics.
    
    Args:
        signal: Ambisonic signals of shape (4, n_samples)
        yaw: Rotation around vertical axis in radians
        pitch: Rotation around side axis in radians
        roll: Rotation around front axis in radians
    
    Returns:
        Rotated ambisonic signals of shape (4, n_samples)
    """
    if signal.shape[0] != 4:
        # This function only works for first-order (4-channel) ambisonics
        return signal
    
    # Get rotation matrix
    rot_matrix = fast_rotation_matrix(yaw, pitch, roll)
    
    # Apply rotation
    result = np.zeros_like(signal)
    for i in range(4):
        for j in range(4):
            result[i] += rot_matrix[i, j] * signal[j]
    
    return result


@numba.njit(fastmath=True, parallel=True)
def fast_binauralize_simple(ambisonics: np.ndarray, hrtf_matrix: np.ndarray) -> np.ndarray:
    """
    Simple binaural rendering for first-order ambisonics using basic HRTF approximation.
    
    This is a highly simplified version that applies a pre-computed HRTF matrix without
    frequency-dependent processing, suitable for very low-latency applications.
    
    Args:
        ambisonics: Ambisonic signals of shape (n_channels, n_samples)
        hrtf_matrix: Pre-computed HRTF matrix of shape (2, n_channels)
    
    Returns:
        Binaural signals of shape (2, n_samples)
    """
    n_channels, n_samples = ambisonics.shape
    result = np.zeros((2, n_samples), dtype=np.float32)
    
    # Apply HRTF matrix
    for ear in range(2):
        for ch in numba.prange(n_channels):
            result[ear] += hrtf_matrix[ear, ch] * ambisonics[ch]
    
    return result


def precompute_sh_matrix(position: Tuple[float, float, float], order: int) -> np.ndarray:
    """
    Precompute spherical harmonic coefficients for a given position.
    
    Args:
        position: (azimuth, elevation, distance) in radians and meters
        order: Ambisonic order
    
    Returns:
        Spherical harmonic coefficients of shape (n_channels,)
    """
    # This would call into the appropriate SH calculation function
    # For demonstration, we'll create a simple placeholder
    n_channels = (order + 1) ** 2
    azimuth, elevation, distance = position
    
    # Simple spherical harmonic calculation (placeholder)
    sh_values = np.zeros(n_channels, dtype=np.float32)
    sh_values[0] = 1.0  # W channel is always 1
    
    if order >= 1:
        # First-order terms
        sh_values[1] = np.sin(azimuth) * np.cos(elevation)  # Y
        sh_values[2] = np.sin(elevation)                     # Z
        sh_values[3] = np.cos(azimuth) * np.cos(elevation)  # X
    
    # Scale by distance if needed
    distance_attenuation = 1.0 / max(0.1, distance)
    sh_values *= distance_attenuation
    
    return sh_values


def precompute_hrtf_matrix(order: int) -> np.ndarray:
    """
    Precompute a basic HRTF matrix for ambisonic to binaural conversion.
    
    This is a simplified model for demo purposes. A real implementation would
    load and process actual HRTF measurements.
    
    Args:
        order: Ambisonic order
    
    Returns:
        HRTF matrix of shape (2, n_channels)
    """
    n_channels = (order + 1) ** 2
    hrtf_matrix = np.zeros((2, n_channels), dtype=np.float32)
    
    # Simple HRTF model (placeholder)
    # For a real implementation, this would use measured HRTF data
    
    # W channel contributes equally to both ears
    hrtf_matrix[0, 0] = 0.7071  # Left ear
    hrtf_matrix[1, 0] = 0.7071  # Right ear
    
    if order >= 1:
        # Y channel (left-right): positive for left ear, negative for right ear
        hrtf_matrix[0, 1] = 0.5   # Left ear
        hrtf_matrix[1, 1] = -0.5  # Right ear
        
        # Z channel (up-down): equal contribution to both ears
        hrtf_matrix[0, 2] = 0.3   # Left ear
        hrtf_matrix[1, 2] = 0.3   # Right ear
        
        # X channel (front-back): equal contribution to both ears
        hrtf_matrix[0, 3] = 0.5   # Left ear
        hrtf_matrix[1, 3] = 0.5   # Right ear
    
    return hrtf_matrix
"""