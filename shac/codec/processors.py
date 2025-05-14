"""
Sound Field Processing Module

This module contains functions for manipulating and processing ambisonic
sound fields, including rotation, decoding to speakers, and format conversion.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union

from .math_utils import real_spherical_harmonic, sh_rotation_matrix, AmbisonicNormalization


def rotate_ambisonics(ambi_signals: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Rotate an ambisonic sound field.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        yaw: Rotation around vertical axis (positive = left) in radians
        pitch: Rotation around side axis (positive = up) in radians
        roll: Rotation around front axis (positive = tilt right) in radians
        
    Returns:
        Rotated ambisonic signals, shape (n_channels, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    if (order + 1) ** 2 != n_channels:
        raise ValueError(f"Number of channels {n_channels} does not correspond to a complete ambisonic order")
    
    # Compute the rotation matrix
    R = sh_rotation_matrix(order, yaw, pitch, roll)
    
    # Apply the rotation
    rotated_signals = np.zeros_like(ambi_signals)
    for i in range(n_channels):
        for j in range(n_channels):
            rotated_signals[i] += R[i, j] * ambi_signals[j]
    
    return rotated_signals


def decode_to_speakers(ambi_signals: np.ndarray, speaker_positions: np.ndarray,
                      decoding_method: str = 'projection') -> np.ndarray:
    """
    Decode ambisonic signals to a set of loudspeakers.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        speaker_positions: Array of (azimuth, elevation) for each speaker, shape (n_speakers, 2)
        decoding_method: 'projection', 'mode_matching', 'energy_preserving', or 'max_re'
        
    Returns:
        Speaker signals, shape (n_speakers, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    n_speakers = speaker_positions.shape[0]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    # Extract azimuth and elevation
    azimuths = speaker_positions[:, 0]
    elevations = speaker_positions[:, 1]
    
    # Calculate the decoding matrix
    if decoding_method == 'projection':
        # Basic projection decoding (pseudo-inverse of the encoding matrix)
        Y = np.zeros((n_speakers, n_channels))
        
        # Calculate the encoding matrix for each speaker
        for i in range(n_speakers):
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    acn = l * l + l + m
                    Y[i, acn] = real_spherical_harmonic(l, m, azimuths[i], elevations[i])
        
        # Calculate pseudo-inverse of Y
        Y_pinv = np.linalg.pinv(Y)
        
        # Calculate decoding matrix
        D = Y_pinv.T
    
    elif decoding_method == 'mode_matching':
        # Mode matching decoder (exact reconstruction for ideal speaker layouts)
        Y = np.zeros((n_speakers, n_channels))
        
        # Calculate the encoding matrix for each speaker
        for i in range(n_speakers):
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    acn = l * l + l + m
                    Y[i, acn] = real_spherical_harmonic(l, m, azimuths[i], elevations[i])
        
        # Calculate decoding matrix
        D = Y.T / n_speakers
    
    elif decoding_method == 'energy_preserving':
        # Energy preserving decoder (Gerzon's energy vector optimization)
        Y = np.zeros((n_speakers, n_channels))
        
        # Calculate the encoding matrix for each speaker
        for i in range(n_speakers):
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    acn = l * l + l + m
                    Y[i, acn] = real_spherical_harmonic(l, m, azimuths[i], elevations[i])
        
        # Calculate energy-preserving decoding matrix
        # Using singular value decomposition for robustness
        U, S, Vh = np.linalg.svd(Y, full_matrices=False)
        S_inv = np.diag(1.0 / S)
        D = (Vh.T @ S_inv @ U.T) / np.sqrt(n_speakers)
    
    elif decoding_method == 'max_re':
        # Max-rE decoder (maximizes energy concentration)
        Y = np.zeros((n_speakers, n_channels))
        
        # Calculate the encoding matrix for each speaker
        for i in range(n_speakers):
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    acn = l * l + l + m
                    # Apply Max-rE weighting to higher-order components
                    gain = 1.0
                    if l > 0:
                        # Max-rE weights (optimized for energy concentration)
                        max_re_gains = {
                            1: 0.775, 2: 0.4, 3: 0.105, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0
                        }
                        gain = max_re_gains.get(l, 0.0)
                    
                    Y[i, acn] = gain * real_spherical_harmonic(l, m, azimuths[i], elevations[i])
        
        # Calculate decoding matrix
        U, S, Vh = np.linalg.svd(Y, full_matrices=False)
        S_inv = np.diag(1.0 / S)
        D = (Vh.T @ S_inv @ U.T) / np.sqrt(n_speakers)
    
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")
    
    # Apply the decoding matrix to get speaker signals
    speaker_signals = np.zeros((n_speakers, n_samples))
    for i in range(n_speakers):
        for j in range(n_channels):
            speaker_signals[i] += D[j, i] * ambi_signals[j]
    
    return speaker_signals