"""
Encoding and Source Placement Functions

This module contains functions for encoding mono and stereo sources into 
ambisonics as well as utilities for spatial positioning.
"""

import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Union

from .math_utils import real_spherical_harmonic, AmbisonicNormalization


def convert_to_spherical(cartesian: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, distance).
    
    Uses the convention:
    - Azimuth: angle in x-z plane (0 = front, π/2 = left, π = back, 3π/2 = right)
    - Elevation: angle from x-z plane (-π/2 = down, 0 = horizon, π/2 = up)
    - Distance: distance from origin
    
    Args:
        cartesian: (x, y, z) coordinates
        
    Returns:
        (azimuth, elevation, distance) in radians and same distance unit as input
    """
    x, y, z = cartesian
    
    # Calculate distance
    distance = math.sqrt(x*x + y*y + z*z)
    
    # Handle the origin
    if distance < 1e-10:
        return (0.0, 0.0, 0.0)
    
    # Calculate elevation (latitude)
    elevation = math.asin(y / distance)
    
    # Calculate azimuth (longitude)
    azimuth = math.atan2(x, z)
    
    return (azimuth, elevation, distance)


def convert_to_cartesian(spherical: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates (azimuth, elevation, distance) to Cartesian coordinates (x, y, z).
    
    Uses the convention:
    - Azimuth: angle in x-z plane (0 = front, π/2 = left, π = back, 3π/2 = right)
    - Elevation: angle from x-z plane (-π/2 = down, 0 = horizon, π/2 = up)
    - Distance: distance from origin
    
    Args:
        spherical: (azimuth, elevation, distance) in radians and distance unit
        
    Returns:
        (x, y, z) in the same distance unit as input
    """
    azimuth, elevation, distance = spherical
    
    # Calculate Cartesian coordinates
    x = distance * math.sin(azimuth) * math.cos(elevation)
    y = distance * math.sin(elevation)
    z = distance * math.cos(azimuth) * math.cos(elevation)
    
    return (x, y, z)


def encode_mono_source(audio: np.ndarray, position: Tuple[float, float, float], 
                      order: int, normalize: AmbisonicNormalization = AmbisonicNormalization.SN3D) -> np.ndarray:
    """
    Encode a mono audio source into ambisonic signals.
    
    Args:
        audio: Mono audio signal, shape (n_samples,)
        position: (azimuth, elevation, distance) in radians and meters
        order: Ambisonic order
        normalize: Normalization convention
        
    Returns:
        Ambisonic signals, shape ((order+1)², n_samples)
    """
    azimuth, elevation, distance = position
    n_sh = (order + 1) ** 2
    n_samples = len(audio)
    
    # Initialize ambisonic signals
    ambi_signals = np.zeros((n_sh, n_samples))
    
    # Apply distance attenuation (simplified inverse distance law)
    if distance < 1.0:
        distance = 1.0  # Prevent division by zero or amplification
    
    distance_gain = 1.0 / distance
    attenuated_audio = audio * distance_gain
    
    # Encode to each ambisonic channel
    for l in range(order + 1):
        for m in range(-l, l + 1):
            # Calculate ACN index
            acn = l * l + l + m
            
            # Calculate the spherical harmonic coefficient
            sh_val = real_spherical_harmonic(l, m, azimuth, elevation, normalize)
            
            # Apply encoding
            ambi_signals[acn] = attenuated_audio * sh_val
    
    return ambi_signals


def encode_stereo_source(left_audio: np.ndarray, right_audio: np.ndarray, 
                        position: Tuple[float, float, float], width: float,
                        order: int, normalize: AmbisonicNormalization = AmbisonicNormalization.SN3D) -> np.ndarray:
    """
    Encode a stereo audio source into ambisonic signals with appropriate width.
    
    Args:
        left_audio: Left channel audio signal, shape (n_samples,)
        right_audio: Right channel audio signal, shape (n_samples,)
        position: (azimuth, elevation, distance) of the center position in radians and meters
        width: Angular width of the stereo field in radians
        order: Ambisonic order
        normalize: Normalization convention
        
    Returns:
        Ambisonic signals, shape ((order+1)², n_samples)
    """
    # Check if audio lengths match
    if len(left_audio) != len(right_audio):
        raise ValueError("Left and right audio must have the same length")
    
    # Extract center and side signals
    mid = (left_audio + right_audio) * 0.5
    side = (left_audio - right_audio) * 0.5
    
    # Calculate positions for left and right
    azimuth, elevation, distance = position
    left_azimuth = azimuth + width / 2
    right_azimuth = azimuth - width / 2
    
    # Encode mid and side signals
    mid_ambi = encode_mono_source(mid, (azimuth, elevation, distance), order, normalize)
    left_ambi = encode_mono_source(side, (left_azimuth, elevation, distance), order, normalize)
    right_ambi = encode_mono_source(-side, (right_azimuth, elevation, distance), order, normalize)
    
    # Mix the signals
    return mid_ambi + left_ambi + right_ambi