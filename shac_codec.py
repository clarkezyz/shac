"""
Spherical Harmonic Audio Codec (SHAC)

A complete implementation of a spatial audio codec optimized for real-time interactive 
navigation and manipulation of 3D sound fields.

This module implements a comprehensive system for encoding, processing, and decoding
spatial audio using spherical harmonics (ambisonics), with specific optimizations for
interactive applications.

Features:
- Mono source encoding with precise spherical positioning
- Efficient head rotation directly in the ambisonic domain
- Hierarchical encoding with adaptive resolution based on perceptual importance
- Optimized binaural rendering with customizable HRTF support
- Distance-based attenuation with atmospheric modeling
- Advanced frequency-dependent processing
- Support for streaming operation with minimal latency

Author: Claude
License: MIT License
"""

import numpy as np
from scipy import signal, special
from enum import Enum
import math
import warnings
import threading
import queue
import time
import struct
import io
import os
import typing
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

# =====================================================================================
# Constants and Type Definitions
# =====================================================================================

class AmbisonicNormalization(Enum):
    """Defines the normalization convention for spherical harmonics"""
    SN3D = 1  # Schmidt semi-normalized (most common, N3D / sqrt(2n+1))
    N3D = 2   # Fully normalized (orthonormal basis)
    FUMA = 3  # FuMa (legacy B-format) normalization


class AmbisonicOrdering(Enum):
    """Defines the channel ordering convention for ambisonics"""
    ACN = 1    # Ambisonic Channel Number (most common)
    FUMA = 2   # FuMa (legacy B-format) ordering
    CUSTOM = 3 # User-defined ordering


class HRTFInterpolationMethod(Enum):
    """HRTF interpolation methods between measured points"""
    NEAREST = 1      # Nearest neighbor (fastest, lowest quality)
    BILINEAR = 2     # Bilinear interpolation
    SPHERICAL = 3    # Spherical weighted average
    MAGNITUDE = 4    # Magnitude interpolation with phase adjustment
    MODAL = 5        # Modal decomposition and reconstruction


@dataclass
class SourceAttributes:
    """Attributes defining a mono sound source in 3D space"""
    position: Tuple[float, float, float]  # (azimuth, elevation, distance) in radians/meters
    directivity: float = 0.0  # 0.0 = omnidirectional, 1.0 = cardioid, etc.
    directivity_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Direction of max radiation
    width: float = 0.0  # 0.0 = point source, π = 180° spread
    doppler: bool = True  # Whether to apply Doppler effect for moving sources
    air_absorption: bool = True  # Whether to apply frequency-dependent air absorption
    early_reflections: bool = True  # Whether to generate early reflections
    occlusion: float = 0.0  # 0.0 = no occlusion, 1.0 = fully occluded
    obstruction: float = 0.0  # 0.0 = no obstruction, 1.0 = fully obstructed


@dataclass
class RoomAttributes:
    """Attributes defining the acoustic properties of a virtual room"""
    dimensions: Tuple[float, float, float]  # Room dimensions in meters
    reflection_coefficients: Dict[str, float]  # Reflection coeffs for each surface
    scattering_coefficients: Dict[str, float]  # Scattering coeffs for each surface
    absorption_coefficients: Dict[str, Dict[int, float]]  # Freq-dependent absorption
    humidity: float = 50.0  # Relative humidity (%)
    temperature: float = 20.0  # Temperature (°C)
    air_density: float = 1.2  # Air density (kg/m³)


@dataclass
class BinauralRendererConfig:
    """Configuration for binaural rendering from ambisonics"""
    hrtf_database: str  # Path to HRTF database
    interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL
    nearfield_compensation: bool = True  # Apply nearfield compensation for close sources
    crossfade_time: float = 0.1  # Time to crossfade when HRTF changes drastically
    

# =====================================================================================
# Core Mathematical Functions for Spherical Harmonics
# =====================================================================================

def factorial(n: int) -> int:
    """Compute factorial, optimized for small integers"""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    """Compute double factorial n!! = n * (n-2) * (n-4) * ... """
    if n < 0:
        raise ValueError("Double factorial not defined for negative numbers")
    if n <= 1:
        return 1
    
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


def associated_legendre(l: int, m: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the associated Legendre polynomial P_l^m(x) for spherical harmonics.
    
    This is a custom implementation optimized for spherical harmonics, avoiding
    the phase issue in scipy's implementation and handling the normalization correctly.
    
    Args:
        l: Degree of the spherical harmonic (l >= 0)
        m: Order of the spherical harmonic (-l <= m <= l)
        x: Value or array where -1 <= x <= 1
        
    Returns:
        The associated Legendre polynomial value(s)
    """
    m_abs = abs(m)
    
    if m_abs > l:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    
    # Handle the case where l = m = 0 explicitly
    if l == 0 and m == 0:
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
    
    # For m > 0, use the recurrence relationship
    # P_l^m(x) = (2l-1) * sqrt(1-x²) * P_{l-1}^{m-1}(x) - (l+m-1) * P_{l-2}^m(x) / (l-m)
    
    # First compute P_m^m
    pmm = 1.0
    somx2 = np.sqrt((1.0 - x) * (1.0 + x)) if isinstance(x, np.ndarray) else math.sqrt((1.0 - x) * (1.0 + x))
    fact = 1.0
    
    for i in range(1, m_abs + 1):
        pmm *= (-fact) * somx2
        fact += 2.0
    
    if l == m_abs:
        return pmm
    
    # Compute P_{m+1}^m
    pmmp1 = x * (2.0 * m_abs + 1.0) * pmm
    
    if l == m_abs + 1:
        return pmmp1
    
    # Use the recurrence relationship to get higher degrees
    pll = 0.0
    for ll in range(m_abs + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m_abs - 1.0) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll
    
    # Apply Condon-Shortley phase for m < 0
    if m < 0:
        pll *= (-1)**m_abs * factorial(l - m_abs) / factorial(l + m_abs)
    
    return pll


def real_spherical_harmonic(l: int, m: int, theta: float, phi: float, 
                          normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D) -> float:
    """
    Compute the real-valued spherical harmonic Y_l^m(theta, phi) for given degree l and order m.
    
    Args:
        l: Degree of the spherical harmonic (l >= 0)
        m: Order of the spherical harmonic (-l <= m <= l)
        theta: Azimuthal angle in radians [0, 2π)
        phi: Polar angle in radians [0, π]
        normalization: Normalization convention to use
        
    Returns:
        The value of the real spherical harmonic
    """
    # Ensure valid range for parameters
    if l < 0:
        raise ValueError("Degree l must be non-negative")
    if abs(m) > l:
        raise ValueError("Order m must satisfy -l <= m <= l")
    
    # Compute normalization factor
    if normalization == AmbisonicNormalization.SN3D:
        # Schmidt semi-normalized (SN3D)
        if m == 0:
            norm = math.sqrt((2 * l + 1) / (4 * math.pi))
        else:
            norm = math.sqrt((2 * l + 1) / (2 * math.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    elif normalization == AmbisonicNormalization.N3D:
        # Fully normalized (N3D)
        norm = math.sqrt((2 * l + 1) * factorial(l - abs(m)) / (4 * math.pi * factorial(l + abs(m))))
    elif normalization == AmbisonicNormalization.FUMA:
        # FuMa normalization (for legacy B-format)
        if l == 0 and m == 0:
            norm = 1.0 / math.sqrt(2)  # W channel scaling
        elif l == 1:
            norm = 1.0  # X, Y, Z channels
        else:
            # Higher order channels match SN3D but with empirical scaling
            norm = math.sqrt((2 * l + 1) / (2 * math.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    
    # Convert to Cartesian coordinates for associatedLegendre
    x = math.cos(phi)
    
    # Compute the associated Legendre polynomial
    plm = associated_legendre(l, abs(m), x)
    
    # Compute the spherical harmonic
    if m == 0:
        return norm * plm
    elif m > 0:
        return norm * math.sqrt(2) * plm * math.cos(m * theta)
    else:  # m < 0
        return norm * math.sqrt(2) * plm * math.sin(abs(m) * theta)


def spherical_harmonic_matrix(degree: int, thetas: np.ndarray, phis: np.ndarray, 
                             normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D) -> np.ndarray:
    """
    Compute a matrix of spherical harmonics for a set of directions.
    
    Args:
        degree: Maximum degree of spherical harmonics to compute
        thetas: Array of azimuthal angles in radians [0, 2π)
        phis: Array of polar angles in radians [0, π]
        normalization: Normalization convention to use
        
    Returns:
        Matrix of shape (len(thetas), (degree+1)²) where each row contains
        all spherical harmonic values for a specific direction, ordered by ACN.
    """
    n_dirs = len(thetas)
    n_sh = (degree + 1) ** 2
    
    # Initialize the matrix
    Y = np.zeros((n_dirs, n_sh))
    
    # Compute for each direction
    for i in range(n_dirs):
        theta = thetas[i]
        phi = phis[i]
        
        # Compute each spherical harmonic
        for l in range(degree + 1):
            for m in range(-l, l + 1):
                # ACN index
                acn = l * l + l + m
                
                # Compute the spherical harmonic
                Y[i, acn] = real_spherical_harmonic(l, m, theta, phi, normalization)
    
    return Y


def sh_rotation_matrix(degree: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute a rotation matrix for spherical harmonic coefficients.
    
    This implements the full rotation matrix calculation for arbitrarily high orders
    using Wigner D-matrices.
    
    Args:
        degree: Maximum degree of spherical harmonics
        alpha: First Euler angle (yaw) in radians
        beta: Second Euler angle (pitch) in radians
        gamma: Third Euler angle (roll) in radians
        
    Returns:
        Rotation matrix of shape ((degree+1)², (degree+1)²)
    """
    n_sh = (degree + 1) ** 2
    R = np.zeros((n_sh, n_sh))
    
    # Handle degree 0 (omnidirectional component) explicitly
    R[0, 0] = 1.0
    
    # For each degree l
    for l in range(1, degree + 1):
        # Precompute the Wigner D-matrix for this degree
        wigner_d = _compute_wigner_d(l, beta)
        
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                # ACN indices
                acn_m = l * l + l + m
                acn_n = l * l + l + n
                
                # Apply the Euler angles
                R[acn_m, acn_n] = wigner_d[l+m, l+n] * np.exp(-1j * m * alpha) * np.exp(-1j * n * gamma)
    
    # Ensure the result is real (should be within numerical precision)
    return np.real(R)


def _compute_wigner_d(l: int, beta: float) -> np.ndarray:
    """
    Compute the Wigner d-matrix for rotation around the y-axis.
    
    Args:
        l: Degree of spherical harmonics
        beta: Rotation angle around y-axis (pitch) in radians
        
    Returns:
        Wigner d-matrix of shape (2l+1, 2l+1)
    """
    size = 2 * l + 1
    d = np.zeros((size, size), dtype=np.complex128)
    
    # Compute half-angle values
    cos_beta_2 = np.cos(beta / 2)
    sin_beta_2 = np.sin(beta / 2)
    
    # For each pair of orders (m, n)
    for m_idx in range(size):
        m = m_idx - l
        for n_idx in range(size):
            n = n_idx - l
            
            # Apply the Wigner formula
            j_min = max(0, n - m)
            j_max = min(l + n, l - m)
            
            d_val = 0.0
            for j in range(j_min, j_max + 1):
                num = math.sqrt(factorial(l + n) * factorial(l - n) * factorial(l + m) * factorial(l - m))
                denom = factorial(j) * factorial(l + n - j) * factorial(l - m - j) * factorial(j + m - n)
                
                d_val += ((-1) ** (j + n - m)) * (num / denom) * \
                        (cos_beta_2 ** (2 * l - 2 * j + m - n)) * \
                        (sin_beta_2 ** (2 * j + n - m))
            
            d[m_idx, n_idx] = d_val
    
    return d


def convert_acn_to_fuma(ambi_acn: np.ndarray) -> np.ndarray:
    """
    Convert ambisonic signals from ACN ordering to FuMa ordering.
    
    Args:
        ambi_acn: Ambisonic signals in ACN ordering, shape (n_channels, n_samples)
        
    Returns:
        Ambisonic signals in FuMa ordering
    """
    n_channels = ambi_acn.shape[0]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    if (order + 1) ** 2 != n_channels:
        raise ValueError(f"Number of channels {n_channels} does not correspond to a complete ambisonic order")
    
    # Initialize FuMa array
    ambi_fuma = np.zeros_like(ambi_acn)
    
    # Conversion table (from ACN to FuMa)
    # W, Y, Z, X, V, T, R, S, U, Q, O, M, K, L, N, P
    fuma_to_acn = {
        0: 0,   # W
        1: 3,   # X
        2: 1,   # Y
        3: 2,   # Z
        4: 6,   # R
        5: 8,   # S
        6: 4,   # T
        7: 5,   # U
        8: 7,   # V
        9: 15,  # K
        10: 13, # L
        11: 11, # M
        12: 12, # N
        13: 10, # O
        14: 14, # P
        15: 9   # Q
    }
    
    # Conversion factor table (from ACN/SN3D to FuMa)
    # Each degree has a specific scaling factor
    conversion_factors = {
        0: math.sqrt(2),    # W (degree 0)
        1: 1.0,             # X, Y, Z (degree 1)
        2: 1.0 / math.sqrt(3),  # R, S, T, U, V (degree 2)
        3: 1.0 / math.sqrt(5),  # K, L, M, N, O, P, Q (degree 3)
        # Add more if needed for higher orders
    }
    
    # Apply the conversion (only up to 3rd order for now)
    for fuma_idx in range(min(n_channels, 16)):
        acn_idx = fuma_to_acn[fuma_idx]
        
        if acn_idx < n_channels:
            # Determine the degree based on ACN index
            degree = math.floor(math.sqrt(acn_idx))
            
            # Apply the conversion
            ambi_fuma[fuma_idx] = ambi_acn[acn_idx] * conversion_factors[degree]
    
    return ambi_fuma


def convert_fuma_to_acn(ambi_fuma: np.ndarray) -> np.ndarray:
    """
    Convert ambisonic signals from FuMa ordering to ACN ordering with SN3D normalization.
    
    Args:
        ambi_fuma: Ambisonic signals in FuMa ordering, shape (n_channels, n_samples)
        
    Returns:
        Ambisonic signals in ACN ordering with SN3D normalization
    """
    n_channels = ambi_fuma.shape[0]
    
    # Maximum supported channels for FuMa (up to 3rd order)
    max_channels = 16
    if n_channels > max_channels:
        raise ValueError(f"FuMa conversion only supported up to 3rd order (16 channels), got {n_channels}")
    
    # Determine the order from the number of channels
    if n_channels <= 1:
        order = 0
    elif n_channels <= 4:
        order = 1
    elif n_channels <= 9:
        order = 2
    elif n_channels <= 16:
        order = 3
    else:
        order = math.floor(math.sqrt(n_channels)) - 1
    
    # Initialize ACN array with the appropriate number of channels
    acn_channels = (order + 1) ** 2
    ambi_acn = np.zeros((acn_channels, ambi_fuma.shape[1]))
    
    # Conversion table (from FuMa to ACN)
    fuma_to_acn = {
        0: 0,   # W
        1: 3,   # X
        2: 1,   # Y
        3: 2,   # Z
        4: 6,   # R
        5: 8,   # S
        6: 4,   # T
        7: 5,   # U
        8: 7,   # V
        9: 15,  # K
        10: 13, # L
        11: 11, # M
        12: 12, # N
        13: 10, # O
        14: 14, # P
        15: 9   # Q
    }
    
    # Conversion factor table (from FuMa to ACN/SN3D)
    conversion_factors = {
        0: 1.0 / math.sqrt(2),  # W (degree 0)
        1: 1.0,                 # X, Y, Z (degree 1)
        2: math.sqrt(3),        # R, S, T, U, V (degree 2)
        3: math.sqrt(5),        # K, L, M, N, O, P, Q (degree 3)
        # Add more if needed for higher orders
    }
    
    # Apply the conversion
    for fuma_idx in range(min(n_channels, 16)):
        acn_idx = fuma_to_acn[fuma_idx]
        
        if acn_idx < acn_channels:
            # Determine the degree based on ACN index
            degree = math.floor(math.sqrt(acn_idx))
            
            # Apply the conversion
            ambi_acn[acn_idx] = ambi_fuma[fuma_idx] * conversion_factors[degree]
    
    return ambi_acn


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


# =====================================================================================
# Core Ambisonics Processing Functions
# =====================================================================================

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


def binauralize_ambisonics(ambi_signals: np.ndarray, hrtf_database: Union[str, Dict], 
                           normalize: bool = True) -> np.ndarray:
    """
    Convert ambisonic signals to binaural stereo using HRTF convolution.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        hrtf_database: Path to HRTF database or dictionary with HRTF data
        normalize: Whether to normalize the output
        
    Returns:
        Binaural stereo signals, shape (2, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    # Load HRTF data if needed
    if isinstance(hrtf_database, str):
        hrtf_data = load_hrtf_database(hrtf_database)
    else:
        hrtf_data = hrtf_database
    
    # Get the HRTF filters in SH domain
    sh_hrtfs = hrtf_data['sh_hrtfs']
    if sh_hrtfs.shape[1] < n_channels:
        raise ValueError(f"HRTF database only supports up to order {math.floor(math.sqrt(sh_hrtfs.shape[1])) - 1}, but got signals of order {order}")
    
    # Apply SH-domain convolution
    binaural = np.zeros((2, n_samples + sh_hrtfs.shape[2] - 1))
    for ch in range(n_channels):
        binaural[0] += np.convolve(ambi_signals[ch], sh_hrtfs[0, ch])
        binaural[1] += np.convolve(ambi_signals[ch], sh_hrtfs[1, ch])
    
    # Truncate the result to the original length
    binaural = binaural[:, :n_samples]
    
    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(binaural))
        if max_val > 0.0:
            binaural = binaural / max_val * 0.99
    
    return binaural


def load_hrtf_database(hrtf_path: str) -> Dict:
    """
    Load and prepare an HRTF database for use with the binauralize_ambisonics function.
    
    Args:
        hrtf_path: Path to the HRTF database
        
    Returns:
        Dictionary containing HRTF data in suitable format for binauralization
    """
    # This is a placeholder for a real HRTF database loader
    # In practice, this would load from a specific format like SOFA
    
    # For demonstration, we'll create a simplified HRTF in the SH domain
    # In reality, this would come from measurements
    
    # Parameters for the synthetic HRTF
    max_order = 3
    n_channels = (max_order + 1) ** 2
    hrtf_length = 256
    
    # Create synthetic SH-domain HRTFs
    sh_hrtfs = np.zeros((2, n_channels, hrtf_length))
    
    # W channel (order 0)
    # Simple approximation: omnidirectional with ITD
    sh_hrtfs[0, 0] = np.hstack([np.zeros(5), np.exp(-np.arange(hrtf_length-5)/20)])
    sh_hrtfs[1, 0] = np.hstack([np.zeros(8), np.exp(-np.arange(hrtf_length-8)/20)])
    
    # Y channel (order 1, m=-1): front-back
    # Affect the coloration differences between front and back
    sh_hrtfs[0, 1] = np.hstack([np.zeros(5), 0.5 * np.exp(-np.arange(hrtf_length-5)/15)])
    sh_hrtfs[1, 1] = np.hstack([np.zeros(8), 0.5 * np.exp(-np.arange(hrtf_length-8)/15)])
    
    # Z channel (order 1, m=0): up-down
    # Affect the coloration differences between up and down
    sh_hrtfs[0, 2] = np.hstack([np.zeros(5), 0.3 * np.exp(-np.arange(hrtf_length-5)/10)])
    sh_hrtfs[1, 2] = np.hstack([np.zeros(8), 0.3 * np.exp(-np.arange(hrtf_length-8)/10)])
    
    # X channel (order 1, m=1): left-right
    # Strongest ILD component
    sh_hrtfs[0, 3] = np.hstack([np.zeros(5), -0.7 * np.exp(-np.arange(hrtf_length-5)/25)])
    sh_hrtfs[1, 3] = np.hstack([np.zeros(8), 0.7 * np.exp(-np.arange(hrtf_length-8)/25)])
    
    # Higher order components (simplified)
    for ch in range(4, n_channels):
        l = math.floor(math.sqrt(ch))
        # Reduce the amplitude with increasing order
        gain = 0.2 * (1.0 / l)
        
        # Add some randomness to the higher order components
        # In a real HRTF, these would have specific patterns
        np.random.seed(ch)  # For reproducibility
        sh_hrtfs[0, ch] = gain * np.random.randn(hrtf_length) * np.exp(-np.arange(hrtf_length)/10)
        sh_hrtfs[1, ch] = gain * np.random.randn(hrtf_length) * np.exp(-np.arange(hrtf_length)/10)
    
    # Normalize
    for ear in range(2):
        max_val = np.max(np.abs(sh_hrtfs[ear, 0]))
        if max_val > 0:
            sh_hrtfs[ear, 0] = sh_hrtfs[ear, 0] / max_val
    
    # Create the output dictionary
    hrtf_data = {
        'sh_hrtfs': sh_hrtfs,
        'sample_rate': 48000,
        'max_order': max_order
    }
    
    return hrtf_data


def apply_frequency_dependent_effects(ambi_signals: np.ndarray, sample_rate: int, 
                                     distance: float, air_absorption: bool = True,
                                     source_directivity: Optional[Dict] = None) -> np.ndarray:
    """
    Apply frequency-dependent effects to ambisonic signals.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        sample_rate: Sample rate in Hz
        distance: Distance from listener in meters
        air_absorption: Whether to apply air absorption with distance
        source_directivity: Optional source directivity parameters
        
    Returns:
        Processed ambisonic signals, shape (n_channels, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    # Apply frequency-dependent processing
    # For efficiency, we'll use FFT-based processing
    
    # Convert to frequency domain
    ambi_freq = np.fft.rfft(ambi_signals, axis=1)
    n_freqs = ambi_freq.shape[1]
    
    # Calculate frequency vector
    freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)
    
    # Process each frequency bin
    for f_idx, freq in enumerate(freqs):
        # Skip DC and very low frequencies
        if freq < 20:
            continue
        
        # 1. Air absorption with distance
        if air_absorption and distance > 1.0:
            # Simplified air absorption model
            # Higher frequencies are attenuated more with distance
            absorption = 1.0 - np.clip(0.0001 * freq * (distance - 1.0), 0.0, 0.95)
            ambi_freq[:, f_idx] *= absorption
        
        # 2. Apply source directivity if provided
        if source_directivity is not None:
            # Extract directivity parameters
            pattern = source_directivity.get('pattern', 'omnidirectional')
            directivity_order = source_directivity.get('order', 1.0)
            axis = source_directivity.get('axis', (1.0, 0.0, 0.0))  # Default to front
            freq_dependent = source_directivity.get('frequency_dependent', False)
            
            # Apply frequency-dependent weighting to directivity
            if freq_dependent:
                # Higher frequencies are often more directional
                if freq < 250:
                    weight = 0.2
                elif freq < 1000:
                    weight = 0.5
                elif freq < 4000:
                    weight = 0.8
                else:
                    weight = 1.0
                
                # Scale the directivity order by the frequency weight
                effective_order = directivity_order * weight
            else:
                effective_order = directivity_order
            
            # Apply the directivity pattern
            if pattern == 'cardioid' and effective_order > 0.0:
                # Cardioid pattern primarily affects first-order components
                # Scale Y, Z, X channels based on the pattern's orientation
                for l in range(1, min(2, order + 1)):
                    for m in range(-l, l + 1):
                        acn = l * l + l + m
                        # Simplified directivity scaling
                        ambi_freq[acn, f_idx] *= 1.0 - 0.5 * effective_order
    
    # Convert back to time domain
    processed_signals = np.fft.irfft(ambi_freq, n=n_samples, axis=1)
    
    return processed_signals


def apply_early_reflections(ambi_signals: np.ndarray, source_position: Tuple[float, float, float],
                          room_dimensions: Tuple[float, float, float], reflection_coefficients: Dict[str, float],
                          sample_rate: int, max_order: int = 1) -> np.ndarray:
    """
    Add early reflections to ambisonic signals based on a simple shoebox room model.
    
    Args:
        ambi_signals: Direct path ambisonic signals, shape (n_channels, n_samples)
        source_position: (x, y, z) position of the source in meters
        room_dimensions: (width, height, length) of the room in meters
        reflection_coefficients: Reflection coefficients for each surface
        sample_rate: Sample rate in Hz
        max_order: Maximum reflection order to compute
        
    Returns:
        Ambisonic signals with early reflections, shape (n_channels, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    
    # Convert source position to Cartesian if needed
    if len(source_position) == 3 and isinstance(source_position[0], float):
        if source_position[2] > 100:  # Assume it's spherical if distance is large
            source_position = convert_to_cartesian(source_position)
    
    # Extract room parameters
    width, height, length = room_dimensions
    
    # Listener is at the center of the room
    listener_position = (width/2, height/2, length/2)
    
    # Calculate direct path distance
    sx, sy, sz = source_position
    lx, ly, lz = listener_position
    direct_distance = math.sqrt((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2)
    
    # Calculate direct path delay in samples
    speed_of_sound = 343.0  # m/s
    direct_delay_samples = int(direct_distance / speed_of_sound * sample_rate)
    
    # Initialize output signals with the direct path
    output_signals = ambi_signals.copy()
    
    # Compute early reflections
    for x_order in range(-max_order, max_order + 1):
        for y_order in range(-max_order, max_order + 1):
            for z_order in range(-max_order, max_order + 1):
                # Skip direct path
                if x_order == 0 and y_order == 0 and z_order == 0:
                    continue
                
                # Calculate image source position
                image_x = sx
                image_y = sy
                image_z = sz
                
                # Apply reflections in x dimension
                if x_order != 0:
                    if x_order % 2 == 0:  # Even reflections
                        image_x = 2 * width * (x_order // 2) + sx
                    else:  # Odd reflections
                        image_x = 2 * width * (x_order // 2) + (2 * width - sx)
                
                # Apply reflections in y dimension
                if y_order != 0:
                    if y_order % 2 == 0:  # Even reflections
                        image_y = 2 * height * (y_order // 2) + sy
                    else:  # Odd reflections
                        image_y = 2 * height * (y_order // 2) + (2 * height - sy)
                
                # Apply reflections in z dimension
                if z_order != 0:
                    if z_order % 2 == 0:  # Even reflections
                        image_z = 2 * length * (z_order // 2) + sz
                    else:  # Odd reflections
                        image_z = 2 * length * (z_order // 2) + (2 * length - sz)
                
                # Calculate reflection distance
                reflection_distance = math.sqrt((image_x-lx)**2 + (image_y-ly)**2 + (image_z-lz)**2)
                
                # Calculate delay in samples
                delay_samples = int(reflection_distance / speed_of_sound * sample_rate)
                
                # Skip if delay is too long
                if delay_samples >= n_samples:
                    continue
                
                # Calculate attenuation based on distance and reflection coefficients
                distance_attenuation = direct_distance / reflection_distance
                
                # Apply reflection coefficients
                reflection_attenuation = 1.0
                if abs(x_order) > 0:
                    wall = 'left' if (x_order % 2) == 1 else 'right'
                    reflection_attenuation *= reflection_coefficients.get(wall, 0.9) ** abs(x_order)
                
                if abs(y_order) > 0:
                    wall = 'floor' if (y_order % 2) == 1 else 'ceiling'
                    reflection_attenuation *= reflection_coefficients.get(wall, 0.9) ** abs(y_order)
                
                if abs(z_order) > 0:
                    wall = 'front' if (z_order % 2) == 1 else 'back'
                    reflection_attenuation *= reflection_coefficients.get(wall, 0.9) ** abs(z_order)
                
                # Total attenuation
                attenuation = distance_attenuation * reflection_attenuation
                
                # Calculate direction to image source
                direction_x = image_x - lx
                direction_y = image_y - ly
                direction_z = image_z - lz
                
                # Convert to spherical coordinates
                azimuth = math.atan2(direction_x, direction_z)
                elevation = math.atan2(direction_y, math.sqrt(direction_x**2 + direction_z**2))
                
                # Encode the reflection as a delayed, attenuated copy
                reflection_signal = np.zeros_like(ambi_signals)
                
                # Extract the W channel (omnidirectional component)
                w_channel = ambi_signals[0]
                
                # Create a delayed version of the W channel
                delayed_w = np.zeros_like(w_channel)
                if delay_samples < len(delayed_w):
                    delayed_w[delay_samples:] = w_channel[:len(delayed_w)-delay_samples]
                
                # Encode the delayed signal to ambisonics
                for l in range(n_channels):
                    for m in range(-l, l + 1):
                        acn = l * l + l + m
                        if acn < n_channels:
                            sh_val = real_spherical_harmonic(l, m, azimuth, elevation)
                            reflection_signal[acn] = delayed_w * sh_val * attenuation
                
                # Add the reflection to the output
                output_signals += reflection_signal
    
    # Normalize if necessary
    max_val = np.max(np.abs(output_signals))
    if max_val > 0.99:
        output_signals = output_signals * 0.99 / max_val
    
    return output_signals


def apply_diffuse_reverberation(ambi_signals: np.ndarray, rt60: float, sample_rate: int,
                              room_volume: float, early_reflection_delay: int = 0) -> np.ndarray:
    """
    Add diffuse reverberation to ambisonic signals.
    
    Args:
        ambi_signals: Ambisonic signals with direct path and early reflections
        rt60: Reverberation time (time for level to drop by 60dB) in seconds
        sample_rate: Sample rate in Hz
        room_volume: Room volume in cubic meters
        early_reflection_delay: Delay in samples before the reverb tail starts
        
    Returns:
        Ambisonic signals with diffuse reverberation, shape (n_channels, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    
    # Calculate reverberation parameters
    reverb_samples = int(rt60 * sample_rate)
    if reverb_samples <= 0:
        return ambi_signals
    
    # Create a frequency-dependent reverb tail
    # For simplicity, we'll use exponentially decaying noise
    reverb_tail = np.zeros((n_channels, n_samples + reverb_samples))
    
    # Copy the input signals
    reverb_tail[:, :n_samples] = ambi_signals
    
    # Generate the diffuse reverb tail
    for ch in range(n_channels):
        # Skip the W channel (index 0) as it's omnidirectional
        if ch == 0:
            continue
        
        # Create decorrelated noise for each channel
        np.random.seed(ch)  # For reproducibility
        noise = np.random.randn(reverb_samples)
        
        # Apply exponential decay
        decay = np.exp(-6.91 * np.arange(reverb_samples) / reverb_samples)
        
        # Apply frequency-dependent filtering
        # Higher-order components decay faster in reverb
        l = math.floor(math.sqrt(ch))
        freq_decay = np.exp(-0.5 * l)
        
        # Create the tail
        reverb_tail[ch, n_samples:] += noise * decay * freq_decay * 0.1
    
    # Add more energy to the W channel
    np.random.seed(0)
    w_noise = np.random.randn(reverb_samples)
    decay = np.exp(-6.91 * np.arange(reverb_samples) / reverb_samples)
    reverb_tail[0, n_samples:] += w_noise * decay * 0.2
    
    # Trim to original length
    output_signals = reverb_tail[:, :n_samples]
    
    # Normalize if necessary
    max_val = np.max(np.abs(output_signals))
    if max_val > 0.99:
        output_signals = output_signals * 0.99 / max_val
    
    return output_signals


# =====================================================================================
# File Format Implementation
# =====================================================================================

class SHACFileWriter:
    """
    Writer for the Spherical Harmonic Audio Codec (SHAC) file format.
    """
    
    def __init__(self, order: int, sample_rate: int, normalize: AmbisonicNormalization = AmbisonicNormalization.SN3D):
        """
        Initialize the SHAC file writer.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            normalize: Normalization convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.n_channels = (order + 1) ** 2
        self.layers = {}
        self.layer_metadata = {}
    
    def add_layer(self, layer_id: str, ambi_signals: np.ndarray, metadata: Dict = None):
        """
        Add a layer to the SHAC file.
        
        Args:
            layer_id: Unique identifier for the layer
            ambi_signals: Ambisonic signals for this layer
            metadata: Optional metadata for the layer
        """
        if ambi_signals.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {ambi_signals.shape[0]}")
        
        self.layers[layer_id] = ambi_signals
        
        if metadata is None:
            metadata = {}
        
        self.layer_metadata[layer_id] = metadata
    
    def write_file(self, filename: str, bit_depth: int = 32) -> None:
        """
        Write the SHAC file to disk.
        
        Args:
            filename: Output filename
            bit_depth: Bit depth (16 or 32)
        """
        # Validate bit depth
        if bit_depth not in [16, 32]:
            raise ValueError("Bit depth must be 16 or 32")
        
        # Determine the sample format code
        sample_format = 1 if bit_depth == 16 else 2
        
        # Find the maximum number of frames across all layers
        max_frames = max(layer.shape[1] for layer in self.layers.values()) if self.layers else 0
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(b'SHAC')  # Magic number
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<I', self.order))  # Ambisonics order
            f.write(struct.pack('<I', self.sample_rate))  # Sample rate
            f.write(struct.pack('<I', self.n_channels))  # Number of channels
            f.write(struct.pack('<I', sample_format))  # Sample format
            f.write(struct.pack('<I', max_frames))  # Number of frames
            f.write(struct.pack('<I', 0))  # Reserved
            
            # Write channel metadata section
            channel_metadata_size = 16 * self.n_channels + 4
            f.write(struct.pack('<I', channel_metadata_size))  # Section size
            
            for ch in range(self.n_channels):
                l = math.floor(math.sqrt(ch))
                m = ch - l*l - l
                
                f.write(struct.pack('<I', ch))  # Channel index
                f.write(struct.pack('<I', l))  # Spherical harmonic degree
                f.write(struct.pack('<I', m))  # Spherical harmonic order
                f.write(struct.pack('<I', self.normalize.value))  # Normalization type
            
            # Write layer information section
            layer_section_size = 8  # Base size (section size + number of layers)
            for layer_id, metadata in self.layer_metadata.items():
                layer_section_size += 8  # Layer ID + name length
                layer_section_size += len(layer_id.encode('utf-8'))  # Layer name
                layer_section_size += 4 + (self.n_channels + 7) // 8  # Channel mask size + mask
            
            f.write(struct.pack('<I', layer_section_size))  # Section size
            f.write(struct.pack('<I', len(self.layers)))  # Number of layers
            
            # Write each layer's metadata
            for layer_id, metadata in self.layer_metadata.items():
                layer_id_bytes = layer_id.encode('utf-8')
                f.write(struct.pack('<I', hash(layer_id) & 0xFFFFFFFF))  # Layer ID (hashed)
                f.write(struct.pack('<I', len(layer_id_bytes)))  # Layer name length
                f.write(layer_id_bytes)  # Layer name
                
                # Write channel mask (which channels belong to this layer)
                mask_bytes = (self.n_channels + 7) // 8
                mask = bytearray(mask_bytes)
                for i in range(self.n_channels):
                    mask[i // 8] |= (1 << (i % 8))
                
                f.write(struct.pack('<I', mask_bytes))  # Mask size
                f.write(mask)  # Channel mask
            
            # Write audio data section
            # Mix all layers together
            mixed_audio = np.zeros((self.n_channels, max_frames))
            for layer_id, ambi_signals in self.layers.items():
                n_frames = ambi_signals.shape[1]
                mixed_audio[:, :n_frames] += ambi_signals
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.99:
                mixed_audio = mixed_audio * 0.99 / max_val
            
            # Convert to the appropriate format
            if bit_depth == 16:
                mixed_audio = (mixed_audio * 32767).astype(np.int16)
            else:
                mixed_audio = mixed_audio.astype(np.float32)
            
            # Write interleaved audio data
            for frame in range(max_frames):
                for ch in range(self.n_channels):
                    if bit_depth == 16:
                        f.write(struct.pack('<h', mixed_audio[ch, frame]))
                    else:
                        f.write(struct.pack('<f', mixed_audio[ch, frame]))


class SHACFileReader:
    """
    Reader for the Spherical Harmonic Audio Codec (SHAC) file format.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the SHAC file reader.
        
        Args:
            filename: Input filename
        """
        self.filename = filename
        self.file_info = {}
        self.layers = {}
        self.layer_metadata = {}
        self._read_header()
    
    def _read_header(self) -> None:
        """Read the SHAC file header and metadata."""
        with open(self.filename, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'SHAC':
                raise ValueError("Not a valid SHAC file")
            
            version = struct.unpack('<I', f.read(4))[0]
            order = struct.unpack('<I', f.read(4))[0]
            sample_rate = struct.unpack('<I', f.read(4))[0]
            n_channels = struct.unpack('<I', f.read(4))[0]
            sample_format = struct.unpack('<I', f.read(4))[0]
            n_frames = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # Reserved
            
            # Store file info
            self.file_info = {
                'version': version,
                'order': order,
                'sample_rate': sample_rate,
                'n_channels': n_channels,
                'sample_format': sample_format,
                'n_frames': n_frames
            }
            
            # Read channel metadata section
            section_size = struct.unpack('<I', f.read(4))[0]
            channel_metadata = []
            
            for _ in range(n_channels):
                channel_idx = struct.unpack('<I', f.read(4))[0]
                sh_degree = struct.unpack('<I', f.read(4))[0]
                sh_order = struct.unpack('<I', f.read(4))[0]
                norm_type = struct.unpack('<I', f.read(4))[0]
                
                channel_metadata.append({
                    'channel_idx': channel_idx,
                    'sh_degree': sh_degree,
                    'sh_order': sh_order,
                    'normalization': AmbisonicNormalization(norm_type)
                })
            
            self.file_info['channel_metadata'] = channel_metadata
            
            # Read layer information section
            section_size = struct.unpack('<I', f.read(4))[0]
            n_layers = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(n_layers):
                layer_id = struct.unpack('<I', f.read(4))[0]
                name_length = struct.unpack('<I', f.read(4))[0]
                layer_name = f.read(name_length).decode('utf-8')
                
                mask_size = struct.unpack('<I', f.read(4))[0]
                channel_mask = f.read(mask_size)
                
                # Convert mask to list of channels
                channels = []
                for byte_idx, byte_val in enumerate(channel_mask):
                    for bit_idx in range(8):
                        if byte_val & (1 << bit_idx):
                            channel_idx = byte_idx * 8 + bit_idx
                            if channel_idx < n_channels:
                                channels.append(channel_idx)
                
                self.layer_metadata[layer_name] = {
                    'layer_id': layer_id,
                    'channels': channels
                }
            
            # Store the offset to the audio data
            self.audio_data_offset = f.tell()
    
    def read_audio_data(self) -> np.ndarray:
        """
        Read the complete audio data from the file.
        
        Returns:
            Ambisonic signals, shape (n_channels, n_frames)
        """
        n_channels = self.file_info['n_channels']
        n_frames = self.file_info['n_frames']
        sample_format = self.file_info['sample_format']
        
        # Determine data type and scaling
        if sample_format == 1:  # 16-bit PCM
            dtype = np.int16
            scale = 1.0 / 32768.0
        elif sample_format == 2:  # 32-bit float
            dtype = np.float32
            scale = 1.0
        else:
            raise ValueError(f"Unsupported sample format: {sample_format}")
        
        with open(self.filename, 'rb') as f:
            # Seek to audio data
            f.seek(self.audio_data_offset)
            
            # Read interleaved audio data
            audio_data = np.zeros((n_channels, n_frames), dtype=np.float32)
            
            # Read frame by frame
            for frame in range(n_frames):
                for ch in range(n_channels):
                    if sample_format == 1:  # 16-bit PCM
                        sample = struct.unpack('<h', f.read(2))[0]
                    else:  # 32-bit float
                        sample = struct.unpack('<f', f.read(4))[0]
                    
                    audio_data[ch, frame] = sample * scale
        
        return audio_data
    
    def read_layer(self, layer_name: str) -> Optional[np.ndarray]:
        """
        Read a specific layer from the file.
        
        Args:
            layer_name: Name of the layer to read
            
        Returns:
            Ambisonic signals for the specified layer, or None if not found
        """
        if layer_name not in self.layer_metadata:
            return None
        
        # Get the full audio data
        full_audio = self.read_audio_data()
        
        # Extract the channels for this layer
        channels = self.layer_metadata[layer_name]['channels']
        layer_audio = full_audio[channels]
        
        return layer_audio
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the file.
        
        Returns:
            List of layer names
        """
        return list(self.layer_metadata.keys())
    
    def get_file_info(self) -> Dict:
        """
        Get information about the SHAC file.
        
        Returns:
            Dictionary with file information
        """
        return self.file_info


# =====================================================================================
# SHAC Codec Implementation
# =====================================================================================

class SHACCodec:
    """
    Main class for the Spherical Harmonic Audio Codec (SHAC).
    
    This class provides a high-level interface for encoding, processing,
    and decoding spatial audio using spherical harmonics.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, 
                normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D,
                ordering: AmbisonicOrdering = AmbisonicOrdering.ACN):
        """
        Initialize the SHAC codec.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            normalization: Normalization convention
            ordering: Channel ordering convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.normalization = normalization
        self.ordering = ordering
        self.n_channels = (order + 1) ** 2
        
        # Initialize processing state
        self.sources = {}
        self.layers = {}
        self.layer_metadata = {}
        
        # Initialize room model
        self.room = None
        
        # Initialize binaural renderer
        self.binaural_renderer = None
        
        # Load default HRTF if available
        self.hrtf_database = self._load_default_hrtf()
    
    def _load_default_hrtf(self) -> Optional[Dict]:
        """
        Load a default HRTF database if available.
        
        Returns:
            HRTF database or None if not available
        """
        # In a real implementation, this would load from a file
        # Here we'll use our synthetic HRTF generator
        try:
            return load_hrtf_database("")
        except:
            warnings.warn("Could not load default HRTF database")
            return None
    
    def add_mono_source(self, source_id: str, audio: np.ndarray, position: Tuple[float, float, float],
                       attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a mono audio source to the codec.
        
        Args:
            source_id: Unique identifier for the source
            audio: Mono audio signal
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if source_id in self.sources:
            warnings.warn(f"Source {source_id} already exists. Overwriting.")
        
        # Ensure audio is mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * 0.99 / max_val
        
        # Store source data
        self.sources[source_id] = {
            'audio': audio,
            'position': position,
            'attributes': attributes or SourceAttributes(position)
        }
        
        # Encode to ambisonic signals
        ambi_signals = encode_mono_source(audio, position, self.order, self.normalization)
        
        # Apply source directivity and other attributes if provided
        if attributes and (attributes.directivity > 0 or attributes.width > 0):
            source_directivity = {
                'pattern': 'cardioid',
                'order': attributes.directivity,
                'axis': attributes.directivity_axis,
                'frequency_dependent': True
            }
            
            ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                           position[2], attributes.air_absorption,
                                                           source_directivity)
        
        # Create a layer for this source
        self.layers[source_id] = ambi_signals
        
        # Store layer metadata
        self.layer_metadata[source_id] = {
            'type': 'source',
            'position': position,
            'original_gain': 1.0,
            'current_gain': 1.0,
            'muted': False
        }
    
    def add_stereo_source(self, source_id: str, left_audio: np.ndarray, right_audio: np.ndarray,
                         position: Tuple[float, float, float], width: float = 0.6,
                         attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a stereo audio source to the codec.
        
        Args:
            source_id: Unique identifier for the source
            left_audio: Left channel audio signal
            right_audio: Right channel audio signal
            position: (azimuth, elevation, distance) of center position
            width: Angular width in radians
            attributes: Optional source attributes
        """
        if source_id in self.sources:
            warnings.warn(f"Source {source_id} already exists. Overwriting.")
        
        # Ensure audio is mono for each channel
        if len(left_audio.shape) > 1 and left_audio.shape[1] > 1:
            left_audio = np.mean(left_audio, axis=1)
        if len(right_audio.shape) > 1 and right_audio.shape[1] > 1:
            right_audio = np.mean(right_audio, axis=1)
        
        # Ensure same length
        if len(left_audio) != len(right_audio):
            max_len = max(len(left_audio), len(right_audio))
            if len(left_audio) < max_len:
                left_audio = np.pad(left_audio, (0, max_len - len(left_audio)))
            if len(right_audio) < max_len:
                right_audio = np.pad(right_audio, (0, max_len - len(right_audio)))
        
        # Normalize audio to prevent clipping
        max_val = max(np.max(np.abs(left_audio)), np.max(np.abs(right_audio)))
        if max_val > 0.99:
            scale = 0.99 / max_val
            left_audio = left_audio * scale
            right_audio = right_audio * scale
        
        # Store source data
        self.sources[source_id] = {
            'left_audio': left_audio,
            'right_audio': right_audio,
            'position': position,
            'width': width,
            'attributes': attributes or SourceAttributes(position)
        }
        
        # Encode to ambisonic signals
        ambi_signals = encode_stereo_source(left_audio, right_audio, position, width,
                                          self.order, self.normalization)
        
        # Apply source attributes if provided
        if attributes:
            # For stereo sources, only apply distance-based effects
            ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                           position[2], attributes.air_absorption)
        
        # Create a layer for this source
        self.layers[source_id] = ambi_signals
        
        # Store layer metadata
        self.layer_metadata[source_id] = {
            'type': 'stereo_source',
            'position': position,
            'width': width,
            'original_gain': 1.0,
            'current_gain': 1.0,
            'muted': False
        }
    
    def set_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update position in source data
        self.sources[source_id]['position'] = position
        
        # Update position in layer metadata
        self.layer_metadata[source_id]['position'] = position
        
        # Re-encode the source with the new position
        source_data = self.sources[source_id]
        
        if 'audio' in source_data:
            # Mono source
            audio = source_data['audio']
            attributes = source_data['attributes']
            
            # Re-encode
            ambi_signals = encode_mono_source(audio, position, self.order, self.normalization)
            
            # Apply source attributes if needed
            if attributes and (attributes.directivity > 0 or attributes.width > 0):
                source_directivity = {
                    'pattern': 'cardioid',
                    'order': attributes.directivity,
                    'axis': attributes.directivity_axis,
                    'frequency_dependent': True
                }
                
                ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                               position[2], attributes.air_absorption,
                                                               source_directivity)
            
            # Update layer
            self.layers[source_id] = ambi_signals
        
        elif 'left_audio' in source_data:
            # Stereo source
            left_audio = source_data['left_audio']
            right_audio = source_data['right_audio']
            width = source_data['width']
            attributes = source_data['attributes']
            
            # Re-encode
            ambi_signals = encode_stereo_source(left_audio, right_audio, position, width,
                                              self.order, self.normalization)
            
            # Apply source attributes if needed
            if attributes:
                ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                               position[2], attributes.air_absorption)
            
            # Update layer
            self.layers[source_id] = ambi_signals
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.layers:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update gain in layer metadata
        self.layer_metadata[source_id]['current_gain'] = gain
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source.
        
        Args:
            source_id: Identifier of the source to update
            muted: True to mute, False to unmute
        """
        if source_id not in self.layers:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update mute state in layer metadata
        self.layer_metadata[source_id]['muted'] = muted
    
    def set_room_model(self, room_dimensions: Tuple[float, float, float],
                      reflection_coefficients: Dict[str, float],
                      rt60: float) -> None:
        """
        Set a room model for reflections and reverberation.
        
        Args:
            room_dimensions: (width, height, length) in meters
            reflection_coefficients: Coefficients for each surface
            rt60: Reverberation time in seconds
        """
        # Calculate room volume
        width, height, length = room_dimensions
        volume = width * height * length
        
        # Store room model
        self.room = {
            'dimensions': room_dimensions,
            'reflection_coefficients': reflection_coefficients,
            'rt60': rt60,
            'volume': volume
        }
    
    def set_binaural_renderer(self, hrtf_database: Union[str, Dict],
                             interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL) -> None:
        """
        Set the binaural renderer configuration.
        
        Args:
            hrtf_database: Path to HRTF database or dictionary with HRTF data
            interpolation_method: HRTF interpolation method
        """
        if isinstance(hrtf_database, str):
            self.hrtf_database = load_hrtf_database(hrtf_database)
        else:
            self.hrtf_database = hrtf_database
        
        self.binaural_renderer = {
            'interpolation_method': interpolation_method,
            'nearfield_compensation': True,
            'crossfade_time': 0.1
        }
    
    def process(self) -> np.ndarray:
        """
        Process all sources and layers to create the final ambisonic signals.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, n_samples)
        """
        # Determine the maximum number of samples across all layers
        max_samples = 0
        for layer_id, ambi_signals in self.layers.items():
            if not self.layer_metadata[layer_id]['muted']:
                max_samples = max(max_samples, ambi_signals.shape[1])
        
        if max_samples == 0:
            return np.zeros((self.n_channels, 0))
        
        # Initialize output signals
        output_signals = np.zeros((self.n_channels, max_samples))
        
        # Mix all active layers
        for layer_id, ambi_signals in self.layers.items():
            # Skip muted layers
            if self.layer_metadata[layer_id]['muted']:
                continue
            
            # Apply gain
            gain = self.layer_metadata[layer_id]['current_gain']
            scaled_signals = ambi_signals * gain
            
            # Mix into output
            n_samples = scaled_signals.shape[1]
            output_signals[:, :n_samples] += scaled_signals
        
        # Apply room model if available
        if self.room is not None:
            # Process each source for early reflections
            reflections = np.zeros_like(output_signals)
            
            for source_id, source_data in self.sources.items():
                # Skip muted sources
                if source_id not in self.layers or self.layer_metadata[source_id]['muted']:
                    continue
                
                # Get source position
                position = source_data['position']
                
                # Convert to Cartesian
                cart_pos = convert_to_cartesian(position)
                
                # Apply early reflections
                source_reflections = apply_early_reflections(
                    self.layers[source_id],
                    cart_pos,
                    self.room['dimensions'],
                    self.room['reflection_coefficients'],
                    self.sample_rate
                )
                
                # Apply gain
                gain = self.layer_metadata[source_id]['current_gain']
                reflections += source_reflections * gain
            
            # Mix in reflections
            output_signals = output_signals + reflections
            
            # Apply diffuse reverberation
            output_signals = apply_diffuse_reverberation(
                output_signals,
                self.room['rt60'],
                self.sample_rate,
                self.room['volume']
            )
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_signals))
        if max_val > 0.99:
            output_signals = output_signals * 0.99 / max_val
        
        return output_signals
    
    def rotate(self, ambi_signals: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        Rotate the ambisonic sound field.
        
        Args:
            ambi_signals: Ambisonic signals to rotate
            yaw: Rotation around vertical axis in radians
            pitch: Rotation around side axis in radians
            roll: Rotation around front axis in radians
            
        Returns:
            Rotated ambisonic signals
        """
        return rotate_ambisonics(ambi_signals, yaw, pitch, roll)
    
    def binauralize(self, ambi_signals: np.ndarray) -> np.ndarray:
        """
        Convert ambisonic signals to binaural stereo.
        
        Args:
            ambi_signals: Ambisonic signals to binauralize
            
        Returns:
            Binaural stereo signals, shape (2, n_samples)
        """
        if self.hrtf_database is None:
            raise ValueError("No HRTF database available for binauralization")
        
        return binauralize_ambisonics(ambi_signals, self.hrtf_database)
    
    def save_to_file(self, filename: str, bit_depth: int = 32) -> None:
        """
        Save the processed audio to a SHAC file.
        
        Args:
            filename: Output filename
            bit_depth: Bit depth (16 or 32)
        """
        # Process all sources and layers
        ambi_signals = self.process()
        
        # Create a SHAC file writer
        writer = SHACFileWriter(self.order, self.sample_rate, self.normalization)
        
        # Add a single layer with the processed audio
        writer.add_layer('main', ambi_signals, {'type': 'mixed'})
        
        # Add individual layers if available
        for layer_id, layer_signals in self.layers.items():
            if not self.layer_metadata[layer_id]['muted']:
                writer.add_layer(layer_id, layer_signals, self.layer_metadata[layer_id])
        
        # Write the file
        writer.write_file(filename, bit_depth)
    
    def load_from_file(self, filename: str) -> None:
        """
        Load audio from a SHAC file.
        
        Args:
            filename: Input filename
        """
        # Create a SHAC file reader
        reader = SHACFileReader(filename)
        
        # Get file info
        file_info = reader.get_file_info()
        
        # Update codec parameters
        self.order = file_info['order']
        self.sample_rate = file_info['sample_rate']
        self.n_channels = file_info['n_channels']
        
        # Clear existing layers
        self.layers = {}
        self.layer_metadata = {}
        
        # Load each layer
        for layer_name in reader.get_layer_names():
            layer_audio = reader.read_layer(layer_name)
            if layer_audio is not None:
                self.layers[layer_name] = layer_audio
                self.layer_metadata[layer_name] = {
                    'type': 'loaded',
                    'original_gain': 1.0,
                    'current_gain': 1.0,
                    'muted': False
                }


# =====================================================================================
# Real-time Processing Classes
# =====================================================================================

class SHACStreamProcessor:
    """
    Real-time streaming processor for SHAC audio.
    
    This class handles real-time processing of spatial audio streams
    with minimal latency and efficient CPU usage.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, buffer_size: int = 1024,
                 max_sources: int = 32):
        """
        Initialize the SHAC stream processor.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            buffer_size: Processing buffer size in samples
            max_sources: Maximum number of simultaneous sources
        """
        self.order = order
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        self.n_channels = (order + 1) ** 2
        
        # Create the main codec
        self.codec = SHACCodec(order, sample_rate)
        
        # Initialize streaming state
        self.sources = {}
        self.source_buffers = {}
        self.output_buffer = np.zeros((self.n_channels, buffer_size))
        
        # Initialize processing thread
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
    
    def start(self) -> None:
        """Start the real-time processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self) -> None:
        """Stop the real-time processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def add_source(self, source_id: str, position: Tuple[float, float, float],
                  attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a streaming source.
        
        Args:
            source_id: Unique identifier for the source
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if len(self.sources) >= self.max_sources:
            raise ValueError(f"Maximum number of sources ({self.max_sources}) reached")
        
        self.sources[source_id] = {
            'position': position,
            'attributes': attributes or SourceAttributes(position),
            'gain': 1.0,
            'muted': False
        }
        
        # Initialize source buffer
        self.source_buffers[source_id] = np.zeros(self.buffer_size)
    
    def remove_source(self, source_id: str) -> None:
        """
        Remove a streaming source.
        
        Args:
            source_id: Identifier of the source to remove
        """
        if source_id in self.sources:
            del self.sources[source_id]
        
        if source_id in self.source_buffers:
            del self.source_buffers[source_id]
    
    def update_source(self, source_id: str, audio_chunk: np.ndarray) -> None:
        """
        Update a source with new audio data.
        
        Args:
            source_id: Identifier of the source to update
            audio_chunk: New audio data chunk
        """
        if source_id not in self.sources:
            return
        
        # Copy audio data to source buffer
        n_samples = min(len(audio_chunk), self.buffer_size)
        self.source_buffers[source_id][:n_samples] = audio_chunk[:n_samples]
        
        # If audio chunk is smaller than buffer, zero-pad
        if n_samples < self.buffer_size:
            self.source_buffers[source_id][n_samples:] = 0.0
    
    def update_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['position'] = position
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['gain'] = gain
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source.
        
        Args:
            source_id: Identifier of the source to update
            muted: True to mute, False to unmute
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['muted'] = muted
    
    def set_listener_rotation(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Set the listener's head rotation.
        
        Args:
            yaw: Rotation around vertical axis in radians
            pitch: Rotation around side axis in radians
            roll: Rotation around front axis in radians
        """
        # Store for next processing cycle
        self.codec.listener_rotation = (yaw, pitch, roll)
    
    def process_block(self) -> np.ndarray:
        """
        Process a block of audio.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, buffer_size)
        """
        # Initialize output buffer
        output_buffer = np.zeros((self.n_channels, self.buffer_size))
        
        # Process each source
        for source_id, source_info in self.sources.items():
            # Skip muted sources
            if source_info['muted']:
                continue
            
            # Get source parameters
            position = source_info['position']
            attributes = source_info['attributes']
            gain = source_info['gain']
            
            # Get audio data
            audio = self.source_buffers[source_id]
            
            # Encode to ambisonics
            ambi_source = encode_mono_source(audio, position, self.order, AmbisonicNormalization.SN3D)
            
            # Apply source attributes if needed
            if attributes and (attributes.directivity > 0 or attributes.width > 0):
                source_directivity = {
                    'pattern': 'cardioid',
                    'order': attributes.directivity,
                    'axis': attributes.directivity_axis,
                    'frequency_dependent': True
                }
                
                ambi_source = apply_frequency_dependent_effects(ambi_source, self.sample_rate,
                                                              position[2], attributes.air_absorption,
                                                              source_directivity)
            
            # Apply gain
            ambi_source *= gain
            
            # Mix into output buffer
            output_buffer += ambi_source
        
        # Apply room effects if configured
        if hasattr(self.codec, 'room') and self.codec.room is not None:
            # Add simple diffuse reverberation (simplified for real-time processing)
            # In a full implementation, this would use convolution with pre-computed IRs
            pass
        
        # Apply rotation if needed
        if hasattr(self.codec, 'listener_rotation'):
            yaw, pitch, roll = self.codec.listener_rotation
            output_buffer = rotate_ambisonics(output_buffer, yaw, pitch, roll)
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0.99:
            output_buffer = output_buffer * 0.99 / max_val
        
        return output_buffer
    
    def get_binaural_output(self) -> np.ndarray:
        """
        Get the current block as binaural stereo.
        
        Returns:
            Binaural stereo signals, shape (2, buffer_size)
        """
        # Process the current block
        ambi_block = self.process_block()
        
        # Convert to binaural
        return self.codec.binauralize(ambi_block)
    
    def _processing_loop(self) -> None:
        """Main processing loop for the streaming thread."""
        while self.running:
            # Process the current block
            try:
                # Process the block
                ambi_block = self.process_block()
                
                # Binauralize if needed
                if hasattr(self.codec, 'binaural_renderer') and self.codec.binaural_renderer is not None:
                    binaural_block = self.codec.binauralize(ambi_block)
                    
                    # Add to output queue
                    self.output_queue.put(binaural_block)
                else:
                    # Add ambisonic block to output queue
                    self.output_queue.put(ambi_block)
                
                # Sleep for a bit to avoid burning CPU
                # In a real-time audio system, this would be synchronized with audio callbacks
                time.sleep(self.buffer_size / self.sample_rate / 2)
            
            except Exception as e:
                warnings.warn(f"Error in processing loop: {e}")
                time.sleep(0.1)


# =====================================================================================
# Example usage and demonstration
# =====================================================================================

def create_example_sound_scene():
    """
    Create and process an example 3D sound scene.
    
    This example demonstrates the core functionality of the SHAC codec.
    """
    print("Creating example 3D sound scene...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Create synthetic audio signals
    duration = 5.0  # seconds
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Piano sound (sine wave with harmonics and decay)
    piano_freq = 440.0  # A4
    piano_audio = 0.5 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
    piano_audio += 0.25 * np.sin(2 * np.pi * 2 * piano_freq * t) * np.exp(-t/1.5)
    piano_audio += 0.125 * np.sin(2 * np.pi * 3 * piano_freq * t) * np.exp(-t)
    
    # Drum sound (impulses with decay)
    drum_audio = np.zeros_like(t)
    for i in range(0, len(t), 12000):  # Four beats
        if i + 2000 < len(drum_audio):
            drum_audio[i:i+2000] = 0.8 * np.exp(-np.linspace(0, 10, 2000))
    
    # Ambient sound (filtered noise)
    np.random.seed(42)  # For reproducibility
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 0.1)
    ambient_audio = signal.filtfilt(b, a, noise) * 0.2
    
    # Add sources to the codec
    # Position format: (azimuth, elevation, distance) in radians and meters
    
    # Piano in front left
    piano_position = (-np.pi/4, 0.0, 3.0)
    piano_attributes = SourceAttributes(
        position=piano_position,
        directivity=0.7,
        directivity_axis=(0.0, 0.0, 1.0),
        width=0.2
    )
    codec.add_mono_source("piano", piano_audio, piano_position, piano_attributes)
    
    # Drum in front right
    drum_position = (np.pi/4, -0.1, 2.5)
    drum_attributes = SourceAttributes(
        position=drum_position,
        directivity=0.3,
        width=0.4
    )
    codec.add_mono_source("drum", drum_audio, drum_position, drum_attributes)
    
    # Ambient sound above
    ambient_position = (0.0, np.pi/3, 5.0)
    ambient_attributes = SourceAttributes(
        position=ambient_position,
        directivity=0.0,
        width=1.0
    )
    codec.add_mono_source("ambient", ambient_audio, ambient_position, ambient_attributes)
    
    # Set up a room model
    room_dimensions = (10.0, 3.0, 8.0)  # width, height, length in meters
    reflection_coefficients = {
        'left': 0.7,
        'right': 0.7,
        'floor': 0.4,
        'ceiling': 0.8,
        'front': 0.6,
        'back': 0.6
    }
    rt60 = 1.2  # seconds
    codec.set_room_model(room_dimensions, reflection_coefficients, rt60)
    
    # Process the scene
    print("Processing audio...")
    ambi_signals = codec.process()
    
    # Apply head rotation (as if user is looking to the left)
    print("Applying head rotation...")
    yaw = np.pi/3  # 60 degrees to the left
    rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
    
    # Convert to binaural
    print("Converting to binaural...")
    binaural_output = codec.binauralize(rotated_ambi)
    
    # Save outputs
    print("Saving outputs...")
    try:
        import soundfile as sf
        sf.write("shac_example_binaural.wav", binaural_output.T, sample_rate)
        print("Saved binaural output to shac_example_binaural.wav")
    except ImportError:
        print("Could not save audio files: soundfile module not available")
    
    # Save to SHAC file
    print("Saving to SHAC file...")
    codec.save_to_file("example_scene.shac")
    print("Saved SHAC file to example_scene.shac")
    
    print("Done!")
    return codec


def demonstrate_interactive_navigation():
    """
    Demonstrate interactive navigation through a 3D sound scene.
    
    This example creates a sequence of binaural renders while
    navigating through a sound scene.
    """
    print("Demonstrating interactive navigation...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Load scene from file if it exists, otherwise create a new one
    if os.path.exists("example_scene.shac"):
        print("Loading scene from file...")
        codec.load_from_file("example_scene.shac")
    else:
        print("Creating new scene...")
        create_example_sound_scene()
    
    # Define a navigation path
    yaw_angles = np.linspace(0, 2*np.pi, 8)  # Full 360° rotation in 8 steps
    
    # Process each step in the path
    for i, yaw in enumerate(yaw_angles):
        print(f"Step {i+1}/{len(yaw_angles)}: Yaw = {yaw:.2f} radians")
        
        # Process the scene
        ambi_signals = codec.process()
        
        # Apply rotation for this step
        rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
        
        # Convert to binaural
        binaural_output = codec.binauralize(rotated_ambi)
        
        # Save this step
        try:
            import soundfile as sf
            sf.write(f"navigation_step_{i+1}.wav", binaural_output.T, codec.sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
    
    print("Navigation demonstration complete!")


def demonstrate_streaming_processor():
    """
    Demonstrate the real-time streaming processor.
    
    This example shows how to use the SHAC stream processor for
    real-time audio processing.
    """
    print("Demonstrating streaming processor...")
    
    # Create a streaming processor
    processor = SHACStreamProcessor(order=3, sample_rate=48000, buffer_size=1024)
    
    # Create synthetic audio signals (single cycle of a sine wave)
    sample_rate = 48000
    buffer_size = 1024
    
    # Create three sources with different frequencies
    source1_freq = 440.0  # A4
    source1_audio = 0.5 * np.sin(2 * np.pi * source1_freq * np.arange(buffer_size) / sample_rate)
    
    source2_freq = 261.63  # C4
    source2_audio = 0.5 * np.sin(2 * np.pi * source2_freq * np.arange(buffer_size) / sample_rate)
    
    source3_freq = 329.63  # E4
    source3_audio = 0.5 * np.sin(2 * np.pi * source3_freq * np.arange(buffer_size) / sample_rate)
    
    # Add sources to the processor
    processor.add_source("source1", (-np.pi/4, 0.0, 2.0))
    processor.add_source("source2", (np.pi/4, 0.0, 2.0))
    processor.add_source("source3", (0.0, np.pi/4, 3.0))
    
    # Start the processor
    processor.start()
    
    # Simulate real-time processing for a few blocks
    for i in range(10):
        print(f"Processing block {i+1}/10")
        
        # Update sources with new audio data
        processor.update_source("source1", source1_audio)
        processor.update_source("source2", source2_audio)
        processor.update_source("source3", source3_audio)
        
        # Set listener rotation (changing over time)
        yaw = i * np.pi / 5  # Rotate gradually
        processor.set_listener_rotation(yaw, 0.0, 0.0)
        
        # Get binaural output
        binaural_output = processor.get_binaural_output()
        
        # Save this block
        try:
            import soundfile as sf
            sf.write(f"streaming_block_{i+1}.wav", binaural_output.T, sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
        
        # In a real application, this would feed audio to the sound card
        # For this demo, we sleep to simulate real-time processing
        time.sleep(buffer_size / sample_rate)
    
    # Stop the processor
    processor.stop()
    
    print("Streaming demonstration complete!")


def main():
    """Main function to demonstrate the SHAC codec."""
    print("SHAC Codec Demonstration")
    print("=======================")
    
    # Create and process an example sound scene
    create_example_sound_scene()
    
    # Demonstrate interactive navigation
    demonstrate_interactive_navigation()
    
    # Demonstrate streaming processor
    demonstrate_streaming_processor()


if __name__ == "__main__":
    main()
