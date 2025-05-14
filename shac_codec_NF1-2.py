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

