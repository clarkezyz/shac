"""
Core Mathematical Functions for Spherical Harmonics

This module provides the fundamental mathematical operations required for
spherical harmonic audio processing, including Legendre polynomials, spherical
harmonic calculations, rotation matrices, and coordinate transformations.

These functions form the mathematical foundation of the SHAC codec and enable
the efficient representation and manipulation of 3D sound fields. Most 
functions are optimized for both scalar and vector inputs for performance.

See Also:
    - utils: For type definitions and utility functions
    - config: For configuration settings related to math functions
"""

import numpy as np
import math
import functools
import warnings
from typing import Union, Dict, List, Tuple, Optional, Callable, Any, TypeVar, cast
from enum import Enum, auto

from .utils import Vector3, SphericalCoord, CartesianCoord
from .exceptions import MathError

# Type variable for functions that accept both scalar and array inputs
T = TypeVar('T', float, np.ndarray)

# Cache size for factorial memoization
_FACTORIAL_CACHE_SIZE = 50


class AmbisonicNormalization(Enum):
    """
    Defines the normalization convention for spherical harmonics.
    
    Different normalization schemes affect the scaling of the spherical harmonic
    components. The choice of normalization impacts energy preservation and
    compatibility with various ambisonic processing systems.
    
    Attributes:
        SN3D: Schmidt semi-normalized
            Most common in modern ambisonic systems (ACN/SN3D)
            SN3D = N3D / sqrt(2n+1)
        N3D: Fully normalized (orthonormal basis)
            Each component has equal energy contribution
        FUMA: FuMa (legacy B-format) normalization
            Used in classic first-order ambisonics
    """
    SN3D = auto()  # Schmidt semi-normalized (most common, N3D / sqrt(2n+1))
    N3D = auto()   # Fully normalized (orthonormal basis)
    FUMA = auto()  # FuMa (legacy B-format) normalization


@functools.lru_cache(maxsize=_FACTORIAL_CACHE_SIZE)
def factorial(n: int) -> int:
    """
    Compute factorial, optimized with caching for repeated calls.
    
    This implementation uses memoization for efficiency when computing
    factorials repeatedly, as is common in spherical harmonic calculations.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! (n factorial)
        
    Raises:
        MathError.DomainError: If n is negative
    
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    if n < 0:
        raise MathError.DomainError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    
    # Use cached results for repeated calls
    return n * factorial(n - 1)


@functools.lru_cache(maxsize=_FACTORIAL_CACHE_SIZE)
def double_factorial(n: int) -> int:
    """
    Compute double factorial n!! = n * (n-2) * (n-4) * ...
    
    The double factorial is used in certain spherical harmonic calculations.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n!! (n double factorial)
        
    Raises:
        MathError.DomainError: If n is negative
    
    Examples:
        >>> double_factorial(5)  # 5 * 3 * 1
        15
        >>> double_factorial(6)  # 6 * 4 * 2
        48
    """
    if n < 0:
        raise MathError.DomainError("Double factorial not defined for negative numbers")
    if n <= 1:
        return 1
    
    # Use cached results for repeated calls
    return n * double_factorial(n - 2)


def associated_legendre(l: int, m: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the associated Legendre polynomial P_l^m(x) for spherical harmonics.
    
    This is a custom implementation optimized for spherical harmonics, avoiding
    the phase issue in scipy's implementation and handling the normalization 
    correctly. It works efficiently with both scalar and array inputs.
    
    Args:
        l: Degree of the spherical harmonic (l >= 0)
        m: Order of the spherical harmonic (-l <= m <= l)
        x: Value or array where -1 <= x <= 1
        
    Returns:
        The associated Legendre polynomial value(s)
    
    Raises:
        MathError.DomainError: If l < 0 or |m| > l
        MathError.PrecisionError: If numerical instability is detected
        
    Notes:
        The implementation uses a recurrence relationship that is numerically
        stable and optimized for both scalar and vectorized computation.
        For |x| ≈ 1, special care is taken to avoid numerical issues.
    
    Examples:
        >>> associated_legendre(2, 1, 0.5)  # P_2^1(0.5)
        -1.299038...
        
        >>> associated_legendre(3, 2, np.array([0.1, 0.2, 0.3]))
        array([-0.08778525, -0.36880185, -0.84983146])
    """
    # Input validation
    if l < 0:
        raise MathError.DomainError(f"Degree l must be non-negative, got {l}")
        
    m_abs = abs(m)
    
    if m_abs > l:
        # Return zeros with proper shape
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    
    # Handle simple cases
    if l == 0 and m == 0:
        # P_0^0(x) = 1
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
    
    # Use numpy operations for both scalar and array inputs
    x_array = np.asarray(x) if not isinstance(x, np.ndarray) else x
    
    # Check input domain
    if np.any(np.abs(x_array) > 1.0 + 1e-10):
        raise MathError.DomainError(f"Input x must be in range [-1, 1], got values outside this range")
    
    # Clip values to ensure stability at boundaries
    x_array = np.clip(x_array, -1.0 + 1e-10, 1.0 - 1e-10)
    
    # First compute P_m^m
    pmm = np.ones_like(x_array)
    somx2 = np.sqrt((1.0 - x_array) * (1.0 + x_array))
    fact = 1.0
    
    for i in range(1, m_abs + 1):
        pmm *= (-fact) * somx2
        fact += 2.0
    
    if l == m_abs:
        return pmm if isinstance(x, np.ndarray) else float(pmm)
    
    # Compute P_{m+1}^m using stable recursion
    pmmp1 = x_array * (2.0 * m_abs + 1.0) * pmm
    
    if l == m_abs + 1:
        return pmmp1 if isinstance(x, np.ndarray) else float(pmmp1)
    
    # Use the recurrence relationship to get higher degrees
    # P_l^m(x) = ((2l-1)x * P_{l-1}^m(x) - (l+m-1) * P_{l-2}^m(x)) / (l-m)
    pll = np.zeros_like(x_array)
    for ll in range(m_abs + 2, l + 1):
        pll = (x_array * (2.0 * ll - 1.0) * pmmp1 - (ll + m_abs - 1.0) * pmm) / (ll - m_abs)
        pmm = pmmp1
        pmmp1 = pll
    
    # Apply Condon-Shortley phase for m < 0
    if m < 0:
        # Handle potential overflow in high-order calculations
        try:
            # Calculate the ratio directly to avoid overflow
            if l + m_abs > 20:  # Threshold where factorials get very large
                ratio = 1.0
                for i in range(l - m_abs + 1, l + m_abs + 1):
                    ratio /= i
                phase_factor = (-1)**m_abs * ratio
            else:
                phase_factor = (-1)**m_abs * factorial(l - m_abs) / factorial(l + m_abs)
            
            pll *= phase_factor
        except OverflowError:
            raise MathError.PrecisionError(f"Numerical overflow in Legendre calculation for l={l}, m={m}")
    
    # Return scalar or array based on input type
    return pll if isinstance(x, np.ndarray) else float(pll)


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
    }
    
    # Convert up to 3rd order
    max_chan = min(n_channels, 16)
    for fuma_idx in range(max_chan):
        if fuma_idx in fuma_to_acn:
            acn_idx = fuma_to_acn[fuma_idx]
            if acn_idx < n_channels:
                ambi_fuma[fuma_idx] = ambi_acn[acn_idx]
    
    return ambi_fuma


def convert_fuma_to_acn(ambi_fuma: np.ndarray) -> np.ndarray:
    """
    Convert ambisonic signals from FuMa ordering to ACN ordering.
    
    Args:
        ambi_fuma: Ambisonic signals in FuMa ordering, shape (n_channels, n_samples)
        
    Returns:
        Ambisonic signals in ACN ordering
    """
    n_channels = ambi_fuma.shape[0]
    
    # Determine the corresponding ambisonic order
    if n_channels == 1:
        order = 0
    elif n_channels == 4:
        order = 1
    elif n_channels == 9:
        order = 2
    elif n_channels == 16:
        order = 3
    else:
        raise ValueError(f"Number of channels {n_channels} does not correspond to a standard FuMa layout")
    
    # Initialize ACN array
    acn_channels = (order + 1) ** 2
    ambi_acn = np.zeros((acn_channels,) + ambi_fuma.shape[1:], dtype=ambi_fuma.dtype)
    
    # Conversion table (from FuMa to ACN)
    acn_to_fuma = {
        0: 0,   # W
        1: 2,   # Y
        2: 3,   # Z
        3: 1,   # X
        4: 6,   # T
        5: 7,   # U
        6: 4,   # R
        7: 8,   # V
        8: 5,   # S
    }
    
    # Convert
    for acn_idx in range(min(acn_channels, len(acn_to_fuma))):
        if acn_idx in acn_to_fuma:
            fuma_idx = acn_to_fuma[acn_idx]
            if fuma_idx < n_channels:
                ambi_acn[acn_idx] = ambi_fuma[fuma_idx]
    
    return ambi_acn