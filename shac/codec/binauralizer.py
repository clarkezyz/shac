"""
Binaural Rendering Module

This module contains functions for binaural rendering of ambisonic signals
using Head-Related Transfer Functions (HRTFs).

It supports various HRTF formats including SOFA (Spatially Oriented Format
for Acoustics) and provides different interpolation methods for accurate
3D audio spatialization.
"""

import numpy as np
import math
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from .utils import HRTFInterpolationMethod, SphericalCoord, Vector3, HRTFData
from . import sofa_support

# Set up logging
logger = logging.getLogger(__name__)

# Flag for whether we've warned about synthetic HRTF use
_synthetic_hrtf_warning_shown = False


def generate_synthetic_hrtf_database(sample_rate: int = 48000) -> Dict:
    """
    Generate a simple synthetic HRTF database when no measured data is available.
    This provides very basic 3D cues but is not as realistic as measured HRTFs.
    
    Args:
        sample_rate: The sample rate in Hz
        
    Returns:
        A synthetic HRTF database compatible with the binauralizer
    """
    global _synthetic_hrtf_warning_shown
    
    if not _synthetic_hrtf_warning_shown:
        logger.warning("Using synthetic HRTF database. For better spatial audio quality, "
                     "install either 'python-sofa', 'netCDF4', or 'h5netcdf' and use a real SOFA file.")
        _synthetic_hrtf_warning_shown = True
    
    # Create positions in a grid
    azimuths = np.linspace(0, 2*np.pi, 12, endpoint=False)
    elevations = np.linspace(-np.pi/2, np.pi/2, 6)
    
    # Create position array
    positions = []
    for azimuth in azimuths:
        for elevation in elevations:
            positions.append((azimuth, elevation, 1.0))  # Fixed distance of 1m
    
    positions = np.array(positions)
    
    # Create HRTF dictionary
    hrtf_length = 128
    hrtf_dict = {}
    
    for i, (azimuth, elevation, distance) in enumerate(positions):
        # Create a simple head model for ITD (interaural time difference)
        # ITD = (r/c) * (sin(azimuth) + elevation/π)
        head_radius = 0.0875  # meters
        speed_of_sound = 343.0  # m/s
        
        # Calculate ITD in samples
        itd_max = head_radius / speed_of_sound * sample_rate
        itd = itd_max * np.sin(azimuth) * np.cos(elevation)
        itd_samples = int(abs(itd))
        
        # Create left and right HRTFs with interaural differences
        left_hrtf = np.zeros(hrtf_length)
        right_hrtf = np.zeros(hrtf_length)
        
        # Center impulse
        center = hrtf_length // 4
        
        # Create basic HRTF with ITD and ILD
        if itd >= 0:  # Sound is more to the right
            left_hrtf[center + itd_samples] = 0.8
            right_hrtf[center] = 1.0
        else:  # Sound is more to the left
            left_hrtf[center] = 1.0
            right_hrtf[center + itd_samples] = 0.8
        
        # Apply basic elevation cues (very simplified)
        elev_factor = 0.5 + 0.5 * np.sin(elevation)
        freq = 2 + elev_factor * 10  # Higher frequencies for higher elevations
        
        # Apply minimal frequency shaping
        t = np.arange(hrtf_length)
        env = np.exp(-0.5 * t / hrtf_length)
        left_hrtf = left_hrtf * env
        right_hrtf = right_hrtf * env
        
        # Store in dictionary
        hrtf_dict[(azimuth, elevation, distance)] = {
            'left': left_hrtf,
            'right': right_hrtf
        }
    
    # Create the full database structure
    hrtf_database = {
        'sample_rate': sample_rate,
        'positions': positions,
        'hrtf_dict': hrtf_dict,
        'max_order': 3,  # Reasonable default
        'sh_hrtfs': None,  # Will be computed on demand
        'convention': "Synthetic",
        'version': "1.0"
    }
    
    return hrtf_database


def binauralize_ambisonics(ambi_signals: np.ndarray, hrtf_database: Union[str, Dict], 
                          normalize: bool = True, 
                          interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL) -> np.ndarray:
    """
    Convert ambisonic signals to binaural stereo using HRTF convolution.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        hrtf_database: Path to HRTF database (SOFA file) or dictionary with HRTF data
        normalize: Whether to normalize the output
        interpolation_method: Method to use for HRTF interpolation
        
    Returns:
        Binaural stereo signals, shape (2, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    order = math.floor(math.sqrt(n_channels)) - 1
    
    # Load HRTF data if needed
    if isinstance(hrtf_database, str):
        try:
            # First try to load as SOFA file
            hrtf_data = sofa_support.load_sofa_file(hrtf_database)
        except Exception as e:
            logger.warning(f"Failed to load SOFA file: {str(e)}. Using legacy loader.")
            hrtf_data = load_hrtf_database(hrtf_database)
    else:
        # Assume it's already loaded as a dictionary
        hrtf_data = hrtf_database
    
    # Get the HRTF filters in SH domain
    sh_hrtfs = hrtf_data['sh_hrtfs']
    if sh_hrtfs.shape[1] < n_channels:
        logger.warning(f"HRTF database only supports up to order {math.floor(math.sqrt(sh_hrtfs.shape[1])) - 1}, " +
                      f"but got signals of order {order}. Truncating to available order.")
        # Truncate the ambisonic signals to the available order
        n_channels = sh_hrtfs.shape[1]
        ambi_signals = ambi_signals[:n_channels]
    
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
    
    This function first tries to load the path as a SOFA file. If that fails,
    it falls back to the legacy synthetic HRTF generation. If the file doesn't
    exist and has a .sofa extension, it attempts to download a default HRTF database.
    
    Args:
        hrtf_path: Path to the HRTF database or SOFA file
        
    Returns:
        Dictionary containing HRTF data in suitable format for binauralization
    """
    # Check if path exists
    if not os.path.exists(hrtf_path):
        # If path doesn't exist but has .sofa extension, try to download a default
        if hrtf_path.lower().endswith('.sofa'):
            logger.info(f"HRTF file not found at {hrtf_path}, attempting to download default database")
            try:
                # Try to download default HRTF database
                target_dir = os.path.dirname(hrtf_path)
                if not target_dir:
                    target_dir = None  # Use default directory
                sofa_file = sofa_support.download_default_hrtf_database(target_dir)
                return sofa_support.load_sofa_file(sofa_file)
            except Exception as e:
                logger.warning(f"Failed to download HRTF database: {str(e)}. Using synthetic HRTF.")
                return sofa_support.get_default_hrtf_database()
        else:
            logger.warning(f"HRTF file not found at {hrtf_path}. Using synthetic HRTF.")
            return generate_synthetic_hrtf_database()
    
    # Try to load as SOFA file
    try:
        return sofa_support.load_sofa_file(hrtf_path)
    except Exception as e:
        logger.warning(f"Failed to load {hrtf_path} as SOFA file: {str(e)}. Using synthetic HRTF.")
        return generate_synthetic_hrtf_database()


def _create_synthetic_hrtf_database() -> Dict:
    """
    Create a simplified synthetic HRTF database for demonstration purposes.
    
    Returns:
        Dictionary containing synthetic HRTF data
    """
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


def binauralize_mono_source(mono_signal: np.ndarray, position: SphericalCoord, 
                           hrtf_database: Union[str, Dict],
                           interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL) -> np.ndarray:
    """
    Render a mono sound source to binaural stereo at a specific position.
    
    This is a direct binaural rendering that bypasses the ambisonic encoding.
    It's more efficient for rendering a single source and can provide better
    quality for near-field sources.
    
    Args:
        mono_signal: Mono audio signal, shape (n_samples,)
        position: Source position as (azimuth, elevation, distance)
        hrtf_database: Path to HRTF database or dictionary with HRTF data
        interpolation_method: Method to use for HRTF interpolation
        
    Returns:
        Binaural stereo signal, shape (2, n_samples)
    """
    # Load HRTF data if needed
    if isinstance(hrtf_database, str):
        try:
            # First try to load as SOFA file
            hrtf_data = sofa_support.load_sofa_file(hrtf_database)
        except Exception as e:
            logger.warning(f"Failed to load SOFA file: {str(e)}. Using legacy loader.")
            hrtf_data = load_hrtf_database(hrtf_database)
    else:
        # Assume it's already loaded as a dictionary
        hrtf_data = hrtf_database
    
    # Get interpolated HRTF for the specific direction
    if 'hrtf_dict' in hrtf_data and 'positions' in hrtf_data:
        # Use SOFA-based interpolation
        interpolated_hrtf = sofa_support.interpolate_hrtf(position, hrtf_data, interpolation_method)
        left_hrtf = interpolated_hrtf['left']
        right_hrtf = interpolated_hrtf['right']
    else:
        # Fall back to basic synthetic HRTF
        logger.warning("HRTF database does not contain position-based HRTFs. Using simplified approach.")
        left_hrtf, right_hrtf = _create_synthetic_direction_hrtfs(position)
    
    # Apply distance attenuation if needed
    _, _, distance = position
    if distance > 1.0:
        # Simple 1/r distance attenuation
        gain = 1.0 / distance
        mono_signal = mono_signal * gain
    
    # Apply the HRTFs via convolution
    binaural = np.zeros((2, len(mono_signal) + len(left_hrtf) - 1))
    binaural[0] = np.convolve(mono_signal, left_hrtf)
    binaural[1] = np.convolve(mono_signal, right_hrtf)
    
    # Truncate to original length
    binaural = binaural[:, :len(mono_signal)]
    
    return binaural


def _create_synthetic_direction_hrtfs(position: SphericalCoord) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic HRTFs for a specific direction.
    
    This is a simplified approach that creates basic directional HRTFs
    when no measured data is available. It models ITD and ILD based on
    direction but lacks the complex spectral cues of real HRTFs.
    
    Args:
        position: Source position as (azimuth, elevation, distance)
        
    Returns:
        Tuple of (left HRTF, right HRTF) as numpy arrays
    """
    azimuth, elevation, _ = position
    hrtf_length = 256
    
    # Calculate ITD based on direction
    # Simplified spherical head model
    # ITD is approximately 0.65 ms (about 31 samples at 48 kHz) for sources at 90°
    itd_max_samples = 31
    itd_samples = int(itd_max_samples * math.sin(azimuth) * math.cos(elevation))
    
    left_delay = max(0, -itd_samples)
    right_delay = max(0, itd_samples)
    
    # Create basic impulse responses with delays
    left_hrtf = np.zeros(hrtf_length)
    right_hrtf = np.zeros(hrtf_length)
    
    # Create exponential decay after the delay
    left_hrtf[left_delay:] = np.exp(-np.arange(hrtf_length-left_delay)/20)
    right_hrtf[right_delay:] = np.exp(-np.arange(hrtf_length-right_delay)/20)
    
    # Apply ILD (Interaural Level Difference) based on direction
    # Sources to the left are louder in the left ear and vice versa
    ild_factor = 0.6  # Maximum level difference
    left_gain = 1.0 - ild_factor * math.sin(azimuth) * math.cos(elevation) / 2
    right_gain = 1.0 + ild_factor * math.sin(azimuth) * math.cos(elevation) / 2
    
    left_hrtf *= left_gain
    right_hrtf *= right_gain
    
    # Apply elevation effects
    # Higher elevations have more high-frequency content
    if elevation != 0:
        # Create a basic spectral tilt
        elevation_factor = math.sin(elevation)
        spectral_tilt = np.linspace(0, 1, hrtf_length)
        
        # Apply the tilt with different polarities based on whether sound is above or below
        left_hrtf *= (1 + elevation_factor * spectral_tilt * 0.5)
        right_hrtf *= (1 + elevation_factor * spectral_tilt * 0.5)
    
    # Normalize
    left_max = np.max(np.abs(left_hrtf))
    right_max = np.max(np.abs(right_hrtf))
    
    if left_max > 0:
        left_hrtf /= left_max
    if right_max > 0:
        right_hrtf /= right_max
    
    return left_hrtf, right_hrtf


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