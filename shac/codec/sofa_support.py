"""
SOFA (Spatially Oriented Format for Acoustics) Support Module

This module provides functionality for working with SOFA files,
which store Head-Related Transfer Functions (HRTFs) and other 
spatially-oriented acoustic data.

The SOFA format is standardized as AES69-2022 and provides a
consistent way to store and exchange HRTF data across different
platforms and applications.

References:
    https://www.sofaconventions.org/
    https://www.aes.org/publications/standards/search.cfm?docID=99
"""

import os
import numpy as np
import math
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from .utils import HRTFInterpolationMethod, SphericalCoord, Vector3, HRTFData

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_HRTF_LENGTH = 256
DEFAULT_MAX_ORDER = 3

# Try to import SOFA libraries with fallbacks
SOFA_AVAILABLE = False
NETCDF4_AVAILABLE = False
H5NETCDF_AVAILABLE = False

try:
    import sofa
    SOFA_AVAILABLE = True
except ImportError:
    logger.info("python-sofa not found, trying netCDF4...")
    try:
        from netCDF4 import Dataset
        NETCDF4_AVAILABLE = True
    except ImportError:
        logger.info("netCDF4 not found, trying h5netcdf...")
        try:
            import h5netcdf
            H5NETCDF_AVAILABLE = True
        except ImportError:
            logger.warning("No SOFA library found. Install 'python-sofa', 'netCDF4', or 'h5netcdf' for SOFA support.")


class SOFAConvention(Enum):
    """Supported SOFA conventions."""
    SIMPLE_HRTF = "SimpleFreeFieldHRIR"
    GENERAL_HRTF = "GeneralFIR"
    SIMPLE_HRTF_DEPRECATED = "SimpleFreeFieldHRTF"  # Older files might use this


class SOFACoordinateSystem(Enum):
    """Coordinate systems used in SOFA files."""
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"


def load_sofa_file(file_path: str) -> Dict[str, Any]:
    """
    Load a SOFA file and extract HRTF data.
    
    Args:
        file_path: Path to the SOFA file
        
    Returns:
        Dictionary containing the HRTF data and metadata
        
    Raises:
        ImportError: If no SOFA library is available
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid SOFA file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SOFA file not found: {file_path}")
    
    logger.info(f"Loading SOFA file: {file_path}")
    
    # Use available library to load the file
    if SOFA_AVAILABLE:
        return _load_with_python_sofa(file_path)
    elif NETCDF4_AVAILABLE:
        return _load_with_netcdf4(file_path)
    elif H5NETCDF_AVAILABLE:
        return _load_with_h5netcdf(file_path)
    else:
        raise ImportError("No SOFA library available. Install 'python-sofa', 'netCDF4', or 'h5netcdf'.")


def _load_with_python_sofa(file_path: str) -> Dict[str, Any]:
    """
    Load a SOFA file using the python-sofa library.
    
    Args:
        file_path: Path to the SOFA file
        
    Returns:
        Dictionary containing the HRTF data and metadata
    """
    try:
        # Open the SOFA file
        sofa_obj = sofa.Database.open(file_path)
        
        # Get basic metadata
        convention = sofa_obj.GLOBAL_SOFAConventions
        version = sofa_obj.GLOBAL_SOFAVersion
        sample_rate = float(sofa_obj.Data_SamplingRate.get_values()[0])
        
        # Get dimensions
        n_measurements = sofa_obj.API_M  # Number of measurements/positions
        n_receivers = sofa_obj.API_R     # Number of receivers (typically 2 for binaural)
        n_samples = sofa_obj.API_N       # Number of samples per measurement
        
        # Get impulse responses (IRs)
        ir_data = sofa_obj.Data_IR.get_values()  # Shape: [M, R, N]
        
        # Get source positions
        source_positions = sofa_obj.SourcePosition.get_values()
        source_pos_type = sofa_obj.SourcePosition_Type
        source_pos_units = sofa_obj.SourcePosition_Units
        
        # Convert positions to consistent format (spherical coordinates)
        spherical_positions = []
        if source_pos_type.lower() == 'cartesian':
            # Convert Cartesian to spherical
            for pos in source_positions:
                x, y, z = pos
                azimuth = math.atan2(y, x)
                elevation = math.atan2(z, math.sqrt(x*x + y*y))
                distance = math.sqrt(x*x + y*y + z*z)
                spherical_positions.append((azimuth, elevation, distance))
        else:
            # Assume already spherical but might need unit conversion
            for pos in source_positions:
                # Check units and convert if needed
                azimuth, elevation, distance = pos
                if 'degree' in source_pos_units:
                    # Convert degrees to radians
                    azimuth = math.radians(azimuth)
                    elevation = math.radians(elevation)
                
                spherical_positions.append((azimuth, elevation, distance))
        
        # Create a dictionary with all the extracted data
        hrtf_data = {
            'convention': convention,
            'version': version,
            'sample_rate': sample_rate,
            'n_measurements': n_measurements,
            'n_receivers': n_receivers,
            'n_samples': n_samples,
            'ir_data': ir_data,
            'source_positions': np.array(source_positions),
            'spherical_positions': np.array(spherical_positions)
        }
        
        # Get listener position and orientation if available
        try:
            hrtf_data['listener_position'] = sofa_obj.ListenerPosition.get_values()
            hrtf_data['listener_view'] = sofa_obj.ListenerView.get_values()
            hrtf_data['listener_up'] = sofa_obj.ListenerUp.get_values()
        except:
            logger.info("Listener orientation data not found in SOFA file.")
            
        # Close the file
        sofa_obj.close()
        
        # Convert to format expected by our binauralizer
        return _prepare_hrtf_data(hrtf_data)
    
    except Exception as e:
        logger.error(f"Error loading SOFA file with python-sofa: {str(e)}")
        raise ValueError(f"Could not load SOFA file: {str(e)}")


def _load_with_netcdf4(file_path: str) -> Dict[str, Any]:
    """
    Load a SOFA file using the netCDF4 library.
    
    Args:
        file_path: Path to the SOFA file
        
    Returns:
        Dictionary containing the HRTF data and metadata
    """
    try:
        # Open the SOFA file
        sofa_data = Dataset(file_path, 'r', format='NETCDF4')
        
        # Extract basic metadata
        convention = sofa_data.getncattr('SOFAConventions')
        version = sofa_data.getncattr('SOFAVersion')
        
        # Extract sampling rate
        sample_rate = float(sofa_data.variables['Data.SamplingRate'][:][0])
        
        # Extract IRs
        ir_data = sofa_data.variables['Data.IR'][:]
        n_measurements, n_receivers, n_samples = ir_data.shape
        
        # Extract source positions
        source_positions = sofa_data.variables['SourcePosition'][:]
        source_pos_type = sofa_data.variables['SourcePosition'].type
        source_pos_units = sofa_data.variables['SourcePosition'].units
        
        # Convert positions to consistent format (spherical coordinates)
        spherical_positions = []
        if source_pos_type.lower() == 'cartesian':
            # Convert Cartesian to spherical
            for pos in source_positions:
                x, y, z = pos
                azimuth = math.atan2(y, x)
                elevation = math.atan2(z, math.sqrt(x*x + y*y))
                distance = math.sqrt(x*x + y*y + z*z)
                spherical_positions.append((azimuth, elevation, distance))
        else:
            # Assume already spherical but might need unit conversion
            for pos in source_positions:
                # Check units and convert if needed
                azimuth, elevation, distance = pos
                if 'degree' in source_pos_units:
                    # Convert degrees to radians
                    azimuth = math.radians(azimuth)
                    elevation = math.radians(elevation)
                
                spherical_positions.append((azimuth, elevation, distance))
        
        # Create a dictionary with all the extracted data
        hrtf_data = {
            'convention': convention,
            'version': version,
            'sample_rate': sample_rate,
            'n_measurements': n_measurements,
            'n_receivers': n_receivers,
            'n_samples': n_samples,
            'ir_data': ir_data,
            'source_positions': np.array(source_positions),
            'spherical_positions': np.array(spherical_positions)
        }
        
        # Get listener position and orientation if available
        try:
            hrtf_data['listener_position'] = sofa_data.variables['ListenerPosition'][:]
            hrtf_data['listener_view'] = sofa_data.variables['ListenerView'][:]
            hrtf_data['listener_up'] = sofa_data.variables['ListenerUp'][:]
        except:
            logger.info("Listener orientation data not found in SOFA file.")
            
        # Close the file
        sofa_data.close()
        
        # Convert to format expected by our binauralizer
        return _prepare_hrtf_data(hrtf_data)
    
    except Exception as e:
        logger.error(f"Error loading SOFA file with netCDF4: {str(e)}")
        raise ValueError(f"Could not load SOFA file: {str(e)}")


def _load_with_h5netcdf(file_path: str) -> Dict[str, Any]:
    """
    Load a SOFA file using the h5netcdf library.
    
    Args:
        file_path: Path to the SOFA file
        
    Returns:
        Dictionary containing the HRTF data and metadata
    """
    try:
        # Open the SOFA file
        with h5netcdf.File(file_path, 'r') as sofa_data:
            # Extract basic metadata
            convention = sofa_data.attrs['SOFAConventions']
            version = sofa_data.attrs['SOFAVersion']
            
            # Extract sampling rate
            sample_rate = float(sofa_data['Data.SamplingRate'][:][0])
            
            # Extract IRs
            ir_data = sofa_data['Data.IR'][:]
            n_measurements, n_receivers, n_samples = ir_data.shape
            
            # Extract source positions
            source_positions = sofa_data['SourcePosition'][:]
            try:
                source_pos_type = sofa_data['SourcePosition'].attrs['Type']
                source_pos_units = sofa_data['SourcePosition'].attrs['Units']
            except:
                # Default to spherical if not specified
                source_pos_type = 'spherical'
                source_pos_units = 'degree, degree, meter'
            
            # Convert positions to consistent format (spherical coordinates)
            spherical_positions = []
            if source_pos_type.lower() == 'cartesian':
                # Convert Cartesian to spherical
                for pos in source_positions:
                    x, y, z = pos
                    azimuth = math.atan2(y, x)
                    elevation = math.atan2(z, math.sqrt(x*x + y*y))
                    distance = math.sqrt(x*x + y*y + z*z)
                    spherical_positions.append((azimuth, elevation, distance))
            else:
                # Assume already spherical but might need unit conversion
                for pos in source_positions:
                    # Check units and convert if needed
                    azimuth, elevation, distance = pos
                    if 'degree' in source_pos_units:
                        # Convert degrees to radians
                        azimuth = math.radians(azimuth)
                        elevation = math.radians(elevation)
                    
                    spherical_positions.append((azimuth, elevation, distance))
            
            # Create a dictionary with all the extracted data
            hrtf_data = {
                'convention': convention,
                'version': version,
                'sample_rate': sample_rate,
                'n_measurements': n_measurements,
                'n_receivers': n_receivers,
                'n_samples': n_samples,
                'ir_data': ir_data,
                'source_positions': np.array(source_positions),
                'spherical_positions': np.array(spherical_positions)
            }
            
            # Get listener position and orientation if available
            try:
                hrtf_data['listener_position'] = sofa_data['ListenerPosition'][:]
                hrtf_data['listener_view'] = sofa_data['ListenerView'][:]
                hrtf_data['listener_up'] = sofa_data['ListenerUp'][:]
            except:
                logger.info("Listener orientation data not found in SOFA file.")
        
        # Convert to format expected by our binauralizer
        return _prepare_hrtf_data(hrtf_data)
    
    except Exception as e:
        logger.error(f"Error loading SOFA file with h5netcdf: {str(e)}")
        raise ValueError(f"Could not load SOFA file: {str(e)}")


def _prepare_hrtf_data(raw_hrtf_data: Dict[str, Any]) -> HRTFData:
    """
    Convert raw HRTF data from SOFA file to format needed by binauralizer.
    
    Args:
        raw_hrtf_data: Dictionary with raw HRTF data from SOFA file
        
    Returns:
        Dictionary formatted for use with binauralize_ambisonics function
    """
    # Extract necessary data
    ir_data = raw_hrtf_data['ir_data']  # Shape: [M, R, N]
    positions = raw_hrtf_data['spherical_positions']  # Shape: [M, 3]
    sample_rate = raw_hrtf_data['sample_rate']
    
    # Create HRTF database structure
    # In our application, we need the HRTFs in spherical harmonic (SH) domain
    # To do this properly, we'd need to:
    # 1. Decompose the spatial distribution of HRTFs into SH coefficients
    # 2. Store the SH coefficients for each ear

    # For simplicity in this first implementation, we'll create a structure
    # with the original IRs and positions, and do the SH encoding when needed
    
    max_order = DEFAULT_MAX_ORDER
    n_sh_channels = (max_order + 1) ** 2
    
    # Organize data for easy lookup by position
    hrtf_dict = {}
    for i, pos in enumerate(positions):
        # Use position as key
        pos_key = tuple(pos)
        
        # Store IRs for left and right ears
        left_ir = ir_data[i, 0, :]
        right_ir = ir_data[i, 1, :]
        
        hrtf_dict[pos_key] = {
            'left': left_ir,
            'right': right_ir
        }
    
    # Create the output data structure
    hrtf_data = {
        'sample_rate': sample_rate,
        'max_order': max_order,
        'positions': positions,
        'hrtf_dict': hrtf_dict,
        'convention': raw_hrtf_data['convention'],
        'version': raw_hrtf_data['version'],
        'original_data': raw_hrtf_data
    }
    
    # For the binauralizer, we also need SH-domain HRTFs
    # This is a simplification - a proper implementation would properly encode the HRTFs
    # into the SH domain based on their spatial distribution
    hrtf_data['sh_hrtfs'] = _convert_to_sh_domain(hrtf_dict, positions, max_order)
    
    return hrtf_data


def _convert_to_sh_domain(hrtf_dict: Dict, positions: np.ndarray, max_order: int) -> np.ndarray:
    """
    Convert HRTFs from spatial domain to spherical harmonic domain.
    
    This is a simplified implementation that creates a basic SH domain
    representation of the HRTFs. A more advanced implementation would use
    proper spherical harmonic decomposition based on the available measurements.
    
    Args:
        hrtf_dict: Dictionary of HRTFs indexed by position
        positions: Array of positions in spherical coordinates
        max_order: Maximum spherical harmonic order
        
    Returns:
        SH-domain HRTFs as array with shape [2, (max_order+1)², hrtf_length]
    """
    # Count the number of HRTF measurements
    n_measurements = len(hrtf_dict)
    
    if n_measurements == 0:
        raise ValueError("No HRTF measurements found")
    
    # Get the length of the HRTF from the first measurement
    first_pos = tuple(positions[0])
    hrtf_length = len(hrtf_dict[first_pos]['left'])
    
    # Number of SH channels
    n_sh_channels = (max_order + 1) ** 2
    
    # Initialize array for SH-domain HRTFs
    sh_hrtfs = np.zeros((2, n_sh_channels, hrtf_length))
    
    # For now, we'll use a simple approximation that doesn't require advanced SH processing
    # This is a placeholder for a proper SH encoding
    
    # W channel (order 0) - omnidirectional component
    # Use the average of all HRTFs
    w_left = np.zeros(hrtf_length)
    w_right = np.zeros(hrtf_length)
    
    for pos_key in hrtf_dict:
        w_left += hrtf_dict[pos_key]['left']
        w_right += hrtf_dict[pos_key]['right']
    
    if n_measurements > 0:
        w_left /= n_measurements
        w_right /= n_measurements
    
    sh_hrtfs[0, 0] = w_left
    sh_hrtfs[1, 0] = w_right
    
    # Only proceed if we have at least 4 measurements for first-order components
    if n_measurements >= 4 and max_order >= 1:
        # Y channel (order 1, m=-1): front-back axis
        y_left = np.zeros(hrtf_length)
        y_right = np.zeros(hrtf_length)
        
        # Z channel (order 1, m=0): up-down axis
        z_left = np.zeros(hrtf_length)
        z_right = np.zeros(hrtf_length)
        
        # X channel (order 1, m=1): left-right axis
        x_left = np.zeros(hrtf_length)
        x_right = np.zeros(hrtf_length)
        
        # Approximate with weighted averages based on direction
        for i, pos in enumerate(positions):
            pos_key = tuple(pos)
            azimuth, elevation, distance = pos
            
            # Get the IRs
            left_ir = hrtf_dict[pos_key]['left']
            right_ir = hrtf_dict[pos_key]['right']
            
            # Weight by direction components
            x_weight = math.sin(azimuth) * math.cos(elevation)  # Left-right
            y_weight = math.cos(azimuth) * math.cos(elevation)  # Front-back
            z_weight = math.sin(elevation)                      # Up-down
            
            # Accumulate weighted IRs
            x_left += x_weight * left_ir
            x_right += x_weight * right_ir
            
            y_left += y_weight * left_ir
            y_right += y_weight * right_ir
            
            z_left += z_weight * left_ir
            z_right += z_weight * right_ir
        
        # Normalize and store
        norm_factor = math.sqrt(4 * math.pi / 3)  # SH normalization factor for order 1
        
        # Y channel (ACN 1)
        sh_hrtfs[0, 1] = norm_factor * y_left / n_measurements
        sh_hrtfs[1, 1] = norm_factor * y_right / n_measurements
        
        # Z channel (ACN 2)
        sh_hrtfs[0, 2] = norm_factor * z_left / n_measurements
        sh_hrtfs[1, 2] = norm_factor * z_right / n_measurements
        
        # X channel (ACN 3)
        sh_hrtfs[0, 3] = norm_factor * x_left / n_measurements
        sh_hrtfs[1, 3] = norm_factor * x_right / n_measurements
    
    # Higher order components would require proper SH decomposition
    # For this initial implementation, we'll use simplified approximations
    if max_order >= 2 and n_measurements >= 9:
        # Here we would implement proper SH decomposition for orders 2 and higher
        # This is simplified for now
        for ch in range(4, n_sh_channels):
            l = math.floor(math.sqrt(ch))
            # Reduce amplitude with increasing order
            gain = 0.2 * (1.0 / l)
            
            # Simple approximation - use a filtered version of the omnidirectional component
            # with some randomness to create different directional patterns
            np.random.seed(ch)  # For reproducibility
            sh_hrtfs[0, ch] = gain * w_left * np.exp(-np.arange(hrtf_length)/10) * (1 + 0.2 * np.random.randn(hrtf_length))
            sh_hrtfs[1, ch] = gain * w_right * np.exp(-np.arange(hrtf_length)/10) * (1 + 0.2 * np.random.randn(hrtf_length))
    
    # Normalize to have consistent energy
    for ear in range(2):
        max_val = np.max(np.abs(sh_hrtfs[ear, 0]))
        if max_val > 0:
            sh_hrtfs[ear, 0] = sh_hrtfs[ear, 0] / max_val
    
    return sh_hrtfs


def find_nearest_position(target_position: SphericalCoord, available_positions: np.ndarray) -> Tuple[int, float]:
    """
    Find the index of the nearest available position to the target.
    
    Args:
        target_position: Target position as (azimuth, elevation, distance)
        available_positions: Array of available positions, shape [n_positions, 3]
        
    Returns:
        Tuple of (index of nearest position, distance to that position)
    """
    target_azimuth, target_elevation, target_distance = target_position
    
    # Calculate angular distance (great circle distance) for each position
    distances = []
    for i, pos in enumerate(available_positions):
        pos_azimuth, pos_elevation, pos_distance = pos
        
        # Calculate great circle distance (angular distance)
        cos_angle = (math.sin(target_elevation) * math.sin(pos_elevation) + 
                     math.cos(target_elevation) * math.cos(pos_elevation) * 
                     math.cos(target_azimuth - pos_azimuth))
        
        # Clamp to valid range for acos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angular_dist = math.acos(cos_angle)
        
        # Also consider distance difference
        distance_diff = abs(target_distance - pos_distance)
        
        # Combined metric (weighted sum)
        combined_dist = angular_dist + 0.1 * distance_diff
        
        distances.append(combined_dist)
    
    # Find the minimum distance and its index
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    
    return min_dist_idx, min_dist


def interpolate_hrtf(target_position: SphericalCoord, hrtf_data: HRTFData, 
                    method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL) -> Dict[str, np.ndarray]:
    """
    Interpolate HRTF for a specific target position using available measurements.
    
    Args:
        target_position: Target position as (azimuth, elevation, distance)
        hrtf_data: HRTF database
        method: Interpolation method to use
        
    Returns:
        Dictionary with 'left' and 'right' interpolated HRTFs
    """
    positions = hrtf_data['positions']
    hrtf_dict = hrtf_data['hrtf_dict']
    
    if method == HRTFInterpolationMethod.NEAREST:
        # Nearest neighbor interpolation (simplest)
        nearest_idx, _ = find_nearest_position(target_position, positions)
        nearest_pos = tuple(positions[nearest_idx])
        return {
            'left': hrtf_dict[nearest_pos]['left'].copy(),
            'right': hrtf_dict[nearest_pos]['right'].copy()
        }
    
    elif method == HRTFInterpolationMethod.SPHERICAL:
        # Spherical weighted average interpolation
        target_azimuth, target_elevation, target_distance = target_position
        
        # Find K nearest neighbors (K=4 is a good compromise)
        K = 4
        distances = []
        for i, pos in enumerate(positions):
            pos_azimuth, pos_elevation, pos_distance = pos
            
            # Calculate great circle distance (angular distance)
            cos_angle = (math.sin(target_elevation) * math.sin(pos_elevation) + 
                        math.cos(target_elevation) * math.cos(pos_elevation) * 
                        math.cos(target_azimuth - pos_azimuth))
            
            # Clamp to valid range for acos
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angular_dist = math.acos(cos_angle)
            
            # Also consider distance difference
            distance_diff = abs(target_distance - pos_distance)
            
            # Combined metric (weighted sum)
            combined_dist = angular_dist + 0.1 * distance_diff
            
            distances.append((i, combined_dist))
        
        # Sort by distance and take K nearest
        distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, _ in distances[:K]]
        
        # Get the nearest positions and their HRTFs
        nearest_positions = [tuple(positions[idx]) for idx in nearest_indices]
        nearest_hrtfs_left = [hrtf_dict[pos]['left'] for pos in nearest_positions]
        nearest_hrtfs_right = [hrtf_dict[pos]['right'] for pos in nearest_positions]
        
        # Calculate weights based on inverse distance
        nearest_distances = [distances[idx][1] for idx in range(K)]
        
        # Avoid division by zero
        nearest_distances = [max(d, 1e-6) for d in nearest_distances]
        
        # Inverse distance weighting
        weights = [1.0 / d for d in nearest_distances]
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Weighted average of HRTFs
        left_ir = np.zeros_like(nearest_hrtfs_left[0])
        right_ir = np.zeros_like(nearest_hrtfs_right[0])
        
        for i in range(K):
            left_ir += normalized_weights[i] * nearest_hrtfs_left[i]
            right_ir += normalized_weights[i] * nearest_hrtfs_right[i]
        
        return {
            'left': left_ir,
            'right': right_ir
        }
    
    elif method == HRTFInterpolationMethod.BILINEAR:
        # Find 4 surrounding points (2 in azimuth, 2 in elevation)
        # This is a simplified version and assumes a grid-like distribution
        # of measurements, which might not be the case for all HRTF databases
        target_azimuth, target_elevation, target_distance = target_position
        
        # Find nearest points in azimuth and elevation
        azimuths = np.array([p[0] for p in positions])
        elevations = np.array([p[1] for p in positions])
        
        # Find closest azimuth below and above target
        az_below_idx = np.where(azimuths <= target_azimuth)[0]
        az_above_idx = np.where(azimuths > target_azimuth)[0]
        
        if len(az_below_idx) == 0 or len(az_above_idx) == 0:
            # Fall back to nearest neighbor if we can't do bilinear
            return interpolate_hrtf(target_position, hrtf_data, HRTFInterpolationMethod.NEAREST)
        
        az_below = azimuths[az_below_idx[np.argmax(azimuths[az_below_idx])]]
        az_above = azimuths[az_above_idx[np.argmin(azimuths[az_above_idx])]]
        
        # Find closest elevation below and above target
        el_below_idx = np.where(elevations <= target_elevation)[0]
        el_above_idx = np.where(elevations > target_elevation)[0]
        
        if len(el_below_idx) == 0 or len(el_above_idx) == 0:
            # Fall back to nearest neighbor if we can't do bilinear
            return interpolate_hrtf(target_position, hrtf_data, HRTFInterpolationMethod.NEAREST)
        
        el_below = elevations[el_below_idx[np.argmax(elevations[el_below_idx])]]
        el_above = elevations[el_above_idx[np.argmin(elevations[el_above_idx])]]
        
        # Find indices of the four surrounding points
        bl_idx = np.where((azimuths == az_below) & (elevations == el_below))[0]
        br_idx = np.where((azimuths == az_above) & (elevations == el_below))[0]
        tl_idx = np.where((azimuths == az_below) & (elevations == el_above))[0]
        tr_idx = np.where((azimuths == az_above) & (elevations == el_above))[0]
        
        if len(bl_idx) == 0 or len(br_idx) == 0 or len(tl_idx) == 0 or len(tr_idx) == 0:
            # Fall back to nearest neighbor if we can't find all four points
            return interpolate_hrtf(target_position, hrtf_data, HRTFInterpolationMethod.NEAREST)
        
        # Get positions and HRTFs for these points
        bl_pos = tuple(positions[bl_idx[0]])
        br_pos = tuple(positions[br_idx[0]])
        tl_pos = tuple(positions[tl_idx[0]])
        tr_pos = tuple(positions[tr_idx[0]])
        
        bl_hrtf_left = hrtf_dict[bl_pos]['left']
        bl_hrtf_right = hrtf_dict[bl_pos]['right']
        br_hrtf_left = hrtf_dict[br_pos]['left']
        br_hrtf_right = hrtf_dict[br_pos]['right']
        tl_hrtf_left = hrtf_dict[tl_pos]['left']
        tl_hrtf_right = hrtf_dict[tl_pos]['right']
        tr_hrtf_left = hrtf_dict[tr_pos]['left']
        tr_hrtf_right = hrtf_dict[tr_pos]['right']
        
        # Calculate interpolation weights
        az_weight = (target_azimuth - az_below) / (az_above - az_below) if az_above != az_below else 0.5
        el_weight = (target_elevation - el_below) / (el_above - el_below) if el_above != el_below else 0.5
        
        # Bilinear interpolation
        left_ir = ((1-az_weight)*(1-el_weight)*bl_hrtf_left + 
                  az_weight*(1-el_weight)*br_hrtf_left +
                  (1-az_weight)*el_weight*tl_hrtf_left +
                  az_weight*el_weight*tr_hrtf_left)
        
        right_ir = ((1-az_weight)*(1-el_weight)*bl_hrtf_right + 
                   az_weight*(1-el_weight)*br_hrtf_right +
                   (1-az_weight)*el_weight*tl_hrtf_right +
                   az_weight*el_weight*tr_hrtf_right)
        
        return {
            'left': left_ir,
            'right': right_ir
        }
    
    elif method == HRTFInterpolationMethod.MAGNITUDE:
        # This would implement a more sophisticated method that interpolates
        # magnitude and phase separately in the frequency domain.
        # For now, we'll fall back to spherical interpolation
        logger.warning("Magnitude interpolation not fully implemented, falling back to spherical")
        return interpolate_hrtf(target_position, hrtf_data, HRTFInterpolationMethod.SPHERICAL)
    
    elif method == HRTFInterpolationMethod.MODAL:
        # This would implement modal interpolation using spherical harmonic
        # decomposition of the HRTF, which is the most accurate method.
        # For now, we'll fall back to spherical interpolation
        logger.warning("Modal interpolation not fully implemented, falling back to spherical")
        return interpolate_hrtf(target_position, hrtf_data, HRTFInterpolationMethod.SPHERICAL)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def get_default_hrtf_database() -> HRTFData:
    """
    Create a simple default HRTF database when no SOFA file is available.
    
    Returns:
        Dictionary containing synthetic HRTF data
    """
    # Note: We need to avoid circular imports
    # Import here rather than at module level
    import importlib
    binauralizer = importlib.import_module(".binauralizer", package="shac.codec")
    
    # Use the generate_synthetic_hrtf_database function from binauralizer
    logger.info("No SOFA support available. Using synthetic HRTF database.")
    return binauralizer.generate_synthetic_hrtf_database(DEFAULT_SAMPLE_RATE)


def download_default_hrtf_database(target_dir: Optional[str] = None) -> str:
    """
    Download a default HRTF database (MIT KEMAR) from the internet.
    
    Args:
        target_dir: Directory to save the downloaded file, defaults to ~/.shac/hrtf
        
    Returns:
        Path to the downloaded SOFA file
    """
    import urllib.request
    import tempfile
    import shutil
    
    # Default target directory
    if target_dir is None:
        target_dir = os.path.expanduser("~/.shac/hrtf")
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Target file path
    target_file = os.path.join(target_dir, "mit_kemar.sofa")
    
    # Check if file already exists
    if os.path.exists(target_file):
        logger.info(f"HRTF database already exists at {target_file}")
        return target_file
    
    # URL for MIT KEMAR HRTF in SOFA format
    # This is the compact version from the SOFA database
    url = "https://sofacoustics.org/data/database/mit/mit_kemar_compact.sofa"
    
    logger.info(f"Downloading HRTF database from {url}...")
    
    # Download to temporary file first
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            # Download the file
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, temp_file)
            
            # Move to final location
            shutil.move(temp_file.name, target_file)
            logger.info(f"HRTF database downloaded to {target_file}")
            
            return target_file
        
        except Exception as e:
            # Clean up temp file on error
            os.unlink(temp_file.name)
            logger.error(f"Error downloading HRTF database: {str(e)}")
            raise
    
    return target_file


def find_sofa_files(search_dir: Optional[str] = None) -> List[str]:
    """
    Find all SOFA files in the specified directory and its subdirectories.
    
    Args:
        search_dir: Directory to search, defaults to ~/.shac/hrtf
        
    Returns:
        List of paths to SOFA files
    """
    # Default search directory
    if search_dir is None:
        search_dir = os.path.expanduser("~/.shac/hrtf")
    
    # Check if directory exists
    if not os.path.exists(search_dir):
        logger.warning(f"Search directory does not exist: {search_dir}")
        return []
    
    # Find all SOFA files
    sofa_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(".sofa"):
                sofa_files.append(os.path.join(root, file))
    
    return sofa_files