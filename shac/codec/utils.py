"""
General Utility Functions and Definitions

This module contains utility functions, type definitions, constants,
and data classes used across the SHAC codebase.

This module defines the fundamental data structures and enumerations
that are used by other modules in the codec.

See Also:
    - config: For centralized configuration management
    - math_utils: For mathematical utility functions
"""

import math
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable, TypeVar, Protocol, Generic, Any, TypedDict

# Type aliases for improved readability
Vector3 = Tuple[float, float, float]  # (x, y, z) or (azimuth, elevation, distance)
SphericalCoord = Tuple[float, float, float]  # (azimuth, elevation, distance) in radians/meters
CartesianCoord = Tuple[float, float, float]  # (x, y, z) in meters
AmbisonicSignal = np.ndarray  # Shape: (n_channels, n_samples)
MonoSignal = np.ndarray  # Shape: (n_samples,)
StereoSignal = np.ndarray  # Shape: (2, n_samples)
HRTFData = Dict[str, Any]  # HRTF data structure


class AmbisonicOrdering(Enum):
    """
    Defines the channel ordering convention for ambisonics.
    
    Different ordering schemes arrange the spherical harmonic components
    in different orders in the channel array.
    
    Attributes:
        ACN: Ambisonic Channel Number (most common standard)
            Channels are ordered by degree and index: 0,1-3,4-8...
        FUMA: FuMa (legacy B-format) ordering
            Uses traditional naming: W,X,Y,Z...
        CUSTOM: User-defined ordering
            For specialized applications requiring custom channel order
    """
    ACN = auto()    # Ambisonic Channel Number (most common)
    FUMA = auto()   # FuMa (legacy B-format) ordering
    CUSTOM = auto() # User-defined ordering


class HRTFInterpolationMethod(Enum):
    """
    HRTF interpolation methods between measured points.
    
    Determines how HRTFs (Head-Related Transfer Functions) are interpolated
    for directions that don't exactly match measured points.
    
    Attributes:
        NEAREST: Nearest neighbor interpolation (fastest, lowest quality)
        BILINEAR: Bilinear interpolation between adjacent points
        SPHERICAL: Spherical weighted average (good balance of quality/speed)
        MAGNITUDE: Magnitude interpolation with phase adjustment (high quality)
        MODAL: Modal decomposition and reconstruction (highest quality)
    """
    NEAREST = auto()      # Nearest neighbor (fastest, lowest quality)
    BILINEAR = auto()     # Bilinear interpolation
    SPHERICAL = auto()    # Spherical weighted average
    MAGNITUDE = auto()    # Magnitude interpolation with phase adjustment
    MODAL = auto()        # Modal decomposition and reconstruction


class DirectivityPattern(Enum):
    """
    Source directivity patterns for sound radiation models.
    
    Defines how sound radiates from a source in different directions.
    
    Attributes:
        OMNIDIRECTIONAL: Equal radiation in all directions
        CARDIOID: Heart-shaped pattern with max radiation in front
        FIGURE_8: Bidirectional pattern with max radiation front and back
        HYPERCARDIOID: Tighter directional pattern than cardioid
        CUSTOM: User-defined directivity via a function
    """
    OMNIDIRECTIONAL = auto()  # Equal radiation in all directions
    CARDIOID = auto()         # Heart-shaped pattern (standard directional)
    FIGURE_8 = auto()         # Bidirectional pattern
    HYPERCARDIOID = auto()    # Tighter directional pattern than cardioid
    CUSTOM = auto()           # User-defined directivity function


@dataclass
class SourceAttributes:
    """
    Attributes defining a mono sound source in 3D space.
    
    Encapsulates all properties that define how a sound source behaves
    in the virtual acoustic environment.
    
    Attributes:
        position: (azimuth, elevation, distance) in radians/meters
        directivity: Directivity factor (0.0=omnidirectional, 1.0=full pattern)
        directivity_pattern: The radiation pattern type
        directivity_axis: Direction of maximum radiation as a unit vector
        width: Angular width in radians (0.0=point source, π=180° spread)
        doppler: Whether to apply Doppler effect for moving sources
        air_absorption: Whether to apply frequency-dependent air absorption
        early_reflections: Whether to generate early reflections
        occlusion: Occlusion factor (0.0=not occluded, 1.0=fully occluded)
        obstruction: Obstruction factor (0.0=not obstructed, 1.0=fully obstructed)
        priority: Rendering priority (higher values = higher priority)
    """
    position: SphericalCoord  # (azimuth, elevation, distance) in radians/meters
    directivity: float = 0.0  # 0.0 = omnidirectional, 1.0 = full pattern
    directivity_pattern: DirectivityPattern = DirectivityPattern.CARDIOID
    directivity_axis: Vector3 = (0.0, 0.0, 1.0)  # Direction of max radiation
    width: float = 0.0  # 0.0 = point source, π = 180° spread
    doppler: bool = True  # Whether to apply Doppler effect for moving sources
    air_absorption: bool = True  # Whether to apply frequency-dependent air absorption
    early_reflections: bool = True  # Whether to generate early reflections
    occlusion: float = 0.0  # 0.0 = no occlusion, 1.0 = fully occluded
    obstruction: float = 0.0  # 0.0 = no obstruction, 1.0 = fully obstructed
    priority: int = 0  # Rendering priority (higher values = higher priority)
    
    def __post_init__(self):
        """Validate and normalize inputs after initialization."""
        # Normalize directivity value
        self.directivity = max(0.0, min(1.0, self.directivity))
        
        # Normalize directivity axis to unit vector
        dx, dy, dz = self.directivity_axis
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
        if magnitude > 0:
            self.directivity_axis = (dx/magnitude, dy/magnitude, dz/magnitude)
        else:
            self.directivity_axis = (0.0, 0.0, 1.0)  # Default to forward
        
        # Normalize width
        self.width = max(0.0, min(math.pi, self.width))
        
        # Normalize occlusion/obstruction
        self.occlusion = max(0.0, min(1.0, self.occlusion))
        self.obstruction = max(0.0, min(1.0, self.obstruction))


class FrequencyBand(TypedDict):
    """
    Frequency band definition for acoustic properties.
    
    Used to define frequency-dependent acoustic coefficients
    like absorption, scattering, etc.
    
    Attributes:
        center: Center frequency in Hz
        coefficient: The acoustic coefficient value (0.0-1.0)
    """
    center: float  # Center frequency in Hz
    coefficient: float  # Coefficient value (0.0-1.0)


@dataclass
class AcousticMaterial:
    """
    Acoustic material properties for room simulation.
    
    Defines how a material or surface interacts with sound waves.
    
    Attributes:
        name: Material name
        absorption: Absorption coefficients by frequency
        scattering: Scattering coefficients by frequency
        transmission: Transmission coefficients by frequency
    """
    name: str  # Material name (e.g., "brick", "carpet")
    absorption: Dict[int, float] = field(default_factory=dict)  # By frequency in Hz
    scattering: Dict[int, float] = field(default_factory=dict)  # By frequency in Hz
    transmission: Dict[int, float] = field(default_factory=dict)  # By frequency in Hz
    
    def get_absorption(self, frequency: float) -> float:
        """Get interpolated absorption coefficient for a frequency."""
        return self._interpolate_coefficient(self.absorption, frequency)
    
    def get_scattering(self, frequency: float) -> float:
        """Get interpolated scattering coefficient for a frequency."""
        return self._interpolate_coefficient(self.scattering, frequency)
    
    def get_transmission(self, frequency: float) -> float:
        """Get interpolated transmission coefficient for a frequency."""
        return self._interpolate_coefficient(self.transmission, frequency)
    
    def _interpolate_coefficient(self, coefficients: Dict[int, float], frequency: float) -> float:
        """Interpolate coefficient value for a given frequency."""
        # Check for exact match
        if int(frequency) in coefficients:
            return coefficients[int(frequency)]
        
        # Find nearest frequency bands
        freqs = sorted(coefficients.keys())
        if not freqs:
            return 0.0
        
        if frequency <= freqs[0]:
            return coefficients[freqs[0]]
        if frequency >= freqs[-1]:
            return coefficients[freqs[-1]]
        
        # Find surrounding frequency bands
        lower_freq = max(f for f in freqs if f <= frequency)
        upper_freq = min(f for f in freqs if f >= frequency)
        
        # Linear interpolation
        if lower_freq == upper_freq:
            return coefficients[lower_freq]
        
        t = (frequency - lower_freq) / (upper_freq - lower_freq)
        return coefficients[lower_freq] * (1 - t) + coefficients[upper_freq] * t


@dataclass
class RoomAttributes:
    """
    Attributes defining the acoustic properties of a virtual room.
    
    Encapsulates all properties that define the acoustic environment.
    
    Attributes:
        dimensions: Room dimensions (width, height, length) in meters
        materials: Mapping of surface names to material properties
        reflection_coefficients: Simplified reflection coeffs for each surface
        scattering_coefficients: Simplified scattering coeffs for each surface
        absorption_coefficients: Freq-dependent absorption coefficients
        humidity: Relative humidity percentage
        temperature: Temperature in Celsius
        air_density: Air density in kg/m³
    """
    dimensions: Vector3  # Room dimensions (width, height, length) in meters
    materials: Dict[str, AcousticMaterial] = field(default_factory=dict)  # By surface name
    reflection_coefficients: Dict[str, float] = field(default_factory=dict)  # Simplified coeffs
    scattering_coefficients: Dict[str, float] = field(default_factory=dict)  # Simplified coeffs
    absorption_coefficients: Dict[str, Dict[int, float]] = field(default_factory=dict)  # By freq
    humidity: float = 50.0  # Relative humidity (%)
    temperature: float = 20.0  # Temperature (°C)
    air_density: float = 1.2  # Air density (kg/m³)
    
    def __post_init__(self):
        """Validate and setup room attributes after initialization."""
        # Ensure dimensions are positive
        if any(d <= 0 for d in self.dimensions):
            raise ValueError("Room dimensions must be positive")
        
        # Ensure reflection and scattering coefficients are within range
        for surface, coef in self.reflection_coefficients.items():
            self.reflection_coefficients[surface] = max(0.0, min(1.0, coef))
            
        for surface, coef in self.scattering_coefficients.items():
            self.scattering_coefficients[surface] = max(0.0, min(1.0, coef))
        
        # Calculate default coefficients for standard surfaces if not specified
        standard_surfaces = {'left', 'right', 'floor', 'ceiling', 'front', 'back'}
        for surface in standard_surfaces:
            if surface not in self.reflection_coefficients:
                # Default reflection coefficients
                self.reflection_coefficients[surface] = 0.7
            if surface not in self.scattering_coefficients:
                # Default scattering coefficients
                self.scattering_coefficients[surface] = 0.2


@dataclass
class BinauralRendererConfig:
    """
    Configuration for binaural rendering from ambisonics.
    
    Controls how ambisonic signals are converted to binaural stereo output.
    
    Attributes:
        hrtf_database: Path to HRTF database or dictionary with HRTF data
        interpolation_method: Method used to interpolate between HRTF points
        nearfield_compensation: Whether to apply nearfield compensation
        crossfade_time: Time to crossfade when HRTF changes drastically
        hrtf_preloading: Whether to preload all HRTFs at initialization
        personalization: HRTF personalization parameters
    """
    hrtf_database: Union[str, Dict[str, Any]]  # Path or data dict
    interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL
    nearfield_compensation: bool = True  # Apply nearfield compensation for close sources
    crossfade_time: float = 0.1  # Time in seconds to crossfade when HRTF changes
    hrtf_preloading: bool = True  # Whether to preload all HRTFs at initialization
    personalization: Dict[str, Any] = field(default_factory=dict)  # HRTF personalization parameters
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.crossfade_time < 0:
            raise ValueError("Crossfade time must be non-negative")
            
        if isinstance(self.hrtf_database, str) and not self.hrtf_database:
            raise ValueError("HRTF database path cannot be empty")


# Common utility functions

def validate_spherical_coordinates(coords: SphericalCoord) -> SphericalCoord:
    """
    Validate and normalize spherical coordinates.
    
    Args:
        coords: (azimuth, elevation, distance) in radians/meters
        
    Returns:
        Normalized spherical coordinates with azimuth in [0, 2π),
        elevation in [-π/2, π/2], and positive distance.
    
    Raises:
        ValueError: If distance is negative
    """
    azimuth, elevation, distance = coords
    
    # Normalize azimuth to [0, 2π)
    azimuth = azimuth % (2 * math.pi)
    
    # Normalize elevation to [-π/2, π/2]
    elevation = max(-math.pi/2, min(math.pi/2, elevation))
    
    # Ensure distance is positive
    if distance < 0:
        raise ValueError("Distance must be non-negative")
    
    return (azimuth, elevation, distance)


def angle_between_vectors(v1: Vector3, v2: Vector3) -> float:
    """
    Calculate the angle between two 3D vectors in radians.
    
    Args:
        v1: First vector (x, y, z)
        v2: Second vector (x, y, z)
        
    Returns:
        Angle between vectors in radians
    """
    # Compute dot product
    dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    
    # Compute magnitudes
    mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
    
    # Handle zero vectors or numerical issues
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0
    
    # Compute cosine of angle
    cos_angle = dot_product / (mag1 * mag2)
    
    # Handle potential floating-point errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # Return angle in radians
    return math.acos(cos_angle)