"""
Configuration Management Module

This module provides centralized configuration management for the SHAC codec,
including constants, default settings, and configuration utilities.
"""

from typing import Dict, Any, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import os


# =====================================================================================
# Constants
# =====================================================================================

# Physics constants
SPEED_OF_SOUND = 343.0  # m/s at room temperature
AIR_DENSITY = 1.2  # kg/m³ at room temperature

# Default sample rates
DEFAULT_SAMPLE_RATE = 48000  # Hz
SUPPORTED_SAMPLE_RATES = [44100, 48000, 96000]

# Default buffer sizes
DEFAULT_BUFFER_SIZE = 1024  # samples
SUPPORTED_BUFFER_SIZES = [256, 512, 1024, 2048, 4096]

# Ambisonic settings
DEFAULT_AMBISONIC_ORDER = 3
MAX_SUPPORTED_ORDER = 7  # Higher orders may have numerical precision issues

# Default paths
DEFAULT_HRTF_PATH = os.path.join(os.path.dirname(__file__), 'data', 'hrtf')

# Default room settings
DEFAULT_ROOM_RT60 = 0.5  # seconds

# File format settings
SHAC_FILE_MAGIC = b'SHAC'
SHAC_FILE_VERSION = 1
DEFAULT_BIT_DEPTH = 32


# =====================================================================================
# Configuration Classes
# =====================================================================================

@dataclass
class ProcessingConfig:
    """Configuration for audio processing parameters"""
    
    # General settings
    sample_rate: int = DEFAULT_SAMPLE_RATE
    buffer_size: int = DEFAULT_BUFFER_SIZE
    
    # Ambisonic settings
    order: int = DEFAULT_AMBISONIC_ORDER
    
    # Processing flags
    enable_doppler: bool = True
    enable_air_absorption: bool = True
    enable_distance_attenuation: bool = True
    
    # Performance settings
    use_vectorization: bool = True
    use_threading: bool = True
    thread_count: int = 0  # 0 = auto-detect
    
    # Precision
    precision: np.dtype = np.float32
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Sample rate {self.sample_rate} not supported. Use one of: {SUPPORTED_SAMPLE_RATES}")
        
        if self.buffer_size not in SUPPORTED_BUFFER_SIZES:
            raise ValueError(f"Buffer size {self.buffer_size} not supported. Use one of: {SUPPORTED_BUFFER_SIZES}")
        
        if self.order < 0 or self.order > MAX_SUPPORTED_ORDER:
            raise ValueError(f"Ambisonic order must be between 0 and {MAX_SUPPORTED_ORDER}")
        
        if self.thread_count < 0:
            raise ValueError("Thread count must be non-negative (0 for auto-detect)")


@dataclass
class BinauralConfig:
    """Configuration for binaural rendering"""
    
    # HRTF settings
    hrtf_path: str = DEFAULT_HRTF_PATH
    interpolation_quality: int = 2  # 0=nearest, 1=bilinear, 2=spherical, 3=magnitude
    
    # Rendering settings
    enable_crossfade: bool = True
    crossfade_time: float = 0.1  # seconds
    nearfield_compensation: bool = True
    
    # Performance settings
    convolution_mode: str = 'fft'  # 'fft' or 'time'
    hrtf_cache_size: int = 32  # Number of HRTFs to cache
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.interpolation_quality < 0 or self.interpolation_quality > 3:
            raise ValueError("Interpolation quality must be between 0 and 3")
        
        if self.crossfade_time < 0:
            raise ValueError("Crossfade time must be non-negative")
        
        if self.convolution_mode not in ['fft', 'time']:
            raise ValueError("Convolution mode must be 'fft' or 'time'")
        
        if self.hrtf_cache_size < 1:
            raise ValueError("HRTF cache size must be at least 1")


@dataclass
class RoomConfig:
    """Configuration for room acoustics simulation"""
    
    # Room dimensions in meters
    dimensions: tuple = (10.0, 3.0, 8.0)  # width, height, length
    
    # Acoustic properties
    reflection_coefficients: Dict[str, float] = field(default_factory=lambda: {
        'left': 0.7, 'right': 0.7, 'floor': 0.4, 
        'ceiling': 0.8, 'front': 0.6, 'back': 0.6
    })
    
    # Reverberation settings
    rt60: float = DEFAULT_ROOM_RT60
    early_reflection_count: int = 8
    late_reflections_enabled: bool = True
    
    # Environmental factors
    temperature: float = 20.0  # °C
    humidity: float = 50.0  # Relative humidity (%)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.dimensions) != 3:
            raise ValueError("Room dimensions must be a tuple of (width, height, length)")
        
        if any(d <= 0 for d in self.dimensions):
            raise ValueError("Room dimensions must be positive")
        
        if self.rt60 < 0:
            raise ValueError("RT60 must be non-negative")
        
        required_surfaces = {'left', 'right', 'floor', 'ceiling', 'front', 'back'}
        if not required_surfaces.issubset(self.reflection_coefficients.keys()):
            raise ValueError(f"Reflection coefficients must include all surfaces: {required_surfaces}")
        
        if any(not (0 <= rc <= 1) for rc in self.reflection_coefficients.values()):
            raise ValueError("Reflection coefficients must be between 0 and 1")


@dataclass
class SHACConfig:
    """Complete configuration for the SHAC codec"""
    
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    binaural: BinauralConfig = field(default_factory=BinauralConfig)
    room: Optional[RoomConfig] = field(default_factory=RoomConfig)
    
    # File I/O settings
    output_bit_depth: int = DEFAULT_BIT_DEPTH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'processing': {
                'sample_rate': self.processing.sample_rate,
                'buffer_size': self.processing.buffer_size,
                'order': self.processing.order,
                'enable_doppler': self.processing.enable_doppler,
                'enable_air_absorption': self.processing.enable_air_absorption,
                'enable_distance_attenuation': self.processing.enable_distance_attenuation,
                'use_vectorization': self.processing.use_vectorization,
                'use_threading': self.processing.use_threading,
                'thread_count': self.processing.thread_count,
                'precision': str(self.processing.precision)
            },
            'binaural': {
                'hrtf_path': self.binaural.hrtf_path,
                'interpolation_quality': self.binaural.interpolation_quality,
                'enable_crossfade': self.binaural.enable_crossfade,
                'crossfade_time': self.binaural.crossfade_time,
                'nearfield_compensation': self.binaural.nearfield_compensation,
                'convolution_mode': self.binaural.convolution_mode,
                'hrtf_cache_size': self.binaural.hrtf_cache_size
            },
            'room': None if self.room is None else {
                'dimensions': self.room.dimensions,
                'reflection_coefficients': self.room.reflection_coefficients,
                'rt60': self.room.rt60,
                'early_reflection_count': self.room.early_reflection_count,
                'late_reflections_enabled': self.room.late_reflections_enabled,
                'temperature': self.room.temperature,
                'humidity': self.room.humidity
            },
            'output_bit_depth': self.output_bit_depth
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SHACConfig':
        """Create configuration from dictionary"""
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        binaural_config = BinauralConfig(**config_dict.get('binaural', {}))
        
        room_dict = config_dict.get('room')
        room_config = None if room_dict is None else RoomConfig(**room_dict)
        
        return cls(
            processing=processing_config,
            binaural=binaural_config,
            room=room_config,
            output_bit_depth=config_dict.get('output_bit_depth', DEFAULT_BIT_DEPTH)
        )
    
    def save(self, file_path: str) -> None:
        """Save configuration to file"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'SHACConfig':
        """Load configuration from file"""
        import json
        with open(file_path, 'r') as f:
            return cls.from_dict(json.load(f))


# Create a default configuration
default_config = SHACConfig()