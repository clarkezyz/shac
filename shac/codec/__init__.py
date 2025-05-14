"""
Spherical Harmonic Audio Codec (SHAC) Package

A modular implementation of a spatial audio codec optimized for real-time 
interactive navigation and manipulation of 3D sound fields.
"""

from .core import SHACCodec
from .math_utils import real_spherical_harmonic, associated_legendre, factorial, double_factorial
from .streaming import SHACStreamProcessor
from .examples import create_example_sound_scene, demonstrate_interactive_navigation, demonstrate_streaming_processor, main

__version__ = '0.1.0'