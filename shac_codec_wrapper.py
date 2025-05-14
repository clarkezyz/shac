"""
Spherical Harmonic Audio Codec (SHAC) - Wrapper Module

This module re-exports the functionality from the new modular structure
to maintain backward compatibility with existing code that imports from
the original shac_codec.py file.

To migrate to the new structure, use direct imports from shac.codec instead.
"""

# Re-export main class and constants
from shac.codec.core import SHACCodec
from shac.codec.math_utils import (
    AmbisonicNormalization,
    factorial, 
    double_factorial,
    associated_legendre, 
    real_spherical_harmonic,
    spherical_harmonic_matrix,
    sh_rotation_matrix,
    convert_acn_to_fuma,
    convert_fuma_to_acn
)
from shac.codec.utils import (
    AmbisonicOrdering,
    HRTFInterpolationMethod,
    SourceAttributes,
    RoomAttributes,
    BinauralRendererConfig
)
from shac.codec.encoders import (
    convert_to_spherical,
    convert_to_cartesian,
    encode_mono_source,
    encode_stereo_source
)
from shac.codec.processors import (
    rotate_ambisonics,
    decode_to_speakers
)
from shac.codec.binauralizer import (
    binauralize_ambisonics,
    load_hrtf_database,
    apply_frequency_dependent_effects
)
from shac.codec.io import (
    SHACFileWriter,
    SHACFileReader
)
from shac.codec.core import (
    apply_early_reflections,
    apply_diffuse_reverberation
)
from shac.codec.streaming import SHACStreamProcessor
from shac.codec.examples import (
    create_example_sound_scene,
    demonstrate_interactive_navigation,
    demonstrate_streaming_processor,
    main
)

# Set version
__version__ = '0.1.0'