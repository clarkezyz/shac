"""
Audio Format Conversion Module

This module provides functions for converting between different audio formats,
with a focus on creating and working with the SHAC spatial audio format.

It enables conversion of standard audio formats (WAV, MP3, etc.) to the SHAC
format, as well as exporting SHAC data to various output formats including
binaural stereo.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from .io import SHACFileWriter, SHACFileReader
from .math_utils import real_spherical_harmonic, AmbisonicNormalization
from .utils import SphericalCoord, Vector3, SourceAttributes, DirectivityPattern
from .binauralizer import binauralize_ambisonics

# Import file_loader from the parent package
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from file_loader import load_audio_file, list_audio_files

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ConversionOptions:
    """Options for file conversion."""
    
    # Input options
    normalize_input: bool = True          # Normalize input audio to peak at 1.0
    target_sample_rate: int = 48000       # Sample rate for conversion
    mono_conversion: bool = True          # Convert stereo to mono for non-stereo sources
    
    # SHAC encoding options
    ambisonic_order: int = 3              # Order of ambisonic representation
    normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D
    
    # Output options
    bit_depth: int = 32                   # Bit depth for output (16 or 32)
    
    # Processing options
    chunk_size: int = 8192                # Process audio in chunks of this size
    show_progress: bool = True            # Show progress during conversion
    
    # Source attributes
    default_distance: float = 2.0         # Default distance for sources without position
    default_directivity: float = 0.0      # Default directivity (0.0 = omnidirectional)
    default_directivity_pattern: DirectivityPattern = DirectivityPattern.CARDIOID
    
    # Default position when none provided
    # (azimuth, elevation, distance) in radians/meters
    default_position: SphericalCoord = (0.0, 0.0, 2.0)  # Default: directly in front, 2m away


def mono_to_shac(input_file: str, output_file: str, 
                position: Optional[SphericalCoord] = None,
                source_name: Optional[str] = None,
                options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """
    Convert a mono audio file to SHAC format with a specific position.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output SHAC file
        position: Source position as (azimuth, elevation, distance)
                 Default is directly in front
        source_name: Name for the source (uses filename if None)
        options: Conversion options
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Use default position if none provided
    if position is None:
        position = options.default_position
    
    # Load audio file
    logger.info(f"Loading audio file: {input_file}")
    audio_info = load_audio_file(input_file, options.target_sample_rate)
    
    # Ensure mono audio
    audio_data = audio_info['audio_data']
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        if options.mono_conversion:
            # Convert to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
            logger.info(f"Converted stereo audio to mono")
        else:
            raise ValueError("Input must be mono audio unless mono_conversion is enabled")
    
    # Normalize if requested
    if options.normalize_input:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
    
    # Determine source name if not provided
    if source_name is None:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create SHAC writer
    writer = SHACFileWriter(options.ambisonic_order, options.target_sample_rate, options.normalization)
    
    # Encode mono source to ambisonics
    logger.info(f"Encoding mono source '{source_name}' at position {position}")
    ambi_signals = _encode_mono_source(audio_data, position, options.ambisonic_order, options.normalization)
    
    # Create metadata for the layer
    metadata = {
        'name': source_name,
        'position': position,
        'original_file': input_file,
        'source_type': 'mono',
        'directivity': options.default_directivity,
        'directivity_pattern': options.default_directivity_pattern.name,
    }
    
    # Add layer to the SHAC file
    writer.add_layer(source_name, ambi_signals, metadata)
    
    # Write the SHAC file
    logger.info(f"Writing SHAC file: {output_file}")
    writer.write_file(output_file, options.bit_depth)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'source_name': source_name,
        'position': position,
        'ambisonic_order': options.ambisonic_order,
        'sample_rate': options.target_sample_rate,
        'bit_depth': options.bit_depth,
        'duration': audio_info['duration'],
    }


def stereo_to_shac(input_file: str, output_file: str,
                 base_position: Optional[SphericalCoord] = None,
                 source_width: float = 0.3,
                 source_name: Optional[str] = None,
                 options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """
    Convert a stereo audio file to SHAC format.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output SHAC file
        base_position: Base position for stereo field as (azimuth, elevation, distance)
                      Default is directly in front
        source_width: Width of the stereo image in radians
        source_name: Name for the source (uses filename if None)
        options: Conversion options
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Use default position if none provided
    if base_position is None:
        base_position = options.default_position
    
    # Load audio file
    logger.info(f"Loading audio file: {input_file}")
    audio_info = load_audio_file(input_file, options.target_sample_rate)
    
    # Check if audio is stereo
    audio_data = audio_info['audio_data']
    if len(audio_data.shape) == 1 or audio_data.shape[1] == 1:
        logger.warning(f"Input file {input_file} is mono, not stereo. Using mono_to_shac instead.")
        return mono_to_shac(input_file, output_file, base_position, source_name, options)
    
    # Ensure we have exactly 2 channels
    if audio_data.shape[1] != 2:
        raise ValueError(f"Expected stereo audio (2 channels), but got {audio_data.shape[1]} channels")
    
    # Normalize if requested
    if options.normalize_input:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
    
    # Determine source name if not provided
    if source_name is None:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create SHAC writer
    writer = SHACFileWriter(options.ambisonic_order, options.target_sample_rate, options.normalization)
    
    # Encode stereo source to ambisonics
    logger.info(f"Encoding stereo source '{source_name}' at position {base_position} with width {source_width:.2f}")
    ambi_signals = _encode_stereo_source(audio_data, base_position, source_width, 
                                        options.ambisonic_order, options.normalization)
    
    # Create metadata for the layer
    metadata = {
        'name': source_name,
        'position': base_position,
        'original_file': input_file,
        'source_type': 'stereo',
        'source_width': source_width,
        'directivity': options.default_directivity,
        'directivity_pattern': options.default_directivity_pattern.name,
    }
    
    # Add layer to the SHAC file
    writer.add_layer(source_name, ambi_signals, metadata)
    
    # Write the SHAC file
    logger.info(f"Writing SHAC file: {output_file}")
    writer.write_file(output_file, options.bit_depth)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'source_name': source_name,
        'base_position': base_position,
        'source_width': source_width,
        'ambisonic_order': options.ambisonic_order,
        'sample_rate': options.target_sample_rate,
        'bit_depth': options.bit_depth,
        'duration': audio_info['duration'],
    }


def multichannel_to_shac(input_file: str, output_file: str,
                       channel_positions: List[SphericalCoord],
                       source_name: Optional[str] = None,
                       options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """
    Convert a multichannel audio file to SHAC format.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output SHAC file
        channel_positions: List of positions for each channel
        source_name: Name for the source (uses filename if None)
        options: Conversion options
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Load audio file
    logger.info(f"Loading audio file: {input_file}")
    audio_info = load_audio_file(input_file, options.target_sample_rate)
    
    # Get audio data
    audio_data = audio_info['audio_data']
    
    # Handle different array shapes
    if len(audio_data.shape) == 1:
        # Mono file
        n_channels = 1
        audio_data = audio_data.reshape(len(audio_data), 1)
    else:
        # Multi-channel file
        n_channels = audio_data.shape[1]
    
    # Check if we have enough positions for all channels
    if len(channel_positions) < n_channels:
        raise ValueError(f"Not enough positions provided: got {len(channel_positions)}, "
                        f"need {n_channels} for all channels")
    
    # Normalize if requested
    if options.normalize_input:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
    
    # Determine source name if not provided
    if source_name is None:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create SHAC writer
    writer = SHACFileWriter(options.ambisonic_order, options.target_sample_rate, options.normalization)
    
    # Initialize ambisonic signals
    n_samples = len(audio_data)
    n_ambi_channels = (options.ambisonic_order + 1) ** 2
    ambi_signals = np.zeros((n_ambi_channels, n_samples))
    
    # Process each channel
    for ch in range(n_channels):
        channel_audio = audio_data[:, ch]
        channel_position = channel_positions[ch]
        
        # Encode this channel
        logger.info(f"Encoding channel {ch} at position {channel_position}")
        channel_ambi = _encode_mono_source(channel_audio, channel_position, 
                                         options.ambisonic_order, options.normalization)
        
        # Add to the total
        ambi_signals += channel_ambi
    
    # Create metadata
    metadata = {
        'name': source_name,
        'original_file': input_file,
        'source_type': 'multichannel',
        'n_channels': n_channels,
        'channel_positions': channel_positions,
    }
    
    # Add layer to the SHAC file
    writer.add_layer(source_name, ambi_signals, metadata)
    
    # Write the SHAC file
    logger.info(f"Writing SHAC file: {output_file}")
    writer.write_file(output_file, options.bit_depth)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'source_name': source_name,
        'channel_positions': channel_positions,
        'n_channels': n_channels,
        'ambisonic_order': options.ambisonic_order,
        'sample_rate': options.target_sample_rate,
        'bit_depth': options.bit_depth,
        'duration': audio_info['duration'],
    }


def ambisonic_to_shac(input_file: str, output_file: str,
                    input_order: int,
                    input_normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D,
                    source_name: Optional[str] = None,
                    options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """
    Convert an ambisonic audio file to SHAC format.
    
    Args:
        input_file: Path to input ambisonic audio file
        output_file: Path to output SHAC file
        input_order: Order of the input ambisonic audio
        input_normalization: Normalization of the input ambisonic audio
        source_name: Name for the source (uses filename if None)
        options: Conversion options
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Load audio file
    logger.info(f"Loading ambisonic file: {input_file}")
    audio_info = load_audio_file(input_file, options.target_sample_rate)
    
    # Get audio data
    audio_data = audio_info['audio_data']
    
    # Check channel count against expected ambisonic channels
    expected_channels = (input_order + 1) ** 2
    
    if len(audio_data.shape) == 1:
        actual_channels = 1
    else:
        actual_channels = audio_data.shape[1]
    
    if actual_channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels for order {input_order}, "
                        f"but input file has {actual_channels} channels")
    
    # Reshape audio data to expected shape for ambisonic signals
    if len(audio_data.shape) == 1:
        # Mono file (1 channel)
        ambi_signals = audio_data.reshape(1, len(audio_data))
    else:
        # Multi-channel file
        ambi_signals = audio_data.T  # Transpose to get (channels, samples)
    
    # Normalize if requested
    if options.normalize_input:
        max_val = np.max(np.abs(ambi_signals))
        if max_val > 0:
            ambi_signals = ambi_signals / max_val
    
    # Convert normalization if needed
    if input_normalization != options.normalization:
        logger.info(f"Converting normalization from {input_normalization.name} to {options.normalization.name}")
        ambi_signals = _convert_normalization(ambi_signals, input_order, 
                                            input_normalization, options.normalization)
    
    # Determine source name if not provided
    if source_name is None:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create SHAC writer
    writer = SHACFileWriter(max(input_order, options.ambisonic_order), 
                           options.target_sample_rate, options.normalization)
    
    # Create metadata
    metadata = {
        'name': source_name,
        'original_file': input_file,
        'source_type': 'ambisonic',
        'input_order': input_order,
        'input_normalization': input_normalization.name,
    }
    
    # Add layer to the SHAC file
    writer.add_layer(source_name, ambi_signals, metadata)
    
    # Write the SHAC file
    logger.info(f"Writing SHAC file: {output_file}")
    writer.write_file(output_file, options.bit_depth)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'source_name': source_name,
        'input_order': input_order,
        'output_order': max(input_order, options.ambisonic_order),
        'sample_rate': options.target_sample_rate,
        'bit_depth': options.bit_depth,
        'duration': audio_info['duration'],
    }


def multi_source_to_shac(input_files: List[str], output_file: str,
                        positions: List[SphericalCoord],
                        source_names: Optional[List[str]] = None,
                        options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """
    Convert multiple audio files to a single SHAC file with multiple layers.
    
    Args:
        input_files: List of input audio file paths
        output_file: Path to output SHAC file
        positions: List of positions for each source
        source_names: List of names for each source (uses filenames if None)
        options: Conversion options
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Check if we have enough positions for all sources
    if len(positions) < len(input_files):
        raise ValueError(f"Not enough positions provided: got {len(positions)}, "
                        f"need {len(input_files)} for all sources")
    
    # Create source names if not provided
    if source_names is None:
        source_names = [os.path.splitext(os.path.basename(file))[0] for file in input_files]
    elif len(source_names) < len(input_files):
        raise ValueError(f"Not enough source names provided: got {len(source_names)}, "
                        f"need {len(input_files)} for all sources")
    
    # Create SHAC writer
    writer = SHACFileWriter(options.ambisonic_order, options.target_sample_rate, options.normalization)
    
    # Process each source
    sources_info = []
    
    for i, input_file in enumerate(input_files):
        source_name = source_names[i]
        position = positions[i]
        
        # Load audio file
        logger.info(f"Loading audio file: {input_file}")
        try:
            audio_info = load_audio_file(input_file, options.target_sample_rate)
            
            # Get audio data
            audio_data = audio_info['audio_data']
            
            # Handle stereo files
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                if options.mono_conversion:
                    # Convert to mono by averaging channels
                    audio_data = np.mean(audio_data, axis=1)
                    logger.info(f"Converted stereo audio to mono for source '{source_name}'")
                else:
                    raise ValueError(f"Input '{input_file}' is not mono and mono_conversion is disabled")
            
            # Normalize if requested
            if options.normalize_input:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
            
            # Encode to ambisonics
            logger.info(f"Encoding source '{source_name}' at position {position}")
            ambi_signals = _encode_mono_source(audio_data, position, 
                                            options.ambisonic_order, options.normalization)
            
            # Create metadata
            metadata = {
                'name': source_name,
                'position': position,
                'original_file': input_file,
                'source_type': 'mono',
                'directivity': options.default_directivity,
                'directivity_pattern': options.default_directivity_pattern.name,
            }
            
            # Add layer to the SHAC file
            writer.add_layer(source_name, ambi_signals, metadata)
            
            # Store source info
            sources_info.append({
                'name': source_name,
                'input_file': input_file,
                'position': position,
                'duration': audio_info['duration'],
            })
            
        except Exception as e:
            logger.error(f"Error processing source '{source_name}': {str(e)}")
            logger.error(f"Skipping source and continuing with others")
    
    # Write the SHAC file
    logger.info(f"Writing SHAC file: {output_file}")
    writer.write_file(output_file, options.bit_depth)
    
    return {
        'output_file': output_file,
        'n_sources': len(sources_info),
        'sources': sources_info,
        'ambisonic_order': options.ambisonic_order,
        'sample_rate': options.target_sample_rate,
        'bit_depth': options.bit_depth,
    }


def shac_to_binaural(input_file: str, output_file: str,
                    hrtf_file: Optional[str] = None,
                    head_rotation: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """
    Convert a SHAC file to binaural stereo WAV.
    
    Args:
        input_file: Path to input SHAC file
        output_file: Path to output WAV file
        hrtf_file: Path to HRTF file (SOFA format)
        head_rotation: Optional (yaw, pitch, roll) rotation in radians
        
    Returns:
        Dictionary with conversion information
    """
    try:
        import scipy.io.wavfile
    except ImportError:
        raise ImportError("scipy is required for WAV file export. Install with: pip install scipy")
    
    # Load SHAC file
    logger.info(f"Loading SHAC file: {input_file}")
    reader = SHACFileReader(input_file)
    file_info = reader.get_file_info()
    
    # Get all layers
    layer_names = reader.get_layer_names()
    logger.info(f"Found {len(layer_names)} layers in SHAC file")
    
    # Get sample rate and bit depth
    sample_rate = file_info['sample_rate']
    bit_depth = file_info.get('bit_depth', 32)
    
    # Mix all layers
    mixed_ambisonic = None
    
    for layer_name in layer_names:
        # Read layer audio
        layer_audio = reader.read_layer(layer_name)
        
        # If this is the first layer, initialize the mixed audio
        if mixed_ambisonic is None:
            mixed_ambisonic = layer_audio
        else:
            # Mix with existing audio (simple addition)
            # In a more advanced implementation, we'd need to handle different layer lengths
            min_length = min(mixed_ambisonic.shape[1], layer_audio.shape[1])
            mixed_ambisonic[:, :min_length] += layer_audio[:, :min_length]
    
    if mixed_ambisonic is None:
        raise ValueError("No audio data found in SHAC file")
    
    # Apply rotation if specified
    if head_rotation is not None:
        from .processors import rotate_ambisonics
        yaw, pitch, roll = head_rotation
        logger.info(f"Applying head rotation: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
        mixed_ambisonic = rotate_ambisonics(mixed_ambisonic, yaw, pitch, roll)
    
    # Render to binaural
    logger.info(f"Rendering to binaural using HRTF: {hrtf_file if hrtf_file else 'default'}")
    binaural_audio = binauralize_ambisonics(mixed_ambisonic, hrtf_file if hrtf_file else None)
    
    # Write WAV file
    logger.info(f"Writing binaural audio to: {output_file}")
    if bit_depth == 16:
        # Convert to 16-bit integer
        audio_int16 = (binaural_audio * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_file, sample_rate, audio_int16.T)
    else:
        # Use 32-bit float
        scipy.io.wavfile.write(output_file, sample_rate, binaural_audio.T.astype(np.float32))
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'n_layers': len(layer_names),
        'sample_rate': sample_rate,
        'bit_depth': bit_depth,
        'head_rotation': head_rotation,
    }


def shac_to_ambisonic(input_file: str, output_file: str,
                     output_order: Optional[int] = None,
                     output_normalization: Optional[AmbisonicNormalization] = None) -> Dict[str, Any]:
    """
    Convert a SHAC file to a standard ambisonic audio file.
    
    Args:
        input_file: Path to input SHAC file
        output_file: Path to output ambisonic WAV file
        output_order: Order for the output ambisonic file (defaults to input order)
        output_normalization: Normalization for output (defaults to input normalization)
        
    Returns:
        Dictionary with conversion information
    """
    try:
        import scipy.io.wavfile
    except ImportError:
        raise ImportError("scipy is required for WAV file export. Install with: pip install scipy")
    
    # Load SHAC file
    logger.info(f"Loading SHAC file: {input_file}")
    reader = SHACFileReader(input_file)
    file_info = reader.get_file_info()
    
    # Get all layers
    layer_names = reader.get_layer_names()
    logger.info(f"Found {len(layer_names)} layers in SHAC file")
    
    # Get sample rate and other info
    sample_rate = file_info['sample_rate']
    bit_depth = file_info.get('bit_depth', 32)
    input_order = file_info['order']
    input_normalization = file_info.get('normalization', AmbisonicNormalization.SN3D)
    
    # Use input defaults if not specified
    if output_order is None:
        output_order = input_order
    
    if output_normalization is None:
        output_normalization = input_normalization
    
    # Mix all layers
    mixed_ambisonic = None
    
    for layer_name in layer_names:
        # Read layer audio
        layer_audio = reader.read_layer(layer_name)
        
        # If this is the first layer, initialize the mixed audio
        if mixed_ambisonic is None:
            mixed_ambisonic = layer_audio
        else:
            # Mix with existing audio (simple addition)
            min_length = min(mixed_ambisonic.shape[1], layer_audio.shape[1])
            mixed_ambisonic[:, :min_length] += layer_audio[:, :min_length]
    
    if mixed_ambisonic is None:
        raise ValueError("No audio data found in SHAC file")
    
    # Convert order if needed
    input_channels = mixed_ambisonic.shape[0]
    output_channels = (output_order + 1) ** 2
    
    if output_order != input_order:
        logger.info(f"Converting ambisonic order: {input_order} -> {output_order}")
        
        if output_order < input_order:
            # Truncate to lower order
            mixed_ambisonic = mixed_ambisonic[:output_channels]
        else:
            # Extend to higher order (with zeros)
            n_samples = mixed_ambisonic.shape[1]
            extended_ambisonic = np.zeros((output_channels, n_samples))
            extended_ambisonic[:input_channels] = mixed_ambisonic
            mixed_ambisonic = extended_ambisonic
    
    # Convert normalization if needed
    if output_normalization != input_normalization:
        logger.info(f"Converting normalization: {input_normalization.name} -> {output_normalization.name}")
        mixed_ambisonic = _convert_normalization(mixed_ambisonic, output_order, 
                                               input_normalization, output_normalization)
    
    # Write WAV file
    logger.info(f"Writing ambisonic audio to: {output_file}")
    
    # Transpose to (samples, channels) for WAV output
    output_audio = mixed_ambisonic.T
    
    if bit_depth == 16:
        # Convert to 16-bit integer
        audio_int16 = (output_audio * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_file, sample_rate, audio_int16)
    else:
        # Use 32-bit float
        scipy.io.wavfile.write(output_file, sample_rate, output_audio.astype(np.float32))
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'n_layers': len(layer_names),
        'input_order': input_order,
        'output_order': output_order,
        'sample_rate': sample_rate,
        'bit_depth': bit_depth,
    }


def _encode_mono_source(mono_signal: np.ndarray, position: SphericalCoord, 
                      order: int, normalization: AmbisonicNormalization) -> np.ndarray:
    """
    Encode a mono signal to ambisonic signals based on position.
    
    Args:
        mono_signal: Mono audio signal
        position: Source position as (azimuth, elevation, distance)
        order: Ambisonic order
        normalization: Ambisonic normalization
        
    Returns:
        Ambisonic signals with shape [(order+1)², samples]
    """
    azimuth, elevation, distance = position
    n_channels = (order + 1) ** 2
    n_samples = len(mono_signal)
    
    # Apply simple distance attenuation
    if distance > 0:
        attenuated_signal = mono_signal * (1.0 / max(1.0, distance))
    else:
        attenuated_signal = mono_signal
    
    # Initialize ambisonic signals
    ambi_signals = np.zeros((n_channels, n_samples))
    
    # Encode each ambisonic channel
    for n in range(order + 1):
        for m in range(-n, n + 1):
            # Calculate ACN index
            acn = n * n + n + m
            
            # Calculate spherical harmonic value for this direction
            sh_val = real_spherical_harmonic(n, m, azimuth, elevation, normalization)
            
            # Encode the mono signal to this ambisonic channel
            ambi_signals[acn] = attenuated_signal * sh_val
    
    return ambi_signals


def _encode_stereo_source(stereo_signal: np.ndarray, base_position: SphericalCoord,
                        width: float, order: int, normalization: AmbisonicNormalization) -> np.ndarray:
    """
    Encode a stereo signal to ambisonics with a position and width.
    
    Args:
        stereo_signal: Stereo audio signal with shape [samples, 2]
        base_position: Base position of the stereo field as (azimuth, elevation, distance)
        width: Width of the stereo field in radians
        order: Ambisonic order
        normalization: Ambisonic normalization
        
    Returns:
        Ambisonic signals with shape [(order+1)², samples]
    """
    base_azimuth, elevation, distance = base_position
    n_channels = (order + 1) ** 2
    n_samples = len(stereo_signal)
    
    # Initialize ambisonic signals
    ambi_signals = np.zeros((n_channels, n_samples))
    
    # Split left and right channels
    left_signal = stereo_signal[:, 0]
    right_signal = stereo_signal[:, 1]
    
    # Calculate left and right positions
    left_azimuth = (base_azimuth - width/2) % (2 * np.pi)
    right_azimuth = (base_azimuth + width/2) % (2 * np.pi)
    
    left_position = (left_azimuth, elevation, distance)
    right_position = (right_azimuth, elevation, distance)
    
    # Encode left and right channels
    left_ambi = _encode_mono_source(left_signal, left_position, order, normalization)
    right_ambi = _encode_mono_source(right_signal, right_position, order, normalization)
    
    # Mix left and right ambisonic signals
    ambi_signals = left_ambi + right_ambi
    
    return ambi_signals


def _convert_normalization(ambi_signals: np.ndarray, order: int,
                         input_norm: AmbisonicNormalization,
                         output_norm: AmbisonicNormalization) -> np.ndarray:
    """
    Convert ambisonic signals from one normalization scheme to another.
    
    Args:
        ambi_signals: Ambisonic signals with shape [channels, samples]
        order: Ambisonic order
        input_norm: Input normalization
        output_norm: Output normalization
        
    Returns:
        Converted ambisonic signals
    """
    if input_norm == output_norm:
        return ambi_signals
    
    n_channels = (order + 1) ** 2
    if ambi_signals.shape[0] < n_channels:
        raise ValueError(f"Ambisonic signals has {ambi_signals.shape[0]} channels, "
                        f"but order {order} requires {n_channels} channels")
    
    converted_signals = ambi_signals.copy()
    
    # Apply conversion factors based on ACN channel ordering
    for n in range(order + 1):
        for m in range(-n, n + 1):
            acn = n * n + n + m
            
            # Skip if beyond the available channels
            if acn >= ambi_signals.shape[0]:
                continue
            
            # Get conversion factor for this channel
            factor = 1.0
            
            # SN3D to N3D: Multiply by sqrt(2n+1)
            if input_norm == AmbisonicNormalization.SN3D and output_norm == AmbisonicNormalization.N3D:
                factor = np.sqrt(2 * n + 1)
            
            # N3D to SN3D: Divide by sqrt(2n+1)
            elif input_norm == AmbisonicNormalization.N3D and output_norm == AmbisonicNormalization.SN3D:
                factor = 1.0 / np.sqrt(2 * n + 1)
            
            # FuMa conversions require more complex handling of channel ordering and normalization
            elif input_norm == AmbisonicNormalization.FUMA or output_norm == AmbisonicNormalization.FUMA:
                # FuMa only defined up to 3rd order
                if n > 3:
                    logger.warning(f"FuMa conversion not defined for order > 3. Channel {acn} skipped.")
                    continue
                
                # For simplicity we only handle conversion between SN3D and FuMa here
                # A complete implementation would handle all combinations
                if input_norm == AmbisonicNormalization.SN3D and output_norm == AmbisonicNormalization.FUMA:
                    # Factors depend on degree and order
                    if n == 0:  # W
                        factor = 1.0 / np.sqrt(2)
                    elif n == 1:  # Y, Z, X
                        factor = 1.0
                    elif n == 2:  # V, T, R, S, U
                        factor = 2.0 / np.sqrt(3) if m == 0 else np.sqrt(2.0 / 3.0)
                    elif n == 3:  # 3rd order
                        if m == 0:
                            factor = np.sqrt(8.0 / 5.0)
                        elif abs(m) == 1 or abs(m) == 3:
                            factor = np.sqrt(9.0 / 5.0)
                        else:  # |m| == 2
                            factor = np.sqrt(6.0 / 5.0)
                
                elif input_norm == AmbisonicNormalization.FUMA and output_norm == AmbisonicNormalization.SN3D:
                    # Inverse of the above factors
                    if n == 0:  # W
                        factor = np.sqrt(2)
                    elif n == 1:  # Y, Z, X
                        factor = 1.0
                    elif n == 2:  # V, T, R, S, U
                        factor = np.sqrt(3) / 2.0 if m == 0 else np.sqrt(3.0 / 2.0)
                    elif n == 3:  # 3rd order
                        if m == 0:
                            factor = np.sqrt(5.0 / 8.0)
                        elif abs(m) == 1 or abs(m) == 3:
                            factor = np.sqrt(5.0 / 9.0)
                        else:  # |m| == 2
                            factor = np.sqrt(5.0 / 6.0)
            
            # Apply conversion factor to this channel
            converted_signals[acn] = ambi_signals[acn] * factor
    
    return converted_signals


def batch_convert_directory(input_dir: str, output_dir: str,
                          file_type: str = 'mono',
                          recursive: bool = False,
                          options: Optional[ConversionOptions] = None,
                          position_func: Optional[Callable[[str], SphericalCoord]] = None) -> Dict[str, Any]:
    """
    Convert all audio files in a directory to SHAC format.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory for output SHAC files
        file_type: Type of input files ('mono', 'stereo', 'ambisonic')
        recursive: Whether to search subdirectories
        options: Conversion options
        position_func: Function that returns a position for each file path
        
    Returns:
        Dictionary with conversion information
    """
    # Use default options if none provided
    if options is None:
        options = ConversionOptions()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio files
    audio_files = []
    
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
                audio_files.append(os.path.join(input_dir, file))
    
    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Process each file
    successful = []
    failed = []
    
    for input_file in audio_files:
        try:
            # Generate output filename
            rel_path = os.path.relpath(input_file, input_dir)
            output_file = os.path.join(output_dir, 
                                    os.path.splitext(rel_path)[0] + '.shac')
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Get position for this file
            if position_func is not None:
                position = position_func(input_file)
            else:
                # Use default position with some variation based on the filename
                # This spreads sources around the listener for better spatial separation
                import hashlib
                file_hash = hashlib.md5(input_file.encode()).hexdigest()
                hash_val = int(file_hash[:8], 16) / (16**8)  # Value between 0 and 1
                
                # Use hash to generate azimuth (full circle)
                azimuth = hash_val * 2 * np.pi
                
                # Use second part of hash for elevation (limited range)
                hash_val2 = int(file_hash[8:16], 16) / (16**8)
                elevation = (hash_val2 * 60 - 30) * np.pi / 180  # -30° to +30°
                
                # Fixed distance
                distance = options.default_distance
                
                position = (azimuth, elevation, distance)
            
            # Convert file based on type
            if file_type == 'mono':
                result = mono_to_shac(input_file, output_file, position, options=options)
                successful.append(result)
                logger.info(f"Converted {input_file} to {output_file}")
            
            elif file_type == 'stereo':
                result = stereo_to_shac(input_file, output_file, position, options=options)
                successful.append(result)
                logger.info(f"Converted {input_file} to {output_file}")
            
            elif file_type == 'ambisonic':
                # For ambisonic files, we need to know the order
                # Here we assume 1st order, but this should be configurable
                input_order = 1
                result = ambisonic_to_shac(input_file, output_file, input_order, options=options)
                successful.append(result)
                logger.info(f"Converted {input_file} to {output_file}")
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
        except Exception as e:
            logger.error(f"Failed to convert {input_file}: {str(e)}")
            failed.append({
                'input_file': input_file,
                'error': str(e)
            })
    
    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'n_files': len(audio_files),
        'n_successful': len(successful),
        'n_failed': len(failed),
        'successful': successful,
        'failed': failed
    }


# Utility functions for command-line use

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SHAC Audio Format Converter')
    
    # Main arguments
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', help='Output file or directory')
    
    # Operation mode
    parser.add_argument('--mode', choices=['mono', 'stereo', 'ambisonic', 'batch', 'binaural'], 
                      default='mono', help='Conversion mode')
    
    # Common options
    parser.add_argument('--order', type=int, default=3, help='Ambisonic order')
    parser.add_argument('--sr', type=int, default=48000, help='Target sample rate')
    parser.add_argument('--bit-depth', type=int, default=32, choices=[16, 32], help='Bit depth')
    
    # Positional parameters
    parser.add_argument('--azimuth', type=float, help='Azimuth in degrees (0=front, 90=left)')
    parser.add_argument('--elevation', type=float, help='Elevation in degrees (0=horizon, 90=up)')
    parser.add_argument('--distance', type=float, default=2.0, help='Distance in meters')
    parser.add_argument('--width', type=float, default=30.0, help='Stereo width in degrees')
    
    # Ambisonic options
    parser.add_argument('--input-order', type=int, help='Input ambisonic order')
    parser.add_argument('--normalization', choices=['sn3d', 'n3d', 'fuma'], 
                      default='sn3d', help='Ambisonic normalization')
    
    # Batch options
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories for batch mode')
    
    # Binaural options
    parser.add_argument('--hrtf', help='HRTF file for binaural rendering')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw rotation in degrees')
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch rotation in degrees')
    parser.add_argument('--roll', type=float, default=0.0, help='Roll rotation in degrees')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create conversion options
    options = ConversionOptions(
        ambisonic_order=args.order,
        target_sample_rate=args.sr,
        bit_depth=args.bit_depth,
        default_distance=args.distance
    )
    
    # Set normalization
    if args.normalization == 'sn3d':
        norm = AmbisonicNormalization.SN3D
    elif args.normalization == 'n3d':
        norm = AmbisonicNormalization.N3D
    elif args.normalization == 'fuma':
        norm = AmbisonicNormalization.FUMA
    else:
        norm = AmbisonicNormalization.SN3D
    
    options.normalization = norm
    
    # Convert angles to radians
    azimuth_rad = np.radians(args.azimuth) if args.azimuth is not None else 0.0
    elevation_rad = np.radians(args.elevation) if args.elevation is not None else 0.0
    width_rad = np.radians(args.width) if args.width is not None else np.radians(30.0)
    
    # Create position tuple
    position = (azimuth_rad, elevation_rad, args.distance)
    
    try:
        # Perform conversion based on mode
        if args.mode == 'mono':
            result = mono_to_shac(args.input, args.output, position, options=options)
            logger.info(f"Converted mono file to SHAC: {args.output}")
        
        elif args.mode == 'stereo':
            result = stereo_to_shac(args.input, args.output, position, width_rad, options=options)
            logger.info(f"Converted stereo file to SHAC: {args.output}")
        
        elif args.mode == 'ambisonic':
            input_order = args.input_order if args.input_order is not None else args.order
            result = ambisonic_to_shac(args.input, args.output, input_order, norm, options=options)
            logger.info(f"Converted ambisonic file to SHAC: {args.output}")
        
        elif args.mode == 'batch':
            result = batch_convert_directory(args.input, args.output, 'mono', 
                                          args.recursive, options)
            logger.info(f"Batch conversion completed: {result['n_successful']} successful, "
                      f"{result['n_failed']} failed")
        
        elif args.mode == 'binaural':
            # Convert rotation angles to radians
            yaw_rad = np.radians(args.yaw)
            pitch_rad = np.radians(args.pitch)
            roll_rad = np.radians(args.roll)
            
            rotation = (yaw_rad, pitch_rad, roll_rad)
            result = shac_to_binaural(args.input, args.output, args.hrtf, rotation)
            logger.info(f"Converted SHAC file to binaural: {args.output}")
        
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()