"""
Core SHAC Codec Module

This module contains the main SHACCodec class that ties together all the
functionality provided by the other modules.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Union, Callable
import queue
import threading

from .math_utils import AmbisonicNormalization, real_spherical_harmonic
from .encoders import encode_mono_source, encode_stereo_source, convert_to_cartesian, convert_to_spherical
from .processors import rotate_ambisonics, decode_to_speakers
from .binauralizer import binauralize_ambisonics, load_hrtf_database, apply_frequency_dependent_effects
from .utils import SourceAttributes, RoomAttributes, BinauralRendererConfig, HRTFInterpolationMethod, AmbisonicOrdering
from .io import SHACFileWriter, SHACFileReader


def apply_early_reflections(ambi_signals: np.ndarray, source_position: Tuple[float, float, float],
                         room_dimensions: Tuple[float, float, float], reflection_coefficients: Dict[str, float],
                         sample_rate: int, max_order: int = 1) -> np.ndarray:
    """
    Add early reflections to ambisonic signals based on a simple shoebox room model.
    
    Args:
        ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
        source_position: Source position (x, y, z) in meters
        room_dimensions: Room dimensions (width, height, length) in meters
        reflection_coefficients: Dictionary of reflection coefficients for each surface
        sample_rate: Sample rate in Hz
        max_order: Maximum reflection order (1 = first-order reflections only)
        
    Returns:
        Ambisonic signals with early reflections, shape (n_channels, n_samples)
    """
    n_channels = ambi_signals.shape[0]
    n_samples = ambi_signals.shape[1]
    
    # Initialize output signals
    output_signals = np.copy(ambi_signals)
    
    # Define room surfaces
    surfaces = {
        'floor': {'normal': (0, 1, 0), 'distance': 0},
        'ceiling': {'normal': (0, -1, 0), 'distance': room_dimensions[1]},
        'left': {'normal': (1, 0, 0), 'distance': 0},
        'right': {'normal': (-1, 0, 0), 'distance': room_dimensions[0]},
        'front': {'normal': (0, 0, 1), 'distance': 0},
        'back': {'normal': (0, 0, -1), 'distance': room_dimensions[2]}
    }
    
    # Extract source coordinates
    sx, sy, sz = source_position
    
    # Speed of sound in m/s
    c = 343.0
    
    # For each reflection order
    for order in range(1, max_order + 1):
        # Calculate all possible reflection paths of this order
        # For simplicity, we'll just enumerate all surface combinations
        surface_combinations = []
        if order == 1:
            surface_combinations = [(surface,) for surface in surfaces.keys()]
        else:
            # Generate combinations for higher orders
            # This could be made more efficient
            pass
        
        # Process each reflection path
        for path in surface_combinations:
            # Calculate reflection attenuation and delay
            attenuation = 1.0
            image_x, image_y, image_z = sx, sy, sz
            
            # Calculate image source position
            for surface_name in path:
                surface = surfaces[surface_name]
                normal = surface['normal']
                distance = surface['distance']
                
                # Apply reflection
                if normal[0] != 0:
                    image_x = 2 * distance - image_x if normal[0] < 0 else -image_x
                if normal[1] != 0:
                    image_y = 2 * distance - image_y if normal[1] < 0 else -image_y
                if normal[2] != 0:
                    image_z = 2 * distance - image_z if normal[2] < 0 else -image_z
                
                # Apply surface reflection coefficient
                attenuation *= reflection_coefficients.get(surface_name, 0.5)
            
            # Calculate distance to image source
            listener_position = (0, 0, 0)  # Assumes listener at origin
            lx, ly, lz = listener_position
            distance = math.sqrt((image_x - lx)**2 + (image_y - ly)**2 + (image_z - lz)**2)
            
            # Calculate delay in samples
            delay_samples = int(distance / c * sample_rate)
            
            # Apply distance attenuation
            attenuation *= 1.0 / max(1.0, distance)
            
            # Apply early reflection
            if delay_samples < n_samples:
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


class SHACCodec:
    """
    Spherical Harmonic Audio Codec (SHAC) main class.
    
    This class provides a complete API for encoding, processing, and decoding
    spatial audio using spherical harmonics.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000,
               normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D,
               channel_ordering: AmbisonicOrdering = AmbisonicOrdering.ACN):
        """
        Initialize the SHAC codec.
        
        Args:
            order: Ambisonic order (higher = more spatial resolution)
            sample_rate: Audio sample rate in Hz
            normalization: Spherical harmonic normalization convention
            channel_ordering: Channel ordering convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.normalization = normalization
        self.channel_ordering = channel_ordering
        
        # Calculate number of channels
        self.n_channels = (order + 1) ** 2
        
        # Initialize storage for sources and layers
        self.sources = {}  # Raw mono sources
        self.layers = {}   # Encoded ambisonic layers
        self.layer_metadata = {}  # Metadata for layers
        
        # Processing parameters
        self.hrtf_database = None
        self.binaural_renderer = None
        self.room = None
        
        # Initialize with default listener orientation
        self.listener_orientation = (0.0, 0.0, 0.0)  # yaw, pitch, roll
        
        # Frame size for real-time processing
        self.frame_size = 1024
        
        # For real-time processing
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        print(f"SHAC Codec initialized: order {order}, {self.n_channels} channels")
    
    def add_mono_source(self, source_id: str, audio: np.ndarray, position: Tuple[float, float, float],
                      attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a mono source to the SHAC encoder.
        
        Args:
            source_id: Unique identifier for the source
            audio: Mono audio signal, shape (n_samples,)
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if audio.ndim != 1:
            raise ValueError("Audio must be mono (1-dimensional)")
        
        # Store the source
        self.sources[source_id] = {
            'audio': audio,
            'position': position,
            'attributes': attributes if attributes is not None else SourceAttributes(position)
        }
        
        # Encode to ambisonics
        ambi_signals = encode_mono_source(audio, position, self.order, self.normalization)
        
        # Store as a layer
        self.layers[source_id] = ambi_signals
        self.layer_metadata[source_id] = {
            'type': 'mono_source',
            'position': position,
            'muted': False,
            'gain': 1.0,
            'current_gain': 1.0,  # For smooth gain changes
            'attributes': attributes
        }
    
    def add_stereo_source(self, source_id: str, left_audio: np.ndarray, right_audio: np.ndarray,
                        position: Tuple[float, float, float], width: float = 0.35) -> None:
        """
        Add a stereo source to the SHAC encoder.
        
        Args:
            source_id: Unique identifier for the source
            left_audio: Left channel audio, shape (n_samples,)
            right_audio: Right channel audio, shape (n_samples,)
            position: (azimuth, elevation, distance) in radians and meters
            width: Angular width of the stereo field in radians
        """
        if left_audio.ndim != 1 or right_audio.ndim != 1:
            raise ValueError("Audio channels must be mono (1-dimensional)")
        
        if len(left_audio) != len(right_audio):
            raise ValueError("Left and right channels must have the same length")
        
        # Store the source
        self.sources[source_id] = {
            'left_audio': left_audio,
            'right_audio': right_audio,
            'position': position,
            'width': width
        }
        
        # Encode to ambisonics
        ambi_signals = encode_stereo_source(left_audio, right_audio, position, width, self.order, self.normalization)
        
        # Store as a layer
        self.layers[source_id] = ambi_signals
        self.layer_metadata[source_id] = {
            'type': 'stereo_source',
            'position': position,
            'width': width,
            'muted': False,
            'gain': 1.0,
            'current_gain': 1.0
        }
    
    def add_ambisonic_layer(self, layer_id: str, ambi_signals: np.ndarray, metadata: Dict = None) -> None:
        """
        Add an existing ambisonic layer to the SHAC encoder.
        
        Args:
            layer_id: Unique identifier for the layer
            ambi_signals: Ambisonic signals, shape (n_channels, n_samples)
            metadata: Optional metadata for the layer
        """
        if ambi_signals.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {ambi_signals.shape[0]}")
        
        # Store the layer
        self.layers[layer_id] = ambi_signals
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {'type': 'ambisonic_layer'}
        
        # Add standard metadata fields if not present
        if 'muted' not in metadata:
            metadata['muted'] = False
        if 'gain' not in metadata:
            metadata['gain'] = 1.0
        if 'current_gain' not in metadata:
            metadata['current_gain'] = 1.0
        
        self.layer_metadata[layer_id] = metadata
    
    def update_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Source identifier
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            raise ValueError(f"Source not found: {source_id}")
        
        # Update the source position
        self.sources[source_id]['position'] = position
        
        # Update layer metadata
        if source_id in self.layer_metadata:
            self.layer_metadata[source_id]['position'] = position
        
        # Re-encode if it's a mono or stereo source
        source_data = self.sources[source_id]
        
        if 'audio' in source_data:
            # Mono source
            ambi_signals = encode_mono_source(source_data['audio'], position, self.order, self.normalization)
            self.layers[source_id] = ambi_signals
        
        elif 'left_audio' in source_data and 'right_audio' in source_data:
            # Stereo source
            width = source_data.get('width', 0.35)
            ambi_signals = encode_stereo_source(
                source_data['left_audio'], source_data['right_audio'], 
                position, width, self.order, self.normalization
            )
            self.layers[source_id] = ambi_signals
    
    def update_listener_orientation(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Update the listener's orientation.
        
        Args:
            yaw: Rotation around vertical axis (positive = left) in radians
            pitch: Rotation around side axis (positive = up) in radians
            roll: Rotation around front axis (positive = tilt right) in radians
        """
        self.listener_orientation = (yaw, pitch, roll)
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source or layer.
        
        Args:
            source_id: Source or layer identifier
            gain: Gain value (1.0 = unity gain)
        """
        if source_id in self.layer_metadata:
            self.layer_metadata[source_id]['gain'] = gain
            # For smooth transitions, actual gain change happens in process()
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source or layer.
        
        Args:
            source_id: Source or layer identifier
            muted: Whether to mute (True) or unmute (False)
        """
        if source_id in self.layer_metadata:
            self.layer_metadata[source_id]['muted'] = muted
    
    def set_room_model(self, room_dimensions: Tuple[float, float, float], 
                     reflection_coefficients: Dict[str, float], rt60: float) -> None:
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
    
    def set_listener_rotation(self, rotation: Tuple[float, float, float]) -> None:
        """
        Set the listener's head rotation.
        
        Args:
            rotation: Tuple of (yaw, pitch, roll) in radians
        """
        self.listener_orientation = rotation
    
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
            layer_metadata = reader.get_layer_metadata(layer_name)
            
            self.add_ambisonic_layer(layer_name, layer_audio, layer_metadata)
    
    def start_realtime_processing(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Start real-time processing thread.
        
        Args:
            callback: Function to call with processed audio frames
        """
        if self.running:
            return
        
        self.running = True
        
        def processing_loop():
            while self.running:
                try:
                    # Get input frame from queue
                    frame_data = self.processing_queue.get(timeout=0.1)
                    
                    if frame_data is None:
                        continue
                    
                    # Process the frame
                    output_frame = self.process_frame(frame_data)
                    
                    # Call the callback
                    callback(output_frame)
                    
                except queue.Empty:
                    # No input data available
                    pass
                except Exception as e:
                    print(f"Error in processing thread: {e}")
        
        self.processing_thread = threading.Thread(target=processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_realtime_processing(self) -> None:
        """Stop the real-time processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
    
    def process_frame(self, frame_data: Dict) -> np.ndarray:
        """Process a single frame of audio data.
        
        Args:
            frame_data: Dictionary with input data
            
        Returns:
            Processed audio frame
        """
        # Example implementation - this would vary based on your real-time needs
        output_frame = np.zeros((self.n_channels, self.frame_size))
        
        # Process source updates
        for source_id, source_data in frame_data.get('sources', {}).items():
            if source_id in self.sources:
                if 'position' in source_data:
                    self.update_source_position(source_id, source_data['position'])
                if 'gain' in source_data:
                    self.set_source_gain(source_id, source_data['gain'])
                if 'muted' in source_data:
                    self.mute_source(source_id, source_data['muted'])
        
        # Process listener updates
        listener_data = frame_data.get('listener', {})
        if 'orientation' in listener_data:
            yaw, pitch, roll = listener_data['orientation']
            self.update_listener_orientation(yaw, pitch, roll)
        
        # Process audio data
        if 'audio_data' in frame_data:
            # Actual processing would go here
            # This is just a placeholder
            pass
        
        return output_frame