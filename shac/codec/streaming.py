"""
Real-time Streaming Module

This module contains the SHACStreamProcessor class for real-time
streaming audio processing with minimal latency.
"""

import numpy as np
import threading
import queue
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable

from .core import SHACCodec
from .math_utils import AmbisonicNormalization
from .encoders import encode_mono_source
from .processors import rotate_ambisonics
from .binauralizer import apply_frequency_dependent_effects
from .utils import SourceAttributes


class SHACStreamProcessor:
    """
    Real-time streaming processor for SHAC audio.
    
    This class handles real-time processing of spatial audio streams
    with minimal latency and efficient CPU usage.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, buffer_size: int = 1024,
                 max_sources: int = 32):
        """
        Initialize the SHAC stream processor.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            buffer_size: Processing buffer size in samples
            max_sources: Maximum number of simultaneous sources
        """
        self.order = order
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        self.n_channels = (order + 1) ** 2
        
        # Create the main codec
        self.codec = SHACCodec(order, sample_rate)
        
        # Initialize streaming state
        self.sources = {}
        self.source_buffers = {}
        self.output_buffer = np.zeros((self.n_channels, buffer_size))
        
        # Initialize processing thread
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
    
    def start(self) -> None:
        """Start the real-time processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self) -> None:
        """Stop the real-time processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def add_source(self, source_id: str, position: Tuple[float, float, float],
                  attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a streaming source.
        
        Args:
            source_id: Unique identifier for the source
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if len(self.sources) >= self.max_sources:
            raise ValueError(f"Maximum number of sources ({self.max_sources}) reached")
        
        self.sources[source_id] = {
            'position': position,
            'attributes': attributes or SourceAttributes(position),
            'gain': 1.0,
            'muted': False
        }
        
        # Initialize source buffer
        self.source_buffers[source_id] = np.zeros(self.buffer_size)
    
    def remove_source(self, source_id: str) -> None:
        """
        Remove a streaming source.
        
        Args:
            source_id: Identifier of the source to remove
        """
        if source_id in self.sources:
            del self.sources[source_id]
        
        if source_id in self.source_buffers:
            del self.source_buffers[source_id]
    
    def update_source(self, source_id: str, audio_chunk: np.ndarray) -> None:
        """
        Update a source with new audio data.
        
        Args:
            source_id: Identifier of the source to update
            audio_chunk: New audio data chunk
        """
        if source_id not in self.sources:
            return
        
        # Copy audio data to source buffer
        n_samples = min(len(audio_chunk), self.buffer_size)
        self.source_buffers[source_id][:n_samples] = audio_chunk[:n_samples]
        
        # If audio chunk is smaller than buffer, zero-pad
        if n_samples < self.buffer_size:
            self.source_buffers[source_id][n_samples:] = 0.0
    
    def update_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['position'] = position
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['gain'] = gain
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source.
        
        Args:
            source_id: Identifier of the source to update
            muted: True to mute, False to unmute
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['muted'] = muted
    
    def set_listener_rotation(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Set the listener's head rotation.
        
        Args:
            yaw: Rotation around vertical axis in radians
            pitch: Rotation around side axis in radians
            roll: Rotation around front axis in radians
        """
        # Store for next processing cycle
        self.codec.listener_rotation = (yaw, pitch, roll)
    
    def process_block(self) -> np.ndarray:
        """
        Process a block of audio.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, buffer_size)
        """
        # Initialize output buffer
        output_buffer = np.zeros((self.n_channels, self.buffer_size))
        
        # Process each source
        for source_id, source_info in self.sources.items():
            # Skip muted sources
            if source_info['muted']:
                continue
            
            # Get source parameters
            position = source_info['position']
            attributes = source_info['attributes']
            gain = source_info['gain']
            
            # Get audio data
            audio = self.source_buffers[source_id]
            
            # Encode to ambisonics
            ambi_source = encode_mono_source(audio, position, self.order, AmbisonicNormalization.SN3D)
            
            # Apply source attributes if needed
            if attributes and (attributes.directivity > 0 or attributes.width > 0):
                source_directivity = {
                    'pattern': 'cardioid',
                    'order': attributes.directivity,
                    'axis': attributes.directivity_axis,
                    'frequency_dependent': True
                }
                
                ambi_source = apply_frequency_dependent_effects(ambi_source, self.sample_rate,
                                                              position[2], attributes.air_absorption,
                                                              source_directivity)
            
            # Apply gain
            ambi_source *= gain
            
            # Mix into output buffer
            output_buffer += ambi_source
        
        # Apply room effects if configured
        if hasattr(self.codec, 'room') and self.codec.room is not None:
            # Add simple diffuse reverberation (simplified for real-time processing)
            # In a full implementation, this would use convolution with pre-computed IRs
            pass
        
        # Apply rotation if needed
        if hasattr(self.codec, 'listener_rotation'):
            yaw, pitch, roll = self.codec.listener_rotation
            output_buffer = rotate_ambisonics(output_buffer, yaw, pitch, roll)
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0.99:
            output_buffer = output_buffer * 0.99 / max_val
        
        return output_buffer
    
    def get_binaural_output(self) -> np.ndarray:
        """
        Get the current block as binaural stereo.
        
        Returns:
            Binaural stereo signals, shape (2, buffer_size)
        """
        # Process the current block
        ambi_block = self.process_block()
        
        # Convert to binaural
        return self.codec.binauralize(ambi_block)
    
    def _processing_loop(self) -> None:
        """Main processing loop for the streaming thread."""
        while self.running:
            # Process the current block
            try:
                # Process the block
                ambi_block = self.process_block()
                
                # Binauralize if needed
                if hasattr(self.codec, 'binaural_renderer') and self.codec.binaural_renderer is not None:
                    binaural_block = self.codec.binauralize(ambi_block)
                    
                    # Add to output queue
                    self.output_queue.put(binaural_block)
                else:
                    # Add ambisonic block to output queue
                    self.output_queue.put(ambi_block)
                
                # Sleep for a bit to avoid burning CPU
                # In a real-time audio system, this would be synchronized with audio callbacks
                time.sleep(self.buffer_size / self.sample_rate / 2)
            
            except Exception as e:
                warnings.warn(f"Error in processing loop: {e}")
                time.sleep(0.1)