"""
Real-time Streaming Module (Optimized)

This module contains an optimized SHACStreamProcessor class for real-time
streaming audio processing with minimal latency, efficient memory usage,
and high performance.
"""

import numpy as np
import threading
import queue
import time
import warnings
import collections
from typing import Dict, List, Tuple, Optional, Union, Callable, Deque

from .core import SHACCodec
from .math_utils import AmbisonicNormalization
from .encoders import encode_mono_source
from .processors import rotate_ambisonics
from .binauralizer import apply_frequency_dependent_effects
from .utils import SourceAttributes


class BufferPool:
    """
    A memory pool for efficient buffer reuse to avoid frequent allocations and garbage collection.
    
    This helps reduce memory fragmentation and CPU usage during real-time processing.
    """
    
    def __init__(self, buffer_shape: Tuple[int, ...], max_buffers: int = 32, dtype=np.float32):
        """
        Initialize the buffer pool.
        
        Args:
            buffer_shape: Shape of each buffer
            max_buffers: Maximum number of buffers to keep in the pool
            dtype: NumPy data type for the buffers
        """
        self.buffer_shape = buffer_shape
        self.max_buffers = max_buffers
        self.dtype = dtype
        self.free_buffers: Deque[np.ndarray] = collections.deque(maxlen=max_buffers)
        self.active_buffers = 0
    
    def get(self) -> np.ndarray:
        """
        Get a buffer from the pool or create a new one if none available.
        
        Returns:
            A numpy array with the configured shape and dtype
        """
        if self.free_buffers:
            buf = self.free_buffers.popleft()
            # Zero out the buffer for safety
            buf.fill(0)
        else:
            buf = np.zeros(self.buffer_shape, dtype=self.dtype)
            self.active_buffers += 1
        
        return buf
    
    def release(self, buf: np.ndarray) -> None:
        """
        Return a buffer to the pool for reuse.
        
        Args:
            buf: Buffer to return to the pool
        """
        if buf.shape != self.buffer_shape or buf.dtype != self.dtype:
            warnings.warn(f"Buffer with incorrect shape or dtype returned to pool: {buf.shape}, {buf.dtype}")
            return
        
        if len(self.free_buffers) < self.max_buffers:
            self.free_buffers.append(buf)
    
    def stats(self) -> Dict[str, int]:
        """
        Get statistics about buffer usage.
        
        Returns:
            Dictionary with buffer usage statistics
        """
        return {
            'active_buffers': self.active_buffers,
            'free_buffers': len(self.free_buffers),
            'total_buffers': self.active_buffers,
            'pool_capacity': self.max_buffers
        }


class SHACStreamProcessor:
    """
    Real-time streaming processor for SHAC audio.
    
    This class handles real-time processing of spatial audio streams
    with minimal latency and efficient CPU usage through buffer pooling,
    vectorized operations, and adaptive timing.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, buffer_size: int = 1024,
                 max_sources: int = 32, pool_size: int = 64):
        """
        Initialize the SHAC stream processor.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            buffer_size: Processing buffer size in samples
            max_sources: Maximum number of simultaneous sources
            pool_size: Size of the buffer memory pool
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
        self.source_positions_cache = {}  # Cache for position calculations
        
        # Create buffer pools for efficient memory reuse
        self.mono_buffer_pool = BufferPool((buffer_size,), pool_size)
        self.ambi_buffer_pool = BufferPool((self.n_channels, buffer_size), pool_size)
        self.output_buffer_pool = BufferPool((2, buffer_size), pool_size)  # For binaural output
        
        # Source buffers will be managed through the mono_buffer_pool
        self.source_buffers = {}
        
        # Output buffer (reused from pool during processing)
        self.output_buffer = self.ambi_buffer_pool.get()
        
        # Performance monitoring
        self.process_times = collections.deque(maxlen=100)  # Track processing times
        self.average_process_time = 0.01  # Start with a reasonable default
        self.target_cpu_usage = 0.7  # Target CPU usage (fraction of available time)
        
        # Initialize processing thread
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=4)  # Limit queue size to prevent backlog
        self.running = False
        self.processing_thread = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
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
            
        # Return all buffers to the pools
        with self.lock:
            # Clear source buffers
            for source_id, buffer in self.source_buffers.items():
                self.mono_buffer_pool.release(buffer)
            self.source_buffers.clear()
            
            # Return the output buffer
            self.ambi_buffer_pool.release(self.output_buffer)
    
    def add_source(self, source_id: str, position: Tuple[float, float, float],
                  attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a streaming source.
        
        Args:
            source_id: Unique identifier for the source
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        with self.lock:
            if len(self.sources) >= self.max_sources:
                raise ValueError(f"Maximum number of sources ({self.max_sources}) reached")
            
            # Create the source data
            self.sources[source_id] = {
                'position': position,
                'attributes': attributes or SourceAttributes(position),
                'gain': 1.0,
                'muted': False,
                'last_update': time.time()  # Track when the source was last updated
            }
            
            # Cache the position for efficient encoding
            self._update_position_cache(source_id, position)
            
            # Initialize source buffer from the pool
            self.source_buffers[source_id] = self.mono_buffer_pool.get()
    
    def remove_source(self, source_id: str) -> None:
        """
        Remove a streaming source.
        
        Args:
            source_id: Identifier of the source to remove
        """
        with self.lock:
            if source_id in self.sources:
                del self.sources[source_id]
            
            if source_id in self.source_positions_cache:
                del self.source_positions_cache[source_id]
            
            # Return the buffer to the pool
            if source_id in self.source_buffers:
                self.mono_buffer_pool.release(self.source_buffers[source_id])
                del self.source_buffers[source_id]
    
    def update_source(self, source_id: str, audio_chunk: np.ndarray) -> None:
        """
        Update a source with new audio data.
        
        Args:
            source_id: Identifier of the source to update
            audio_chunk: New audio data chunk
        """
        with self.lock:
            if source_id not in self.sources:
                return
            
            # Update the last update timestamp
            self.sources[source_id]['last_update'] = time.time()
            
            # Get the source buffer
            source_buffer = self.source_buffers[source_id]
            
            # Copy audio data to source buffer efficiently
            n_samples = min(len(audio_chunk), self.buffer_size)
            source_buffer[:n_samples] = audio_chunk[:n_samples]
            
            # If audio chunk is smaller than buffer, zero-pad
            if n_samples < self.buffer_size:
                source_buffer[n_samples:] = 0.0
    
    def _update_position_cache(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position cache for a source.
        
        This pre-computes values needed for encoding to avoid recalculating them each frame.
        
        Args:
            source_id: Source ID
            position: (azimuth, elevation, distance) in radians and meters
        """
        # Here we would pre-compute any values needed for the encoding process
        # For example, spherical harmonics values for this position
        self.source_positions_cache[source_id] = {
            'position': position,
            # Additional pre-computed values could be stored here
        }
    
    def update_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        with self.lock:
            if source_id not in self.sources:
                return
            
            # Update the source position
            self.sources[source_id]['position'] = position
            
            # Update the position cache
            self._update_position_cache(source_id, position)
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        with self.lock:
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
        with self.lock:
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
        start_time = time.time()
        
        # Get an output buffer from the pool
        output_buffer = self.ambi_buffer_pool.get()
        output_buffer.fill(0)  # Ensure it's zeroed
        
        with self.lock:
            # Process each source using vectorized operations where possible
            active_sources = [(source_id, source_info) for source_id, source_info in self.sources.items() 
                              if not source_info['muted']]
            
            # Process sources in batches for efficiency
            for source_id, source_info in active_sources:
                # Get source parameters
                position = source_info['position']
                attributes = source_info['attributes']
                gain = source_info['gain']
                
                # Get audio data
                audio = self.source_buffers[source_id]
                
                # Encode to ambisonics (using precomputed values when possible)
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
                
                # Mix into output buffer using vectorized addition
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
        
        # Normalize if necessary using vectorized operations
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0.99:
            # Use in-place multiplication for efficiency
            output_buffer *= 0.99 / max_val
        
        # Track processing time for adaptive timing
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        self.average_process_time = sum(self.process_times) / len(self.process_times)
        
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
                # Measure overall processing time
                start_time = time.time()
                
                # Get fresh buffers from the pool
                ambi_block = self.process_block()
                
                # Binauralize if needed
                if hasattr(self.codec, 'binaural_renderer') and self.codec.binaural_renderer is not None:
                    binaural_block = self.codec.binauralize(ambi_block)
                    output_block = binaural_block
                else:
                    # Use ambisonic block as output
                    output_block = ambi_block
                
                # Return the ambisonic buffer to the pool
                self.ambi_buffer_pool.release(ambi_block)
                
                # Add to output queue if there's room (non-blocking)
                try:
                    self.output_queue.put_nowait(output_block)
                except queue.Full:
                    # If queue is full, just drop this block and release the buffer
                    if isinstance(output_block, np.ndarray):
                        if output_block.shape[0] == 2:  # Binaural
                            self.output_buffer_pool.release(output_block)
                        else:  # Ambisonic
                            self.ambi_buffer_pool.release(output_block)
                
                # Adaptive sleep time calculation
                total_time = time.time() - start_time
                target_time = self.buffer_size / self.sample_rate
                
                # Sleep for the remaining time, targeting our CPU usage goal
                remaining_time = target_time - total_time
                if remaining_time > 0:
                    # Scale sleep time by target CPU usage (higher usage = less sleep)
                    sleep_time = remaining_time * (1.0 - self.target_cpu_usage)
                    time.sleep(max(0.001, sleep_time))  # Minimum sleep to avoid busy-waiting
            
            except Exception as e:
                warnings.warn(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def get_next_output_block(self) -> Optional[np.ndarray]:
        """
        Get the next output block from the queue.
        
        Returns:
            Next output block or None if queue is empty
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Union[float, int, Dict]]:
        """
        Get performance statistics for monitoring.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'average_process_time': self.average_process_time,
            'target_frame_time': self.buffer_size / self.sample_rate,
            'cpu_usage': self.average_process_time / (self.buffer_size / self.sample_rate),
            'active_sources': len(self.sources),
            'output_queue_size': self.output_queue.qsize(),
            'buffer_pools': {
                'mono': self.mono_buffer_pool.stats(),
                'ambi': self.ambi_buffer_pool.stats(),
                'output': self.output_buffer_pool.stats()
            }
        }
"""