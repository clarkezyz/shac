"""
SHAC Codec - Minimal implementation for demonstration purposes

This is a simplified version of the SHAC codec with just enough functionality
to make the demo work. For the full implementation, refer to shac_codec.py.
"""

import numpy as np
import time

class SHACCodec:
    """Simplified SHAC codec for demonstration"""
    
    def __init__(self, order=3, sample_rate=48000, normalization=None, ordering=None):
        """
        Initialize the SHAC codec
        
        Parameters:
        - order: Ambisonic order
        - sample_rate: Sample rate in Hz
        - normalization: Normalization convention (ignored in minimal implementation)
        - ordering: Channel ordering convention (ignored in minimal implementation)
        """
        self.order = order
        self.sample_rate = sample_rate
        self.n_channels = (order + 1) ** 2
        
        # Initialize processing state
        self.sources = {}
        self.listener_position = (0.0, 0.0, 0.0)
        self.listener_rotation = (0.0, 0.0, 0.0)
        
        print(f"Initialized SHAC codec (minimal) with order {order} ({self.n_channels} channels)")
    
    def add_mono_source(self, source_id, audio, position, directivity=0.0, directivity_axis=None):
        """
        Add a mono audio source to the codec
        
        Parameters:
        - source_id: Unique identifier for the source
        - audio: Mono audio signal
        - position: (azimuth, elevation, distance) in radians and meters
        - directivity: Directivity factor (0.0 = omnidirectional, 1.0 = cardioid)
        - directivity_axis: Direction of maximum radiation (defaults to source direction)
        """
        # Ensure audio is mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * 0.99 / max_val
        
        # Set default directivity axis if not provided
        if directivity_axis is None:
            directivity_axis = position
        
        # Store source data
        self.sources[source_id] = {
            'audio': audio,
            'position': position,
            'directivity': directivity,
            'directivity_axis': directivity_axis,
            'gain': 1.0,
            'muted': False
        }
        
        print(f"Added source: {source_id} at position {position}")
    
    def set_source_position(self, source_id, position):
        """
        Update the position of a source
        
        Parameters:
        - source_id: Identifier of the source to update
        - position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update position
        self.sources[source_id]['position'] = position
    
    def set_source_gain(self, source_id, gain):
        """
        Set the gain for a source
        
        Parameters:
        - source_id: Identifier of the source to update
        - gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update gain
        self.sources[source_id]['gain'] = gain
    
    def mute_source(self, source_id, muted=True):
        """
        Mute or unmute a source
        
        Parameters:
        - source_id: Identifier of the source to update
        - muted: True to mute, False to unmute
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update mute state
        self.sources[source_id]['muted'] = muted
    
    def set_listener_position(self, position):
        """
        Set the listener position
        
        Parameters:
        - position: (x, y, z) in meters
        """
        self.listener_position = position
    
    def set_listener_rotation(self, rotation):
        """
        Set the listener rotation
        
        Parameters:
        - rotation: (yaw, pitch, roll) in radians
        """
        self.listener_rotation = rotation
    
    def process(self):
        """
        Process all sources to create the final ambisonic signals
        
        Returns:
        - Processed ambisonic signals, shape (n_channels, n_samples)
        """
        # Find the maximum number of samples across all sources
        max_samples = 0
        for source_id, source_data in self.sources.items():
            if not source_data['muted']:
                max_samples = max(max_samples, len(source_data['audio']))
        
        if max_samples == 0:
            return np.zeros((self.n_channels, 0))
        
        # Initialize output signals (ambisonic channels)
        output_signals = np.zeros((self.n_channels, max_samples))
        
        # In a real implementation, we would process each source through spherical harmonics
        # For this minimal demo, we'll just distribute the audio across channels based on position
        
        # Process each source
        for source_id, source_data in self.sources.items():
            # Skip muted sources
            if source_data['muted']:
                continue
            
            # Get source parameters
            audio = source_data['audio']
            azimuth, elevation, distance = source_data['position']
            gain = source_data['gain']
            
            # Apply distance attenuation
            if distance < 1.0:
                distance = 1.0  # Prevent division by zero
            
            distance_gain = 1.0 / distance
            attenuated_audio = audio * distance_gain * gain
            
            # Number of samples to mix
            n_samples = min(len(attenuated_audio), output_signals.shape[1])
            
            # Mix into output channels (simplified distribution)
            # We'll use a very simplified approach for this demo
            
            # W channel (omnidirectional) always gets the signal
            output_signals[0, :n_samples] += attenuated_audio[:n_samples] * 0.7
            
            # Directional channels
            if self.n_channels > 1:
                # Front-back (Y) - channel 1
                front_back = np.cos(azimuth) * np.cos(elevation)
                output_signals[1, :n_samples] += attenuated_audio[:n_samples] * front_back * 0.5
                
                # Up-down (Z) - channel 2
                up_down = np.sin(elevation)
                output_signals[2, :n_samples] += attenuated_audio[:n_samples] * up_down * 0.5
                
                # Left-right (X) - channel 3
                left_right = np.sin(azimuth) * np.cos(elevation)
                output_signals[3, :n_samples] += attenuated_audio[:n_samples] * left_right * 0.5
                
                # Higher order channels get progressively smaller contributions
                for i in range(4, self.n_channels):
                    # Just distribute some energy to higher channels for demonstration
                    output_signals[i, :n_samples] += attenuated_audio[:n_samples] * 0.2 / i
        
        # Apply head rotation (simplified)
        # In a real implementation, we would apply a proper rotation matrix
        # For this demo, we just slightly adjust the mix between channels
        
        yaw, pitch, roll = self.listener_rotation
        
        if abs(yaw) > 0.01 or abs(pitch) > 0.01 or abs(roll) > 0.01:
            # Very simplified rotation effect - just for demonstration
            if self.n_channels > 3:
                # Adjust left-right balance based on yaw
                yaw_factor = np.sin(yaw)
                output_signals[1, :] = output_signals[1, :] * np.cos(yaw)
                output_signals[3, :] = output_signals[3, :] - yaw_factor * output_signals[1, :]
                
                # Adjust up-down balance based on pitch
                pitch_factor = np.sin(pitch)
                output_signals[2, :] = output_signals[2, :] + pitch_factor * output_signals[1, :]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_signals))
        if max_val > 0.99:
            output_signals = output_signals * 0.99 / max_val
        
        return output_signals
    
    def binauralize(self, ambi_signals):
        """
        Convert ambisonic signals to binaural stereo
        
        Parameters:
        - ambi_signals: Ambisonic signals to binauralize
        
        Returns:
        - Binaural stereo signals, shape (2, n_samples)
        """
        if ambi_signals.shape[0] == 0 or ambi_signals.shape[1] == 0:
            return np.zeros((2, 0))
        
        n_channels = ambi_signals.shape[0]
        n_samples = ambi_signals.shape[1]
        
        # Initialize binaural output
        binaural = np.zeros((2, n_samples))
        
        # Extremely simplified binaural rendering for demonstration
        # In a real implementation, we would use proper HRTFs
        
        # W channel (omnidirectional) contributes to both ears equally
        binaural[0, :] += 0.7071 * ambi_signals[0, :]  # Left
        binaural[1, :] += 0.7071 * ambi_signals[0, :]  # Right
        
        if n_channels > 1:
            # Y channel (front-back)
            binaural[0, :] += 0.5 * ambi_signals[1, :]  # Left
            binaural[1, :] += 0.5 * ambi_signals[1, :]  # Right
            
            # Z channel (up-down)
            binaural[0, :] += 0.0 * ambi_signals[2, :]  # Left (neutral)
            binaural[1, :] += 0.0 * ambi_signals[2, :]  # Right (neutral)
            
            # X channel (left-right)
            binaural[0, :] += -0.5 * ambi_signals[3, :]  # Left (negative)
            binaural[1, :] += 0.5 * ambi_signals[3, :]   # Right (positive)
        
        # Higher order components would have more sophisticated processing
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(binaural))
        if max_val > 0.99:
            binaural = binaural * 0.99 / max_val
        
        return binaural


class SHACStreamProcessor:
    """
    Real-time stream processor for SHAC audio.
    Simplified version for the demo.
    """
    
    def __init__(self, order=3, sample_rate=48000, buffer_size=1024):
        """
        Initialize the stream processor
        
        Parameters:
        - order: Ambisonic order
        - sample_rate: Sample rate in Hz
        - buffer_size: Processing buffer size in samples
        """
        self.order = order
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.n_channels = (order + 1) ** 2
        
        # Create core codec
        self.codec = SHACCodec(order, sample_rate)
        
        # Initialize streaming state
        self.sources = {}
        self.source_buffers = {}
        self.output_buffer = np.zeros((self.n_channels, buffer_size))
        
        # Processing parameters
        self.listener_rotation = (0.0, 0.0, 0.0)
        
        print(f"Initialized SHAC stream processor with buffer size {buffer_size}")
    
    def add_source(self, source_id, position, directivity=0.0, directivity_axis=None):
        """
        Add a streaming source
        
        Parameters:
        - source_id: Unique identifier for the source
        - position: (azimuth, elevation, distance) in radians and meters
        - directivity: Directivity factor (0.0 = omnidirectional, 1.0 = cardioid)
        - directivity_axis: Direction of maximum radiation (defaults to source direction)
        """
        # Set default directivity axis if not provided
        if directivity_axis is None:
            directivity_axis = position
        
        # Store source information
        self.sources[source_id] = {
            'position': position,
            'directivity': directivity,
            'directivity_axis': directivity_axis,
            'gain': 1.0,
            'muted': False
        }
        
        # Initialize source buffer
        self.source_buffers[source_id] = np.zeros(self.buffer_size)
    
    def update_source(self, source_id, audio_chunk):
        """
        Update a source with new audio data
        
        Parameters:
        - source_id: Identifier of the source to update
        - audio_chunk: New audio data chunk
        """
        if source_id not in self.sources:
            return
        
        # Ensure audio is mono
        if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # Copy audio data to source buffer
        n_samples = min(len(audio_chunk), self.buffer_size)
        self.source_buffers[source_id][:n_samples] = audio_chunk[:n_samples]
        
        # If audio chunk is smaller than buffer, zero-pad
        if n_samples < self.buffer_size:
            self.source_buffers[source_id][n_samples:] = 0.0
    
    def process_block(self):
        """
        Process a block of audio
        
        Returns:
        - Processed ambisonic signals, shape (n_channels, buffer_size)
        """
        # Initialize output buffer
        output_buffer = np.zeros((self.n_channels, self.buffer_size))
        
        # Process each source (simplified for the demo)
        for source_id, source_info in self.sources.items():
            # Skip muted sources
            if source_info['muted']:
                continue
            
            # Get parameters
            position = source_info['position']
            gain = source_info['gain']
            
            # Get audio data
            audio = self.source_buffers[source_id]
            
            # Apply gain
            audio = audio * gain
            
            # Mix into output buffer (very simplified for demo)
            # In a real implementation, we would encode to ambisonics properly
            
            # W channel (omnidirectional)
            output_buffer[0, :] += audio * 0.7
            
            # Directional channels (if we have enough channels)
            if self.n_channels > 1:
                azimuth, elevation, distance = position
                
                # Apply distance attenuation
                if distance < 1.0:
                    distance = 1.0
                distance_gain = 1.0 / distance
                audio = audio * distance_gain
                
                # Front-back (Y)
                output_buffer[1, :] += audio * np.cos(azimuth) * np.cos(elevation) * 0.5
                
                # Up-down (Z)
                output_buffer[2, :] += audio * np.sin(elevation) * 0.5
                
                # Left-right (X)
                output_buffer[3, :] += audio * np.sin(azimuth) * np.cos(elevation) * 0.5
        
        # Apply rotation if needed (simplified)
        yaw, pitch, roll = self.listener_rotation
        if yaw != 0 or pitch != 0 or roll != 0:
            # Very simple rotation effect for demo
            if self.n_channels > 3:
                # Just adjust the balance between channels
                yaw_factor = np.sin(yaw)
                output_buffer[1, :] = output_buffer[1, :] * np.cos(yaw)
                output_buffer[3, :] = output_buffer[3, :] - yaw_factor * output_buffer[1, :]
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0.99:
            output_buffer = output_buffer * 0.99 / max_val
        
        return output_buffer
    
    def get_binaural_output(self):
        """
        Get the current block as binaural stereo
        
        Returns:
        - Binaural stereo signals, shape (2, buffer_size)
        """
        # Process the current block
        ambi_block = self.process_block()
        
        # Convert to binaural (simplified)
        binaural = np.zeros((2, self.buffer_size))
        
        # W channel to both ears
        binaural[0, :] += ambi_block[0, :] * 0.7
        binaural[1, :] += ambi_block[0, :] * 0.7
        
        # Directional information if available
        if self.n_channels > 3:
            # Left-right (X) creates intensity difference
            binaural[0, :] -= ambi_block[3, :] * 0.5  # Left ear gets negative X
            binaural[1, :] += ambi_block[3, :] * 0.5  # Right ear gets positive X
            
            # Front-back (Y) is similar for both ears with slight phase difference
            # This is extremely simplified and would normally use HRTFs
            binaural[0, :] += ambi_block[1, :] * 0.3
            binaural[1, :] += ambi_block[1, :] * 0.3
        
        # Normalize if necessary
        max_val = np.max(np.abs(binaural))
        if max_val > 0.99:
            binaural = binaural * 0.99 / max_val
            
        return binaural