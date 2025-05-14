"""
SHAC Sound Layer Manager

This module provides a layer management system for working with spatial audio layers
in the SHAC system. It serves as an interface between the controller and the audio codec.

Author: Claude
License: MIT License
"""

import numpy as np

class SoundLayerManager:
    """
    Manages audio layers in a spatial sound field
    
    This class provides an interface between the controller interface and the SHAC codec.
    It maintains layer state and handles operations on layers.
    """
    
    def __init__(self, codec):
        """
        Initialize the sound layer manager
        
        Parameters:
        - codec: SHACCodec instance
        """
        self.codec = codec
        self.layers = {}  # Dictionary of layer info by ID
        self.frequency_bands = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high', 'very_high']
        
        # Frequency ranges for each band (in Hz)
        self.band_ranges = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'very_high': (8000, 20000)
        }
        
        # Shelf and peaking filter parameters
        self.filter_params = {
            'sub_bass': {'type': 'low_shelf', 'q': 0.7},
            'bass': {'type': 'low_shelf', 'q': 0.8},
            'low_mid': {'type': 'peaking', 'q': 1.0},
            'mid': {'type': 'peaking', 'q': 1.0},
            'high_mid': {'type': 'peaking', 'q': 1.0},
            'high': {'type': 'high_shelf', 'q': 0.8},
            'very_high': {'type': 'high_shelf', 'q': 0.7}
        }
    
    def add_layer(self, layer_id, name, audio_data, position, properties=None):
        """
        Add a new audio layer
        
        Parameters:
        - layer_id: Unique ID for the layer
        - name: Display name for the layer
        - audio_data: Audio samples
        - position: (azimuth, elevation, distance) tuple
        - properties: Additional properties dictionary
        """
        if properties is None:
            properties = {}
        
        # Add to our codec
        self.codec.add_mono_source(layer_id, audio_data, position)
        
        # Store layer info
        self.layers[layer_id] = {
            'name': name,
            'position': position,
            'current_gain': 1.0,
            'muted': False,
            'eq_settings': {band: 1.0 for band in self.frequency_bands},
            'properties': properties
        }
        
        return layer_id
    
    def remove_layer(self, layer_id):
        """
        Remove a layer
        
        Parameters:
        - layer_id: Layer ID to remove
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Remove from our records
        del self.layers[layer_id]
        
        # In a real implementation, we'd need to add this method to the codec
        # self.codec.remove_source(layer_id)
        
        # For now, just mute it
        self.codec.mute_source(layer_id, True)
    
    def get_layer_info(self, layer_id):
        """
        Get layer information
        
        Parameters:
        - layer_id: Layer ID
        
        Returns:
        - Dictionary with layer information
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        return self.layers[layer_id]
    
    def get_all_layers(self):
        """
        Get information about all layers
        
        Returns:
        - Dictionary of layer info by ID
        """
        return self.layers
    
    def move_layer(self, layer_id, position):
        """
        Move a layer to a new position
        
        Parameters:
        - layer_id: Layer ID
        - position: (azimuth, elevation, distance) tuple
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Update our local record
        self.layers[layer_id]['position'] = position
        
        # Update in the codec
        self.codec.set_source_position(layer_id, position)
    
    def adjust_layer_gain(self, layer_id, gain):
        """
        Adjust the gain of a layer
        
        Parameters:
        - layer_id: Layer ID
        - gain: New gain value (1.0 = unity gain)
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Update our local record
        self.layers[layer_id]['current_gain'] = gain
        
        # Update in the codec
        self.codec.set_source_gain(layer_id, gain)
    
    def mute_layer(self, layer_id, muted):
        """
        Mute or unmute a layer
        
        Parameters:
        - layer_id: Layer ID
        - muted: True to mute, False to unmute
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Update our local record
        self.layers[layer_id]['muted'] = muted
        
        # Update in the codec
        self.codec.mute_source(layer_id, muted)
    
    def solo_layer(self, layer_id):
        """
        Solo a layer (mute all others)
        
        Parameters:
        - layer_id: Layer ID to solo
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Mute all layers except the soloed one
        for lid in self.layers:
            self.mute_layer(lid, lid != layer_id)
    
    def adjust_layer_eq(self, layer_id, band, gain):
        """
        Adjust the EQ for a specific frequency band
        
        Parameters:
        - layer_id: Layer ID
        - band: Frequency band name
        - gain: Gain value for the band
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        if band not in self.frequency_bands:
            raise ValueError(f"Band {band} not recognized")
        
        # Update our local record
        self.layers[layer_id]['eq_settings'][band] = gain
        
        # In a real implementation, this would apply the EQ to the codec
        # We'd need to implement a per-source EQ in the codec for this
        # For now, just print the change
        print(f"Adjusted {band} to {gain} for layer {layer_id}")
    
    def reset_layer_eq(self, layer_id):
        """
        Reset all EQ settings for a layer to defaults
        
        Parameters:
        - layer_id: Layer ID
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Reset all bands to unity gain
        for band in self.frequency_bands:
            self.adjust_layer_eq(layer_id, band, 1.0)
    
    def set_layer_directivity(self, layer_id, directivity, axis=None):
        """
        Set the directivity pattern for a layer
        
        Parameters:
        - layer_id: Layer ID
        - directivity: Directivity factor (0.0 = omnidirectional, 1.0 = cardioid)
        - axis: Optional directivity axis override
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Use current position as axis if not specified
        if axis is None:
            axis = self.layers[layer_id]['position']
        
        # Update local record
        self.layers[layer_id]['directivity'] = directivity
        self.layers[layer_id]['directivity_axis'] = axis
        
        # In a real implementation, this would call the codec method
        # self.codec.set_source_directivity(layer_id, directivity, axis)
        
        # For now, just print the change
        print(f"Set directivity to {directivity} for layer {layer_id}")
    
    def apply_layer_effect(self, layer_id, effect_type, parameters):
        """
        Apply an audio effect to a layer
        
        Parameters:
        - layer_id: Layer ID
        - effect_type: Type of effect (e.g., 'reverb', 'delay', 'modulation')
        - parameters: Dictionary of effect parameters
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        # Update local record
        if 'effects' not in self.layers[layer_id]:
            self.layers[layer_id]['effects'] = {}
        
        self.layers[layer_id]['effects'][effect_type] = parameters
        
        # In a real implementation, this would apply the effect in the codec
        print(f"Applied {effect_type} effect to layer {layer_id}")
    
    def remove_layer_effect(self, layer_id, effect_type):
        """
        Remove an audio effect from a layer
        
        Parameters:
        - layer_id: Layer ID
        - effect_type: Type of effect to remove
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        
        if 'effects' in self.layers[layer_id] and effect_type in self.layers[layer_id]['effects']:
            del self.layers[layer_id]['effects'][effect_type]
            print(f"Removed {effect_type} effect from layer {layer_id}")
    
    def set_listener_position(self, position):
        """
        Set the listener's position
        
        Parameters:
        - position: (x, y, z) tuple in meters
        """
        # In a real implementation, this would call the codec method
        self.codec.set_listener_position(position)
    
    def set_listener_rotation(self, yaw, pitch, roll):
        """
        Set the listener's head rotation
        
        Parameters:
        - yaw: Rotation around vertical axis in radians
        - pitch: Rotation around side axis in radians
        - roll: Rotation around front axis in radians
        """
        # Store rotation in the layer manager
        self.listener_rotation = (yaw, pitch, roll)
        
        # Add fallback if the method doesn't exist
        if not hasattr(self.codec, 'set_listener_rotation'):
            # Just store it in listener_orientation directly
            self.codec.listener_orientation = self.listener_rotation
        else:
            # Call the existing method
            try:
                self.codec.set_listener_rotation(self.listener_rotation)
            except TypeError:
                # If signature mismatch, just update directly
                self.codec.listener_orientation = self.listener_rotation
    
    def process_audio(self):
        """
        Process all layers and return the final audio output
        
        Returns:
        - Processed audio data
        """
        # In a real implementation, this would call the codec's process method
        ambisonic_signals = self.codec.process()
        
        # Convert to binaural if needed
        binaural_output = self.codec.binauralize(ambisonic_signals)
        
        return binaural_output
    
    def create_snapshot(self):
        """
        Create a snapshot of the current layer configuration
        
        Returns:
        - Dictionary with complete layer state
        """
        return {
            'layers': self.layers.copy(),
            # Include additional state if needed
        }
    
    def restore_snapshot(self, snapshot):
        """
        Restore from a previously saved snapshot
        
        Parameters:
        - snapshot: Snapshot dictionary from create_snapshot()
        """
        # Restore layers
        for layer_id, layer_info in snapshot['layers'].items():
            # If the layer already exists, update its properties
            if layer_id in self.layers:
                self.move_layer(layer_id, layer_info['position'])
                self.adjust_layer_gain(layer_id, layer_info['current_gain'])
                self.mute_layer(layer_id, layer_info['muted'])
                
                # Restore EQ settings
                for band, gain in layer_info['eq_settings'].items():
                    self.adjust_layer_eq(layer_id, band, gain)
            
            # Otherwise, we'd need audio data to recreate it
            # This would require storing audio in the snapshot or having a reference