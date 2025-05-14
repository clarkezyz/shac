"""
SHAC Xbox Controller Interface

This module provides an interface for controlling spatial audio scenes using a standard
gaming controller (Xbox, PlayStation, etc.). It enables navigation through a 3D sound field
and manipulation of audio layers using controller inputs.

Author: Claude
License: MIT License
"""

import pygame
import numpy as np
import time

class ControllerInterface:
    """
    Maps a standard gaming controller to sound navigation and manipulation functions
    """
    
    def __init__(self, layer_manager):
        """
        Initialize the controller interface
        
        Parameters:
        - layer_manager: SoundLayerManager instance
        """
        self.layer_manager = layer_manager
        
        # Initialize pygame for controller input
        pygame.init()
        pygame.joystick.init()
        
        # Check for available controllers
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Controller connected: {self.controller.get_name()}")
        else:
            print("No controller found. Using keyboard fallback.")
        
        # Navigation state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # yaw, pitch, roll
        
        # Movement parameters
        self.movement_speed = 1.0  # Base speed in meters per second
        self.rotation_speed = 1.0  # Base speed in radians per second
        
        # Current mode (navigation, layer selection, manipulation)
        self.mode = 'navigation'
        
        # Currently selected layer
        self.selected_layer_id = None
        
        # Last event time (for calculating deltas)
        self.last_time = time.time()
        
        # For keyboard control - store previous key states
        self.prev_keys = {}
        
        # Controller button mappings
        self.button_map = {
            'a_button': 0,  # A button (Xbox) / X button (PlayStation)
            'b_button': 1,  # B button (Xbox) / Circle button (PlayStation)
            'x_button': 2,  # X button (Xbox) / Square button (PlayStation)
            'y_button': 3,  # Y button (Xbox) / Triangle button (PlayStation)
            'left_shoulder': 4,  # Left bumper
            'right_shoulder': 5,  # Right bumper
            'back_button': 6,  # Back/Select button
            'start_button': 7,  # Start button
            'left_stick_button': 8,  # Left stick press
            'right_stick_button': 9,  # Right stick press
        }
        
        # Controller axis mappings
        self.axis_map = {
            'left_stick_x': 0,  # Left stick horizontal
            'left_stick_y': 1,  # Left stick vertical
            'right_stick_x': 2,  # Right stick horizontal
            'right_stick_y': 3,  # Right stick vertical
            'left_trigger': 4,  # Left trigger
            'right_trigger': 5,  # Right trigger
        }
        
        # D-pad is typically represented as a hat
        
    def process_input(self):
        """
        Process controller input and update the sound environment accordingly
        
        Returns:
        - updated: True if the sound environment was updated
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        updated = False
        
        # Poll for events
        pygame.event.pump()
        
        if self.controller is None:
            # Keyboard fallback
            keys = pygame.key.get_pressed()
            
            # Movement speed and rotation speed for keyboard (faster for better responsiveness)
            kb_move_speed = 5.0
            kb_rot_speed = 5.0
            
            # Check which mode we're in
            if self.mode == 'navigation':
                # WASD for movement in the direction we're facing
                move_x = 0
                move_z = 0
                
                if keys[pygame.K_w]:  # Forward
                    move_z -= kb_move_speed * dt
                if keys[pygame.K_s]:  # Backward
                    move_z += kb_move_speed * dt
                if keys[pygame.K_a]:  # Left
                    move_x -= kb_move_speed * dt
                if keys[pygame.K_d]:  # Right
                    move_x += kb_move_speed * dt
                    
                # Arrow keys for rotation
                if keys[pygame.K_LEFT]:  # Turn left
                    self.orientation[0] -= kb_rot_speed * dt
                    updated = True
                if keys[pygame.K_RIGHT]:  # Turn right
                    self.orientation[0] += kb_rot_speed * dt
                    updated = True
                if keys[pygame.K_UP]:  # Look up
                    self.orientation[1] -= kb_rot_speed * dt
                    self.orientation[1] = np.clip(self.orientation[1], -np.pi/2, np.pi/2)
                    updated = True
                if keys[pygame.K_DOWN]:  # Look down
                    self.orientation[1] += kb_rot_speed * dt
                    self.orientation[1] = np.clip(self.orientation[1], -np.pi/2, np.pi/2)
                    updated = True
                    
                # Up/down movement with Q/E
                if keys[pygame.K_q]:  # Move up
                    self.position[1] += kb_move_speed * dt
                    updated = True
                if keys[pygame.K_e]:  # Move down
                    self.position[1] -= kb_move_speed * dt
                    updated = True
                
                # Apply movement if any keys pressed
                if move_x != 0 or move_z != 0:
                    # Apply orientation to movement (so we move in the direction we're facing)
                    yaw = self.orientation[0]
                    self.position[0] += move_x * np.cos(yaw) - move_z * np.sin(yaw)
                    self.position[2] += move_x * np.sin(yaw) + move_z * np.cos(yaw)
                    updated = True
                
                # Mode switching with Space
                if keys[pygame.K_SPACE] and not self.prev_keys.get(pygame.K_SPACE, False):
                    self._switch_to_layer_selection()
                    updated = True
            
            elif self.mode == 'layer_selection':
                # Layer navigation with arrow keys
                if keys[pygame.K_UP] and not self.prev_keys.get(pygame.K_UP, False):
                    self._select_prev_layer()
                    updated = True
                if keys[pygame.K_DOWN] and not self.prev_keys.get(pygame.K_DOWN, False):
                    self._select_next_layer()
                    updated = True
                
                # Enter layer edit mode with Enter
                if keys[pygame.K_RETURN] and not self.prev_keys.get(pygame.K_RETURN, False):
                    self._switch_to_layer_edit()
                    updated = True
                
                # Back to navigation with Space
                if keys[pygame.K_SPACE] and not self.prev_keys.get(pygame.K_SPACE, False):
                    self._switch_to_navigation()
                    updated = True
            
            elif self.mode == 'layer_edit':
                # Layer manipulation with WASD
                if keys[pygame.K_w]:  # Move source forward
                    self._move_selected_layer(0, 0, -kb_move_speed * dt)
                    updated = True
                if keys[pygame.K_s]:  # Move source backward
                    self._move_selected_layer(0, 0, kb_move_speed * dt)
                    updated = True
                if keys[pygame.K_a]:  # Move source left
                    self._move_selected_layer(-kb_move_speed * dt, 0, 0)
                    updated = True
                if keys[pygame.K_d]:  # Move source right
                    self._move_selected_layer(kb_move_speed * dt, 0, 0)
                    updated = True
                
                # Up/down with Q/E
                if keys[pygame.K_q]:  # Move source up
                    self._move_selected_layer(0, kb_move_speed * dt, 0)
                    updated = True
                if keys[pygame.K_e]:  # Move source down
                    self._move_selected_layer(0, -kb_move_speed * dt, 0)
                    updated = True
                
                # Volume control with arrow keys
                if keys[pygame.K_UP]:  # Increase volume
                    self._adjust_selected_layer_gain(0.5 * dt)
                    updated = True
                if keys[pygame.K_DOWN]:  # Decrease volume
                    self._adjust_selected_layer_gain(-0.5 * dt)
                    updated = True
                
                # Mute toggle with M
                if keys[pygame.K_m] and not self.prev_keys.get(pygame.K_m, False):
                    self._toggle_selected_layer_mute()
                    updated = True
                
                # Back to layer selection with Space
                if keys[pygame.K_SPACE] and not self.prev_keys.get(pygame.K_SPACE, False):
                    self._switch_to_layer_selection()
                    updated = True
            
            # Store current key state for next frame
            self.prev_keys = {}
            for key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_UP, pygame.K_DOWN, pygame.K_m]:
                self.prev_keys[key] = keys[key]
            
            if updated:
                self._update_sound_field()
                
            return updated
        
        # Process controller input based on current mode
        if self.mode == 'navigation':
            # Left stick: Move in XZ plane (forward/back, left/right)
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1:
                left_x = 0
            if abs(left_y) < 0.1:
                left_y = 0
            
            # Right stick: Control orientation (yaw/pitch)
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_x) < 0.1:
                right_x = 0
            if abs(right_y) < 0.1:
                right_y = 0
                
            # Update position based on left stick input
            # Move in the direction we're facing
            move_x = -left_x * self.movement_speed * dt
            move_z = -left_y * self.movement_speed * dt
            
            # Apply orientation to movement (so we move in the direction we're facing)
            yaw = self.orientation[0]
            self.position[0] += move_x * np.cos(yaw) - move_z * np.sin(yaw)
            self.position[2] += move_x * np.sin(yaw) + move_z * np.cos(yaw)
            
            # Update orientation based on right stick input
            self.orientation[0] += right_x * self.rotation_speed * dt  # Yaw (look left/right)
            self.orientation[1] += right_y * self.rotation_speed * dt  # Pitch (look up/down)
            
            # Clamp pitch to avoid flipping over
            self.orientation[1] = np.clip(self.orientation[1], -np.pi/2, np.pi/2)
            
            # Triggers: Move up/down (Y axis)
            left_trigger = self.controller.get_axis(self.axis_map['left_trigger'])
            right_trigger = self.controller.get_axis(self.axis_map['right_trigger'])
            
            # Normalize trigger inputs (they might be in different ranges on different controllers)
            left_trigger = (left_trigger + 1) / 2  # Convert from [-1, 1] to [0, 1]
            right_trigger = (right_trigger + 1) / 2
            
            # Move up with right trigger, down with left trigger
            self.position[1] += (right_trigger - left_trigger) * self.movement_speed * dt
            
            # If position or orientation changed, update sound field
            if move_x != 0 or move_z != 0 or self.position[1] != 0 or right_x != 0 or right_y != 0:
                # Convert our position and orientation to rotation and translation for the sound field
                self._update_sound_field()
                updated = True
            
            # Mode switching
            if self.controller.get_button(self.button_map['y_button']):
                self.mode = 'layer_selection'
                print("Mode: Layer Selection")
                updated = True
        
        elif self.mode == 'layer_selection':
            # D-pad or left stick to scroll through layers
            hat = self.controller.get_hat(0)  # Assuming only one hat (d-pad)
            
            # Process D-pad for layer selection
            if hat[1] == 1:  # D-pad up
                self._select_previous_layer()
                updated = True
            elif hat[1] == -1:  # D-pad down
                self._select_next_layer()
                updated = True
            
            # A button to select layer
            if self.controller.get_button(self.button_map['a_button']):
                if self.selected_layer_id is not None:
                    print(f"Selected layer: {self.layer_manager.get_layer_info(self.selected_layer_id)['name']}")
                    self.mode = 'layer_manipulation'
                    updated = True
            
            # B button to go back to navigation
            if self.controller.get_button(self.button_map['b_button']):
                self.mode = 'navigation'
                print("Mode: Navigation")
                updated = True
        
        elif self.mode == 'layer_manipulation':
            # Layer manipulation controls
            
            # Left stick: adjust layer position
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1:
                left_x = 0
            if abs(left_y) < 0.1:
                left_y = 0
            
            if left_x != 0 or left_y != 0:
                # Get current position
                azimuth, elevation, distance = self.layer_manager.get_layer_info(self.selected_layer_id)['position']
                
                # Update azimuth based on left_x
                azimuth += left_x * self.rotation_speed * dt
                # Keep azimuth in range [-π, π]
                azimuth = ((azimuth + np.pi) % (2 * np.pi)) - np.pi
                
                # Update elevation based on left_y
                elevation -= left_y * self.rotation_speed * dt
                # Clamp elevation to prevent going beyond the poles
                elevation = np.clip(elevation, -np.pi/2, np.pi/2)
                
                # Apply the updated position
                self.layer_manager.move_layer(self.selected_layer_id, (azimuth, elevation, distance))
                updated = True
            
            # Right stick Y: adjust layer distance
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_y) < 0.1:
                right_y = 0
            
            if right_y != 0:
                # Get current position
                azimuth, elevation, distance = self.layer_manager.get_layer_info(self.selected_layer_id)['position']
                
                # Update distance based on right_y (negative = closer)
                distance_change_rate = 2.0  # meters per second
                distance -= right_y * distance_change_rate * dt
                # Ensure distance stays positive and reasonable
                distance = np.clip(distance, 0.1, 20.0)
                
                # Apply the updated position
                self.layer_manager.move_layer(self.selected_layer_id, (azimuth, elevation, distance))
                updated = True
            
            # Right stick X: adjust layer gain
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1:
                right_x = 0
            
            if right_x != 0:
                # Get current gain
                current_gain = self.layer_manager.get_layer_info(self.selected_layer_id)['current_gain']
                
                # Update gain exponentially for better control
                gain_change_rate = 2.0  # dB per second
                gain_change = np.exp(right_x * gain_change_rate * dt / 8.686)  # Convert dB to ratio
                new_gain = current_gain * gain_change
                
                # Clamp gain to reasonable range
                new_gain = np.clip(new_gain, 0.0, 4.0)
                
                # Apply the updated gain
                self.layer_manager.adjust_layer_gain(self.selected_layer_id, new_gain)
                updated = True
            
            # Shoulder buttons: adjust EQ
            left_shoulder = self.controller.get_button(self.button_map['left_shoulder'])
            right_shoulder = self.controller.get_button(self.button_map['right_shoulder'])
            
            if left_shoulder:
                # Enhance bass
                self.layer_manager.adjust_layer_eq(self.selected_layer_id, 'bass', 1.5)
                self.layer_manager.adjust_layer_eq(self.selected_layer_id, 'sub_bass', 1.7)
                updated = True
            
            if right_shoulder:
                # Enhance treble
                self.layer_manager.adjust_layer_eq(self.selected_layer_id, 'high', 1.5)
                self.layer_manager.adjust_layer_eq(self.selected_layer_id, 'very_high', 1.7)
                updated = True
            
            # A button: toggle mute
            if self.controller.get_button(self.button_map['a_button']):
                # Toggle mute state
                current_mute = self.layer_manager.get_layer_info(self.selected_layer_id)['muted']
                self.layer_manager.mute_layer(self.selected_layer_id, not current_mute)
                updated = True
                print(f"Layer {self.selected_layer_id} {'muted' if not current_mute else 'unmuted'}")
            
            # B button: return to layer selection
            if self.controller.get_button(self.button_map['b_button']):
                self.mode = 'layer_selection'
                print("Mode: Layer Selection")
                updated = True
            
            # X button: reset layer to default
            if self.controller.get_button(self.button_map['x_button']):
                # Reset gain
                self.layer_manager.adjust_layer_gain(self.selected_layer_id, 1.0)
                
                # Reset EQ
                for band in self.layer_manager.frequency_bands:
                    self.layer_manager.adjust_layer_eq(self.selected_layer_id, band, 1.0)
                
                # Reset mute
                self.layer_manager.mute_layer(self.selected_layer_id, False)
                
                updated = True
                print(f"Layer {self.selected_layer_id} reset to defaults")
        
        return updated
    
    def _move_selected_layer(self, dx, dy, dz):
        """Move the selected layer by the specified amounts"""
        if self.selected_layer_id is None:
            return
            
        # Get current position
        azimuth, elevation, distance = self.layer_manager.get_layer_info(self.selected_layer_id)['position']
        
        # Convert position to Cartesian coordinates
        x = distance * np.sin(azimuth) * np.cos(elevation)
        y = distance * np.sin(elevation)
        z = distance * np.cos(azimuth) * np.cos(elevation)
        
        # Update position
        x += dx
        y += dy
        z += dz
        
        # Convert back to spherical
        new_distance = np.sqrt(x*x + y*y + z*z)
        new_elevation = np.arcsin(y / new_distance) if new_distance > 0 else 0
        new_azimuth = np.arctan2(x, z)
        
        # Apply the updated position
        self.layer_manager.move_layer(self.selected_layer_id, (new_azimuth, new_elevation, new_distance))
    
    def _adjust_selected_layer_gain(self, dB_change):
        """Adjust the gain of the selected layer"""
        if self.selected_layer_id is None:
            return
            
        # Get current gain
        current_gain = self.layer_manager.get_layer_info(self.selected_layer_id)['current_gain']
        
        # Update gain exponentially for better control
        gain_change = np.exp(dB_change / 8.686)  # Convert dB to ratio
        new_gain = current_gain * gain_change
        
        # Clamp gain to reasonable range
        new_gain = np.clip(new_gain, 0.0, 4.0)
        
        # Apply the updated gain
        self.layer_manager.adjust_layer_gain(self.selected_layer_id, new_gain)
    
    def _toggle_selected_layer_mute(self):
        """Toggle mute for the selected layer"""
        if self.selected_layer_id is None:
            return
            
        # Toggle mute state
        current_mute = self.layer_manager.get_layer_info(self.selected_layer_id)['muted']
        self.layer_manager.mute_layer(self.selected_layer_id, not current_mute)
        print(f"Layer {self.selected_layer_id} {'muted' if not current_mute else 'unmuted'}")
    
    def _switch_to_navigation(self):
        """Switch to navigation mode"""
        self.mode = 'navigation'
        print("Mode: Navigation")
        
    def _switch_to_layer_selection(self):
        """Switch to layer selection mode"""
        self.mode = 'layer_selection'
        
        # Initialize layer selection if needed
        if not hasattr(self.layer_manager, 'layers') or len(self.layer_manager.layers) == 0:
            return
            
        # Select the first layer if none is selected
        if self.selected_layer_id is None:
            self.selected_layer_id = list(self.layer_manager.layers.keys())[0]
            
        print("Mode: Layer Selection")
        
    def _switch_to_layer_edit(self):
        """Switch to layer edit mode"""
        if self.selected_layer_id is None:
            self._switch_to_layer_selection()
            return
            
        self.mode = 'layer_edit'
        layer_info = self.layer_manager.get_layer_info(self.selected_layer_id)
        print(f"Mode: Layer Edit - {layer_info['name']}")
    def _select_previous_layer(self):
        """Select the previous layer in the list"""
        layer_ids = list(self.layer_manager.layers.keys())
        if not layer_ids:
            return
            
        if self.selected_layer_id is None:
            self.selected_layer_id = layer_ids[0]
        else:
            current_index = layer_ids.index(self.selected_layer_id)
            prev_index = (current_index - 1) % len(layer_ids)
            self.selected_layer_id = layer_ids[prev_index]
            
        layer_info = self.layer_manager.get_layer_info(self.selected_layer_id)
        print(f"Selected layer: {layer_info['name']}")
    
    def _select_next_layer(self):
        """Select the next layer in the list"""
        layer_ids = list(self.layer_manager.layers.keys())
        if not layer_ids:
            return
            
        if self.selected_layer_id is None:
            self.selected_layer_id = layer_ids[0]
        else:
            current_index = layer_ids.index(self.selected_layer_id)
            next_index = (current_index + 1) % len(layer_ids)
            self.selected_layer_id = layer_ids[next_index]
            
        layer_info = self.layer_manager.get_layer_info(self.selected_layer_id)
        print(f"Selected layer: {layer_info['name']}")
    
    def _update_sound_field(self):
        """Update the sound field based on position and orientation"""
        # Create rotation matrix from our orientation
        yaw, pitch, roll = self.orientation
        
        # Apply the rotation to the layer manager's codec
        self.layer_manager.set_listener_rotation(yaw, pitch, roll)
        
        # For demonstration purposes, we'll just print the update
        print(f"Position: {self.position}, Orientation: {self.orientation}")


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
    
    def get_layer_info(self, layer_id):
        """Get layer information"""
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} not found")
        return self.layers[layer_id]
    
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
        print(f"Adjusted {band} to {gain} for layer {layer_id}")
    
    def set_listener_rotation(self, yaw, pitch, roll):
        """
        Set the listener's head rotation
        
        Parameters:
        - yaw: Rotation around vertical axis in radians
        - pitch: Rotation around side axis in radians
        - roll: Rotation around front axis in radians
        """
        self.codec.set_listener_rotation((yaw, pitch, roll))


def create_example_controller_interface():
    """
    Create an example controller interface setup with sample sounds
    """
    from shac_codec import SHACCodec
    import numpy as np
    
    # Create codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Create layer manager
    layer_manager = SoundLayerManager(codec)
    
    # Create some example layers
    
    # Piano sound
    duration = 5.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    piano_audio = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t/2)
    piano_audio += 0.25 * np.sin(2 * np.pi * 880 * t) * np.exp(-t/1.5)
    layer_manager.add_layer('piano', 'Piano', piano_audio, (-np.pi/4, 0.0, 3.0))
    
    # Drum sound
    drum_audio = np.zeros_like(t)
    for i in range(0, len(t), int(sample_rate / 4)):
        if i + 5000 < len(drum_audio):
            drum_audio[i:i+5000] = 0.8 * np.exp(-np.linspace(0, 10, 5000))
    layer_manager.add_layer('drum', 'Drums', drum_audio, (np.pi/4, -0.1, 2.5))
    
    # Ambient sound
    np.random.seed(42)
    noise = np.random.randn(len(t))
    ambient_audio = np.convolve(noise, np.ones(20)/20, mode='same') * 0.2
    layer_manager.add_layer('ambient', 'Ambient', ambient_audio, (0.0, np.pi/3, 5.0))
    
    # Create controller interface
    controller = ControllerInterface(layer_manager)
    
    return controller


if __name__ == "__main__":
    # Example usage
    controller = create_example_controller_interface()
    
    # Main loop
    try:
        print("Controller interface running. Press Ctrl+C to exit.")
        while True:
            # Process controller input
            updated = controller.process_input()
            
            # If sound environment was updated, we would want to render audio
            if updated:
                pass  # In a real implementation, this would trigger audio rendering
            
            # Limit update rate
            time.sleep(0.016)  # ~60 Hz
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit()