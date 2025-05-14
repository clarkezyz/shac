"""
SHAC Simple Demo with Xbox Controller and Headphones - Final Version

A simplified demonstration of the SHAC system that works with minimal dependencies.
This version fixes all reported issues and improves audio quality.

Author: Claude
License: MIT License
"""

import numpy as np
import pygame
import threading
import time
import sys
import os
import math

# Initialize pygame for controller and audio
pygame.init()
pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=1024)

class SimpleDemo:
    """A simplified SHAC demo that works with just pygame"""
    
    def __init__(self):
        """Initialize the demo with basic components"""
        self.sample_rate = 48000
        self.buffer_size = 1024
        
        # Initialize controller
        pygame.joystick.init()
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Controller connected: {self.controller.get_name()}")
        else:
            print("No controller found. Using keyboard fallback.")
        
        # Controller mapping (for Xbox controller)
        self.button_map = {
            'a_button': 0,      # A button
            'b_button': 1,      # B button
            'x_button': 2,      # X button
            'y_button': 3,      # Y button
            'left_shoulder': 4, # LB
            'right_shoulder': 5 # RB
        }
        
        self.axis_map = {
            'left_stick_x': 0,  # Left stick horizontal
            'left_stick_y': 1,  # Left stick vertical
            'right_stick_x': 2, # Right stick horizontal
            'right_stick_y': 3, # Right stick vertical
            # Triggers might be mapped differently based on controller
        }
        
        # Navigation state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # yaw, pitch, roll
        self.mode = 'navigation'
        
        # Sound sources
        self.sources = {}
        self.selected_source_id = None
        
        # Create sample sounds (higher quality)
        self._create_sound_sources()
        
        # Audio output
        self.sounds = {}
        self.looping_sounds = True  # Control for sound looping
        
        # Create sound channels and start playback
        self._initialize_audio()
        
        # Setup visualization
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("SHAC Simple Demo")
        self.font = pygame.font.Font(None, 36)
        
        # Thread control
        self.running = False
        
        # Last button states to prevent immediate repeat presses
        self.last_button_states = {name: False for name in self.button_map.keys()}
        
        # Time tracking for delta-based movement
        self.last_time = time.time()
        
        # Trigger states (for controllers that map triggers differently)
        self.trigger_states = {'left': 0.0, 'right': 0.0}
        
        print("Simple SHAC demo initialized")
    
    def _create_sound_sources(self):
        """Create sample sound sources with better audio quality"""
        duration = 10.0  # longer sounds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano-like sound (tone with harmonics and decay)
        piano_freq = 440  # A4
        piano_audio = np.zeros_like(t)
        # Add multiple notes at different times
        for start_time in [0.0, 2.5, 5.0, 7.5]:
            # Convert time to sample index
            start_idx = int(start_time * self.sample_rate)
            if start_idx >= len(t):
                continue
                
            # Create note with harmonics
            note_len = min(len(t) - start_idx, int(2.5 * self.sample_rate))
            note_t = np.linspace(0, 2.5, note_len)
            
            # Base frequency
            note = 0.5 * np.sin(2 * np.pi * piano_freq * note_t) * np.exp(-note_t/2)
            # First harmonic
            note += 0.25 * np.sin(2 * np.pi * piano_freq * 2 * note_t) * np.exp(-note_t/1.5)
            # Second harmonic
            note += 0.125 * np.sin(2 * np.pi * piano_freq * 3 * note_t) * np.exp(-note_t)
            # Third harmonic
            note += 0.0625 * np.sin(2 * np.pi * piano_freq * 4 * note_t) * np.exp(-note_t/0.8)
            
            # Add note to main audio
            piano_audio[start_idx:start_idx+note_len] += note
            
        # Normalize
        piano_audio = piano_audio / np.max(np.abs(piano_audio)) * 0.8
        
        self.sources['piano'] = {
            'audio': piano_audio,
            'position': np.array([-3.0, 0.0, 3.0]),  # x, y, z (left, up, front)
            'name': 'Piano',
            'color': (255, 0, 0),
            'gain': 1.0,
            'muted': False
        }
        
        # Bass-like sound (dynamic bass line)
        bass_notes = [110, 146.8, 164.8, 196]  # A2, D3, E3, G3
        bass_audio = np.zeros_like(t)
        
        # Create a simple bass line
        for i, note in enumerate(bass_notes * 3):
            start_idx = i * int(self.sample_rate * 0.8)  # 0.8 seconds per note
            if start_idx >= len(t):
                break
                
            note_len = min(len(t) - start_idx, int(0.7 * self.sample_rate))
            note_t = np.linspace(0, 0.7, note_len)
            
            # Generate note with slight envelope
            envelope = 0.8 * (1 - np.exp(-note_t*5)) * np.exp(-note_t/0.5)
            note_sound = np.sin(2 * np.pi * note * note_t) * envelope
            
            # Add some harmonics
            note_sound += 0.3 * np.sin(2 * np.pi * note * 2 * note_t) * envelope
            
            # Add to main audio
            bass_audio[start_idx:start_idx+note_len] += note_sound
        
        # Normalize
        bass_audio = bass_audio / np.max(np.abs(bass_audio)) * 0.9
        
        self.sources['bass'] = {
            'audio': bass_audio,
            'position': np.array([3.0, -1.0, 4.0]),  # x, y, z (right, down, front)
            'name': 'Bass',
            'color': (0, 0, 255),
            'gain': 1.0,
            'muted': False
        }
        
        # Drum-like sound (realistic pattern)
        drum_audio = np.zeros_like(t)
        
        # Define drum patterns (time in seconds, type)
        drum_pattern = [
            (0.0, 'kick'), (0.5, 'snare'), (1.0, 'kick'), (1.5, 'snare'),
            (2.0, 'kick'), (2.5, 'snare'), (3.0, 'kick'), (3.25, 'kick'), (3.5, 'snare'),
            (4.0, 'kick'), (4.5, 'snare'), (5.0, 'kick'), (5.5, 'snare'),
            (6.0, 'kick'), (6.5, 'snare'), (7.0, 'kick'), (7.25, 'kick'), (7.5, 'snare'),
            (8.0, 'kick'), (8.5, 'snare'), (9.0, 'kick'), (9.5, 'snare')
        ]
        
        # Add hi-hats
        for i in range(40):
            drum_pattern.append((i * 0.25, 'hat'))
        
        # Generate each drum sound
        for time_pos, drum_type in drum_pattern:
            start_idx = int(time_pos * self.sample_rate)
            if start_idx >= len(t):
                continue
                
            if drum_type == 'kick':
                # Kick drum (low thump with fast decay)
                env_len = int(0.15 * self.sample_rate)
                if start_idx + env_len > len(drum_audio):
                    env_len = len(drum_audio) - start_idx
                
                env_t = np.linspace(0, 0.15, env_len)
                env = np.exp(-env_t * 30)
                
                # Frequency sweep from 150Hz to 50Hz
                freq = 150 * np.exp(-env_t * 20) + 50
                phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
                kick = 0.8 * np.sin(phase) * env
                
                drum_audio[start_idx:start_idx+env_len] += kick
                
            elif drum_type == 'snare':
                # Snare (mix of tone and noise)
                env_len = int(0.2 * self.sample_rate)
                if start_idx + env_len > len(drum_audio):
                    env_len = len(drum_audio) - start_idx
                
                env_t = np.linspace(0, 0.2, env_len)
                env = np.exp(-env_t * 20)
                
                # Tone component (200Hz)
                tone = 0.3 * np.sin(2 * np.pi * 200 * env_t) * env
                
                # Noise component
                noise = 0.7 * np.random.randn(env_len) * env
                
                drum_audio[start_idx:start_idx+env_len] += tone + noise
                
            elif drum_type == 'hat':
                # Hi-hat (filtered noise, short)
                env_len = int(0.08 * self.sample_rate)
                if start_idx + env_len > len(drum_audio):
                    env_len = len(drum_audio) - start_idx
                
                env_t = np.linspace(0, 0.08, env_len)
                env = np.exp(-env_t * 50)
                
                # High-pass filtered noise
                noise = np.random.randn(env_len)
                # Apply very simple high-pass filter
                for i in range(1, len(noise)):
                    noise[i] = 0.1 * noise[i] + 0.9 * noise[i-1]
                
                hat = 0.3 * noise * env
                drum_audio[start_idx:start_idx+env_len] += hat
        
        # Normalize
        drum_audio = drum_audio / np.max(np.abs(drum_audio)) * 0.9
        
        self.sources['drums'] = {
            'audio': drum_audio,
            'position': np.array([0.0, -1.0, 2.0]),  # x, y, z (center, down, front)
            'name': 'Drums',
            'color': (0, 255, 0),
            'gain': 1.0,
            'muted': False
        }
        
        # Ambient sound (evolving texture)
        ambient_audio = np.zeros_like(t)
        
        # Create subtle chord pad
        chord_freqs = [261.63, 329.63, 392.0, 523.25]  # C4, E4, G4, C5
        
        for freq in chord_freqs:
            # Add some slight detuning for thickness
            detune = 1.0 + (np.random.rand() - 0.5) * 0.01
            
            # Generate a sustained tone with slow attack
            tone = np.sin(2 * np.pi * freq * detune * t)
            
            # Add slow amplitude modulation
            mod_freq = 0.1 + (np.random.rand() * 0.2)  # 0.1-0.3 Hz
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            
            # Add slow attack
            attack = 1 - np.exp(-t / 2.0)
            
            # Combine
            ambient_audio += 0.2 * tone * mod * attack
        
        # Add some filtered noise for texture
        noise = np.random.randn(len(t)) * 0.1
        # Apply simple low-pass filter
        filtered_noise = np.zeros_like(noise)
        filtered_noise[0] = noise[0]
        for i in range(1, len(noise)):
            filtered_noise[i] = 0.95 * filtered_noise[i-1] + 0.05 * noise[i]
        
        # Combine with main ambient sound
        ambient_audio += filtered_noise
        
        # Normalize
        ambient_audio = ambient_audio / np.max(np.abs(ambient_audio)) * 0.7
        
        self.sources['ambient'] = {
            'audio': ambient_audio,
            'position': np.array([0.0, 3.0, 8.0]),  # x, y, z (center, up, far)
            'name': 'Ambient',
            'color': (255, 255, 0),
            'gain': 1.0,
            'muted': False
        }
    
    def _initialize_audio(self):
        """Initialize audio playback"""
        for source_id, source_data in self.sources.items():
            # Convert to 16-bit for pygame
            data = (source_data['audio'] * 32767).astype(np.int16)
            # Make it stereo
            stereo_data = np.column_stack((data, data))
            # Create pygame sound
            self.sounds[source_id] = pygame.mixer.Sound(buffer=stereo_data)
            # Create a channel for this sound
            channel = pygame.mixer.find_channel()
            if channel:
                source_data['channel'] = channel
                channel.play(self.sounds[source_id], loops=-1)
                channel.set_volume(0.5, 0.5)  # Start with mid volume
            else:
                print(f"Warning: Could not find free channel for {source_id}")
                source_data['channel'] = None
        
        # Start a thread to monitor and restart any sounds that might stop
        self.sound_monitor_thread = threading.Thread(target=self._monitor_sounds)
        self.sound_monitor_thread.daemon = True
        self.sound_monitor_thread.start()
    
    def _monitor_sounds(self):
        """Monitor and restart any sounds that might stop playing"""
        while self.looping_sounds:
            for source_id, source_data in self.sources.items():
                channel = source_data.get('channel')
                if channel and not channel.get_busy() and not source_data['muted']:
                    # Sound stopped - restart it
                    channel.play(self.sounds[source_id], loops=-1)
                    print(f"Restarted sound: {source_id}")
            # Check every second
            time.sleep(1)
    
    def _button_pressed(self, button_name):
        """
        Check if a button was just pressed (handles debouncing)
        
        Returns True if the button was just pressed (state changed from not pressed to pressed)
        """
        if self.controller is None:
            return False
            
        button_index = self.button_map.get(button_name)
        if button_index is None:
            return False
            
        # Get current state
        current_state = self.controller.get_button(button_index)
        
        # Check if button was just pressed (was up, now down)
        just_pressed = current_state and not self.last_button_states[button_name]
        
        # Update last state
        self.last_button_states[button_name] = current_state
        
        return just_pressed
    
    def process_input(self):
        """Process controller input and update the sound field"""
        # Get delta time for smooth movement
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Cap dt to avoid huge jumps if game pauses
        dt = min(dt, 0.1)
        
        # Poll for events
        pygame.event.pump()
        
        # Process keyboard input if no controller
        if self.controller is None:
            keys = pygame.key.get_pressed()
            # Keyboard controls would go here
            return
        
        # Scale factors for movement and rotation
        move_speed = 3.0  # units per second
        rotate_speed = 2.0  # radians per second
        
        # Get controller input
        if self.mode == 'navigation':
            # Left stick: Move in XZ plane
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            # Update position based on orientation
            yaw = self.orientation[0]
            
            # Calculate forward and right vectors
            forward_x = math.sin(yaw)
            forward_z = math.cos(yaw)
            right_x = math.sin(yaw + math.pi/2)
            right_z = math.cos(yaw + math.pi/2)
            
            # Move relative to current orientation
            self.position[0] += (forward_x * -left_y + right_x * left_x) * move_speed * dt
            self.position[2] += (forward_z * -left_y + right_z * left_x) * move_speed * dt
            
            # Right stick: Look direction (yaw only for simplicity)
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            # Update yaw based on right stick input
            if abs(right_x) > 0:
                self.orientation[0] += right_x * rotate_speed * dt
            
            # Try different ways to handle triggers for vertical movement
            try:
                # Method 1: Direct axis mapping
                try:
                    lt = (self.controller.get_axis(4) + 1) / 2
                    rt = (self.controller.get_axis(5) + 1) / 2
                    self.trigger_states['left'] = lt
                    self.trigger_states['right'] = rt
                except:
                    # Fall back to stored states
                    lt = self.trigger_states['left']
                    rt = self.trigger_states['right']
                
                # Method 2: If triggers are mapped as buttons
                lt_button = self.controller.get_button(6) if self.controller.get_numbuttons() > 6 else 0
                rt_button = self.controller.get_button(7) if self.controller.get_numbuttons() > 7 else 0
                
                if lt_button > 0.5:
                    self.trigger_states['left'] = 1.0
                if rt_button > 0.5:
                    self.trigger_states['right'] = 1.0
                
                # Update Y position
                self.position[1] += (rt - lt) * move_speed * dt
            except:
                # If no trigger control works, use shoulder buttons
                lb = self.controller.get_button(self.button_map['left_shoulder'])
                rb = self.controller.get_button(self.button_map['right_shoulder'])
                self.position[1] += (rb - lb) * move_speed * dt
            
            # Update audio based on listener position and orientation
            self._update_spatial_audio()
            
            # Y button to switch to source selection
            if self._button_pressed('y_button'):
                self.mode = 'source_selection'
                # Select first source if none selected
                if self.selected_source_id is None and self.sources:
                    self.selected_source_id = list(self.sources.keys())[0]
                print(f"Mode: Source Selection - {self.selected_source_id}")
        
        elif self.mode == 'source_selection':
            # D-pad for source selection
            try:
                hat = self.controller.get_hat(0)  # Assuming only one hat (d-pad)
                
                if hat[1] == 1:  # D-pad up
                    self._select_previous_source()
                    time.sleep(0.2)  # Debounce
                elif hat[1] == -1:  # D-pad down
                    self._select_next_source()
                    time.sleep(0.2)  # Debounce
            except:
                # Fallback for d-pad
                pass
            
            # A button to manipulate selected source
            if self._button_pressed('a_button'):
                if self.selected_source_id:
                    self.mode = 'source_manipulation'
                    print(f"Mode: Source Manipulation - {self.selected_source_id}")
            
            # B button to return to navigation
            if self._button_pressed('b_button'):
                self.mode = 'navigation'
                print("Mode: Navigation")
        
        elif self.mode == 'source_manipulation':
            if not self.selected_source_id or self.selected_source_id not in self.sources:
                self.mode = 'navigation'
                return
            
            source = self.sources[self.selected_source_id]
            
            # Left stick: Move source position
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            if abs(left_x) > 0 or abs(left_y) > 0:
                # Update source position (scaled by dt for smooth movement)
                source['position'][0] += left_x * move_speed * dt
                source['position'][2] -= left_y * move_speed * dt
                # Update audio
                self._update_spatial_audio()
            
            # Right stick Y: Adjust distance (Z position)
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_y) < 0.1: right_y = 0
            
            if abs(right_y) > 0:
                # Move source closer or further (scaled by dt)
                source['position'][2] += right_y * move_speed * dt
                # Keep positive distance
                source['position'][2] = max(0.5, source['position'][2])
                # Update audio
                self._update_spatial_audio()
            
            # Right stick X: Adjust volume
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            if abs(right_x) > 0:
                # Adjust gain (scaled by dt)
                source['gain'] += right_x * 2.0 * dt
                # Clamp gain to reasonable range
                source['gain'] = np.clip(source['gain'], 0.0, 2.0)
                # Update audio
                self._update_spatial_audio()
            
            # A button: Toggle mute
            if self._button_pressed('a_button'):
                source['muted'] = not source['muted']
                channel = source.get('channel')
                if channel:
                    if source['muted']:
                        channel.set_volume(0, 0)
                    else:
                        # Restore volumes based on position
                        self._update_spatial_audio()
                print(f"Source {self.selected_source_id} {'muted' if source['muted'] else 'unmuted'}")
            
            # B button: Return to source selection
            if self._button_pressed('b_button'):
                self.mode = 'source_selection'
                print("Mode: Source Selection")
    
    def _select_previous_source(self):
        """Select the previous source in the list"""
        if not self.sources:
            return
            
        source_ids = list(self.sources.keys())
        if self.selected_source_id is None:
            self.selected_source_id = source_ids[0]
        else:
            current_index = source_ids.index(self.selected_source_id)
            prev_index = (current_index - 1) % len(source_ids)
            self.selected_source_id = source_ids[prev_index]
            
        print(f"Selected source: {self.selected_source_id}")
    
    def _select_next_source(self):
        """Select the next source in the list"""
        if not self.sources:
            return
            
        source_ids = list(self.sources.keys())
        if self.selected_source_id is None:
            self.selected_source_id = source_ids[0]
        else:
            current_index = source_ids.index(self.selected_source_id)
            next_index = (current_index + 1) % len(source_ids)
            self.selected_source_id = source_ids[next_index]
            
        print(f"Selected source: {self.selected_source_id}")
    
    def _update_spatial_audio(self):
        """Update audio spatialization based on listener and source positions"""
        # Very simplified spatial audio for demo
        for source_id, source in self.sources.items():
            channel = source.get('channel')
            if not channel:
                continue
                
            if source['muted']:
                channel.set_volume(0, 0)
                continue
                
            # Calculate relative position vector from listener to source
            rel_vector = source['position'] - self.position
            
            # Rotate based on listener orientation
            yaw = self.orientation[0]
            # Apply yaw rotation to determine where the source is relative to facing direction
            rotated_x = rel_vector[0] * math.cos(-yaw) - rel_vector[2] * math.sin(-yaw)
            rotated_z = rel_vector[0] * math.sin(-yaw) + rel_vector[2] * math.cos(-yaw)
            
            # Use the rotated coordinates for audio
            rel_audio_x = rotated_x
            rel_audio_z = rotated_z
            
            # Calculate distance based on 3D position
            distance = math.sqrt(rel_vector[0]**2 + rel_vector[1]**2 + rel_vector[2]**2)
            
            # Calculate stereo panning based on rotated X position
            # This makes the direction you're facing matter for audio
            pan = np.clip(rel_audio_x / 5.0, -1.0, 1.0)
            
            # Apply different distance attenuation models based on source type
            # This makes different sounds behave more realistically
            if source_id == 'ambient':
                # Ambient sounds attenuate more slowly
                distance_gain = 1.0 / max(1.0, math.sqrt(distance))
            elif source_id == 'drums':
                # Drums attenuate faster
                distance_gain = 1.0 / max(1.0, distance**1.5)
            else:
                # Standard inverse distance law
                distance_gain = 1.0 / max(1.0, distance)
            
            # Apply gain from source
            total_gain = distance_gain * source['gain']
            
            # Calculate left and right volumes with improved algorithm
            if pan <= 0:  # Source is to the left
                # Full gain on left, reduced on right based on pan
                left_vol = total_gain 
                right_vol = total_gain * (1.0 + pan)  # pan is negative here
            else:  # Source is to the right
                # Reduced gain on left, full on right based on pan
                left_vol = total_gain * (1.0 - pan)
                right_vol = total_gain
            
            # Apply distance falloff
            left_vol = np.clip(left_vol, 0.0, 1.0)
            right_vol = np.clip(right_vol, 0.0, 1.0)
            
            # Add slight interaural time difference to enhance spatial effect
            # (This would be handled through HRTF in a real spatial audio system)
            # This is just simulated by controlling which channel we update first
            if pan < 0:  # Sound coming from left
                # Update left first (it's closer), right slightly later
                channel.set_volume(left_vol, 0)
                # Short delay to simulate interaural time difference
                time.sleep(0.0001)
                channel.set_volume(left_vol, right_vol)
            else:  # Sound coming from right
                # Update right first (it's closer), left slightly later
                channel.set_volume(0, right_vol)
                # Short delay to simulate interaural time difference
                time.sleep(0.0001)
                channel.set_volume(left_vol, right_vol)
    
    def draw(self):
        """Draw the visualization on screen"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Get screen dimensions
        width, height = self.screen.get_size()
        center_x, center_y = width // 2, height // 2
        
        # Draw coordinate grid
        pygame.draw.line(self.screen, (50, 50, 50), (0, center_y), (width, center_y), 1)  # X axis
        pygame.draw.line(self.screen, (50, 50, 50), (center_x, 0), (center_x, height), 1)  # Z axis
        
        # Draw distance circles
        for d in [1, 3, 5, 10]:
            pygame.draw.circle(self.screen, (30, 30, 30), (center_x, center_y), d * 30, 1)
        
        # Scale factor for drawing
        scale = 30
        
        # Draw listener position
        listener_x = center_x + int(self.position[0] * scale)
        listener_y = center_y - int(self.position[2] * scale)  # Inverted Z for screen coordinates
        pygame.draw.circle(self.screen, (0, 255, 255), (listener_x, listener_y), 10)
        
        # Draw listener orientation
        direction_len = 30
        direction_x = listener_x + int(math.sin(self.orientation[0]) * direction_len)
        direction_y = listener_y - int(math.cos(self.orientation[0]) * direction_len)
        pygame.draw.line(self.screen, (0, 255, 255), (listener_x, listener_y), (direction_x, direction_y), 3)
        
        # Draw listener "sound cone" based on orientation
        cone_width = math.pi / 3  # 60 degree cone
        cone_len = 50
        cone_left_x = listener_x + int(math.sin(self.orientation[0] - cone_width/2) * cone_len)
        cone_left_y = listener_y - int(math.cos(self.orientation[0] - cone_width/2) * cone_len)
        cone_right_x = listener_x + int(math.sin(self.orientation[0] + cone_width/2) * cone_len)
        cone_right_y = listener_y - int(math.cos(self.orientation[0] + cone_width/2) * cone_len)
        
        # Draw sound cone as lines
        pygame.draw.line(self.screen, (0, 150, 150), (listener_x, listener_y), (cone_left_x, cone_left_y), 1)
        pygame.draw.line(self.screen, (0, 150, 150), (listener_x, listener_y), (cone_right_x, cone_right_y), 1)
        
        # Draw "ears" to show what you're hearing
        ear_distance = 8
        left_ear_x = listener_x + int(math.sin(self.orientation[0] - math.pi/2) * ear_distance)
        left_ear_y = listener_y - int(math.cos(self.orientation[0] - math.pi/2) * ear_distance)
        pygame.draw.circle(self.screen, (0, 200, 200), (left_ear_x, left_ear_y), 4)
        
        right_ear_x = listener_x + int(math.sin(self.orientation[0] + math.pi/2) * ear_distance)
        right_ear_y = listener_y - int(math.cos(self.orientation[0] + math.pi/2) * ear_distance)
        pygame.draw.circle(self.screen, (0, 200, 200), (right_ear_x, right_ear_y), 4)
        
        # Draw each sound source
        for source_id, source in self.sources.items():
            # Convert 3D position to screen coordinates
            source_x = center_x + int(source['position'][0] * scale)
            source_y = center_y - int(source['position'][2] * scale)  # Inverted Z
            
            # Determine color and size
            color = source['color']
            if source['muted']:
                # Use darker color for muted sources
                color = tuple(max(0, c // 2) for c in color)
                
            # Draw source as circle with size based on gain
            size = 8 + int(source['gain'] * 5)
            pygame.draw.circle(self.screen, color, (source_x, source_y), size)
            
            # Draw waves emanating from sound sources (animated)
            if not source['muted'] and source['gain'] > 0.1:
                wave_count = 3
                max_wave_size = 30 + size
                for i in range(wave_count):
                    # Calculate wave size based on time
                    wave_phase = (time.time() * 2 + i/wave_count) % 1.0
                    wave_size = int(wave_phase * max_wave_size)
                    if wave_size > size:
                        wave_alpha = int(255 * (1.0 - wave_phase))
                        wave_color = (*color[:3], wave_alpha)
                        # Create a transparent surface for the wave
                        wave_surf = pygame.Surface((wave_size*2, wave_size*2), pygame.SRCALPHA)
                        pygame.draw.circle(wave_surf, wave_color, (wave_size, wave_size), wave_size, 1)
                        self.screen.blit(wave_surf, (source_x - wave_size, source_y - wave_size))
            
            # Highlight selected source
            if source_id == self.selected_source_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (source_x, source_y), size + 4, 2)
            
            # Draw source name
            name_text = self.font.render(source['name'], True, color)
            self.screen.blit(name_text, (source_x + 15, source_y - 15))
            
            # For selected source in edit mode, show position and gain
            if source_id == self.selected_source_id and self.mode == 'source_manipulation':
                info_text = f"Gain: {source['gain']:.1f}, Dist: {source['position'][2]:.1f}"
                info_render = self.font.render(info_text, True, (200, 200, 200))
                self.screen.blit(info_render, (source_x + 15, source_y + 5))
                
                # Draw line from listener to selected source
                pygame.draw.line(self.screen, (100, 100, 100), 
                                (listener_x, listener_y), (source_x, source_y), 1)
        
        # Draw mode indicator
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (255, 255, 255))
        self.screen.blit(mode_text, (20, 20))
        
        # Draw listener position
        pos_text = self.font.render(f"Position: ({self.position[0]:.1f}, {self.position[2]:.1f})", True, (200, 200, 200))
        self.screen.blit(pos_text, (20, 60))
        
        # Draw orientation
        yaw_deg = (math.degrees(self.orientation[0]) + 360) % 360
        orient_text = self.font.render(f"Direction: {yaw_deg:.0f}°", True, (200, 200, 200))
        self.screen.blit(orient_text, (20, 100))
        
        # Draw selected source info
        if self.selected_source_id:
            select_text = self.font.render(f"Selected: {self.selected_source_id}", True, (255, 255, 0))
            self.screen.blit(select_text, (width - 200, 20))
        
        # Draw instructions based on mode
        instruction_text = ""
        if self.mode == 'navigation':
            instruction_text = "Left stick: Move | Right stick: Look | Y: Select sources"
        elif self.mode == 'source_selection':
            instruction_text = "D-pad: Select source | A: Edit source | B: Back"
        elif self.mode == 'source_manipulation':
            instruction_text = "Left: Move source | Right: Volume/Distance | A: Mute | B: Back"
            
        instr = self.font.render(instruction_text, True, (200, 200, 200))
        self.screen.blit(instr, (20, height - 40))
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Run the demo"""
        self.running = True
        clock = pygame.time.Clock()
        
        print("\n== SHAC Simple Demo with Xbox Controller ==")
        print("Controls:")
        print("  Navigation Mode:")
        print("    Left stick: Move in space (relative to where you're facing)")
        print("    Right stick: Turn and look around")
        print("    Triggers/Shoulders: Move up/down")
        print("    Y button: Switch to source selection")
        print("  Source Selection Mode:")
        print("    D-pad up/down: Select source")
        print("    A button: Edit selected source")
        print("    B button: Back to navigation")
        print("  Source Manipulation Mode:")
        print("    Left stick: Move source in space")
        print("    Right stick X: Adjust volume")
        print("    Right stick Y: Adjust distance")
        print("    A button: Mute/unmute")
        print("    B button: Back to selection")
        print("\nClose window or press Esc to exit")
        
        try:
            while self.running:
                # Check for exit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                
                # Process controller input
                self.process_input()
                
                # Draw visualization
                self.draw()
                
                # Limit to 60 FPS
                clock.tick(60)
        
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Stop all sounds
            self.looping_sounds = False
            for source_id, source_data in self.sources.items():
                channel = source_data.get('channel')
                if channel:
                    channel.stop()
            
            # Clean up pygame
            pygame.quit()
            print("Demo stopped")


if __name__ == "__main__":
    # Start the demo
    demo = SimpleDemo()
    demo.run()