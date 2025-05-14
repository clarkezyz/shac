"""
SHAC Simple Demo with Intuitive Controls

A spatial audio demonstration with simplified, intuitive controls
that focuses on natural movement and exploration of the sound field.

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

class SpatialAudioDemo:
    """A simplified spatial audio demo with intuitive controls"""
    
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
        }
        
        # Navigation state - separated into absolute position and facing
        self.position = np.array([0.0, 1.7, 0.0])  # x, y, z (x=right, y=up, z=forward)
        self.facing = 0.0  # Rotation around Y axis (0 = facing +Z, increases clockwise)
        self.mode = 'navigation'
        
        # Camera parameters
        self.head_height = 1.7  # Average human height in meters
        
        # Sound sources
        self.sources = {}
        self.selected_source_id = None
        
        # Create sample sounds
        self._create_sound_sources()
        
        # Audio output
        self.sounds = {}
        self.looping_sounds = True  # Control for sound looping
        
        # Create sound channels and start playback
        self._initialize_audio()
        
        # Setup visualization
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Spatial Audio - Intuitive Demo")
        self.font = pygame.font.Font(None, 36)
        
        # Thread control
        self.running = False
        
        # Last button states to prevent immediate repeat presses
        self.last_button_states = {name: False for name in self.button_map.keys()}
        
        # Time tracking for delta-based movement
        self.last_time = time.time()
        
        # Movement settings
        self.move_speed = 3.0  # meters per second
        self.turn_speed = 2.0  # radians per second
        
        print("Spatial Audio demo initialized")
    
    def _create_sound_sources(self):
        """Create sample sound sources with interesting spatial arrangements"""
        duration = 10.0  # longer sounds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano - creates a sequence of harmonically rich piano notes
        piano_audio = np.zeros_like(t)
        # Notes in C major scale
        notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
        
        # Create a simple melody
        melody = [0, 2, 4, 7, 4, 2, 4, 0]  # C, E, G, C5, G, E, G, C
        note_duration = 1.0  # seconds per note
        
        for i, note_idx in enumerate(melody):
            # Calculate start time and corresponding sample index
            start_time = i * note_duration
            if start_time >= duration:
                break
                
            start_idx = int(start_time * self.sample_rate)
            note_len = int(note_duration * self.sample_rate * 0.9)  # 90% of full duration for slight separation
            
            # Get frequency for this note
            freq = notes[note_idx]
            
            # Create a note with attack, decay, sustain, release envelope
            t_note = np.linspace(0, note_duration, note_len)
            
            # ADSR envelope
            attack = 0.1  # seconds
            decay = 0.2
            sustain_level = 0.7
            release = 0.3
            
            # Generate envelope
            envelope = np.ones_like(t_note)
            attack_samples = int(attack * self.sample_rate)
            decay_samples = int(decay * self.sample_rate)
            release_samples = int(release * self.sample_rate)
            
            # Attack phase
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            # Decay phase
            if decay_samples > 0:
                decay_end = attack_samples + decay_samples
                if decay_end < len(envelope):
                    envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)
            
            # Release phase
            if release_samples > 0 and len(envelope) > release_samples:
                envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
            
            # Generate harmonically rich tone
            note = np.zeros_like(t_note)
            # Fundamental
            note += 0.5 * np.sin(2 * np.pi * freq * t_note)
            # Harmonics with decreasing amplitude
            note += 0.25 * np.sin(2 * np.pi * 2 * freq * t_note)
            note += 0.125 * np.sin(2 * np.pi * 3 * freq * t_note)
            note += 0.0625 * np.sin(2 * np.pi * 4 * freq * t_note)
            
            # Apply envelope
            note *= envelope
            
            # Add to main audio
            if start_idx + len(note) <= len(piano_audio):
                piano_audio[start_idx:start_idx+len(note)] += note
        
        # Normalize
        piano_audio = piano_audio / np.max(np.abs(piano_audio)) * 0.9
        
        # Place piano in front-right
        self.sources['piano'] = {
            'audio': piano_audio,
            'position': np.array([3.0, self.head_height, 5.0]),  # x, y, z
            'name': 'Piano',
            'color': (220, 120, 50),  # Orange-brown
            'gain': 1.0,
            'muted': False
        }
        
        # Water stream sound - continuous babbling brook
        water_audio = np.zeros_like(t)
        for i in range(0, len(t), 1000):
            # Add filtered noise bursts
            length = min(3000, len(t) - i)
            if length <= 0:
                break
                
            noise = np.random.randn(length) * 0.4
            
            # Apply bandpass filter (very simplified)
            filtered = np.zeros_like(noise)
            filtered[0] = noise[0]
            
            # Apply different filter coefficients to create water-like sound
            for j in range(1, len(filtered)):
                filtered[j] = 0.7 * filtered[j-1] + 0.3 * noise[j]
            
            # Apply amplitude modulation
            mod_freq = 2.0 + 0.5 * np.sin(i / self.sample_rate)  # Varying modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * np.linspace(0, length/self.sample_rate, length))
            
            # Add to main audio
            water_audio[i:i+length] += filtered * mod
        
        # Add some high frequency components for sparkle
        sparkle = np.random.randn(len(t)) * 0.1
        for i in range(10, len(sparkle)):
            sparkle[i] = 0.05 * sparkle[i] + 0.95 * sparkle[i-1]
        
        water_audio += sparkle
        
        # Normalize
        water_audio = water_audio / np.max(np.abs(water_audio)) * 0.8
        
        # Place water sound to the left
        self.sources['water'] = {
            'audio': water_audio,
            'position': np.array([-4.0, 0.5, 2.0]),  # x, y, z
            'name': 'Water',
            'color': (50, 150, 255),  # Blue
            'gain': 1.0,
            'muted': False
        }
        
        # Wind sound - ambient environmental sound
        wind_audio = np.zeros_like(t)
        
        # Base noise
        base_noise = np.random.randn(len(t)) * 0.3
        
        # Apply filter for wind-like spectrum
        filtered_wind = np.zeros_like(base_noise)
        filtered_wind[0] = base_noise[0]
        
        # Low-pass filter with time-varying coefficients
        for i in range(1, len(filtered_wind)):
            # Slowly varying filter coefficient
            alpha = 0.97 + 0.02 * np.sin(2 * np.pi * 0.05 * i / self.sample_rate)
            filtered_wind[i] = alpha * filtered_wind[i-1] + (1-alpha) * base_noise[i]
        
        # Add slow amplitude modulation
        t_mod = np.linspace(0, duration, len(t))
        mod1 = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t_mod)
        mod2 = 0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * t_mod + 0.5)
        
        # Apply modulation
        wind_audio = filtered_wind * mod1 * mod2
        
        # Add occasional wind gusts
        for i in range(5):
            # Random gust position
            gust_pos = int(np.random.rand() * (len(t) - self.sample_rate))
            gust_len = int((0.5 + 0.5 * np.random.rand()) * self.sample_rate)
            
            # Create gust envelope
            gust_env = np.zeros(gust_len)
            attack = int(gust_len * 0.3)
            release = int(gust_len * 0.7)
            
            gust_env[:attack] = np.linspace(0, 1, attack)
            gust_env[attack:] = np.linspace(1, 0, gust_len - attack)
            
            # Apply gust
            if gust_pos + gust_len < len(wind_audio):
                wind_audio[gust_pos:gust_pos+gust_len] += 0.5 * np.random.randn(gust_len) * gust_env
        
        # Normalize
        wind_audio = wind_audio / np.max(np.abs(wind_audio)) * 0.7
        
        # Place wind sound all around (distant)
        self.sources['wind'] = {
            'audio': wind_audio,
            'position': np.array([0.0, 4.0, -8.0]),  # x, y, z (behind and above)
            'name': 'Wind',
            'color': (200, 200, 255),  # Light blue-white
            'gain': 0.6,
            'muted': False
        }
        
        # Heartbeat - intermittent low frequency pulses
        heartbeat_audio = np.zeros_like(t)
        
        # Create a rhythmic heartbeat pattern
        beat_interval = 0.8  # seconds between beats
        double_beat = True  # Typical "lub-dub" heartbeat
        
        for i in range(int(duration / beat_interval)):
            # Calculate the position of this beat
            beat_pos = int(i * beat_interval * self.sample_rate)
            
            # Create primary beat
            beat_len = int(0.15 * self.sample_rate)
            if beat_pos + beat_len >= len(heartbeat_audio):
                break
                
            # Beat envelope
            beat_env = np.exp(-np.linspace(0, 8, beat_len))
            
            # Low frequency beat (around 60-80Hz)
            beat_freq = 70
            beat = np.sin(2 * np.pi * beat_freq * np.linspace(0, beat_len/self.sample_rate, beat_len))
            
            # Apply envelope
            beat = beat * beat_env * 0.9
            
            # Add to audio
            heartbeat_audio[beat_pos:beat_pos+beat_len] += beat
            
            # Add second beat (dub) in the double beat pattern
            if double_beat:
                dub_pos = beat_pos + int(0.25 * self.sample_rate)
                if dub_pos + beat_len < len(heartbeat_audio):
                    # Similar beat but shorter and softer
                    dub_env = np.exp(-np.linspace(0, 10, beat_len))
                    dub = np.sin(2 * np.pi * (beat_freq*0.8) * np.linspace(0, beat_len/self.sample_rate, beat_len))
                    dub = dub * dub_env * 0.7
                    
                    heartbeat_audio[dub_pos:dub_pos+beat_len] += dub
        
        # Add slight reverb effect (simple approximation)
        reverb = np.zeros_like(heartbeat_audio)
        for i in range(len(heartbeat_audio)):
            decay = 0.6
            delay = int(0.05 * self.sample_rate)  # 50ms delay
            
            if i >= delay:
                reverb[i] = heartbeat_audio[i-delay] * decay
        
        heartbeat_audio += reverb
        
        # Normalize
        heartbeat_audio = heartbeat_audio / np.max(np.abs(heartbeat_audio)) * 0.9
        
        # Place heartbeat nearby for an ominous effect
        self.sources['heartbeat'] = {
            'audio': heartbeat_audio,
            'position': np.array([0.0, 1.0, -1.5]),  # x, y, z (behind the listener)
            'name': 'Heartbeat',
            'color': (200, 0, 0),  # Red
            'gain': 0.8,
            'muted': False
        }
        
        # Bell chimes - spatial markers at different positions
        bell_audio = np.zeros_like(t)
        
        # Create occasional bell chimes
        chime_times = [0.5, 3.0, 5.5, 8.0]
        bell_freqs = [1000, 1200, 800, 1500]  # Different pitches
        
        for i, chime_time in enumerate(chime_times):
            chime_pos = int(chime_time * self.sample_rate)
            if chime_pos >= len(bell_audio):
                continue
                
            # Bell duration
            chime_len = int(2.0 * self.sample_rate)
            if chime_pos + chime_len > len(bell_audio):
                chime_len = len(bell_audio) - chime_pos
            
            # Bell envelope - sharp attack, long decay
            chime_env = np.exp(-np.linspace(0, 6, chime_len))
            
            # Base frequency and harmonic structure for bell-like sound
            freq = bell_freqs[i % len(bell_freqs)]
            
            # Create bell with complex harmonic structure
            bell = np.zeros(chime_len)
            
            # Fundamental
            bell += 0.6 * np.sin(2 * np.pi * freq * np.linspace(0, chime_len/self.sample_rate, chime_len))
            
            # Minor third partial (creates bell-like quality)
            bell += 0.3 * np.sin(2 * np.pi * freq * 1.2 * np.linspace(0, chime_len/self.sample_rate, chime_len))
            
            # Perfect fifth
            bell += 0.15 * np.sin(2 * np.pi * freq * 1.5 * np.linspace(0, chime_len/self.sample_rate, chime_len))
            
            # Octave
            bell += 0.1 * np.sin(2 * np.pi * freq * 2.0 * np.linspace(0, chime_len/self.sample_rate, chime_len))
            
            # Apply envelope
            bell = bell * chime_env
            
            # Add to audio
            bell_audio[chime_pos:chime_pos+chime_len] += bell
        
        # Normalize
        bell_audio = bell_audio / np.max(np.abs(bell_audio)) * 0.7
        
        # Place bells above and around
        self.sources['bells'] = {
            'audio': bell_audio,
            'position': np.array([2.0, 3.0, -3.0]),  # x, y, z (above and to the right)
            'name': 'Bells',
            'color': (255, 220, 100),  # Gold
            'gain': 0.7,
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
        
        # Get controller input based on current mode
        if self.mode == 'navigation':
            # Greatly simplified controls - left stick directly controls XZ movement
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            # Right stick simply turns
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            # Handle movement first - directly map left stick to XZ movement, 
            # but independent of facing direction for simplicity
            self.position[0] += left_x * self.move_speed * dt  # X axis (left/right)
            self.position[2] -= left_y * self.move_speed * dt  # Z axis (forward/back)
            
            # Separately handle rotation with right stick
            if abs(right_x) > 0:
                self.facing += right_x * self.turn_speed * dt
            
            # Up/Down movement with shoulders
            lb = self.controller.get_button(self.button_map['left_shoulder'])
            rb = self.controller.get_button(self.button_map['right_shoulder'])
            if lb: self.position[1] -= self.move_speed * dt  # Move down
            if rb: self.position[1] += self.move_speed * dt  # Move up
            
            # Keep within reasonable height
            self.position[1] = max(0.5, min(5.0, self.position[1]))
            
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
                source['position'][0] += left_x * self.move_speed * dt
                source['position'][2] -= left_y * self.move_speed * dt
                # Update audio
                self._update_spatial_audio()
            
            # Right stick Y: Adjust height
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_y) < 0.1: right_y = 0
            
            if abs(right_y) > 0:
                # Move source up/down
                source['position'][1] -= right_y * self.move_speed * dt
                # Keep positive height
                source['position'][1] = max(0.1, source['position'][1])
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
                
            # Calculate relative position from listener to source
            rel_vector = source['position'] - self.position
            
            # Calculate distance
            distance = math.sqrt(rel_vector[0]**2 + rel_vector[1]**2 + rel_vector[2]**2)
            
            # Rotate to account for listener's facing direction
            # Find angle between listener orientation and source direction in horizontal plane
            source_direction = math.atan2(rel_vector[0], rel_vector[2])
            
            # Adjust for listener's facing direction
            relative_angle = source_direction - self.facing
            
            # Normalize to [-π, π]
            relative_angle = ((relative_angle + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate binaural panning based on relative angle
            # This is the key factor that makes the sound appear to come from a specific direction
            pan = np.clip(relative_angle / (math.pi/2), -1.0, 1.0)
            
            # Different distance attenuation curves for different source types
            if source_id == 'wind':
                # Wind attenuates very slowly with distance
                distance_gain = 1.0 / max(1.0, math.sqrt(distance/3))
            elif source_id == 'water':
                # Water attenuates at a moderate rate
                distance_gain = 1.0 / max(1.0, distance)
            elif source_id == 'heartbeat':
                # Heartbeat has steep falloff
                distance_gain = 1.0 / max(1.0, distance**2)
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
            
            # Apply elevation factor - sounds from above or below are harder to localize
            elevation_factor = abs(math.atan2(rel_vector[1], math.sqrt(rel_vector[0]**2 + rel_vector[2]**2)))
            if elevation_factor > 0.1:
                # Reduce stereo separation for elevated sounds
                mid_vol = (left_vol + right_vol) / 2
                left_vol = mid_vol + (left_vol - mid_vol) * (1 - elevation_factor/math.pi)
                right_vol = mid_vol + (right_vol - mid_vol) * (1 - elevation_factor/math.pi)
            
            # Apply distance falloff
            left_vol = np.clip(left_vol, 0.0, 1.0)
            right_vol = np.clip(right_vol, 0.0, 1.0)
            
            # Set final volumes
            channel.set_volume(left_vol, right_vol)
    
    def draw(self):
        """Draw the visualization on screen"""
        # Clear screen
        self.screen.fill((10, 10, 30))  # Dark blue background
        
        # Get screen dimensions
        width, height = self.screen.get_size()
        center_x, center_y = width // 2, height // 2
        
        # Draw coordinate grid
        for i in range(0, width, 50):
            alpha = 30 + 20 * (1 if i == center_x else 0)
            pygame.draw.line(self.screen, (alpha, alpha, alpha), (i, 0), (i, height), 1)
        
        for i in range(0, height, 50):
            alpha = 30 + 20 * (1 if i == center_y else 0)
            pygame.draw.line(self.screen, (alpha, alpha, alpha), (0, i), (width, i), 1)
        
        # Draw distance circles
        for d in [1, 3, 5, 10]:
            pygame.draw.circle(self.screen, (40, 40, 60), (center_x, center_y), d * 30, 1)
        
        # Scale factor for drawing
        scale = 30
        
        # Draw and label cardinal directions
        # North (forward)
        north_x = center_x
        north_y = center_y - 150
        pygame.draw.line(self.screen, (150, 150, 150), (center_x, center_y), (north_x, north_y), 1)
        north_label = self.font.render("N", True, (150, 150, 150))
        self.screen.blit(north_label, (north_x - 8, north_y - 25))
        
        # East (right)
        east_x = center_x + 150
        east_y = center_y
        pygame.draw.line(self.screen, (150, 150, 150), (center_x, center_y), (east_x, east_y), 1)
        east_label = self.font.render("E", True, (150, 150, 150))
        self.screen.blit(east_label, (east_x + 10, east_y - 10))
        
        # South (backward)
        south_x = center_x
        south_y = center_y + 150
        pygame.draw.line(self.screen, (150, 150, 150), (center_x, center_y), (south_x, south_y), 1)
        south_label = self.font.render("S", True, (150, 150, 150))
        self.screen.blit(south_label, (south_x - 8, south_y + 10))
        
        # West (left)
        west_x = center_x - 150
        west_y = center_y
        pygame.draw.line(self.screen, (150, 150, 150), (center_x, center_y), (west_x, west_y), 1)
        west_label = self.font.render("W", True, (150, 150, 150))
        self.screen.blit(west_label, (west_x - 25, west_y - 10))
        
        # Draw listener position
        listener_x = center_x + int(self.position[0] * scale)
        listener_y = center_y - int(self.position[2] * scale)  # Inverted Z for screen coordinates
        
        # Draw listener rotation
        # Calculate the endpoint of the facing vector
        facing_len = 20
        facing_x = listener_x + int(math.sin(self.facing) * facing_len)
        facing_y = listener_y - int(math.cos(self.facing) * facing_len)
        
        # Draw listener
        # Head
        pygame.draw.circle(self.screen, (0, 255, 255), (listener_x, listener_y), 10)
        
        # Draw direction indicator (nose)
        pygame.draw.line(self.screen, (0, 255, 255), (listener_x, listener_y), (facing_x, facing_y), 3)
        
        # Draw field of view
        fov = math.pi/2  # 90 degrees field of view
        left_x = listener_x + int(math.sin(self.facing - fov/2) * facing_len * 1.2)
        left_y = listener_y - int(math.cos(self.facing - fov/2) * facing_len * 1.2)
        right_x = listener_x + int(math.sin(self.facing + fov/2) * facing_len * 1.2)
        right_y = listener_y - int(math.cos(self.facing + fov/2) * facing_len * 1.2)
        
        pygame.draw.line(self.screen, (0, 200, 200), (listener_x, listener_y), (left_x, left_y), 1)
        pygame.draw.line(self.screen, (0, 200, 200), (listener_x, listener_y), (right_x, right_y), 1)
        
        # Draw "ears"
        ear_distance = 6
        left_ear_angle = self.facing - math.pi/2
        right_ear_angle = self.facing + math.pi/2
        
        left_ear_x = listener_x + int(math.sin(left_ear_angle) * ear_distance)
        left_ear_y = listener_y - int(math.cos(left_ear_angle) * ear_distance)
        pygame.draw.circle(self.screen, (0, 200, 200), (left_ear_x, left_ear_y), 3)
        
        right_ear_x = listener_x + int(math.sin(right_ear_angle) * ear_distance)
        right_ear_y = listener_y - int(math.cos(right_ear_angle) * ear_distance)
        pygame.draw.circle(self.screen, (0, 200, 200), (right_ear_x, right_ear_y), 3)
        
        # Draw each sound source
        for source_id, source in self.sources.items():
            # Convert 3D position to screen coordinates
            source_x = center_x + int(source['position'][0] * scale)
            source_y = center_y - int(source['position'][2] * scale)  # Inverted Z
            
            # Determine color and size
            color = source['color']
            if source['muted']:
                # Use darker color for muted sources
                color = tuple(max(0, c // 3) for c in color)
                
            # Calculate size based on vertical position (perspective)
            y_perspective = 1 - min(1, max(0, (source['position'][1] - self.position[1]) / 10))
            perspective_scale = 0.5 + 0.5 * y_perspective
            
            # Draw source as circle with size based on gain and perspective
            base_size = 6 + int(source['gain'] * 4)
            size = int(base_size * perspective_scale)
            
            pygame.draw.circle(self.screen, color, (source_x, source_y), size)
            
            # Draw vertical line to show height
            ground_y = source_y + int(source['position'][1] * scale * perspective_scale)
            pygame.draw.line(self.screen, (*color[:3], 100), (source_x, source_y), (source_x, ground_y), 1)
            
            # Draw height marker on ground
            pygame.draw.circle(self.screen, (*color[:3], 100), (source_x, ground_y), 2)
            
            # Draw waves emanating from sound sources (animated)
            if not source['muted'] and source['gain'] > 0.1:
                wave_count = 3
                max_wave_size = 30 + size
                for i in range(wave_count):
                    # Calculate wave size based on time
                    wave_phase = (time.time() * 1.5 + i/wave_count) % 1.0
                    wave_size = int(wave_phase * max_wave_size)
                    if wave_size > size:
                        wave_alpha = int(100 * (1.0 - wave_phase))
                        wave_color = (*color[:3], wave_alpha)
                        # Create a transparent surface for the wave
                        wave_surf = pygame.Surface((wave_size*2, wave_size*2), pygame.SRCALPHA)
                        pygame.draw.circle(wave_surf, wave_color, (wave_size, wave_size), wave_size, 1)
                        self.screen.blit(wave_surf, (source_x - wave_size, source_y - wave_size))
            
            # Highlight selected source
            if source_id == self.selected_source_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (source_x, source_y), size + 4, 2)
            
            # Draw source name with shadow for better readability
            name_text = self.font.render(source['name'], True, color)
            shadow_text = self.font.render(source['name'], True, (0, 0, 0))
            
            # Add shadow offset
            name_x = source_x + 15
            name_y = source_y - 15
            self.screen.blit(shadow_text, (name_x + 1, name_y + 1))
            self.screen.blit(name_text, (name_x, name_y))
            
            # For selected source in edit mode, show position and gain
            if source_id == self.selected_source_id and self.mode == 'source_manipulation':
                height_text = f"Height: {source['position'][1]:.1f}m, Gain: {source['gain']:.1f}"
                info_render = self.font.render(height_text, True, (180, 180, 180))
                shadow_render = self.font.render(height_text, True, (0, 0, 0))
                self.screen.blit(shadow_render, (source_x + 16, source_y + 6))
                self.screen.blit(info_render, (source_x + 15, source_y + 5))
                
                # Draw line from listener to selected source
                pygame.draw.line(self.screen, (100, 100, 100), 
                                (listener_x, listener_y), (source_x, source_y), 1)
                
                # Calculate and show the angle to the source
                source_angle = math.atan2(source['position'][0] - self.position[0], 
                                        source['position'][2] - self.position[2])
                relative_angle = (source_angle - self.facing + math.pi) % (2*math.pi) - math.pi
                angle_deg = math.degrees(relative_angle)
                
                angle_text = f"Angle: {angle_deg:.0f}°"
                angle_render = self.font.render(angle_text, True, (180, 180, 180))
                shadow_angle = self.font.render(angle_text, True, (0, 0, 0))
                self.screen.blit(shadow_angle, (source_x + 16, source_y + 31))
                self.screen.blit(angle_render, (source_x + 15, source_y + 30))
        
        # Draw control mode
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (220, 220, 220))
        mode_shadow = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (0, 0, 0))
        self.screen.blit(mode_shadow, (21, 21))
        self.screen.blit(mode_text, (20, 20))
        
        # Draw listener info
        pos_text = f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})"
        pos_render = self.font.render(pos_text, True, (200, 200, 200))
        pos_shadow = self.font.render(pos_text, True, (0, 0, 0))
        self.screen.blit(pos_shadow, (21, 51))
        self.screen.blit(pos_render, (20, 50))
        
        # Draw orientation
        facing_deg = (math.degrees(self.facing) + 360) % 360
        orient_text = f"Facing: {facing_deg:.0f}°"
        orient_render = self.font.render(orient_text, True, (200, 200, 200))
        orient_shadow = self.font.render(orient_text, True, (0, 0, 0))
        self.screen.blit(orient_shadow, (21, 81))
        self.screen.blit(orient_render, (20, 80))
        
        # Draw selected source info
        if self.selected_source_id:
            select_text = f"Selected: {self.selected_source_id}"
            select_render = self.font.render(select_text, True, (255, 255, 100))
            select_shadow = self.font.render(select_text, True, (0, 0, 0))
            self.screen.blit(select_shadow, (width - 199, 21))
            self.screen.blit(select_render, (width - 200, 20))
        
        # Draw instructions based on mode
        instruction_text = ""
        if self.mode == 'navigation':
            instruction_text = "Left stick: Move | Right stick: Turn | Y: Select sources"
        elif self.mode == 'source_selection':
            instruction_text = "D-pad: Select source | A: Edit source | B: Back"
        elif self.mode == 'source_manipulation':
            instruction_text = "Left: Move XZ | Right: Height/Volume | A: Mute | B: Back"
            
        instr_render = self.font.render(instruction_text, True, (220, 220, 220))
        instr_shadow = self.font.render(instruction_text, True, (0, 0, 0))
        self.screen.blit(instr_shadow, (21, height - 39))
        self.screen.blit(instr_render, (20, height - 40))
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Run the demo"""
        self.running = True
        clock = pygame.time.Clock()
        
        print("\n== Spatial Audio - Intuitive Demo ==")
        print("Controls:")
        print("  Navigation Mode:")
        print("    Left stick: Move left/right/forward/back")
        print("    Right stick: Turn to change facing direction")
        print("    Shoulder buttons: Move up/down")
        print("    Y button: Switch to source selection")
        print("  Source Selection Mode:")
        print("    D-pad up/down: Select source")
        print("    A button: Edit selected source")
        print("    B button: Back to navigation")
        print("  Source Manipulation Mode:")
        print("    Left stick: Move source horizontally")
        print("    Right stick Y: Adjust source height")
        print("    Right stick X: Adjust volume")
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
    demo = SpatialAudioDemo()
    demo.run()