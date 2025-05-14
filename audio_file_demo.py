"""
SHAC Audio File Demo

A demonstration of the SHAC system using real audio files with 
spatial positioning and interactive navigation.

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
from typing import Dict, List, Optional, Tuple

# Import our modules
from file_loader import load_audio_file, list_audio_files, create_audio_directory
from audio_utils import create_sample_audio_files

# Initialize pygame for controller and audio
pygame.init()
pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=1024)

class AudioFileDemo:
    """A spatial audio demo using real audio files"""
    
    def __init__(self):
        """Initialize the demo with real audio files"""
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
        
        # Navigation state
        self.position = np.array([0.0, 1.7, 0.0])  # x, y, z (x=right, y=up, z=forward)
        self.facing = 0.0  # Rotation around Y axis (0 = facing +Z, increases clockwise)
        self.mode = 'navigation'
        
        # Sound sources
        self.sources = {}
        self.selected_source_id = None
        
        # Load audio files and create sound sources
        self._initialize_audio_files()
        
        # Audio output
        self.sounds = {}
        self.looping_sounds = True
        
        # Create sound channels and start playback
        self._initialize_audio()
        
        # Setup visualization
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("SHAC - Audio File Demo")
        self.font = pygame.font.Font(None, 36)
        
        # Thread control
        self.running = False
        
        # Movement settings
        self.move_speed = 3.0  # meters per second
        self.turn_speed = 2.0  # radians per second
        
        # Last button states to prevent immediate repeat presses
        self.last_button_states = {name: False for name in self.button_map.keys()}
        
        # Time tracking for delta-based movement
        self.last_time = time.time()
        
        print("Audio File Demo initialized")
    
    def _initialize_audio_files(self):
        """Load audio files and create sound sources"""
        # Ensure audio directory exists
        audio_dir = create_audio_directory()
        
        # Find audio files
        audio_files = list_audio_files(audio_dir)
        
        # If no files found, create sample files
        if not audio_files:
            print("No audio files found. Creating sample audio files...")
            create_sample_audio_files()
            # Get the newly created files
            audio_files = list_audio_files(audio_dir)
        
        print(f"Found {len(audio_files)} audio files")
        
        # Define positions for audio sources in 3D space
        positions = [
            np.array([3.0, 1.0, 5.0]),    # Front right
            np.array([-3.0, 0.5, 4.0]),   # Front left
            np.array([0.0, 2.0, -4.0]),   # Behind
            np.array([5.0, 1.2, 0.0]),    # Right
            np.array([-5.0, 1.5, 0.0]),   # Left
            np.array([0.0, 0.5, 6.0]),    # Front
        ]
        
        # Define colors for visualization
        colors = [
            (220, 100, 50),   # Orange-red
            (50, 150, 255),   # Blue
            (200, 200, 255),  # Light blue
            (200, 50, 50),    # Red
            (50, 200, 100),   # Green
            (200, 200, 50),   # Yellow
        ]
        
        # Load each audio file and create a source
        for i, file_path in enumerate(audio_files[:min(len(positions), len(audio_files))]):
            try:
                # Get base filename without extension
                filename = os.path.basename(file_path)
                name = os.path.splitext(filename)[0].capitalize()
                
                # Load audio file
                print(f"Loading {file_path}...")
                audio_info = load_audio_file(file_path, self.sample_rate)
                
                # Create source
                self.sources[name] = {
                    'audio': audio_info['audio_data'],
                    'sample_rate': audio_info['sample_rate'],
                    'position': positions[i % len(positions)],
                    'name': name,
                    'color': colors[i % len(colors)],
                    'gain': 1.0,
                    'muted': False,
                    'file_path': file_path
                }
                
                print(f"Added source '{name}' from {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # If no files were successfully loaded, create some synthetic ones
        if not self.sources:
            print("No audio files could be loaded. Creating synthetic sources instead.")
            self._create_synthetic_sources()
    
    def _create_synthetic_sources(self):
        """Create synthetic sound sources as a fallback"""
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano-like sound
        piano_audio = 0.7 * np.sin(2 * np.pi * 440 * t) * np.exp(-t/2)
        self.sources['Piano'] = {
            'audio': piano_audio,
            'sample_rate': self.sample_rate,
            'position': np.array([3.0, 1.0, 5.0]),
            'name': 'Piano',
            'color': (220, 100, 50),
            'gain': 1.0,
            'muted': False
        }
        
        # Bass-like sound
        bass_audio = 0.7 * np.sin(2 * np.pi * 110 * t)
        self.sources['Bass'] = {
            'audio': bass_audio,
            'sample_rate': self.sample_rate,
            'position': np.array([-3.0, 0.5, 4.0]),
            'name': 'Bass',
            'color': (50, 150, 255),
            'gain': 1.0,
            'muted': False
        }
        
        # Drum-like sound
        drum_audio = np.zeros_like(t)
        for i in range(0, len(t), int(self.sample_rate / 2)):
            if i + 1000 < len(drum_audio):
                drum_audio[i:i+1000] = 0.5 * np.exp(-np.linspace(0, 10, 1000)) * np.random.randn(1000)
        self.sources['Drums'] = {
            'audio': drum_audio,
            'sample_rate': self.sample_rate,
            'position': np.array([0.0, 2.0, -4.0]),
            'name': 'Drums',
            'color': (200, 50, 50),
            'gain': 1.0,
            'muted': False
        }
    
    def _initialize_audio(self):
        """Initialize audio playback for each source"""
        for source_id, source_data in self.sources.items():
            # Convert to 16-bit for pygame
            data = (source_data['audio'] * 32767).astype(np.int16)
            
            # Make it stereo
            stereo_data = np.column_stack((data, data))
            
            # Create pygame sound
            try:
                self.sounds[source_id] = pygame.mixer.Sound(buffer=stereo_data)
                
                # Create a channel for this sound
                channel = pygame.mixer.find_channel()
                if channel:
                    source_data['channel'] = channel
                    channel.play(self.sounds[source_id], loops=-1)
                    # Start with spatialized volume
                    self._update_spatial_audio()
                else:
                    print(f"Warning: Could not find free channel for {source_id}")
                    source_data['channel'] = None
            except Exception as e:
                print(f"Error creating sound for {source_id}: {str(e)}")
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
                    if source_id in self.sounds:
                        channel.play(self.sounds[source_id], loops=-1)
                        print(f"Restarted sound: {source_id}")
            # Check every second
            time.sleep(1)
    
    def _button_pressed(self, button_name):
        """Check if a button was just pressed (handles debouncing)"""
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
        
        # Handle keyboard input if no controller
        if self.controller is None:
            keys = pygame.key.get_pressed()
            # Handle WASD and arrow keys for movement
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.position[2] += self.move_speed * dt  # Forward
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.position[2] -= self.move_speed * dt  # Backward
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.position[0] -= self.move_speed * dt  # Left
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.position[0] += self.move_speed * dt  # Right
            if keys[pygame.K_q]:
                self.facing -= self.turn_speed * dt  # Turn left
            if keys[pygame.K_e]:
                self.facing += self.turn_speed * dt  # Turn right
            if keys[pygame.K_r]:
                self.position[1] += self.move_speed * dt  # Up
            if keys[pygame.K_f]:
                self.position[1] -= self.move_speed * dt  # Down
                self.position[1] = max(0.5, self.position[1])  # Don't go below ground
            
            # Update audio based on position
            self._update_spatial_audio()
            return
        
        # Process controller input based on current mode
        if self.mode == 'navigation':
            # Left stick: Move in XZ plane
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            # Move in XZ plane
            self.position[0] += left_x * self.move_speed * dt  # X axis (left/right)
            self.position[2] -= left_y * self.move_speed * dt  # Z axis (forward/back)
            
            # Right stick: Turn (yaw)
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            # Update facing direction
            self.facing += right_x * self.turn_speed * dt
            
            # Shoulder buttons: Move up/down
            if self.controller.get_button(self.button_map['left_shoulder']):
                self.position[1] -= self.move_speed * dt  # Down
            if self.controller.get_button(self.button_map['right_shoulder']):
                self.position[1] += self.move_speed * dt  # Up
            
            # Ensure Y position stays above ground
            self.position[1] = max(0.5, self.position[1])
            
            # Update audio spatialization
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
                hat = self.controller.get_hat(0)  # D-pad
                
                if hat[1] == 1:  # D-pad up
                    self._select_previous_source()
                    time.sleep(0.2)  # Debounce
                elif hat[1] == -1:  # D-pad down
                    self._select_next_source()
                    time.sleep(0.2)  # Debounce
            except:
                # Fallback if d-pad not available
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
            
            # Left stick: Move source in XZ plane
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            if abs(left_x) > 0 or abs(left_y) > 0:
                # Update source position
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
                # Keep positive height (above ground)
                source['position'][1] = max(0.1, source['position'][1])
                # Update audio
                self._update_spatial_audio()
            
            # Right stick X: Adjust volume
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            if abs(right_x) > 0:
                # Adjust gain
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
        for source_id, source in self.sources.items():
            channel = source.get('channel')
            if not channel:
                continue
                
            if source['muted']:
                channel.set_volume(0, 0)
                continue
                
            # Calculate relative position vector
            rel_vector = source['position'] - self.position
            
            # Calculate distance
            distance = math.sqrt(rel_vector[0]**2 + rel_vector[1]**2 + rel_vector[2]**2)
            
            # Calculate angle between listener orientation and source direction in horizontal plane
            source_direction = math.atan2(rel_vector[0], rel_vector[2])
            
            # Adjust for listener's facing direction
            relative_angle = source_direction - self.facing
            
            # Normalize to [-π, π]
            relative_angle = ((relative_angle + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate stereo panning based on relative angle
            pan = np.clip(relative_angle / (math.pi/2), -1.0, 1.0)
            
            # Calculate distance attenuation
            distance_gain = 1.0 / max(1.0, distance)
            
            # Apply gain from source
            total_gain = distance_gain * source['gain']
            
            # Calculate left and right volumes
            if pan <= 0:  # Source is to the left
                left_vol = total_gain
                right_vol = total_gain * (1.0 + pan)  # pan is negative, so this reduces right channel
            else:  # Source is to the right
                left_vol = total_gain * (1.0 - pan)  # reduce left channel
                right_vol = total_gain
            
            # Apply elevation factor - sounds from above or below are harder to localize
            elevation_factor = abs(math.atan2(rel_vector[1], math.sqrt(rel_vector[0]**2 + rel_vector[2]**2)))
            if elevation_factor > 0.1:
                # Reduce stereo separation for elevated sounds
                mid_vol = (left_vol + right_vol) / 2
                left_vol = mid_vol + (left_vol - mid_vol) * (1 - elevation_factor/math.pi)
                right_vol = mid_vol + (right_vol - mid_vol) * (1 - elevation_factor/math.pi)
            
            # Set channel volume
            left_vol = np.clip(left_vol, 0.0, 1.0)
            right_vol = np.clip(right_vol, 0.0, 1.0)
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
        
        # Draw cardinal directions with labels
        directions = [
            ("N", 0, -150),     # North (forward)
            ("E", 150, 0),      # East (right)
            ("S", 0, 150),      # South (backward)
            ("W", -150, 0)      # West (left)
        ]
        
        for label, dx, dy in directions:
            x, y = center_x + dx, center_y + dy
            pygame.draw.line(self.screen, (150, 150, 150), (center_x, center_y), (x, y), 1)
            text = self.font.render(label, True, (150, 150, 150))
            self.screen.blit(text, (x - 8 if label in "NS" else x + 10 if label == "E" else x - 25, 
                                  y - 25 if label == "N" else y + 10 if label == "S" else y - 10))
        
        # Draw listener (with direction indicator)
        listener_x = center_x + int(self.position[0] * scale)
        listener_y = center_y - int(self.position[2] * scale)  # Inverted Z for screen coords
        
        # Calculate facing direction endpoint
        facing_len = 20
        facing_x = listener_x + int(math.sin(self.facing) * facing_len)
        facing_y = listener_y - int(math.cos(self.facing) * facing_len)
        
        # Draw listener circle
        pygame.draw.circle(self.screen, (0, 255, 255), (listener_x, listener_y), 10)
        
        # Draw facing direction
        pygame.draw.line(self.screen, (0, 255, 255), (listener_x, listener_y), (facing_x, facing_y), 3)
        
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
                
            # Draw source as circle
            size = 6 + int(source['gain'] * 4)
            pygame.draw.circle(self.screen, color, (source_x, source_y), size)
            
            # Draw vertical line to show height
            ground_y = source_y + int(source['position'][1] * scale)
            pygame.draw.line(self.screen, (*color[:3], 100), (source_x, source_y), (source_x, ground_y), 1)
            
            # Highlight selected source
            if source_id == self.selected_source_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (source_x, source_y), size + 4, 2)
            
            # Draw source name
            name_text = self.font.render(source['name'], True, color)
            # Add shadow for better readability
            shadow_text = self.font.render(source['name'], True, (0, 0, 0))
            self.screen.blit(shadow_text, (source_x + 16, source_y - 14))
            self.screen.blit(name_text, (source_x + 15, source_y - 15))
        
        # Draw mode and position information
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (220, 220, 220))
        self.screen.blit(mode_text, (20, 20))
        
        pos_text = f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})"
        pos_render = self.font.render(pos_text, True, (200, 200, 200))
        self.screen.blit(pos_render, (20, 50))
        
        # Draw selected source info
        if self.selected_source_id:
            select_text = f"Selected: {self.selected_source_id}"
            select_render = self.font.render(select_text, True, (255, 255, 100))
            self.screen.blit(select_render, (width - 200, 20))
        
        # Draw instructions based on mode
        instruction_text = ""
        if self.mode == 'navigation':
            instruction_text = "Left stick: Move | Right stick: Turn | Y: Select sources"
        elif self.mode == 'source_selection':
            instruction_text = "D-pad: Select source | A: Edit source | B: Back"
        elif self.mode == 'source_manipulation':
            instruction_text = "Left: Move XZ | Right: Height/Volume | A: Mute | B: Back"
            
        instr = self.font.render(instruction_text, True, (220, 220, 220))
        self.screen.blit(instr, (20, height - 40))
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Run the demo"""
        self.running = True
        clock = pygame.time.Clock()
        
        print("\n== SHAC Audio File Demo ==")
        print("Controls:")
        if self.controller:
            print("  Navigation Mode:")
            print("    Left stick: Move in space")
            print("    Right stick: Turn to face different directions")
            print("    Shoulder buttons: Move up/down")
            print("    Y button: Switch to source selection")
            print("  Source Selection Mode:")
            print("    D-pad up/down: Select source")
            print("    A button: Edit selected source")
            print("    B button: Back to navigation")
            print("  Source Manipulation Mode:")
            print("    Left stick: Move source in space")
            print("    Right stick Y: Adjust source height")
            print("    Right stick X: Adjust volume")
            print("    A button: Mute/unmute")
            print("    B button: Back to selection")
        else:
            print("  Keyboard Controls:")
            print("    WASD/Arrows: Move in space")
            print("    Q/E: Turn left/right")
            print("    R/F: Move up/down")
        
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
                
                # Process input
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
    demo = AudioFileDemo()
    demo.run()