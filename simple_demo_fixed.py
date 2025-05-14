"""
SHAC Simple Demo with Xbox Controller and Headphones - Fixed Version

A simplified demonstration of the SHAC system that works with minimal dependencies.
This version fixes several bugs from the original simple_demo.py.

Author: Claude
License: MIT License
"""

import numpy as np
import pygame
import threading
import time
import sys
import os

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
            'left_trigger': 4,  # LT
            'right_trigger': 5  # RT
        }
        
        # Navigation state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # yaw, pitch, roll
        self.mode = 'navigation'
        
        # Sound sources
        self.sources = {}
        self.selected_source_id = None
        
        # Create sample sounds
        self._create_sound_sources()
        
        # Audio output
        self.sounds = {}
        for source_id, source_data in self.sources.items():
            # Convert to 16-bit for pygame
            data = (source_data['audio'] * 32767).astype(np.int16)
            # Make it stereo
            stereo_data = np.column_stack((data, data))
            # Create pygame sound
            self.sounds[source_id] = pygame.mixer.Sound(buffer=stereo_data)
            # Create a channel for this sound
            source_data['channel'] = pygame.mixer.find_channel()
            if source_data['channel']:
                source_data['channel'].play(self.sounds[source_id], loops=-1)
                source_data['channel'].set_volume(0.5, 0.5)  # Start with mid volume
        
        # Setup visualization
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("SHAC Simple Demo")
        self.font = pygame.font.Font(None, 36)
        
        # Thread control
        self.running = False
        
        # Last button states to prevent immediate repeat presses
        self.last_button_states = {name: False for name in self.button_map.keys()}
        
        print("Simple SHAC demo initialized")
    
    def _create_sound_sources(self):
        """Create sample sound sources"""
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano-like sound (pure tone with decay)
        piano_freq = 440  # A4
        piano_audio = 0.7 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
        self.sources['piano'] = {
            'audio': piano_audio,
            'position': np.array([-3.0, 0.0, 3.0]),  # x, y, z (left, up, front)
            'name': 'Piano',
            'color': (255, 0, 0),
            'gain': 1.0,
            'muted': False
        }
        
        # Bass-like sound (low frequency tone)
        bass_freq = 110  # A2
        bass_audio = 0.7 * np.sin(2 * np.pi * bass_freq * t)
        self.sources['bass'] = {
            'audio': bass_audio,
            'position': np.array([3.0, -1.0, 4.0]),  # x, y, z (right, down, front)
            'name': 'Bass',
            'color': (0, 0, 255),
            'gain': 1.0,
            'muted': False
        }
        
        # Drum-like sound (noise bursts)
        drum_audio = np.zeros_like(t)
        for i in range(0, len(t), int(self.sample_rate / 2)):  # Two beats per second
            if i + 1000 < len(drum_audio):
                drum_audio[i:i+1000] = 0.5 * np.exp(-np.linspace(0, 10, 1000)) * np.random.randn(1000)
        self.sources['drums'] = {
            'audio': drum_audio,
            'position': np.array([0.0, -2.0, 2.0]),  # x, y, z (center, down, front)
            'name': 'Drums',
            'color': (0, 255, 0),
            'gain': 1.0,
            'muted': False
        }
        
        # Ambient sound (filtered noise)
        ambient_audio = 0.3 * np.random.randn(len(t))
        # Simple low-pass filter
        for i in range(1, len(ambient_audio)):
            ambient_audio[i] = 0.9 * ambient_audio[i-1] + 0.1 * ambient_audio[i]
        self.sources['ambient'] = {
            'audio': ambient_audio,
            'position': np.array([0.0, 3.0, 8.0]),  # x, y, z (center, up, far)
            'name': 'Ambient',
            'color': (255, 255, 0),
            'gain': 1.0,
            'muted': False
        }
    
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
        # Poll for events
        pygame.event.pump()
        
        # Process keyboard input if no controller
        if self.controller is None:
            keys = pygame.key.get_pressed()
            # (Logic for keyboard would go here)
            return
        
        # Get controller input
        if self.mode == 'navigation':
            # Left stick: Move in XZ plane
            left_x = self.controller.get_axis(self.axis_map['left_stick_x'])
            left_y = self.controller.get_axis(self.axis_map['left_stick_y'])
            
            # Apply deadzone
            if abs(left_x) < 0.1: left_x = 0
            if abs(left_y) < 0.1: left_y = 0
            
            # Update position
            move_speed = 0.1
            self.position[0] += left_x * move_speed  # X axis (left/right)
            self.position[2] -= left_y * move_speed  # Z axis (forward/back)
            
            # Right stick: Look direction
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            if abs(right_y) < 0.1: right_y = 0
            
            # Update orientation
            rotate_speed = 0.05
            if abs(right_x) > 0:
                self.orientation[0] += right_x * rotate_speed  # Yaw (left/right)
                
            if abs(right_y) > 0:
                self.orientation[1] += right_y * rotate_speed  # Pitch (up/down)
                # Clamp pitch to avoid flipping
                self.orientation[1] = np.clip(self.orientation[1], -np.pi/2, np.pi/2)
            
            # Triggers: Move up/down
            try:
                # Note: Some controllers may not have 6 axes
                lt = (self.controller.get_axis(self.axis_map['left_trigger']) + 1) / 2
                rt = (self.controller.get_axis(self.axis_map['right_trigger']) + 1) / 2
                
                # Update Y position
                self.position[1] += (rt - lt) * 0.1
            except:
                # Fallback if triggers are not recognized
                pass
            
            # Update audio based on listener position
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
                # Fallback if d-pad is not recognized
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
                # Update source position
                source['position'][0] += left_x * 0.1  # X axis (left/right)
                source['position'][2] -= left_y * 0.1  # Z axis (forward/back)
                # Update audio
                self._update_spatial_audio()
            
            # Right stick Y: Adjust distance (Z position)
            right_y = self.controller.get_axis(self.axis_map['right_stick_y'])
            
            # Apply deadzone
            if abs(right_y) < 0.1: right_y = 0
            
            if abs(right_y) > 0:
                # Move source closer or further
                source['position'][2] += right_y * 0.2
                # Keep positive distance
                source['position'][2] = max(0.5, source['position'][2])
                # Update audio
                self._update_spatial_audio()
            
            # Right stick X: Adjust volume
            right_x = self.controller.get_axis(self.axis_map['right_stick_x'])
            
            # Apply deadzone
            if abs(right_x) < 0.1: right_x = 0
            
            if abs(right_x) > 0:
                # Adjust gain
                source['gain'] += right_x * 0.02
                # Clamp gain to reasonable range
                source['gain'] = np.clip(source['gain'], 0.0, 2.0)
                # Update audio
                channel = source.get('channel')
                if channel:
                    # Update volumes after gain change
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
            # Apply yaw rotation
            rotated_x = rel_vector[0] * np.cos(yaw) + rel_vector[2] * np.sin(yaw)
            rotated_z = -rel_vector[0] * np.sin(yaw) + rel_vector[2] * np.cos(yaw)
            
            # Use the rotated coordinates for audio
            rel_audio_x = rotated_x
            rel_audio_z = rotated_z
            
            # Calculate distance based on rotated coordinates
            distance = np.sqrt(rel_audio_x**2 + rel_audio_z**2 + rel_vector[1]**2)
            
            # Calculate stereo panning based on rotated X position
            # This makes the direction you're facing matter for audio
            pan = np.clip(rel_audio_x / 5.0, -1.0, 1.0)
            
            # Calculate attenuation based on distance
            distance_gain = 1.0 / max(1.0, distance)
            
            # Apply gain from source
            total_gain = distance_gain * source['gain']
            
            # Calculate left and right volumes
            if pan <= 0:  # Source is to the left
                left_vol = total_gain
                right_vol = total_gain * (1.0 + pan)
            else:  # Source is to the right
                left_vol = total_gain * (1.0 - pan)
                right_vol = total_gain
            
            # Apply distance falloff
            left_vol = np.clip(left_vol, 0.0, 1.0)
            right_vol = np.clip(right_vol, 0.0, 1.0)
            
            # Set channel volume
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
        
        # Scale factor for drawing
        scale = 30
        
        # Draw listener position
        listener_x = center_x + int(self.position[0] * scale)
        listener_y = center_y - int(self.position[2] * scale)  # Inverted Z for screen coordinates
        pygame.draw.circle(self.screen, (0, 255, 255), (listener_x, listener_y), 10)
        
        # Draw listener orientation
        direction_len = 20
        direction_x = listener_x + int(np.sin(self.orientation[0]) * direction_len)
        direction_y = listener_y - int(np.cos(self.orientation[0]) * direction_len)
        pygame.draw.line(self.screen, (0, 255, 255), (listener_x, listener_y), (direction_x, direction_y), 2)
        
        # Draw listener "sound cone" based on orientation
        cone_width = np.pi / 3  # 60 degree cone
        cone_len = 50
        
        # Draw "ears" to show what you're hearing
        left_ear_x = listener_x + int(np.sin(self.orientation[0] - np.pi/2) * 8)
        left_ear_y = listener_y - int(np.cos(self.orientation[0] - np.pi/2) * 8)
        pygame.draw.circle(self.screen, (0, 200, 200), (left_ear_x, left_ear_y), 4)
        
        right_ear_x = listener_x + int(np.sin(self.orientation[0] + np.pi/2) * 8)
        right_ear_y = listener_y - int(np.cos(self.orientation[0] + np.pi/2) * 8)
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
                
            # Draw source as circle
            size = 8 + int(source['gain'] * 5)
            pygame.draw.circle(self.screen, color, (source_x, source_y), size)
            
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
        
        # Draw mode indicator
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (255, 255, 255))
        self.screen.blit(mode_text, (20, 20))
        
        # Draw listener position
        pos_text = self.font.render(f"Position: ({self.position[0]:.1f}, {self.position[2]:.1f})", True, (200, 200, 200))
        self.screen.blit(pos_text, (20, 60))
        
        # Draw orientation
        yaw_deg = np.degrees(self.orientation[0]) % 360
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
        print("    Left stick: Move in space")
        print("    Right stick: Look around")
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
        
        # Clean up
        pygame.quit()
        print("Demo stopped")


if __name__ == "__main__":
    # Start the demo
    demo = SimpleDemo()
    
    try:
        demo.run()
    except Exception as e:
        print(f"Error: {e}")
        # Clean up pygame on error
        pygame.quit()
        # Show exception details
        import traceback
        traceback.print_exc()