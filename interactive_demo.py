"""
SHAC Interactive Demo

This module provides a simple interactive demonstration of the SHAC system
with controller input. It creates a spatial audio scene and allows exploration
and manipulation using a game controller.

Author: Claude
License: MIT License
"""

import pygame
import numpy as np
import time
import sys
import os

# Add parent directory to path to import SHAC modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shac_codec import SHACCodec, create_example_sound_scene
from controller_interface import ControllerInterface
from layer_manager import SoundLayerManager

class SHACDemo:
    """Interactive demonstration of the SHAC system"""
    
    def __init__(self):
        # Initialize pygame for audio and controller
        pygame.init()
        pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=1024)
        
        # Set up audio stream
        self.sample_rate = 48000
        self.buffer_size = 1024
        
        # Create codec
        self.codec = SHACCodec(order=3, sample_rate=self.sample_rate)
        
        # Create layer manager
        self.layer_manager = SoundLayerManager(self.codec)
        
        # Create controller interface
        self.controller = ControllerInterface(self.layer_manager)
        
        # Initialize output buffer for audio
        self.output_buffer = np.zeros((2, self.buffer_size))
        
        # Create a pygame Sound object for output
        self.sound = pygame.mixer.Sound(buffer=np.zeros((self.buffer_size, 2), dtype=np.int16))
        self.sound_channel = self.sound.play(-1)  # Loop the sound
        
        # Set up GUI for visualization
        pygame.display.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("SHAC Interactive Demo")
        self.font = pygame.font.Font(None, 36)
        
        # Create some sample sounds
        self._create_sample_sounds()
        
        # Last update time for frame rate calculation
        self.last_update = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Audio position counter for streaming
        self.audio_position = 0
    
    def _create_sample_sounds(self):
        """Create sample sound sources, using real audio files if available"""
        # Check if we have the Bach file or other audio files
        audio_files = []
        
        # Check in music directory
        if os.path.exists('music'):
            for file in os.listdir('music'):
                if file.lower().endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join('music', file))
                    
        # Check in audio directory
        if os.path.exists('audio'):
            for file in os.listdir('audio'):
                if file.lower().endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join('audio', file))
        
        # Function to load audio file with fallback
        def load_audio(file_path):
            try:
                from file_loader import load_audio_file
                audio_info = load_audio_file(file_path)
                
                # Ensure mono (for simple positioning)
                audio_data = audio_info['audio_data']
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Limit the amount of audio data to 5 seconds for better performance
                max_samples = 5 * audio_info['sample_rate']
                if len(audio_data) > max_samples:
                    # Take the first 5 seconds only to improve performance
                    print(f"Trimming audio to 5 seconds for better performance")
                    audio_data = audio_data[:max_samples]
                    
                # Normalize audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.8
                    
                print(f"Loaded audio file: {file_path}")
                return audio_data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None
        
        # If we have audio files, use them
        if audio_files:
            print(f"Found {len(audio_files)} audio files to use as sources")
            
            # Define positions in 3D space for various sources
            positions = [
                (-np.pi/4, 0.0, 3.0),  # Front left
                (np.pi/4, -0.1, 2.5),  # Front right
                (np.pi, 0.0, 4.0),     # Behind
                (-np.pi/2, 0.0, 2.5),  # Left side
                (np.pi/2, 0.0, 2.5),   # Right side
                (0.0, np.pi/3, 4.0),   # Above front
            ]
            
            # Colors for visualization
            colors = [
                (255, 0, 0),     # Red
                (0, 0, 255),     # Blue
                (0, 255, 0),     # Green
                (255, 255, 0),   # Yellow
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Cyan
            ]
            
            # Add each audio file as a source with a unique position
            for i, file_path in enumerate(audio_files):
                if i >= len(positions):
                    break  # Don't add more sources than positions
                    
                audio_data = load_audio(file_path)
                if audio_data is not None:
                    source_name = os.path.splitext(os.path.basename(file_path))[0]
                    layer_id = f'audio_{i}'
                    position = positions[i]
                    color = colors[i]
                    
                    # Add the source to our layer manager
                    self.layer_manager.add_layer(
                        layer_id, 
                        source_name, 
                        audio_data, 
                        position,
                        {
                            'color': color,
                            'description': f'Audio: {source_name}'
                        }
                    )
                    print(f"Added source '{source_name}' at position {position}")
            
            # Return if we added any audio sources
            if hasattr(self.layer_manager, 'layers') and len(self.layer_manager.layers) > 0:
                return
                
        # If no audio files were found or loaded, create synthetic sources as fallback
        print("Using synthetic audio sources")
        duration = 5.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano sound (sine wave with harmonics and decay)
        piano_freq = 440.0  # A4
        piano_audio = 0.5 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
        piano_audio += 0.25 * np.sin(2 * np.pi * 2 * piano_freq * t) * np.exp(-t/1.5)
        piano_audio += 0.125 * np.sin(2 * np.pi * 3 * piano_freq * t) * np.exp(-t)
        
        # Drum sound (impulses with decay)
        drum_audio = np.zeros_like(t)
        for i in range(0, len(t), int(self.sample_rate / 4)):  # Four beats per second
            if i + 5000 < len(drum_audio):
                drum_audio[i:i+5000] = 0.8 * np.exp(-np.linspace(0, 10, 5000))
        
        # Ambient sound (filtered noise)
        np.random.seed(42)  # For reproducibility
        noise = np.random.randn(len(t))
        b, a = np.array([0.2, 0.2, 0.2, 0.2, 0.2]), np.array([1.0])  # Simple moving average filter
        ambient_audio = np.convolve(noise, b/a, mode='same') * 0.2
        
        # Add layers
        self.layer_manager.add_layer('piano', 'Piano', piano_audio, (-np.pi/4, 0.0, 3.0), {
            'color': (255, 0, 0),  # Red
            'description': 'Piano A4 note with harmonics'
        })
        
        self.layer_manager.add_layer('drum', 'Drums', drum_audio, (np.pi/4, -0.1, 2.5), {
            'color': (0, 0, 255),  # Blue
            'description': 'Rhythm pattern at 120 BPM'
        })
        
        self.layer_manager.add_layer('ambient', 'Ambient', ambient_audio, (0.0, np.pi/3, 5.0), {
            'color': (0, 255, 0),  # Green
            'description': 'Background ambient noise'
        })
    
    def process_audio(self):
        """Generate the next block of audio"""
        try:
            # Create a small chunk of audio to keep UI responsive
            duration = 0.1  # 100ms chunks for better UI responsiveness
            chunk_samples = int(self.sample_rate * duration)
            
            # Get processed audio from the codec
            # If we have a full class implementation, we'd use chunk_size param
            self.output_buffer = np.zeros((2, chunk_samples))  # Stereo
            
            # Process a smaller amount of audio to maintain UI responsiveness
            for layer_id, layer_info in self.layer_manager.layers.items():
                if layer_info['muted']:
                    continue
                
                # Simple panning based on azimuth
                position = layer_info['position']
                azimuth = position[0]
                elevation = position[1]
                distance = position[2]
                
                # Extract a chunk of audio data
                if 'audio' not in layer_info:
                    continue
                    
                audio_data = layer_info['audio']
                start_idx = int(self.audio_position) % len(audio_data)
                end_idx = min(start_idx + chunk_samples, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Pad if needed
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                
                # Gain and distance attenuation
                gain = layer_info['current_gain']
                attenuation = 1.0 / max(1.0, distance / 2.0) 
                
                # Apply basic panning based on azimuth
                pan = np.sin(azimuth)
                left_gain = gain * attenuation * np.clip(1.0 - pan, 0.2, 1.0)
                right_gain = gain * attenuation * np.clip(1.0 + pan, 0.2, 1.0)
                
                # Add to output buffer
                self.output_buffer[0] += chunk * left_gain
                self.output_buffer[1] += chunk * right_gain
            
            # Update position counter
            self.audio_position += chunk_samples
            
            # Normalize if needed
            max_val = np.max(np.abs(self.output_buffer))
            if max_val > 0.99:
                self.output_buffer *= 0.99 / max_val
                
            # Convert to int16 for pygame
            output_int16 = (self.output_buffer.T * 32767).astype(np.int16)
            
            # Update the sound buffer - convert to bytes for pygame
            self.sound = pygame.mixer.Sound(buffer=output_int16.tobytes())
            self.sound_channel.play(self.sound, loops=-1)
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    def draw_visualization(self):
        """Draw visualization of the sound field"""
        self.screen.fill((0, 0, 0))  # Clear screen with black
        
        # Get screen dimensions
        width, height = self.screen.get_size()
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y) - 50
        
        # Draw coordinate system
        pygame.draw.line(self.screen, (100, 100, 100), (center_x - radius, center_y), (center_x + radius, center_y), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (center_x, center_y - radius), (center_x, center_y + radius), 2)
        
        # Draw listener position (center)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), 10)
        
        # Draw orientation
        orientation = self.controller.orientation
        direction_x = np.cos(orientation[0]) * radius * 0.2
        direction_z = np.sin(orientation[0]) * radius * 0.2
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (center_x, center_y), 
                        (center_x + direction_x, center_y - direction_z), 
                        3)
        
        # Draw each layer
        for layer_id, layer_info in self.layer_manager.layers.items():
            if layer_info['muted']:
                continue  # Skip muted layers
                
            position = layer_info['position']
            azimuth, elevation, distance = position
            
            # Scale distance for visualization
            vis_distance = min(1.0, 5.0 / distance) * radius
            
            # Calculate x, y position on screen
            x = center_x + np.sin(azimuth) * np.cos(elevation) * vis_distance
            y = center_y - np.sin(elevation) * vis_distance
            
            # Get layer color
            color = layer_info['properties'].get('color', (255, 255, 255))
            
            # Draw layer
            size = max(5, 20 * layer_info['current_gain'])
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))
            
            # Draw layer name
            name_text = self.font.render(layer_info['name'], True, color)
            self.screen.blit(name_text, (int(x) + 10, int(y) - 20))
            
            # If this is the selected layer, draw a highlight
            if layer_id == self.controller.selected_layer_id:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(size) + 5, 2)
        
        # Draw mode indicator
        mode_text = self.font.render(f"Mode: {self.controller.mode.capitalize()}", True, (255, 255, 255))
        self.screen.blit(mode_text, (20, 20))
        
        # Draw position
        pos_text = self.font.render(f"Position: {np.round(self.controller.position, 2)}", True, (200, 200, 200))
        self.screen.blit(pos_text, (20, 60))
        
        # Draw orientation
        orient_text = self.font.render(f"Orientation: {np.round(self.controller.orientation * 180/np.pi, 1)}°", True, (200, 200, 200))
        self.screen.blit(orient_text, (20, 100))
        
        # Draw FPS
        fps_text = self.font.render(f"FPS: {self.fps:.1f}", True, (150, 150, 150))
        self.screen.blit(fps_text, (width - 120, 20))
        
        # Draw instructions
        if self.controller.mode == 'navigation':
            instr_text = self.font.render("Left stick: Move | Right stick: Look | Y: Layer select", True, (200, 200, 200))
            self.screen.blit(instr_text, (20, height - 40))
        elif self.controller.mode == 'layer_selection':
            instr_text = self.font.render("D-pad: Select layer | A: Edit layer | B: Back", True, (200, 200, 200))
            self.screen.blit(instr_text, (20, height - 40))
        elif self.controller.mode == 'layer_manipulation':
            instr_text = self.font.render("Left: Position | Right: Distance/Gain | A: Mute | X: Reset | B: Back", True, (200, 200, 200))
            self.screen.blit(instr_text, (20, height - 40))
        
        # Update the display
        pygame.display.flip()
        
        # Calculate FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_update >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / (now - self.last_update)
            self.frame_count = 0
            self.last_update = now
    
    def run(self):
        """Run the demo"""
        running = True
        clock = pygame.time.Clock()
        
        print("Starting SHAC Interactive Demo")
        print("\nKeyboard Controls:")
        print("  Navigation Mode:")
        print("    WASD: Move in the direction you're facing")
        print("    Arrow Keys: Look around (change orientation)")
        print("    Q/E: Move up/down")
        print("    Space: Switch to layer selection mode")
        print("\n  Layer Selection Mode:")
        print("    Up/Down: Select previous/next layer")
        print("    Enter: Enter layer editing mode for selected layer")
        print("    Space: Return to navigation mode")
        print("\n  Layer Edit Mode:")
        print("    WASD: Move the sound source around")
        print("    Q/E: Move the sound source up/down")
        print("    Arrow Up/Down: Increase/decrease volume")
        print("    M: Mute/unmute the source")
        print("    Space: Return to layer selection mode")
        print("\nClose window or press Esc to exit")
        
        try:
            # Process audio once before the main loop to ensure it's playing
            self.process_audio()
            
            # Initialize audio timer
            last_audio_time = time.time()
            audio_interval = 0.05  # Update audio every 50ms
            
            while running:
                # Process events 
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Process controller input
                updated = self.controller.process_input()
                
                # Update audio at regular intervals, not tied to controller updates
                current_time = time.time()
                if current_time - last_audio_time >= audio_interval:
                    self.process_audio()
                    last_audio_time = current_time
                
                # Draw visualization - only do this up to 30fps max
                self.draw_visualization()
                
                # Let the system breathe a bit 
                pygame.time.delay(10)  # Short delay for better system responsiveness
        
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            pygame.quit()


if __name__ == "__main__":
    demo = SHACDemo()
    demo.run()