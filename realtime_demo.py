"""
SHAC Realtime Demo with Xbox Controller and Headphones

This script provides a complete demonstration of the SHAC system with:
- Real-time audio output to headphones
- Xbox controller input for navigation and manipulation
- 3D visualization of the sound field

Author: Claude
License: MIT License
"""

import numpy as np
import pygame
import sounddevice as sd
import threading
import time
import sys
import argparse

try:
    from shac_codec import SHACCodec
    from layer_manager import SoundLayerManager
    from controller_interface import ControllerInterface
    from sound_field_visualizer import SoundFieldVisualizer
    from audio_utils import generate_instrument_like, generate_noise, generate_rhythm
except ImportError:
    print("Error importing SHAC modules. Make sure all required files are in the same directory.")
    sys.exit(1)

class RealtimeDemo:
    """
    Real-time demonstration of the SHAC system with audio output and controller input.
    """
    
    def __init__(self, sample_rate=48000, buffer_size=1024, order=3, visualization=True):
        """
        Initialize the real-time demo
        
        Parameters:
        - sample_rate: Audio sample rate in Hz
        - buffer_size: Audio buffer size in samples
        - order: Ambisonic order
        - visualization: Whether to enable visualization
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.order = order
        self.visualization = visualization
        
        # Initialize components
        self.codec = SHACCodec(order=self.order, sample_rate=self.sample_rate)
        self.layer_manager = SoundLayerManager(self.codec)
        self.controller = ControllerInterface(self.layer_manager)
        
        # Create audio sources
        self._create_audio_sources()
        
        # Initialize audio stream
        self.audio_stream = None
        
        # Threading control
        self.running = False
        self.audio_thread = None
        self.visualization_thread = None
        
        # If visualization is enabled, initialize the visualizer
        if self.visualization:
            self.visualizer = SoundFieldVisualizer(self.layer_manager, self.controller)
    
    def _create_audio_sources(self):
        """Create audio sources for demonstration"""
        print("Creating audio sources...")
        
        # Generate test signals
        duration = 10  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Piano sound (C major chord)
        print("Creating piano sound...")
        piano_c = generate_instrument_like('piano', 60, duration, self.sample_rate, 0.7)  # C4
        piano_e = generate_instrument_like('piano', 64, duration, self.sample_rate, 0.5)  # E4
        piano_g = generate_instrument_like('piano', 67, duration, self.sample_rate, 0.6)  # G4
        piano_audio = piano_c + piano_e + piano_g
        
        # Normalize
        if np.max(np.abs(piano_audio)) > 0:
            piano_audio = piano_audio / np.max(np.abs(piano_audio)) * 0.8
            
        # Add to layer manager
        self.layer_manager.add_layer('piano', 'Piano', piano_audio, (-np.pi/4, 0.0, 3.0), 
                                     {'type': 'source', 'color': (255, 0, 0)})
        
        # Violin sound
        print("Creating violin sound...")
        violin_audio = generate_instrument_like('violin', 69, duration, self.sample_rate, 0.6)  # A4
        self.layer_manager.add_layer('violin', 'Violin', violin_audio, (np.pi/4, 0.1, 4.0), 
                                     {'type': 'source', 'color': (0, 255, 0)})
        
        # Bass sound
        print("Creating bass sound...")
        bass_audio = generate_instrument_like('bass', 48, duration, self.sample_rate, 0.7)  # C3
        self.layer_manager.add_layer('bass', 'Bass', bass_audio, (-np.pi/6, -0.1, 3.5), 
                                     {'type': 'source', 'color': (255, 255, 0)})
        
        # Drums
        print("Creating drum pattern...")
        pattern = [1, 0, 0.5, 0, 0.8, 0, 0.5, 0.3]  # Simple rhythm pattern
        drums_audio = generate_rhythm(pattern, 120, duration, self.sample_rate, 0.7)
        self.layer_manager.add_layer('drums', 'Drums', drums_audio, (0.0, -0.2, 2.5), 
                                     {'type': 'source', 'color': (0, 0, 255)})
        
        # Ambient sound
        print("Creating ambient sound...")
        ambient_audio = generate_noise('pink', duration, self.sample_rate, 0.3)
        self.layer_manager.add_layer('ambient', 'Ambient', ambient_audio, (0.0, 0.5, 8.0), 
                                     {'type': 'source', 'color': (0, 255, 255)})
        
        print(f"Created {len(self.layer_manager.layers)} audio sources")
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Audio callback function for sounddevice
        
        This is called by the audio system when it needs more audio data.
        """
        if status:
            print(f"Audio status: {status}")
        
        # Process the current block to get binaural audio
        binaural_output = self.layer_manager.process_audio()
        
        # Get the requested number of frames
        if binaural_output.shape[1] >= frames:
            # If we have enough samples, return the requested number
            outdata[:] = binaural_output[:, :frames].T
        else:
            # If we don't have enough samples, pad with zeros
            outdata[:frames, :] = np.zeros((frames, 2))
            outdata[:binaural_output.shape[1], :] = binaural_output.T
    
    def _audio_thread_function(self):
        """Function for the audio processing thread"""
        print("Starting audio processing thread...")
        
        try:
            # Initialize audio stream
            with sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=2,
                callback=self._audio_callback
            ):
                print("Audio stream started. You should hear sound now.")
                
                # Process controller input until we're told to stop
                while self.running:
                    # Process controller input
                    updated = self.controller.process_input()
                    
                    # Sleep a bit to avoid hammering the CPU
                    time.sleep(0.01)
        
        except Exception as e:
            print(f"Audio error: {e}")
        
        print("Audio thread stopped")
    
    def start(self):
        """Start the demo"""
        if self.running:
            print("Demo is already running")
            return
            
        print("\n== SHAC Real-time Demo with Xbox Controller ==")
        print("Controls:")
        print("  Navigation Mode:")
        print("    Left stick: Move in space")
        print("    Right stick: Look around")
        print("    Y button: Switch to layer selection")
        print("  Layer Selection Mode:")
        print("    D-pad up/down: Select layer")
        print("    A button: Edit selected layer")
        print("    B button: Back to navigation")
        print("  Layer Manipulation Mode:")
        print("    Left stick: Move layer in space")
        print("    Right stick X: Adjust gain")
        print("    Right stick Y: Adjust distance")
        print("    A button: Mute/unmute")
        print("    X button: Reset layer")
        print("    B button: Back to selection")
        print("\nClose visualization window or press Ctrl+C to exit")
        
        # Set running flag
        self.running = True
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_thread_function)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start visualization if enabled
        if self.visualization:
            print("Starting visualization...")
            try:
                self.visualizer.start_animation()
            except Exception as e:
                print(f"Visualization error: {e}")
            
            # When visualization window is closed, stop the demo
            self.stop()
        else:
            print("Visualization disabled. Demo running in console-only mode.")
            print("Press Ctrl+C to exit.")
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                self.stop()
    
    def stop(self):
        """Stop the demo"""
        print("Stopping demo...")
        self.running = False
        
        # Wait for threads to complete
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            
        print("Demo stopped")


def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    # Check for sounddevice
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    # Check for pygame
    try:
        import pygame
    except ImportError:
        missing.append("pygame")
    
    # Check for numpy
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    # Check for matplotlib
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Please install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Main entry point"""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SHAC Real-time Demo')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--order', type=int, default=3, help='Ambisonic order (default: 3)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--buffer-size', type=int, default=1024, help='Buffer size in samples (default: 1024)')
    args = parser.parse_args()
    
    # Create and start the demo
    demo = RealtimeDemo(
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
        order=args.order,
        visualization=not args.no_vis
    )
    
    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        demo.stop()


if __name__ == "__main__":
    main()