"""
Layered Interactive Sound Environment (LISE) Demo

This application demonstrates the SHAC system with interactive sound layer manipulation,
controller input, and 3D visualization. It serves as a complete example of how to use
the SHAC components together.

Author: Claude
License: MIT License
"""

import numpy as np
import time
import os
import threading
import argparse

# Import SHAC components
from shac_codec import SHACCodec, SHACStreamProcessor
from layer_manager import SoundLayerManager
from controller_interface import ControllerInterface
from sound_field_visualizer import SoundFieldVisualizer

class LISEDemo:
    """
    Layered Interactive Sound Environment demonstration application
    """
    
    def __init__(self, use_audio_files=True, sample_rate=48000, buffer_size=1024, order=3):
        """
        Initialize the LISE demo
        
        Parameters:
        - use_audio_files: Whether to load real audio files or generate synthetic signals
        - sample_rate: Audio sample rate in Hz
        - buffer_size: Audio buffer size in samples
        - order: Ambisonic order
        """
        print("Initializing Layered Interactive Sound Environment (LISE) Demo...")
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.order = order
        
        # Initialize the audio codec
        self.codec = SHACCodec(order=self.order, sample_rate=self.sample_rate)
        
        # Initialize the stream processor for real-time audio
        self.stream_processor = SHACStreamProcessor(order=self.order, 
                                                   sample_rate=self.sample_rate, 
                                                   buffer_size=self.buffer_size)
        
        # Initialize the layer manager
        self.layer_manager = SoundLayerManager(self.codec)
        
        # Initialize the controller interface
        self.controller = ControllerInterface(self.layer_manager)
        
        # Initialize the visualizer
        self.visualizer = SoundFieldVisualizer(self.layer_manager, self.controller)
        
        # Create audio sources
        if use_audio_files:
            self._load_audio_files()
        else:
            self._create_synthetic_audio()
        
        # Threading control
        self.running = False
        self.audio_thread = None
        self.visualization_thread = None
    
    def _load_audio_files(self):
        """Load audio files from disk"""
        print("Looking for audio files...")
        
        audio_files = {
            'piano': 'piano.wav',
            'violin': 'violin.wav',
            'drums': 'drums.wav',
            'bass': 'bass.wav',
            'vocals': 'vocals.wav',
            'ambient': 'ambient.wav'
        }
        
        # Check if files exist
        audio_dir = os.path.join(os.path.dirname(__file__), 'audio')
        if not os.path.exists(audio_dir):
            print(f"Audio directory not found: {audio_dir}")
            print("Falling back to synthetic audio.")
            self._create_synthetic_audio()
            return
        
        # Count how many files we have
        found_files = 0
        for name, filename in audio_files.items():
            filepath = os.path.join(audio_dir, filename)
            if os.path.exists(filepath):
                found_files += 1
                print(f"Found: {filename}")
                # In a real implementation, we would load the audio here
                # For now, we'll just use synthetic audio with the appropriate names
                
        if found_files == 0:
            print("No audio files found. Falling back to synthetic audio.")
            self._create_synthetic_audio()
            return
        else:
            print(f"Found {found_files} audio files. Creating layers...")
            # Since we're not actually loading files, we'll use synthetic audio
            # but with the appropriate positioning
            self._create_synthetic_audio(use_names_from_files=True)
    
    def _create_synthetic_audio(self, use_names_from_files=False):
        """Create synthetic audio signals for demonstration"""
        print("Creating synthetic audio signals...")
        
        # Generate test signals
        duration = 10  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create layers based on whether we're using file names or synthetic names
        if use_names_from_files:
            # Create layers with the same names as the audio files
            
            # Piano sound (decaying sine with harmonics)
            piano_freq = 261.63  # C4
            piano_audio = 0.5 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
            piano_audio += 0.25 * np.sin(2 * np.pi * 2 * piano_freq * t) * np.exp(-t/1.5)
            piano_audio += 0.125 * np.sin(2 * np.pi * 3 * piano_freq * t) * np.exp(-t)
            self.layer_manager.add_layer('piano', 'Piano', piano_audio, (-np.pi/4, 0.0, 3.0), 
                                        {'type': 'source', 'color': (255, 0, 0)})
            
            # Violin sound (higher frequency sine with vibrato)
            vib_freq = 5  # 5 Hz vibrato
            vib_amount = 30  # vibrato depth in Hz
            violin_freq = 440  # A4
            violin_audio = 0.5 * np.sin(2 * np.pi * (violin_freq + vib_amount * np.sin(2 * np.pi * vib_freq * t)) * t)
            self.layer_manager.add_layer('violin', 'Violin', violin_audio, (np.pi/4, 0.1, 4.0), 
                                        {'type': 'source', 'color': (0, 255, 0)})
            
            # Drums sound (impulses with decay)
            drums_audio = np.zeros_like(t)
            for i in range(0, len(t), int(self.sample_rate / 4)):  # Four beats per second
                if i + 5000 < len(drums_audio):
                    drums_audio[i:i+5000] = 0.8 * np.exp(-np.linspace(0, 10, 5000))
            self.layer_manager.add_layer('drums', 'Drums', drums_audio, (0.0, -0.2, 2.5), 
                                        {'type': 'source', 'color': (0, 0, 255)})
            
            # Bass sound (low frequency sine)
            bass_freq = 65.41  # C2
            bass_audio = 0.7 * np.sin(2 * np.pi * bass_freq * t)
            bass_audio += 0.2 * np.sin(2 * np.pi * 2 * bass_freq * t)
            self.layer_manager.add_layer('bass', 'Bass', bass_audio, (-np.pi/6, -0.1, 3.5), 
                                        {'type': 'source', 'color': (255, 255, 0)})
            
            # Vocals sound (filtered noise with formants)
            vocals_audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
            # Add some "formants"
            vocals_audio += 0.15 * np.sin(2 * np.pi * 420 * t)
            vocals_audio += 0.1 * np.sin(2 * np.pi * 700 * t)
            # Modulate with a syllable-like envelope
            syllable_rate = 2  # 2 Hz
            envelope = 0.5 + 0.5 * np.cos(2 * np.pi * syllable_rate * t)
            vocals_audio *= envelope
            self.layer_manager.add_layer('vocals', 'Vocals', vocals_audio, (0.0, 0.1, 2.0), 
                                        {'type': 'source', 'color': (255, 0, 255)})
            
            # Ambient sound (filtered noise)
            np.random.seed(42)  # For reproducibility
            noise = np.random.randn(len(t))
            b = np.ones(100) / 100  # Simple moving average filter
            ambient_audio = np.convolve(noise, b, mode='same') * 0.2
            self.layer_manager.add_layer('ambient', 'Ambient', ambient_audio, (0.0, 0.5, 10.0), 
                                        {'type': 'source', 'color': (0, 255, 255)})
        else:
            # Create layers with more descriptive names
            
            # Bell-like sound (decaying sine with beating)
            bell_freq1 = 800
            bell_freq2 = 810  # Slightly detuned for beating
            bell_audio = 0.5 * np.sin(2 * np.pi * bell_freq1 * t) * np.exp(-t/3)
            bell_audio += 0.5 * np.sin(2 * np.pi * bell_freq2 * t) * np.exp(-t/3)
            self.layer_manager.add_layer('bell', 'Bell', bell_audio, (np.pi/2, 0.3, 5.0), 
                                        {'type': 'source', 'color': (255, 0, 0)})
            
            # Bass drone (low sine with slow modulation)
            bass_freq = 60
            mod_freq = 0.2
            bass_audio = 0.7 * np.sin(2 * np.pi * bass_freq * t)
            # Add amplitude modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            bass_audio *= mod
            self.layer_manager.add_layer('bass_drone', 'Bass Drone', bass_audio, (-np.pi/2, -0.2, 7.0), 
                                        {'type': 'source', 'color': (0, 0, 255)})
            
            # Rhythm pattern
            rhythm_audio = np.zeros_like(t)
            pattern = [1, 0, 0.7, 0, 0.5, 0, 0.7, 0]  # Simple rhythm pattern
            beat_duration = int(self.sample_rate * 0.125)  # 1/8th note at 120 BPM
            for i, strength in enumerate(pattern * int(len(t) / (len(pattern) * beat_duration))):
                start = i * beat_duration
                if start + beat_duration < len(rhythm_audio) and strength > 0:
                    rhythm_audio[start:start+beat_duration] = strength * np.exp(-np.linspace(0, 10, beat_duration))
            self.layer_manager.add_layer('rhythm', 'Rhythm', rhythm_audio, (0.0, -0.3, 3.0), 
                                        {'type': 'source', 'color': (0, 255, 0)})
            
            # Ambient pad (filtered noise with slow filter sweep)
            np.random.seed(42)
            noise = np.random.randn(len(t))
            # Create a time-varying filter by changing the filter length
            pad_audio = np.zeros_like(t)
            for i in range(len(t)):
                filter_size = int(10 + 90 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t[i])))
                if i + filter_size < len(noise):
                    pad_audio[i] = np.mean(noise[i:i+filter_size]) * 0.3
            self.layer_manager.add_layer('ambient_pad', 'Ambient Pad', pad_audio, (0.0, 0.5, 8.0), 
                                        {'type': 'source', 'color': (255, 255, 0)})
        
        print(f"Created {len(self.layer_manager.layers)} synthetic audio layers")
    
    def _audio_processing_thread(self):
        """Audio processing thread function"""
        print("Starting audio processing thread...")
        self.running = True
        
        # Initialize audio processing time
        processing_time = 0
        dt = self.buffer_size / self.sample_rate  # Time per buffer
        
        while self.running:
            start_time = time.time()
            
            # Process controller input
            updated = self.controller.process_input()
            
            # If sound environment was updated, regenerate audio
            if updated:
                # Process audio with the codec
                ambi_block = self.stream_processor.process_block()
                binaural_out = self.stream_processor.get_binaural_output()
                
                # In a real implementation, we would send this to the audio output
                # For now, we just simulate the timing
                
                # Update processing time
                processing_time += dt
                
                # Periodically print status
                if int(processing_time) % 10 == 0 and processing_time > 0:
                    print(f"Audio processing time: {processing_time:.1f}s")
            
            # Sleep to maintain real-time processing
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def start(self):
        """Start the demo"""
        if self.running:
            print("Demo is already running")
            return
        
        print("Starting LISE Demo...")
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_processing_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start visualization
        print("Starting visualization... (close window to exit)")
        self.visualizer.start_animation()
        
        # When visualization window is closed, stop the demo
        self.stop()
    
    def start_with_visualization_in_thread(self):
        """Start with visualization in a separate thread"""
        if self.running:
            print("Demo is already running")
            return
        
        print("Starting LISE Demo with threaded visualization...")
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_processing_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start visualization in a thread
        self.visualization_thread = self.visualizer.run_in_thread()
        
        # Return so the caller can do other things
        return self.visualization_thread
    
    def stop(self):
        """Stop the demo"""
        print("Stopping LISE Demo...")
        self.running = False
        
        # Wait for threads to complete
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        print("LISE Demo stopped")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LISE Demo')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic audio instead of loading files')
    parser.add_argument('--order', type=int, default=3, help='Ambisonic order (default: 3)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--buffer-size', type=int, default=1024, help='Buffer size in samples (default: 1024)')
    args = parser.parse_args()
    
    # Create and start the demo
    demo = LISEDemo(
        use_audio_files=not args.synthetic,
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
        order=args.order
    )
    
    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        demo.stop()


if __name__ == "__main__":
    main()