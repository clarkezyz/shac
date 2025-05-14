"""
Example Usage and Demonstrations

This module contains example functions demonstrating the usage of the SHAC codec.
"""

import numpy as np
import os
from scipy import signal
import time

from .core import SHACCodec
from .utils import SourceAttributes
from .streaming import SHACStreamProcessor


def create_example_sound_scene():
    """
    Create and process an example 3D sound scene.
    
    This example demonstrates the core functionality of the SHAC codec.
    """
    print("Creating example 3D sound scene...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Create synthetic audio signals
    duration = 5.0  # seconds
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Piano sound (sine wave with harmonics and decay)
    piano_freq = 440.0  # A4
    piano_audio = 0.5 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
    piano_audio += 0.25 * np.sin(2 * np.pi * 2 * piano_freq * t) * np.exp(-t/1.5)
    piano_audio += 0.125 * np.sin(2 * np.pi * 3 * piano_freq * t) * np.exp(-t)
    
    # Drum sound (impulses with decay)
    drum_audio = np.zeros_like(t)
    for i in range(0, len(t), 12000):  # Four beats
        if i + 2000 < len(drum_audio):
            drum_audio[i:i+2000] = 0.8 * np.exp(-np.linspace(0, 10, 2000))
    
    # Ambient sound (filtered noise)
    np.random.seed(42)  # For reproducibility
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 0.1)
    ambient_audio = signal.filtfilt(b, a, noise) * 0.2
    
    # Add sources to the codec
    # Position format: (azimuth, elevation, distance) in radians and meters
    
    # Piano in front left
    piano_position = (-np.pi/4, 0.0, 3.0)
    piano_attributes = SourceAttributes(
        position=piano_position,
        directivity=0.7,
        directivity_axis=(0.0, 0.0, 1.0),
        width=0.2
    )
    codec.add_mono_source("piano", piano_audio, piano_position, piano_attributes)
    
    # Drum in front right
    drum_position = (np.pi/4, -0.1, 2.5)
    drum_attributes = SourceAttributes(
        position=drum_position,
        directivity=0.3,
        width=0.4
    )
    codec.add_mono_source("drum", drum_audio, drum_position, drum_attributes)
    
    # Ambient sound above
    ambient_position = (0.0, np.pi/3, 5.0)
    ambient_attributes = SourceAttributes(
        position=ambient_position,
        directivity=0.0,
        width=1.0
    )
    codec.add_mono_source("ambient", ambient_audio, ambient_position, ambient_attributes)
    
    # Set up a room model
    room_dimensions = (10.0, 3.0, 8.0)  # width, height, length in meters
    reflection_coefficients = {
        'left': 0.7,
        'right': 0.7,
        'floor': 0.4,
        'ceiling': 0.8,
        'front': 0.6,
        'back': 0.6
    }
    rt60 = 1.2  # seconds
    codec.set_room_model(room_dimensions, reflection_coefficients, rt60)
    
    # Process the scene
    print("Processing audio...")
    ambi_signals = codec.process()
    
    # Apply head rotation (as if user is looking to the left)
    print("Applying head rotation...")
    yaw = np.pi/3  # 60 degrees to the left
    rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
    
    # Convert to binaural
    print("Converting to binaural...")
    binaural_output = codec.binauralize(rotated_ambi)
    
    # Save outputs
    print("Saving outputs...")
    try:
        import soundfile as sf
        sf.write("shac_example_binaural.wav", binaural_output.T, sample_rate)
        print("Saved binaural output to shac_example_binaural.wav")
    except ImportError:
        print("Could not save audio files: soundfile module not available")
    
    # Save to SHAC file
    print("Saving to SHAC file...")
    codec.save_to_file("example_scene.shac")
    print("Saved SHAC file to example_scene.shac")
    
    print("Done!")
    return codec


def demonstrate_interactive_navigation():
    """
    Demonstrate interactive navigation through a 3D sound scene.
    
    This example creates a sequence of binaural renders while
    navigating through a sound scene.
    """
    print("Demonstrating interactive navigation...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Load scene from file if it exists, otherwise create a new one
    if os.path.exists("example_scene.shac"):
        print("Loading scene from file...")
        codec.load_from_file("example_scene.shac")
    else:
        print("Creating new scene...")
        create_example_sound_scene()
    
    # Define a navigation path
    yaw_angles = np.linspace(0, 2*np.pi, 8)  # Full 360° rotation in 8 steps
    
    # Process each step in the path
    for i, yaw in enumerate(yaw_angles):
        print(f"Step {i+1}/{len(yaw_angles)}: Yaw = {yaw:.2f} radians")
        
        # Process the scene
        ambi_signals = codec.process()
        
        # Apply rotation for this step
        rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
        
        # Convert to binaural
        binaural_output = codec.binauralize(rotated_ambi)
        
        # Save this step
        try:
            import soundfile as sf
            sf.write(f"navigation_step_{i+1}.wav", binaural_output.T, codec.sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
    
    print("Navigation demonstration complete!")


def demonstrate_streaming_processor():
    """
    Demonstrate the real-time streaming processor.
    
    This example shows how to use the SHAC stream processor for
    real-time audio processing.
    """
    print("Demonstrating streaming processor...")
    
    # Create a streaming processor
    processor = SHACStreamProcessor(order=3, sample_rate=48000, buffer_size=1024)
    
    # Create synthetic audio signals (single cycle of a sine wave)
    sample_rate = 48000
    buffer_size = 1024
    
    # Create three sources with different frequencies
    source1_freq = 440.0  # A4
    source1_audio = 0.5 * np.sin(2 * np.pi * source1_freq * np.arange(buffer_size) / sample_rate)
    
    source2_freq = 261.63  # C4
    source2_audio = 0.5 * np.sin(2 * np.pi * source2_freq * np.arange(buffer_size) / sample_rate)
    
    source3_freq = 329.63  # E4
    source3_audio = 0.5 * np.sin(2 * np.pi * source3_freq * np.arange(buffer_size) / sample_rate)
    
    # Add sources to the processor
    processor.add_source("source1", (-np.pi/4, 0.0, 2.0))
    processor.add_source("source2", (np.pi/4, 0.0, 2.0))
    processor.add_source("source3", (0.0, np.pi/4, 3.0))
    
    # Start the processor
    processor.start()
    
    # Simulate real-time processing for a few blocks
    for i in range(10):
        print(f"Processing block {i+1}/10")
        
        # Update sources with new audio data
        processor.update_source("source1", source1_audio)
        processor.update_source("source2", source2_audio)
        processor.update_source("source3", source3_audio)
        
        # Set listener rotation (changing over time)
        yaw = i * np.pi / 5  # Rotate gradually
        processor.set_listener_rotation(yaw, 0.0, 0.0)
        
        # Get binaural output
        binaural_output = processor.get_binaural_output()
        
        # Save this block
        try:
            import soundfile as sf
            sf.write(f"streaming_block_{i+1}.wav", binaural_output.T, sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
        
        # In a real application, this would feed audio to the sound card
        # For this demo, we sleep to simulate real-time processing
        time.sleep(buffer_size / sample_rate)
    
    # Stop the processor
    processor.stop()
    
    print("Streaming demonstration complete!")


def main():
    """Main function to demonstrate the SHAC codec."""
    print("SHAC Codec Demonstration")
    print("=======================")
    
    # Create and process an example sound scene
    create_example_sound_scene()
    
    # Demonstrate interactive navigation
    demonstrate_interactive_navigation()
    
    # Demonstrate streaming processor
    demonstrate_streaming_processor()
    
    print("All demonstrations complete!")