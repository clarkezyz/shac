"""
SOFA-based Binaural Rendering Demo

This script demonstrates the SOFA file support for HRTF-based binaural rendering
in the SHAC spatial audio codec. It shows how to load SOFA files, create ambisonic
signals, and render them to binaural stereo using measured HRTFs.

Author: Claude
License: MIT License
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

# Import SHAC modules
from shac.codec import sofa_support
from shac.codec.binauralizer import binauralize_ambisonics, binauralize_mono_source
from shac.codec.math_utils import real_spherical_harmonic, AmbisonicNormalization
from shac.codec.utils import HRTFInterpolationMethod

# Try to import audio playback libraries
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not found. Audio playback will be disabled.")
    print("To enable audio playback, install pygame: pip install pygame")


def generate_test_sound(frequency=440, duration=2.0, sample_rate=48000):
    """Generate a test sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)


def generate_frequency_sweep(start_freq=100, end_freq=8000, duration=5.0, sample_rate=48000):
    """Generate a logarithmic frequency sweep."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Exponential sweep
    k = (end_freq / start_freq) ** (1/duration)
    phase = 2 * np.pi * start_freq * ((k**t - 1) / np.log(k))
    return 0.5 * np.sin(phase)


def encode_mono_source(mono_signal, position, order=3):
    """
    Encode a mono signal to ambisonic signals at a specific position.
    
    Args:
        mono_signal: Mono audio signal
        position: Tuple of (azimuth, elevation, distance) in radians/meters
        order: Ambisonic order
        
    Returns:
        Ambisonic signals with shape [(order+1)², samples]
    """
    azimuth, elevation, distance = position
    n_channels = (order + 1) ** 2
    n_samples = len(mono_signal)
    
    # Apply simple distance attenuation
    if distance > 0:
        attenuated_signal = mono_signal * (1.0 / max(1.0, distance))
    else:
        attenuated_signal = mono_signal
    
    # Initialize ambisonic signals
    ambi_signals = np.zeros((n_channels, n_samples))
    
    # Encode each ambisonic channel
    for n in range(order + 1):
        for m in range(-n, n + 1):
            # Calculate ACN index
            acn = n * n + n + m
            
            # Calculate spherical harmonic value for this direction
            sh_val = real_spherical_harmonic(n, m, azimuth, elevation, AmbisonicNormalization.SN3D)
            
            # Encode the mono signal to this ambisonic channel
            ambi_signals[acn] = attenuated_signal * sh_val
    
    return ambi_signals


def play_audio(binaural_audio, sample_rate=48000):
    """
    Play binaural audio using pygame.
    
    Args:
        binaural_audio: Stereo audio with shape [2, samples]
        sample_rate: Sample rate in Hz
    """
    if not PYGAME_AVAILABLE:
        print("pygame not available. Cannot play audio.")
        return
    
    # Initialize pygame mixer
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=1024)
    
    # Convert to 16-bit integers and proper shape for pygame
    audio_int16 = (binaural_audio.T * 32767).astype(np.int16)
    
    # Create a pygame sound object
    sound = pygame.mixer.Sound(buffer=audio_int16)
    
    # Play the sound
    channel = sound.play()
    
    # Wait until playback is finished
    while channel.get_busy():
        pygame.time.wait(100)


def save_wav(filename, audio, sample_rate=48000):
    """
    Save audio to WAV file.
    
    Args:
        filename: Output filename
        audio: Audio data with shape [channels, samples] or [samples] for mono
        sample_rate: Sample rate in Hz
    """
    try:
        import scipy.io.wavfile
        
        # Convert to proper format for scipy
        if audio.ndim == 2:
            # Stereo: convert to [samples, channels]
            audio_out = audio.T
        else:
            # Mono: keep as [samples]
            audio_out = audio
        
        # Convert to 16-bit int
        audio_int16 = (audio_out * 32767).astype(np.int16)
        
        # Save file
        scipy.io.wavfile.write(filename, sample_rate, audio_int16)
        print(f"Audio saved to {filename}")
    except ImportError:
        print("scipy not found. Cannot save WAV file.")
        print("To save audio files, install scipy: pip install scipy")


def visualize_hrtfs(hrtf_data, num_directions=12, elevation=0):
    """
    Visualize the HRTFs for different directions at a fixed elevation.
    
    Args:
        hrtf_data: HRTF data dictionary from sofa_support
        num_directions: Number of directions to visualize
        elevation: Elevation angle in radians
    """
    if 'hrtf_dict' not in hrtf_data or 'positions' not in hrtf_data:
        print("Cannot visualize HRTFs: Data does not contain position-based HRTFs")
        return
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"HRTF Impulse Responses at {elevation:.2f} radians elevation", fontsize=16)
    
    # Get positions close to specified elevation
    positions = hrtf_data['positions']
    elevation_tolerance = 0.3  # radians
    
    # Filter positions by elevation
    valid_indices = []
    for i, pos in enumerate(positions):
        if abs(pos[1] - elevation) < elevation_tolerance:
            valid_indices.append(i)
    
    # If no positions at requested elevation, use all available
    if not valid_indices:
        print(f"No positions found at elevation {elevation:.2f}. Using all available positions.")
        valid_indices = list(range(len(positions)))
    
    # Select a subset of directions
    if len(valid_indices) > num_directions:
        step = len(valid_indices) // num_directions
        indices = valid_indices[::step][:num_directions]
    else:
        indices = valid_indices
    
    # Set up color cycle
    cmap = plt.cm.get_cmap('hsv')
    colors = [cmap(i/len(indices)) for i in range(len(indices))]
    
    # Plot HRTF impulse responses for left ear
    for i, idx in enumerate(indices):
        pos = tuple(positions[idx])
        hrtf = hrtf_data['hrtf_dict'][pos]
        left_ir = hrtf['left']
        right_ir = hrtf['right']
        
        # Plot left ear IR
        ax1.plot(left_ir, color=colors[i], label=f"Azimuth: {pos[0]:.2f}")
        # Plot right ear IR
        ax2.plot(right_ir, color=colors[i])
    
    # Set axis labels
    ax1.set_title("Left Ear Impulse Responses")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Right Ear Impulse Responses")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    
    # Create a legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hrtf_visualization.png'))
    
    # Show plot
    plt.show()


def visualize_source_movement(output_dir, sample_rate=48000):
    """
    Visualize a source moving around the listener.
    
    Args:
        output_dir: Directory to save visualization and audio files
        sample_rate: Sample rate in Hz
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # Plot listener at center
    ax.plot(0, 0, 'ro', markersize=12, label='Listener')
    
    # Define source path (circle around listener)
    num_positions = 16
    angles = np.linspace(0, 2*np.pi, num_positions, endpoint=False)
    radius = 3.0
    
    # Plot source positions
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    ax.plot(xs, ys, 'bo', markersize=8, label='Source Positions')
    
    # Add arrows to show movement direction
    for i in range(num_positions):
        angle = angles[i]
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        dx = -0.5 * np.sin(angle)
        dy = 0.5 * np.cos(angle)
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc='b', ec='b', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Z Position (meters)')
    ax.set_title('Source Movement Around Listener')
    ax.grid(True)
    ax.legend()
    
    # Add cardinal directions
    ax.text(4.5, 0, 'East', ha='center')
    ax.text(0, 4.5, 'North', ha='center')
    ax.text(-4.5, 0, 'West', ha='center')
    ax.text(0, -4.5, 'South', ha='center')
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'source_movement.png'))
    plt.close()


def moving_source_demo(hrtf_database, output_dir, sample_rate=48000, duration=8.0):
    """
    Create a demo with a source moving around the listener.
    
    Args:
        hrtf_database: HRTF database path or loaded data
        output_dir: Directory to save output files
        sample_rate: Sample rate in Hz
        duration: Total duration in seconds
    
    Returns:
        Binaural audio of the moving source
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HRTF database if it's a string path
    if isinstance(hrtf_database, str):
        hrtf_data = sofa_support.load_sofa_file(hrtf_database)
    else:
        hrtf_data = hrtf_database
    
    # Generate a tone (a chord with three harmonically related frequencies)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1320 * t)
    
    # Apply amplitude envelope for smooth beginning and end
    envelope = np.ones_like(tone)
    fade_samples = int(0.1 * sample_rate)  # 100 ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    tone *= envelope
    
    # Define source movement (circle around listener)
    num_positions = int(duration * 2)  # 2 positions per second
    angles = np.linspace(0, 2*np.pi, num_positions, endpoint=False)
    radius = 3.0  # 3 meters from listener
    elevation = 0.0  # at ear level
    
    # Visualize source movement
    visualize_source_movement(output_dir, sample_rate)
    
    # Create binaural output buffer
    binaural_output = np.zeros((2, n_samples))
    
    # Process audio in blocks
    block_size = int(sample_rate * (duration / num_positions))
    
    for i in range(num_positions):
        # Calculate position for this block
        azimuth = angles[i]
        position = (azimuth, elevation, radius)
        
        # Calculate start and end samples for this block
        start_sample = i * block_size
        end_sample = min(start_sample + block_size, n_samples)
        block_samples = end_sample - start_sample
        
        # Get audio block
        audio_block = tone[start_sample:end_sample]
        
        # Render block to binaural using direct method
        binaural_block = binauralize_mono_source(
            audio_block, 
            position, 
            hrtf_database=hrtf_data,
            interpolation_method=HRTFInterpolationMethod.SPHERICAL
        )
        
        # Add to output with crossfade
        # Simple overlap-add with linear crossfade
        if i > 0:
            # Apply crossfade at the start of the block (overlap with previous block)
            crossfade_samples = min(100, block_samples)  # 100 samples or full block if smaller
            crossfade_weight = np.linspace(0, 1, crossfade_samples).reshape(1, -1)
            
            # Apply crossfade to start of current block
            binaural_output[:, start_sample:start_sample+crossfade_samples] *= (1 - crossfade_weight)
            binaural_block[:, :crossfade_samples] *= crossfade_weight
        
        # Add block to output
        binaural_output[:, start_sample:end_sample] += binaural_block
    
    # Save audio
    save_wav(os.path.join(output_dir, 'moving_source.wav'), binaural_output, sample_rate)
    
    return binaural_output


def ambisonic_rotation_demo(hrtf_database, output_dir, sample_rate=48000, order=3):
    """
    Create a demo showing how ambisonics can be rotated.
    
    This demonstrates encoding sources to ambisonics and then rotating
    the sound field before binaural rendering.
    
    Args:
        hrtf_database: HRTF database path or loaded data
        output_dir: Directory to save output files
        sample_rate: Sample rate in Hz
        order: Ambisonic order
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Duration in seconds
    duration = 5.0
    n_samples = int(sample_rate * duration)
    
    # Define two sources with different sounds
    source1_pos = (0.0, 0.0, 5.0)  # Directly in front
    source2_pos = (np.pi/2, 0.0, 5.0)  # Directly to the right
    
    # Generate different sounds for the two sources
    t = np.linspace(0, duration, n_samples, endpoint=False)
    source1_signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4)
    source2_signal = 0.3 * np.sin(2 * np.pi * 587.33 * t)  # 587.33 Hz (D5)
    
    # Encode both sources to ambisonics
    ambi_source1 = encode_mono_source(source1_signal, source1_pos, order)
    ambi_source2 = encode_mono_source(source2_signal, source2_pos, order)
    
    # Mix the two ambisonic signals
    ambi_signals = ambi_source1 + ambi_source2
    
    # Define rotation angles
    rotation_speeds = [0.0, 0.5, 1.0]  # Rotations per second
    rotation_angles = []
    
    for speed in rotation_speeds:
        # Calculate rotation angle for each sample (radians)
        angle = 2 * np.pi * speed * t
        rotation_angles.append(angle)
    
    # Import ambisonics rotation function
    from shac.codec.processors import rotate_ambisonics
    
    # Render binaural for each rotation speed
    for i, angles in enumerate(rotation_angles):
        speed = rotation_speeds[i]
        
        # Initialize output buffer
        binaural_output = np.zeros((2, n_samples))
        
        # Process in blocks to apply time-varying rotation
        block_size = 4096
        n_blocks = int(np.ceil(n_samples / block_size))
        
        for b in range(n_blocks):
            # Calculate start and end samples for this block
            start_sample = b * block_size
            end_sample = min(start_sample + block_size, n_samples)
            block_samples = end_sample - start_sample
            
            # Get ambisonic block
            ambi_block = ambi_signals[:, start_sample:end_sample]
            
            # Get rotation angle for this block (using middle sample)
            middle_sample = start_sample + block_samples // 2
            if middle_sample < len(angles):
                yaw = angles[middle_sample]
            else:
                yaw = 0.0
                
            # Apply rotation
            if speed > 0:
                rotated_block = rotate_ambisonics(ambi_block, yaw, 0.0, 0.0)
            else:
                rotated_block = ambi_block
            
            # Render to binaural
            binaural_block = binauralize_ambisonics(
                rotated_block, 
                hrtf_database=hrtf_database,
                normalize=False
            )
            
            # Add to output
            binaural_output[:, start_sample:end_sample] = binaural_block
        
        # Normalize output
        max_val = np.max(np.abs(binaural_output))
        if max_val > 0:
            binaural_output = binaural_output / max_val * 0.9
        
        # Save audio
        filename = f"rotating_{speed:.1f}_rps.wav"
        save_wav(os.path.join(output_dir, filename), binaural_output, sample_rate)
        
        print(f"Created demo with {speed:.1f} rotations per second")
    
    return binaural_output


def main():
    """Main function to run the demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SOFA-based binaural rendering demo for SHAC")
    parser.add_argument("--sofa", help="Path to SOFA file (default: download MIT KEMAR)", default=None)
    parser.add_argument("--play", help="Play audio (requires pygame)", action="store_true")
    parser.add_argument("--order", help="Ambisonic order", type=int, default=3)
    parser.add_argument("--demo", help="Demo type", choices=["moving", "rotation", "all"], default="all")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or download SOFA file
    if args.sofa:
        sofa_path = args.sofa
        if not os.path.exists(sofa_path):
            print(f"SOFA file not found: {sofa_path}")
            print("Downloading default SOFA file...")
            sofa_path = sofa_support.download_default_hrtf_database(output_dir)
    else:
        print("No SOFA file specified. Downloading MIT KEMAR HRTF...")
        sofa_path = sofa_support.download_default_hrtf_database(output_dir)
    
    print(f"Loading SOFA file: {sofa_path}")
    hrtf_data = sofa_support.load_sofa_file(sofa_path)
    
    print(f"SOFA file loaded: {hrtf_data['convention']} v{hrtf_data['version']}")
    print(f"Contains {hrtf_data['n_measurements']} measurements at sample rate {hrtf_data['sample_rate']} Hz")
    
    # Visualize HRTFs
    print("Visualizing HRTFs...")
    visualize_hrtfs(hrtf_data)
    
    # Run selected demos
    if args.demo in ["moving", "all"]:
        print("\nRunning moving source demo...")
        binaural_audio = moving_source_demo(hrtf_data, output_dir)
        
        if args.play and PYGAME_AVAILABLE:
            print("Playing moving source demo...")
            play_audio(binaural_audio)
    
    if args.demo in ["rotation", "all"]:
        print("\nRunning ambisonic rotation demo...")
        binaural_audio = ambisonic_rotation_demo(hrtf_data, output_dir, order=args.order)
        
        if args.play and PYGAME_AVAILABLE:
            print("Playing rotation demo (0.5 RPS)...")
            play_audio(binaural_audio)
    
    print(f"\nDone! Output files saved to {output_dir}")
    print("You can play the WAV files with any audio player.")


if __name__ == "__main__":
    main()