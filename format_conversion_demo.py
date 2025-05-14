"""
SHAC Format Conversion Demo

This script demonstrates the file format conversion capabilities of the SHAC spatial audio codec,
showing how to convert standard audio files to the SHAC format and export them as binaural audio.

Author: Claude
License: MIT License
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Import SHAC modules
from shac.codec.converter import (
    mono_to_shac, 
    stereo_to_shac, 
    multi_source_to_shac,
    shac_to_binaural,
    ConversionOptions
)
from shac.codec.utils import HRTFInterpolationMethod

# Import file loading utilities
from file_loader import load_audio_file, list_audio_files, create_audio_directory
from audio_utils import create_sample_audio_files

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_scene(output_dir: str):
    """
    Create a demonstration scene with multiple audio sources positioned in 3D space.
    
    Args:
        output_dir: Directory to save output files
    """
    # Ensure audio directory exists with sample files
    audio_dir = create_audio_directory()
    audio_files = list_audio_files(audio_dir)
    
    # If no audio files found, create sample files
    if not audio_files:
        logger.info("No audio files found. Creating sample audio files...")
        create_sample_audio_files()
        audio_files = list_audio_files(audio_dir)
    
    if len(audio_files) < 3:
        logger.warning(f"Only found {len(audio_files)} audio files. Some demos may be limited.")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up positions for demo scene - circle around listener
    positions = []
    n_positions = min(len(audio_files), 6)  # Use up to 6 files
    
    for i in range(n_positions):
        # Calculate position on a circle around the listener
        angle = 2 * np.pi * i / n_positions
        distance = 3.0  # 3 meters from listener
        elevation = 0.0  # at ear level
        
        # For variety, place some sources higher or lower
        if i % 3 == 0:
            elevation = 0.5  # higher
        elif i % 3 == 1:
            elevation = -0.3  # lower
            
        positions.append((angle, elevation, distance))
    
    # Select a subset of audio files
    selected_files = audio_files[:n_positions]
    
    # Generate source names based on filenames
    source_names = [os.path.splitext(os.path.basename(file))[0] for file in selected_files]
    
    # 1. First demo: Convert a single mono file to SHAC
    if len(selected_files) > 0:
        logger.info("Creating mono file demo...")
        mono_file = selected_files[0]
        mono_position = positions[0]
        mono_name = source_names[0]
        
        # Convert to SHAC
        mono_output = os.path.join(output_dir, "mono_demo.shac")
        mono_result = mono_to_shac(mono_file, mono_output, mono_position, mono_name)
        
        # Export to binaural for listening
        mono_binaural = os.path.join(output_dir, "mono_demo_binaural.wav")
        shac_to_binaural(mono_output, mono_binaural)
        
        logger.info(f"Mono demo created: {mono_output}")
        logger.info(f"Binaural render: {mono_binaural}")
    
    # 2. Second demo: Convert a stereo file with width
    if len(selected_files) > 1:
        logger.info("Creating stereo file demo...")
        stereo_file = selected_files[1]
        stereo_position = positions[1]
        stereo_name = source_names[1]
        
        # Convert to mono first to ensure we have a stereo file
        # (our sample files might be mono)
        temp_file = os.path.join(output_dir, "temp_stereo.wav")
        
        # Create a fake stereo file by duplicating mono with slight delay
        try:
            import scipy.io.wavfile
            
            # Load mono file
            audio_info = load_audio_file(stereo_file)
            audio_data = audio_info['audio_data']
            
            # Make sure it's mono
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Create fake stereo
            stereo_data = np.zeros((len(audio_data), 2))
            stereo_data[:, 0] = audio_data  # Left channel
            
            # Right channel with slight delay
            delay = 5  # samples
            stereo_data[delay:, 1] = audio_data[:-delay] * 0.9  # Right channel
            
            # Add some panning
            pan_law = np.linspace(0.3, 0.7, len(audio_data))
            stereo_data[:, 0] *= (1.0 - pan_law)
            stereo_data[:, 1] *= pan_law
            
            # Save as WAV
            stereo_data_int16 = (stereo_data * 32767).astype(np.int16)
            scipy.io.wavfile.write(temp_file, 48000, stereo_data_int16)
            
            stereo_file = temp_file
        except ImportError:
            logger.warning("scipy not available, using original file as stereo")
        except Exception as e:
            logger.warning(f"Error creating stereo file: {str(e)}")
        
        # Convert to SHAC
        stereo_output = os.path.join(output_dir, "stereo_demo.shac")
        stereo_width = 0.5  # 0.5 radians = ~30 degrees
        stereo_result = stereo_to_shac(stereo_file, stereo_output, stereo_position, 
                                     stereo_width, stereo_name)
        
        # Export to binaural for listening
        stereo_binaural = os.path.join(output_dir, "stereo_demo_binaural.wav")
        shac_to_binaural(stereo_output, stereo_binaural)
        
        logger.info(f"Stereo demo created: {stereo_output}")
        logger.info(f"Binaural render: {stereo_binaural}")
    
    # 3. Third demo: Multi-source scene with all files
    if len(selected_files) >= 3:
        logger.info("Creating multi-source scene demo...")
        
        # Convert to SHAC
        scene_output = os.path.join(output_dir, "scene_demo.shac")
        scene_result = multi_source_to_shac(selected_files, scene_output, 
                                          positions, source_names)
        
        # Export to binaural for listening
        scene_binaural = os.path.join(output_dir, "scene_demo_binaural.wav")
        shac_to_binaural(scene_output, scene_binaural)
        
        logger.info(f"Scene demo created: {scene_output}")
        logger.info(f"Binaural render: {scene_binaural}")
        
        # Create visualization of the scene
        visualize_scene(scene_result, output_dir)
    
    logger.info(f"All demos completed. Output files in {output_dir}")
    logger.info("You can play the binaural WAV files with any audio player.")


def visualize_scene(scene_info: Dict, output_dir: str):
    """
    Create a visualization of the audio scene.
    
    Args:
        scene_info: Scene information from multi_source_to_shac
        output_dir: Directory to save visualization
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Plot each source
    for source in scene_info['sources']:
        name = source['name']
        azimuth, elevation, distance = source['position']
        
        # In polar plot, theta is azimuth and r is distance
        ax.plot(azimuth, distance, 'o', markersize=10, label=name)
        
        # Add text label
        ax.text(azimuth, distance + 0.3, name, 
              ha='center', va='bottom', fontsize=9)
        
        # Use different marker size based on elevation
        # (larger = higher, smaller = lower)
        size = 150 + 100 * elevation
        plt.scatter(azimuth, distance, s=size, alpha=0.3)
    
    # Add listener at center
    ax.plot(0, 0, 'ro', markersize=12)
    ax.text(0, 0, "Listener", ha='center', va='bottom', 
          fontsize=12, weight='bold')
    
    # Set plot properties
    ax.set_title("Audio Scene Visualization", fontsize=14)
    ax.set_rticks([1, 2, 3, 4, 5])
    ax.set_rlabel_position(90)
    ax.grid(True)
    
    # Set azimuth labels to show directions
    ax.set_xticklabels(['Front', 'Right', 'Back', 'Left'])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scene_visualization.png"))
    plt.close()


def head_rotation_demo(shac_file: str, output_dir: str):
    """
    Create a demo of different head rotations from the same SHAC file.
    
    Args:
        shac_file: Path to input SHAC file
        output_dir: Directory to save output files
    """
    logger.info(f"Creating head rotation demo from {shac_file}")
    
    # Define rotation angles (in radians)
    yaw_angles = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]  # 0°, 45°, 90°, 180°, -90°
    names = ["front", "right45", "right90", "back", "left90"]
    
    # Render binaural output for each rotation
    for yaw, name in zip(yaw_angles, names):
        output_file = os.path.join(output_dir, f"rotation_{name}.wav")
        rotation = (yaw, 0, 0)  # (yaw, pitch, roll)
        
        shac_to_binaural(shac_file, output_file, head_rotation=rotation)
        logger.info(f"Created rotation demo: {output_file}")


def convert_audio_file(input_file, output_dir):
    """
    Convert an audio file to SHAC format and create binaural output.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory for output files
    
    Returns:
        Tuple of (shac_file_path, binaural_file_path) or None if failed
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
    
    logger.info(f"Converting {input_file} to SHAC format...")
    
    # Determine output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    shac_file = os.path.join(output_dir, f"{base_name}.shac")
    binaural_file = os.path.join(output_dir, f"{base_name}_binaural.wav")
    
    try:
        from file_loader import load_audio_file
        audio_info = load_audio_file(input_file)
        
        # Check if stereo or mono
        if audio_info['channels'] == 2:
            # Stereo file
            logger.info("Processing stereo file...")
            options = ConversionOptions(
                ambisonic_order=3,
                normalize_input=True,
                target_sample_rate=48000
            )
            stereo_to_shac(input_file, shac_file, options)
        else:
            # Mono file
            logger.info("Processing mono file...")
            options = ConversionOptions(
                ambisonic_order=3,
                normalize_input=True,
                target_sample_rate=48000
            )
            # Use front center position for mono source
            position = (0.0, 0.0, 2.0)  # Front center, 2 meters away
            mono_to_shac(input_file, shac_file, position, None, options)
        
        # Create binaural version
        logger.info("Creating binaural rendering...")
        try:
            # Create directory for output
            os.makedirs(os.path.dirname(binaural_file), exist_ok=True)
            
            # Simple direct audio export - bypass HRTF for now
            logger.info(f"Reading SHAC file for binaural rendering: {shac_file}")
            from shac.codec.io import SHACFileReader
            reader = SHACFileReader(shac_file)
            
            # Get available layers
            layer_names = reader.get_layer_names()
            logger.info(f"Found {len(layer_names)} layers in SHAC file: {layer_names}")
            
            if not layer_names:
                raise ValueError("No layers found in SHAC file")
            
            # Read the first layer
            first_layer = layer_names[0]
            audio_data = reader.read_layer(first_layer)
            logger.info(f"Read layer '{first_layer}' with shape {audio_data.shape}")
            
            # Just take first two channels as a simple stereo mix
            if audio_data.shape[0] >= 2:
                stereo_data = audio_data[:2].T  # Take first two channels as stereo
            else:
                # Duplicate mono to stereo
                stereo_data = np.repeat(audio_data[:1].T, 2, axis=1)
            
            # Normalize
            max_val = np.max(np.abs(stereo_data))
            if max_val > 0:
                stereo_data = stereo_data * 0.9 / max_val
            
            # Convert to int16
            stereo_int16 = (stereo_data * 32767).astype(np.int16)
            
            # Save to WAV
            import wave
            import struct
            with wave.open(binaural_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)
                for sample in stereo_int16:
                    wav_file.writeframes(struct.pack('<hh', sample[0], sample[1]))
                    
            logger.info(f"Created binaural rendering: {binaural_file}")
        except Exception as e:
            logger.error(f"Error creating binaural rendering: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return shac_file, binaural_file
    
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SHAC Format Conversion Demo")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--input-file", help="Input audio file to convert (.wav, .mp3, etc.)")
    parser.add_argument("input_file", nargs="?", help="Input audio file (alternative position)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If an input file is provided (either as positional arg or with --input-file)
    input_file = args.input_file or getattr(args, 'input_file', None)
    
    if input_file:
        result = convert_audio_file(input_file, args.output_dir)
        if result:
            shac_file, binaural_file = result
            logger.info(f"Created SHAC file: {shac_file}")
            logger.info(f"Created binaural file: {binaural_file}")
            logger.info(f"Additional rotation files created in {os.path.join(args.output_dir, 'rotations')}")
    else:
        # Run the default demo scene
        create_demo_scene(args.output_dir)
        
        # Run head rotation demo on the scene file
        scene_file = os.path.join(args.output_dir, "scene_demo.shac")
        if os.path.exists(scene_file):
            head_rotation_demo(scene_file, args.output_dir)
    

if __name__ == "__main__":
    main()