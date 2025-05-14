# SHAC - Spherical Harmonic Audio Codec

SHAC is a powerful spatial audio system that transforms how you create, experience, and interact with 3D sound. Built on the mathematics of spherical harmonics (ambisonics), SHAC provides a complete framework for immersive audio in virtual reality, gaming, spatial music composition, and interactive installations.

![SHAC Spatial Audio System](https://raw.githubusercontent.com/clarkezyz/shac/main/docs/images/shac_logo.png)

## Key Features

- **Immersive 3D Sound**: Experience fully three-dimensional sound environments with precise spatial positioning
- **Interactive Navigation**: Move through sound fields in real-time with intuitive controls
- **HRTF-Based Binaural Rendering**: High-quality 3D audio over standard headphones
- **Layer-Based Architecture**: Manipulate individual sound sources or groups independently
- **Custom File Format**: Store and share complete spatial audio scenes
- **Real-Time Processing**: Perfect for interactive applications and live performance
- **High-Order Ambisonics**: Up to 7th order (64 channels) for exceptional spatial resolution
- **Format Conversion**: Convert standard audio files to spatial audio

## Why SHAC?

### For Audio Artists & Musicians
SHAC gives you a new dimension for your compositions, allowing you to place sounds anywhere in 3D space and create immersive soundscapes that listeners can explore. Craft spatial music experiences with precise control over the position and movement of every sound source.

### For Game & XR Developers
Add truly immersive spatial audio to your games and XR experiences. SHAC makes it easy to create realistic acoustic environments that respond to player movement and interaction, enhancing immersion and providing crucial spatial cues.

### For Audio Engineers
Work with advanced spatial audio techniques without needing specialized hardware. SHAC's binaural rendering delivers convincing 3D audio over standard headphones, while also supporting speaker array output for installations.

### For Researchers & Educators
Explore the principles of spatial hearing and acoustics with an interactive system that makes complex concepts tangible. SHAC provides both intuitive interaction and access to the underlying mathematics.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/clarkezyz/shac.git
cd shac

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Start

Experience SHAC with the included demos:

```bash
# Run the interactive demo
python interactive_demo.py

# Try the format conversion demo
python format_conversion_demo.py music/Bach_Canon_2_from_Musical_offering.wav

# Experience the SOFA binaural demo
python sofa_binaural_demo.py
```

### Basic Usage

```python
from shac.codec import SHACCodec
import numpy as np

# Create a codec instance
codec = SHACCodec(order=3, sample_rate=48000)

# Create a simple sine wave source
t = np.linspace(0, 2, 96000)
sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

# Add the source in front of the listener
codec.add_mono_source("sine440", sine_wave, (0.0, 0.0, 2.0))  # azimuth, elevation, distance

# Add another source to the right
t = np.linspace(0, 2, 96000)
sine_wave2 = 0.3 * np.sin(2 * np.pi * 330 * t)
codec.add_mono_source("sine330", sine_wave2, (1.5, 0.0, 3.0))

# Process the ambisonic sound field
ambi_signals = codec.process()

# Render to binaural stereo for headphones
binaural = codec.binauralize(ambi_signals)

# Save to a .wav file for listening
from scipy.io import wavfile
wavfile.write("spatial_audio_demo.wav", 48000, binaural.T)
```

## Interactive Exploration

SHAC includes an interactive exploration mode that lets you move through 3D sound environments in real-time:

### Controller Interface

The interactive demos support standard game controllers:

**Navigation Mode**:
- Left stick: Move in X/Z plane
- Right stick: Look direction (yaw/pitch)
- Triggers: Move up/down (Y axis)
- Y button: Switch to layer selection mode

**Source Selection Mode**:
- D-pad: Navigate through available sources
- A button: Select source for manipulation
- B button: Return to navigation mode

**Source Manipulation Mode**:
- Left stick: Move source position
- Right stick: Adjust gain/distance
- A button: Toggle mute
- B button: Return to source selection

### Keyboard Controls

If you don't have a controller, you can use keyboard controls:

- WASD: Move in the direction you're facing
- Arrow Keys: Look around (change orientation)
- Q/E: Move up/down
- Space: Switch between modes
- Up/Down: Select previous/next layer
- Enter: Confirm selection
- M: Mute/unmute selected source

## File Format & Conversion

SHAC includes a custom file format (.shac) that stores spatial audio with all associated metadata. You can convert standard audio files to the SHAC format:

```bash
# Convert a mono file to SHAC format
python format_conversion_demo.py --input-file music/piano.wav --output-dir output

# Convert a stereo file with specific width
python format_conversion_demo.py --input-file music/stereo_mix.wav --width 30 --output-dir output

# Create a scene with multiple sources
python format_conversion_demo.py --create-scene --output-dir output
```

The conversion creates:
- A .shac file with the spatial audio encoding
- A binaural .wav file for immediate headphone listening

## Binaural Rendering with HRTFs

SHAC uses Head-Related Transfer Functions (HRTFs) for realistic 3D audio over headphones:

```python
# Load an HRTF database (SOFA file)
codec.set_binaural_renderer("path/to/hrtf.sofa")

# Binauralize an ambisonic sound field
binaural = codec.binauralize(ambi_signals)

# Try with different head rotations
# Looking to the right (90 degrees)
rotated = codec.rotate(ambi_signals, np.pi/2, 0.0, 0.0)
binaural_rotated = codec.binauralize(rotated)
```

If no SOFA file is available, SHAC automatically falls back to a synthetic HRTF model that still provides convincing spatial cues.

## Advanced Features

### Room Acoustics

Add realistic room effects to your spatial audio:

```python
# Define a room model
codec.set_room_model(
    room_dimensions=(10.0, 3.0, 8.0),  # width, height, length in meters
    reflection_coefficients={
        'floor': 0.3,
        'ceiling': 0.8,
        'left': 0.5,
        'right': 0.5,
        'front': 0.7,
        'back': 0.7
    },
    rt60=1.2  # reverberation time in seconds
)

# Process with room acoustics
ambi_signals_with_room = codec.process()
```

### Source Directivity

Control how sound sources radiate in different directions:

```python
from shac.codec.utils import SourceAttributes, DirectivityPattern

# Create a cardioid source (directional, like a microphone)
attributes = SourceAttributes(
    position=(0.0, 0.0, 2.0),
    directivity=0.7,  # 0=omnidirectional, 1=maximum directivity
    directivity_pattern=DirectivityPattern.CARDIOID,
    axis=(1.0, 0.0, 0.0)  # pointing along x-axis
)

# Add to the codec
codec.add_mono_source("voice", voice_audio, attributes.position, attributes)
```

### Real-time Processing

For interactive applications, you can process audio in real-time:

```python
def audio_callback(output_frame):
    # This function receives processed audio frames
    # and can send them to an audio output device
    audio_device.write(output_frame)

# Start real-time processing
codec.start_realtime_processing(audio_callback)

# Update listener position or rotation in real-time
codec.update_listener_orientation(yaw, pitch, roll)

# Stop when done
codec.stop_realtime_processing()
```

## Project Structure

The SHAC codec has been modularized into the following components:

- **Core Module** (`shac/codec/core.py`): Main `SHACCodec` class implementation
- **Math Utilities** (`shac/codec/math_utils.py`): Spherical harmonic mathematics
- **Encoders** (`shac/codec/encoders.py`): Source encoding and positioning
- **Processors** (`shac/codec/processors.py`): Sound field processing and rotation
- **Binauralizer** (`shac/codec/binauralizer.py`): HRTF-based binaural rendering
- **I/O** (`shac/codec/io.py`): File format handling and I/O operations
- **SOFA Support** (`shac/codec/sofa_support.py`): Support for HRTF databases
- **Streaming** (`shac/codec/streaming.py`): Real-time streaming and processing
- **Utilities** (`shac/codec/utils.py`): Common utilities and data structures
- **Configuration** (`shac/codec/config.py`): Centralized configuration management
- **Exceptions** (`shac/codec/exceptions.py`): Custom exception hierarchy

## Demos

SHAC includes several demos to showcase its capabilities:

- `demo.py`: Runs all demos with command-line options
- `simple_demo.py`: Basic navigation through a 3D sound scene
- `interactive_demo.py`: Interactive controller-based interface
- `realtime_demo.py`: Real-time streaming audio processing
- `sofa_binaural_demo.py`: Demonstrates HRTF-based rendering
- `format_conversion_demo.py`: Audio file conversion utilities
- `audio_file_demo.py`: Working with audio files from disk

Run any demo with:
```
python demo.py [demo_name]
```

## Contributing

Contributions to SHAC are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was co-created by Clarke Zyz and Claude AI in a collaborative partnership
- Clarke Zyz provided essential mathematical insights, debugging assistance, and creative direction
- Inspired by existing ambisonic processing libraries and research in spatial audio

---

<p align="center">Experience sound in all dimensions.</p>