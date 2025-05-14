# Getting Started with SHAC

This guide will help you get started with the Spherical Harmonic Audio Codec (SHAC),
a spatial audio codec for interactive 3D sound applications.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shac.git
cd shac

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Basic Usage

### Creating a Codec

```python
from shac.codec import SHACCodec
from shac.codec.math_utils import AmbisonicNormalization

# Create a codec with default settings
codec = SHACCodec()

# Or customize the codec parameters
codec = SHACCodec(
    order=3,                                      # Ambisonic order
    sample_rate=48000,                            # Sample rate in Hz
    normalization=AmbisonicNormalization.SN3D,    # Normalization convention
)
```

### Adding Sources

```python
import numpy as np

# Generate a 1-second mono signal at 440 Hz
sample_rate = 48000
t = np.linspace(0, 1, sample_rate)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

# Position in spherical coordinates (azimuth, elevation, distance) in radians/meters
position = (0.0, 0.0, 2.0)  # Directly in front, 2 meters away

# Add the mono source to the codec
codec.add_mono_source("sine440", audio, position)

# Add a stereo source (requires left and right channels)
left_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
right_audio = 0.5 * np.sin(2 * np.pi * 880 * t)
position = (0.0, 0.0, 3.0)  # Directly in front, 3 meters away
width = np.pi/4  # 45 degrees width

codec.add_stereo_source("stereo_source", left_audio, right_audio, position, width)
```

### Processing the Sound Field

```python
# Process all sources and get the ambisonic output
ambisonic_output = codec.process()

# Apply head rotation
yaw = np.pi/4  # 45 degrees to the left
pitch = 0.0
roll = 0.0
rotated_output = codec.rotate(ambisonic_output, yaw, pitch, roll)

# Convert to binaural stereo for headphone playback
binaural_output = codec.binauralize(rotated_output)
```

### Saving and Loading

```python
# Save to a SHAC file
codec.save_to_file("my_scene.shac")

# Load from a SHAC file
codec.load_from_file("my_scene.shac")
```

## Working with Configurations

```python
from shac.codec.config import SHACConfig

# Create a configuration
config = SHACConfig()

# Customize processing settings
config.processing.sample_rate = 48000
config.processing.order = 3
config.processing.enable_doppler = True

# Customize binaural rendering
config.binaural.interpolation_quality = 2
config.binaural.nearfield_compensation = True

# Create a codec with this configuration
codec = SHACCodec.from_config(config)
```

## Real-time Streaming Processing

```python
from shac.codec.streaming import SHACStreamProcessor

# Create a streaming processor
processor = SHACStreamProcessor(
    order=2, 
    sample_rate=48000, 
    buffer_size=1024
)

# Add sources
processor.add_source("stream1", (0.0, 0.0, 2.0))
processor.add_source("stream2", (np.pi/2, 0.0, 3.0))

# Start processing
processor.start()

# In your audio callback function:
def audio_callback(input_buffer, output_buffer):
    # Update sources with new audio data
    processor.update_source("stream1", input_buffer[0])
    processor.update_source("stream2", input_buffer[1])
    
    # Update listener orientation
    processor.set_listener_rotation(yaw, pitch, roll)
    
    # Get binaural output
    binaural_output = processor.get_binaural_output()
    
    # Copy to output buffer
    output_buffer[:] = binaural_output

# Later, stop processing
processor.stop()
```

## Advanced Features

See the [API Reference](api/index.md) for detailed documentation of all features
and the [Examples](examples.md) page for more complex usage scenarios.