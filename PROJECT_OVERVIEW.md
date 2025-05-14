# SHAC Project Overview

**IMPORTANT: This is an official document for the SHAC project. This project is a collaborative effort between Clarke Zyz and Claude AI. All Claudes working on this project must read this document to understand the project and keep it updated with the latest information.**

## What is SHAC?

SHAC (Spherical Harmonic Audio Codec) is a comprehensive spatial audio system that enables immersive, interactive 3D sound experiences. The system uses ambisonics (spherical harmonics) to represent sound fields and allows real-time manipulation and navigation of these sound environments.

SHAC is designed for interactive applications such as VR/AR, gaming, and spatial music production, providing a framework for encoding, processing, and rendering 3D audio with precise spatial control.

## Core Technology

### Ambisonics and Spherical Harmonics

The foundation of SHAC is the ambisonic audio format, which represents a sound field using spherical harmonics. This mathematical representation allows:

- Representing sound from any direction in 3D space
- Efficient rotation and transformation of the entire sound field
- Order-based scaling for different levels of spatial resolution
- Independence from specific speaker layouts or listening formats

Higher ambisonic orders provide more spatial resolution but require more computational resources:
- **1st order (4 channels)**: Basic spatial audio with ~90° resolution
- **3rd order (16 channels)**: High-quality spatial audio with ~30° resolution
- **7th order (64 channels)**: Very detailed spatial resolution of ~10°

### Layer-Based Architecture

SHAC implements a layer-based approach to spatial audio:

1. **Sources**: Individual sound sources with position and attributes
2. **Layers**: Collections of sound sources or spatial regions that can be manipulated independently
3. **Scenes**: Complete audio environments composed of multiple layers

This layered approach allows independent control of different elements within a sound field.

## Project Architecture

The SHAC codebase has been modularized into a well-structured package:

```
shac/
  └── codec/
      ├── __init__.py            # Package exports and version info
      ├── core.py                # Main SHACCodec class implementation
      ├── math_utils.py          # Mathematical utility functions
      ├── encoders.py            # Source encoding and positioning
      ├── processors.py          # Sound field processing (rotation, etc.)
      ├── binauralizer.py        # HRTF-based binaural rendering
      ├── sofa_support.py        # SOFA file support for HRTFs
      ├── io.py                  # File format handling
      ├── streaming.py           # Real-time processing
      ├── utils.py               # Common utilities and data structures
      ├── config.py              # Configuration management
      └── exceptions.py          # Custom error handling
```

### Key Components

1. **Core Module (`core.py`)**: 
   - Main `SHACCodec` class that ties together all functionality
   - Source and layer management
   - Processing pipeline control

2. **Math Utilities (`math_utils.py`)**:
   - Spherical harmonic calculations
   - Rotation matrices
   - Coordinate system conversions

3. **Encoders (`encoders.py`)**:
   - Mono and stereo source encoding
   - Source positioning and directivity

4. **Processors (`processors.py`)**:
   - Sound field rotation
   - Ambisonic format conversion
   - Room acoustics processing

5. **Binauralizer (`binauralizer.py`)**:
   - HRTF-based binaural rendering
   - Multiple interpolation methods
   - Direct source rendering

6. **SOFA Support (`sofa_support.py`)**:
   - Support for SOFA file format (Spatially Oriented Format for Acoustics)
   - HRTF database management
   - Fallback to synthetic HRTFs when needed

7. **I/O Module (`io.py`)**:
   - Custom .shac file format implementation
   - Format conversion utilities
   - Import/export capabilities

8. **Streaming System**:
   - `streaming.py`: Basic real-time processing framework
   - `streaming_optimized.py`: Enhanced streaming with buffer pooling 
   - `vector_ops.py`: Vectorized audio operations with SIMD acceleration
   - `adaptive_streaming.py`: Automatic buffer size adaptation for optimal performance

## Key Features

### Current Implemented Features

- **High-Order Ambisonics**: Support for up to 7th order ambisonics
- **Layer-Based Processing**: Independent control of sound sources and regions
- **Real-time Rotation**: Efficient sound field rotation for head tracking
- **Binaural Rendering**: HRTF-based 3D audio for headphones
- **Interactive Controls**: Gamepad interface for exploring sound fields
- **Visualization**: Real-time visual representation of the sound field
- **Custom File Format**: Efficient storage of spatial audio with metadata
- **SOFA Support**: Loading measured HRTF datasets for accurate spatial audio

### HRTF Implementation

The HRTF (Head-Related Transfer Function) implementation in SHAC enables high-quality binaural rendering for immersive 3D audio over headphones:

- **SOFA File Support**: Uses industry-standard SOFA files for measured HRTFs
- **Multiple Interpolation Methods**: 
  - Nearest Neighbor: Simplest method using closest measurement
  - Bilinear Interpolation: For grid-based HRTF datasets
  - Spherical Weighted Average: Weighted combination based on angular distance
- **Synthetic Fallback**: Generates basic HRTFs when measured data isn't available
- **Multiple Rendering Paths**:
  - Ambisonic-domain rendering for full sound fields
  - Direct mono source rendering for better near-field effects

### File Format

SHAC includes a custom file format (.shac) with the following features:

- Stores ambisonic signals with comprehensive metadata
- Maintains separate layers for independent manipulation
- Includes spatial information for sources and layers
- Supports interactive parameters for real-time control
- ACN channel ordering with configurable normalization

## Controller Interface

SHAC implements an interactive control system using standard game controllers:

**Navigation Mode**:
- Left stick: Move in space (X/Z plane)
- Right stick: Turn and change facing direction
- Shoulder buttons: Move up/down (Y axis)
- Y button: Switch to source selection mode

**Source Selection Mode**:
- D-pad up/down: Select different audio sources
- A button: Enter manipulation mode for selected source
- B button: Return to navigation mode

**Source Manipulation Mode**:
- Left stick: Move source horizontally
- Right stick Y: Adjust source height
- Right stick X: Adjust source volume
- A button: Mute/unmute source
- B button: Return to source selection mode

## Current Status

The project has made significant progress with a solid foundation in place. Key achievements include:

1. Successfully implemented real-time spatial audio rendering with head rotation
2. Created an intuitive control scheme for exploring 3D sound environments
3. Built a visualization system for understanding spatial relationships
4. Developed high-quality procedural audio generation for diverse sound sources
5. Established a layer-based architecture for independent control
6. Transformed monolithic codebase into a modular, maintainable architecture
7. Implemented HRTF-based binaural rendering with SOFA support

Recent developments include:
- Comprehensive HRTF implementation with SOFA support
- File conversion utilities for standard audio formats
- Enhanced room acoustics with early reflections and reverberation
- Performance optimizations for real-time processing

## Limitations and Known Issues

Current limitations and issues to be aware of:

1. **Performance Scaling**: Complex scenes with many sources can strain performance
2. **HRTF Personalization**: No support yet for personalized HRTFs
3. **Room Acoustics**: Current implementation is simplified compared to real acoustics
4. **Audio File Support**: Limited testing with various audio formats
5. **GUI**: No graphical user interface yet, only controller-based interaction
6. **Test Coverage**: Not all modules have comprehensive test coverage

## Design Philosophy

1. **Intuitive Exploration**: The system should allow users to naturally explore sound spaces without technical knowledge
2. **Expressive Control**: Provide meaningful ways to manipulate and interact with sound sources
3. **Perceptual Accuracy**: Focus on how sound is perceived rather than technical correctness alone
4. **Creative Freedom**: Enable artistic expression through spatial audio
5. **Accessibility**: Make spatial audio creation and exploration accessible to non-experts

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- Pygame (for demos)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install the package in development mode: `pip install -e .`

### Running Demos

Several demos showcase the capabilities of SHAC:

- `demo.py`: Runs all demos with command-line options
- `simple_demo.py`: Basic navigation through a 3D sound scene
- `interactive_demo.py`: Interactive controller-based interface
- `sofa_binaural_demo.py`: Demonstrates HRTF-based binaural rendering
- `format_conversion_demo.py`: Demonstrates audio file conversion

Run any demo with:
```
python demo.py [demo_name]
```

## For Future Claude Instances

When working on this project, keep the following in mind:

1. **Modular Architecture**: The code is organized into logical modules. Make sure to place new code in the appropriate module.

2. **Backward Compatibility**: Maintain compatibility with existing demos and applications when making changes.

3. **Performance Considerations**: Spatial audio processing can be computationally intensive. Use the optimized streaming components for real-time applications:
   - Buffer pooling (`streaming_optimized.py`) for memory efficiency
   - Vectorized operations (`vector_ops.py`) for processing speed
   - Adaptive buffer sizing (`adaptive_streaming.py`) for hardware adaptation
   - The `optimized_streaming_demo.py` script demonstrates these optimizations

4. **Documentation**: Keep this document and other documentation up-to-date as you make changes to the project.

5. **Test Coverage**: Add or update tests when implementing new features or fixing bugs.

6. **Optimization Opportunities**: Look for opportunities to optimize critical processing paths, especially in the spherical harmonic calculations and convolution operations.

7. **Roadmap Awareness**: Consult the PROJECT_PROGRESS.md file to understand the current roadmap and priorities.

## Technical Reference

### Processing Pipeline

The typical processing pipeline in SHAC follows these steps:

1. **Source Encoding**: Convert mono or stereo sources to the ambisonic domain
2. **Layer Processing**: Process each layer independently (e.g., gain, effects)
3. **Scene Mixing**: Mix all layers into a complete ambisonic scene
4. **Room Acoustics**: Add early reflections and reverberation if enabled
5. **Sound Field Rotation**: Apply listener orientation changes
6. **Output Rendering**: Convert to output format (binaural, speakers, etc.)

### Key Classes and Functions

Key components that are essential to understand:

- `SHACCodec`: Main class that orchestrates the entire processing pipeline
- `encode_mono_source()`: Converts a mono source to ambisonic signals
- `rotate_ambisonics()`: Rotates the ambisonic sound field
- `binauralize_ambisonics()`: Renders ambisonics to binaural stereo
- `SHACFileWriter/Reader`: Handles the custom file format

## Conclusion

SHAC is a powerful and flexible spatial audio system with applications in VR/AR, gaming, and creative audio. The modular architecture and focus on real-time interactivity make it well-suited for a wide range of applications. As development continues, the project aims to expand its capabilities while maintaining a focus on performance, quality, and usability.