# SHAC Codec API Reference

This is the API reference for the Spherical Harmonic Audio Codec (SHAC), 
a spatial audio codec optimized for real-time interactive navigation and
manipulation of 3D sound fields.

## Package Structure

The SHAC codec is organized into the following modules:

- `core`: The main `SHACCodec` class and core functionality
- `math_utils`: Mathematical functions for spherical harmonics processing
- `utils`: Utility classes, type definitions, and helper functions
- `encoders`: Functions for encoding mono and stereo sources
- `processors`: Functions for processing and transforming ambisonic signals
- `binauralizer`: Binaural rendering and HRTF processing
- `io`: File format handling and I/O operations
- `streaming`: Real-time streaming and processing
- `config`: Configuration management
- `exceptions`: Custom exception types

## Core Classes

- `SHACCodec`: The main codec class for encoding, processing, and decoding
- `SHACStreamProcessor`: Real-time streaming processor

## Key Concepts

### Ambisonic Order

The ambisonic order determines the spatial resolution of the sound field
representation. Higher orders provide more detailed spatial information
but require more channels:

- Order 0: 1 channel (omnidirectional)
- Order 1: 4 channels (basic directional information)
- Order 2: 9 channels (improved spatial resolution)
- Order 3: 16 channels (high spatial resolution)

### Coordinate Systems

SHAC uses two coordinate systems:

- **Spherical coordinates**: (azimuth, elevation, distance)
  - Azimuth: Horizontal angle (0 = front, π/2 = left, π = back, 3π/2 = right)
  - Elevation: Vertical angle (-π/2 = down, 0 = horizontal, π/2 = up)
  - Distance: Distance from origin in meters

- **Cartesian coordinates**: (x, y, z)
  - X-axis: Positive to the left
  - Y-axis: Positive upward
  - Z-axis: Positive to the front

### Normalization Conventions

Different normalization schemes affect how spherical harmonic components
are scaled:

- **SN3D**: Schmidt semi-normalized (most common)
- **N3D**: Fully normalized (orthonormal basis)
- **FuMa**: FuMa (legacy B-format) normalization

### Channel Ordering

Different ordering schemes determine how channels are arranged:

- **ACN**: Ambisonic Channel Number (most common standard)
- **FuMa**: FuMa (legacy B-format) ordering

## Getting Started

See the [Getting Started Guide](../getting_started.md) for usage instructions
and examples.