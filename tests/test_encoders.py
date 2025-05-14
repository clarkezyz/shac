"""
Unit tests for the encoders module.

These tests verify the encoding of mono and stereo sources into ambisonics.
"""

import pytest
import numpy as np
import math
from shac.codec.encoders import (
    convert_to_spherical, convert_to_cartesian,
    encode_mono_source, encode_stereo_source
)
from shac.codec.math_utils import AmbisonicNormalization


class TestCoordinateConversion:
    """Tests for coordinate conversion functions."""
    
    def test_cartesian_to_spherical(self):
        """Test conversion from Cartesian to spherical coordinates."""
        # Test front (0, 0, 1)
        x, y, z = 0.0, 0.0, 1.0
        azimuth, elevation, distance = convert_to_spherical((x, y, z))
        assert abs(azimuth) < 1e-10
        assert abs(elevation) < 1e-10
        assert abs(distance - 1.0) < 1e-10
        
        # Test left (1, 0, 0)
        x, y, z = 1.0, 0.0, 0.0
        azimuth, elevation, distance = convert_to_spherical((x, y, z))
        assert abs(azimuth - math.pi/2) < 1e-10
        assert abs(elevation) < 1e-10
        assert abs(distance - 1.0) < 1e-10
        
        # Test up (0, 1, 0)
        x, y, z = 0.0, 1.0, 0.0
        azimuth, elevation, distance = convert_to_spherical((x, y, z))
        assert abs(azimuth) < 1e-10
        assert abs(elevation - math.pi/2) < 1e-10
        assert abs(distance - 1.0) < 1e-10
    
    def test_spherical_to_cartesian(self):
        """Test conversion from spherical to Cartesian coordinates."""
        # Test front (0, 0, 1)
        azimuth, elevation, distance = 0.0, 0.0, 1.0
        x, y, z = convert_to_cartesian((azimuth, elevation, distance))
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - 1.0) < 1e-10
        
        # Test left (π/2, 0, 1)
        azimuth, elevation, distance = math.pi/2, 0.0, 1.0
        x, y, z = convert_to_cartesian((azimuth, elevation, distance))
        assert abs(x - 1.0) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z) < 1e-10
        
        # Test up (0, π/2, 1)
        azimuth, elevation, distance = 0.0, math.pi/2, 1.0
        x, y, z = convert_to_cartesian((azimuth, elevation, distance))
        assert abs(x) < 1e-10
        assert abs(y - 1.0) < 1e-10
        assert abs(z) < 1e-10
    
    def test_roundtrip_conversion(self):
        """Test that converting back and forth gives original coordinates."""
        # Test several random positions
        for _ in range(10):
            original_cart = (np.random.uniform(-10, 10), 
                            np.random.uniform(-10, 10), 
                            np.random.uniform(-10, 10))
            # Skip points too close to origin
            if np.linalg.norm(original_cart) < 0.1:
                continue
                
            # Convert to spherical and back
            spherical = convert_to_spherical(original_cart)
            restored_cart = convert_to_cartesian(spherical)
            
            # Check that we got back the original
            for i in range(3):
                assert abs(original_cart[i] - restored_cart[i]) < 1e-10


class TestMonoEncoding:
    """Tests for mono source encoding."""
    
    def test_encode_mono_front(self, test_audio_mono):
        """Test encoding mono source in front position."""
        # Place source directly in front (0, 0, 1)
        position = (0.0, 0.0, 1.0)
        order = 1
        
        # Encode to first order ambisonics
        ambi = encode_mono_source(test_audio_mono, position, order)
        
        # Check shape (4 channels for first order)
        assert ambi.shape == (4, len(test_audio_mono))
        
        # For source in front, we expect:
        # W (omnidirectional) = audio * scaling
        # Y (left-right) = 0
        # Z (up-down) = 0
        # X (front-back) = audio * scaling
        # Check W and X are non-zero, Y and Z are zero
        assert np.all(np.abs(ambi[0]) > 0)  # W
        assert np.all(np.abs(ambi[1]) < 1e-10)  # Y
        assert np.all(np.abs(ambi[2]) < 1e-10)  # Z
        assert np.all(np.abs(ambi[3]) > 0)  # X
    
    def test_encode_mono_left(self, test_audio_mono):
        """Test encoding mono source in left position."""
        # Place source to the left (π/2, 0, 1)
        position = (math.pi/2, 0.0, 1.0)
        order = 1
        
        # Encode to first order ambisonics
        ambi = encode_mono_source(test_audio_mono, position, order)
        
        # For source on the left, we expect:
        # W (omnidirectional) = audio * scaling
        # Y (left-right) = audio * scaling
        # Z (up-down) = 0
        # X (front-back) = 0
        # Check W and Y are non-zero, Z and X are zero
        assert np.all(np.abs(ambi[0]) > 0)  # W
        assert np.all(np.abs(ambi[1]) > 0)  # Y
        assert np.all(np.abs(ambi[2]) < 1e-10)  # Z
        assert np.all(np.abs(ambi[3]) < 1e-10)  # X
    
    def test_distance_attenuation(self, test_audio_mono):
        """Test distance attenuation in mono encoding."""
        # Place source at different distances
        position_near = (0.0, 0.0, 1.0)
        position_far = (0.0, 0.0, 2.0)
        order = 1
        
        # Encode at both distances
        ambi_near = encode_mono_source(test_audio_mono, position_near, order)
        ambi_far = encode_mono_source(test_audio_mono, position_far, order)
        
        # Check that far source is attenuated compared to near source
        assert np.max(np.abs(ambi_near[0])) > np.max(np.abs(ambi_far[0]))
        # Should be roughly half the amplitude at double the distance
        ratio = np.max(np.abs(ambi_near[0])) / np.max(np.abs(ambi_far[0]))
        assert abs(ratio - 2.0) < 0.1


class TestStereoEncoding:
    """Tests for stereo source encoding."""
    
    def test_encode_stereo_front(self, test_audio_stereo):
        """Test encoding stereo source in front position."""
        # Place source in front with some width
        position = (0.0, 0.0, 1.0)
        width = math.pi/4  # 45 degrees width
        order = 1
        
        # Split stereo into left and right
        left = test_audio_stereo[0]
        right = test_audio_stereo[1]
        
        # Encode to first order ambisonics
        ambi = encode_stereo_source(left, right, position, width, order)
        
        # Check shape (4 channels for first order)
        assert ambi.shape == (4, len(left))
        
        # For stereo in front, we expect:
        # W (omnidirectional) = non-zero
        # Y (left-right) = non-zero (due to stereo width)
        # Z (up-down) = near zero
        # X (front-back) = non-zero
        assert np.all(np.abs(ambi[0]) > 0)  # W
        assert np.max(np.abs(ambi[1])) > 0  # Y should have some energy
        assert np.max(np.abs(ambi[2])) < 0.01  # Z should be very small
        assert np.all(np.abs(ambi[3]) > 0)  # X
    
    def test_width_affects_spatial_spread(self, test_audio_stereo):
        """Test that width parameter affects spatial spread."""
        position = (0.0, 0.0, 1.0)
        narrow_width = math.pi/16  # 11.25 degrees
        wide_width = math.pi/4  # 45 degrees
        order = 1
        
        left = test_audio_stereo[0]
        right = test_audio_stereo[1]
        
        # Encode with different widths
        ambi_narrow = encode_stereo_source(left, right, position, narrow_width, order)
        ambi_wide = encode_stereo_source(left, right, position, wide_width, order)
        
        # Wide source should have more energy in the Y (left-right) component
        assert np.max(np.abs(ambi_wide[1])) > np.max(np.abs(ambi_narrow[1]))
        
        # The ratio should roughly correspond to the width ratio
        ratio = np.max(np.abs(ambi_wide[1])) / np.max(np.abs(ambi_narrow[1]))
        width_ratio = wide_width / narrow_width
        assert abs(ratio - width_ratio) < 1.0  # Allow some tolerance