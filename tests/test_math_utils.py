"""
Unit tests for the math_utils module.

These tests verify the mathematical functions used in the SHAC codec,
especially focusing on numerical accuracy and stability.
"""

import pytest
import numpy as np
import math
from shac.codec.math_utils import (
    factorial, double_factorial, associated_legendre, real_spherical_harmonic,
    spherical_harmonic_matrix, sh_rotation_matrix, convert_acn_to_fuma,
    convert_fuma_to_acn, AmbisonicNormalization
)
from shac.codec.exceptions import MathError


class TestFactorials:
    """Tests for factorial and double factorial functions."""
    
    def test_factorial_basic(self):
        """Test basic factorial calculations."""
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800
    
    def test_factorial_negative(self):
        """Test factorial with negative input raises error."""
        with pytest.raises(MathError.DomainError):
            factorial(-1)
    
    def test_double_factorial_basic(self):
        """Test basic double factorial calculations."""
        assert double_factorial(0) == 1
        assert double_factorial(1) == 1
        assert double_factorial(5) == 15  # 5 × 3 × 1
        assert double_factorial(6) == 48  # 6 × 4 × 2
    
    def test_double_factorial_negative(self):
        """Test double factorial with negative input raises error."""
        with pytest.raises(MathError.DomainError):
            double_factorial(-1)
    
    def test_factorial_caching(self):
        """Test that factorial caching works by ensuring we can compute large values."""
        # This would be slow without caching
        for i in range(20):
            factorial(i)
        # Call again to use cached values
        start_values = [factorial(i) for i in range(10)]
        assert start_values == [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]


class TestLegendrePolynomials:
    """Tests for associated Legendre polynomial functions."""
    
    def test_associated_legendre_scalar(self):
        """Test associated Legendre polynomials with scalar inputs."""
        # Test known values
        assert abs(associated_legendre(0, 0, 0.5) - 1.0) < 1e-10
        assert abs(associated_legendre(1, 0, 0.5) - 0.5) < 1e-10
        assert abs(associated_legendre(1, 1, 0.5) - (-0.8660254)) < 1e-6
        assert abs(associated_legendre(2, 0, 0.5) - (-0.125)) < 1e-10
    
    def test_associated_legendre_array(self):
        """Test associated Legendre polynomials with array inputs."""
        x = np.array([0.0, 0.5, -0.5])
        
        # Test for l=0, m=0
        result = associated_legendre(0, 0, x)
        expected = np.ones_like(x)
        np.testing.assert_allclose(result, expected)
        
        # Test for l=1, m=0
        result = associated_legendre(1, 0, x)
        expected = x
        np.testing.assert_allclose(result, expected)
        
        # Test for l=1, m=1
        result = associated_legendre(1, 1, x)
        expected = -np.sqrt(1 - x**2)
        np.testing.assert_allclose(result, expected)
    
    def test_associated_legendre_invalid(self):
        """Test associated Legendre with invalid inputs."""
        with pytest.raises(MathError.DomainError):
            associated_legendre(-1, 0, 0.5)  # l must be non-negative
        
        # m should be in range -l <= m <= l
        assert associated_legendre(2, 3, 0.5) == 0.0  # Should return 0 when |m| > l


class TestSphericalHarmonics:
    """Tests for spherical harmonic functions."""
    
    def test_real_spherical_harmonic_basic(self):
        """Test basic real spherical harmonic calculations."""
        # Test Y_0^0
        y00 = real_spherical_harmonic(0, 0, 0.0, math.pi/2, AmbisonicNormalization.N3D)
        assert abs(y00 - 1.0/math.sqrt(4*math.pi)) < 1e-10
        
        # Test consistency of normalization constants
        sn3d_value = real_spherical_harmonic(2, 1, math.pi/4, math.pi/3, AmbisonicNormalization.SN3D)
        n3d_value = real_spherical_harmonic(2, 1, math.pi/4, math.pi/3, AmbisonicNormalization.N3D)
        
        # SN3D = N3D / sqrt(2l+1)
        assert abs(sn3d_value - n3d_value/math.sqrt(5)) < 1e-10
    
    def test_spherical_harmonic_matrix(self):
        """Test spherical harmonic matrix construction."""
        # Create a matrix for 2 directions
        thetas = np.array([0.0, math.pi/2])
        phis = np.array([math.pi/2, math.pi/3])
        order = 1
        
        Y = spherical_harmonic_matrix(order, thetas, phis)
        
        # Check shape (2 directions, 4 channels for order 1)
        assert Y.shape == (2, 4)
        
        # Check values match individual calculations
        for i in range(2):
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    acn = l * l + l + m
                    expected = real_spherical_harmonic(l, m, thetas[i], phis[i])
                    assert abs(Y[i, acn] - expected) < 1e-10


class TestRotations:
    """Tests for rotation functions."""
    
    def test_sh_rotation_matrix_identity(self):
        """Test that zero rotation produces identity matrix."""
        order = 2
        R = sh_rotation_matrix(order, 0.0, 0.0, 0.0)
        
        n_channels = (order + 1) ** 2
        expected = np.eye(n_channels)
        
        np.testing.assert_allclose(R, expected, atol=1e-10)
    
    def test_sh_rotation_matrix_order1(self):
        """Test rotation matrix for first order."""
        # 90-degree rotation around Y axis
        beta = math.pi/2
        R = sh_rotation_matrix(1, 0.0, beta, 0.0)
        
        # In a 90° Y-rotation, X becomes Z and Z becomes -X
        # The 4 first-order components are: W (0), Y (1), Z (2), X (3) in ACN order
        expected = np.array([
            [1, 0, 0, 0],    # W stays the same
            [0, 1, 0, 0],    # Y stays the same (Y-axis rotation)
            [0, 0, 0, 1],    # Z becomes X
            [0, 0, -1, 0]     # X becomes -Z
        ])
        
        np.testing.assert_allclose(R, expected, atol=1e-10)


class TestFormatConversion:
    """Tests for ambisonic format conversion functions."""
    
    def test_acn_to_fuma_first_order(self):
        """Test conversion from ACN to FuMa ordering for first order."""
        # Create a simple first-order signal
        acn_signals = np.array([
            [1.0, 2.0],  # W
            [3.0, 4.0],  # Y
            [5.0, 6.0],  # Z
            [7.0, 8.0]   # X
        ])
        
        fuma_signals = convert_acn_to_fuma(acn_signals)
        
        # Check FuMa ordering: W, X, Y, Z with appropriate scaling
        np.testing.assert_allclose(fuma_signals[0], acn_signals[0] * math.sqrt(2))  # W * sqrt(2)
        np.testing.assert_allclose(fuma_signals[1], acn_signals[3])  # X (no scaling in first order)
        np.testing.assert_allclose(fuma_signals[2], acn_signals[1])  # Y (no scaling in first order)
        np.testing.assert_allclose(fuma_signals[3], acn_signals[2])  # Z (no scaling in first order)
    
    def test_fuma_to_acn_first_order(self):
        """Test conversion from FuMa to ACN ordering for first order."""
        # Create a simple first-order signal in FuMa ordering
        fuma_signals = np.array([
            [1.0, 2.0],  # W
            [3.0, 4.0],  # X
            [5.0, 6.0],  # Y
            [7.0, 8.0]   # Z
        ])
        
        acn_signals = convert_fuma_to_acn(fuma_signals)
        
        # Check ACN ordering: W, Y, Z, X with appropriate scaling
        np.testing.assert_allclose(acn_signals[0], fuma_signals[0] / math.sqrt(2))  # W / sqrt(2)
        np.testing.assert_allclose(acn_signals[1], fuma_signals[2])  # Y (no scaling in first order)
        np.testing.assert_allclose(acn_signals[2], fuma_signals[3])  # Z (no scaling in first order)
        np.testing.assert_allclose(acn_signals[3], fuma_signals[1])  # X (no scaling in first order)
    
    def test_reversibility(self):
        """Test that conversions are reversible."""
        # Start with ACN ordered signals
        original_acn = np.random.rand(4, 10)  # First order (4 channels)
        
        # Convert to FuMa and back to ACN
        fuma = convert_acn_to_fuma(original_acn)
        restored_acn = convert_fuma_to_acn(fuma)
        
        # Should get back the original signals
        np.testing.assert_allclose(original_acn, restored_acn, atol=1e-10)