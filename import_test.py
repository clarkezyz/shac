"""
Test importing from the new modular structure

This simple script tests importing and using the refactored SHAC modules.
"""

import sys

def test_direct_imports():
    """Test direct imports from the modular structure."""
    try:
        # Import core components
        from shac.codec import SHACCodec, SHACStreamProcessor
        from shac.codec.math_utils import AmbisonicNormalization
        from shac.codec.utils import AmbisonicOrdering, SourceAttributes
        
        print("Successfully imported from shac.codec")
        
        # Create codec instance
        codec = SHACCodec(order=3, sample_rate=48000, 
                        normalization=AmbisonicNormalization.SN3D)
        print(f"Created SHACCodec with {codec.n_channels} channels")
        
        # Create streaming processor
        processor = SHACStreamProcessor(order=2, sample_rate=44100, buffer_size=1024)
        print(f"Created SHACStreamProcessor with {processor.n_channels} channels")
        
        return True
    except ImportError as e:
        print(f"Error importing from modular structure: {e}")
        return False

def test_wrapper_imports():
    """Test imports through the compatibility wrapper."""
    try:
        # Import through wrapper
        from shac_codec_wrapper import (
            SHACCodec, SHACStreamProcessor, AmbisonicNormalization,
            create_example_sound_scene, factorial
        )
        print("Successfully imported from shac_codec_wrapper")
        
        # Create codec instance
        codec = SHACCodec(order=3, sample_rate=48000, 
                        normalization=AmbisonicNormalization.SN3D)
        print(f"Created SHACCodec from wrapper with {codec.n_channels} channels")
        
        # Test a math utility function
        result = factorial(5)
        print(f"factorial(5) = {result}")
        
        return True
    except ImportError as e:
        print(f"Error importing from wrapper: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Testing Direct Imports ===")
    direct_success = test_direct_imports()
    
    print("\n=== Testing Wrapper Imports ===")
    wrapper_success = test_wrapper_imports()
    
    # Exit with status code
    if direct_success and wrapper_success:
        print("\n✅ All import tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some import tests failed.")
        sys.exit(1)