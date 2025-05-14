"""
Custom Exceptions Module

This module defines the exception hierarchy for the SHAC codec,
providing more specific error types for better error handling.
"""

class SHACError(Exception):
    """Base exception class for all SHAC codec errors."""
    pass


class ConfigurationError(SHACError):
    """Error in codec configuration."""
    pass


class InitializationError(SHACError):
    """Error during codec initialization."""
    pass


class EncodingError(SHACError):
    """Error during audio encoding."""
    pass


class ProcessingError(SHACError):
    """Error during audio processing."""
    pass


class DecodingError(SHACError):
    """Error during audio decoding."""
    pass


class BinauralRenderingError(SHACError):
    """Error during binaural rendering."""
    pass


class HRTFError(SHACError):
    """Error related to HRTF processing."""
    pass


class FileFormatError(SHACError):
    """Error in file format handling."""
    pass


class IOError(SHACError):
    """Error during file I/O operations."""
    pass


class MathError(SHACError):
    """Error in mathematical calculations."""
    
    class PrecisionError(SHACError):
        """Error due to numerical precision issues."""
        pass
    
    class DomainError(SHACError):
        """Error due to input values outside the valid domain."""
        pass


class ValidationError(SHACError):
    """Error during parameter validation."""
    pass


class StreamingError(SHACError):
    """Error during real-time streaming operations."""
    pass


class ResourceExhaustedError(SHACError):
    """Error when resources (memory, CPU) are exhausted."""
    pass