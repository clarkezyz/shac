"""
SHAC Audio File Loader

This module provides functions for loading audio files in various formats
for use in the SHAC spatial audio system.

Author: Claude
License: MIT License
"""

import numpy as np
import os
import wave
import struct
from typing import Dict, Tuple, Optional, Union, List

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import optional dependencies
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def load_wav(file_path: str) -> Dict:
    """
    Load audio data from a WAV file

    Parameters:
    - file_path: Path to the WAV file

    Returns:
    - Dictionary with keys:
        - audio_data: Numpy array of samples normalized to [-1.0, 1.0]
        - sample_rate: Sample rate in Hz
        - channels: Number of channels
        - duration: Duration in seconds
    """
    # First try to use scipy.io.wavfile if available (more robust)
    if SCIPY_AVAILABLE:
        try:
            sample_rate, data = wavfile.read(file_path)
            
            # Convert to float and normalize
            if data.dtype == np.int16:
                audio_data = data.astype(np.float32) / 32767.0
            elif data.dtype == np.int32:
                audio_data = data.astype(np.float32) / 2147483647.0
            elif data.dtype == np.uint8:
                audio_data = (data.astype(np.float32) - 128) / 128.0
            else:  # Assume it's already normalized float
                audio_data = data.astype(np.float32)
                
            # Get shape info
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
                # Convert to mono if stereo by averaging channels
                if channels == 2:
                    audio_data = np.mean(audio_data, axis=1)
            
            duration = len(audio_data) / sample_rate
            
            return {
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'channels': 1,  # We always return mono
                'duration': duration
            }
        except Exception as e:
            print(f"Warning: scipy.io.wavfile failed to load {file_path}, falling back to wave module. Error: {str(e)}")
            # Fall back to built-in wave module
            pass
    
    # Use built-in wave module
    try:
        with wave.open(file_path, 'r') as wav_file:
            # Get file parameters
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read all frames
            raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:  # 8-bit unsigned
                data = np.frombuffer(raw_data, dtype=np.uint8)
                audio_data = (data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:  # 16-bit
                data = np.frombuffer(raw_data, dtype=np.int16)
                audio_data = data.astype(np.float32) / 32767.0
            elif sample_width == 3:  # 24-bit
                # Manual unpacking required for 24-bit
                audio_data = np.zeros(n_frames * channels, dtype=np.float32)
                for i in range(n_frames * channels):
                    # Extract 3 bytes and interpret as signed 24-bit (big-endian)
                    start = i * 3
                    if start + 2 < len(raw_data):
                        value = raw_data[start] + (raw_data[start + 1] << 8) + (raw_data[start + 2] << 16)
                        # Convert to signed value
                        if value & 0x800000:
                            value = value - 0x1000000
                        audio_data[i] = value / 8388607.0  # Normalize to [-1, 1]
            elif sample_width == 4:  # 32-bit
                data = np.frombuffer(raw_data, dtype=np.int32)
                audio_data = data.astype(np.float32) / 2147483647.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Reshape if stereo
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                # Convert to mono by averaging channels
                audio_data = np.mean(audio_data, axis=1)
            
            duration = n_frames / sample_rate
            
            return {
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'channels': 1,  # We always return mono
                'duration': duration
            }
    except Exception as e:
        raise ValueError(f"Failed to load WAV file {file_path}: {str(e)}")


def load_audio_file(file_path: str, target_sample_rate: Optional[int] = None) -> Dict:
    """
    Load any supported audio file format
    
    Parameters:
    - file_path: Path to the audio file
    - target_sample_rate: If specified, resample to this rate
    
    Returns:
    - Dictionary with audio data and metadata
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # WAV files - use our built-in function
    if file_ext == '.wav':
        result = load_wav(file_path)
    
    # Other formats - use soundfile if available
    elif SOUNDFILE_AVAILABLE:
        try:
            data, sample_rate = sf.read(file_path, always_2d=False)
            
            # Normalize to [-1, 1] if needed
            if data.dtype != np.float32 and data.dtype != np.float64:
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32767.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483647.0
                else:
                    data = data.astype(np.float32)
            
            # Convert to mono if stereo
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            
            duration = len(data) / sample_rate
            
            result = {
                'audio_data': data.astype(np.float32),
                'sample_rate': sample_rate,
                'channels': 1,  # Always converted to mono
                'duration': duration
            }
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    else:
        raise ValueError(f"Cannot load {file_ext} files without soundfile module. Please install it with 'pip install soundfile'")
    
    # Perform resampling if needed
    if target_sample_rate is not None and result['sample_rate'] != target_sample_rate:
        if SCIPY_AVAILABLE:
            from scipy import signal
            # Calculate resampling ratio
            ratio = target_sample_rate / result['sample_rate']
            # Calculate new length
            new_length = int(len(result['audio_data']) * ratio)
            # Resample
            result['audio_data'] = signal.resample(result['audio_data'], new_length)
            result['sample_rate'] = target_sample_rate
            result['duration'] = new_length / target_sample_rate
        else:
            print(f"Warning: Resampling requested but scipy not available. Using original sample rate {result['sample_rate']}.")
    
    return result


def create_audio_directory():
    """
    Create a directory for audio samples if it doesn't exist
    
    Returns:
    - Path to the audio directory
    """
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir


def list_audio_files(directory: Optional[str] = None) -> List[str]:
    """
    List all audio files in the specified directory
    
    Parameters:
    - directory: Directory to scan (defaults to the 'audio' folder)
    
    Returns:
    - List of audio file paths
    """
    if directory is None:
        directory = create_audio_directory()
    
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aiff']:
                audio_files.append(os.path.join(root, file))
    
    return audio_files


if __name__ == "__main__":
    # Test loading audio files
    audio_dir = create_audio_directory()
    audio_files = list_audio_files(audio_dir)
    
    if not audio_files:
        print("No audio files found. Creating sample audio files...")
        from audio_utils import create_sample_audio_files
        create_sample_audio_files()
        audio_files = list_audio_files(audio_dir)
    
    print(f"Found {len(audio_files)} audio files:")
    for file in audio_files:
        try:
            info = load_audio_file(file)
            print(f"  {os.path.basename(file)}: {info['duration']:.2f}s, {info['sample_rate']}Hz")
        except Exception as e:
            print(f"  Error loading {file}: {str(e)}")