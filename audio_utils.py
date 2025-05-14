"""
SHAC Audio Utilities

This module provides utility functions for audio processing, file handling,
and signal generation for the SHAC spatial audio system.

Author: Claude
License: MIT License
"""

import numpy as np
import os
import wave
import struct
import math

def generate_test_tone(frequency, duration, sample_rate=48000, amplitude=0.8):
    """
    Generate a simple sine wave test tone
    
    Parameters:
    - frequency: Tone frequency in Hz
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - amplitude: Peak amplitude (0.0 to 1.0)
    
    Returns:
    - Numpy array containing the audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_sweep(start_freq, end_freq, duration, sample_rate=48000, amplitude=0.8):
    """
    Generate a frequency sweep (chirp) signal
    
    Parameters:
    - start_freq: Start frequency in Hz
    - end_freq: End frequency in Hz
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - amplitude: Peak amplitude (0.0 to 1.0)
    
    Returns:
    - Numpy array containing the audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Exponential sweep
    if start_freq > 0 and end_freq > 0:
        k = (end_freq / start_freq) ** (1/duration)
        phase = 2 * np.pi * start_freq * ((k**t - 1) / np.log(k))
        return amplitude * np.sin(phase)
    else:
        return np.zeros(int(sample_rate * duration))

def generate_noise(noise_type, duration, sample_rate=48000, amplitude=0.8):
    """
    Generate different types of noise
    
    Parameters:
    - noise_type: Type of noise ('white', 'pink', 'brown')
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - amplitude: Peak amplitude (0.0 to 1.0)
    
    Returns:
    - Numpy array containing the audio samples
    """
    num_samples = int(sample_rate * duration)
    
    if noise_type == 'white':
        # White noise (equal energy per frequency)
        noise = np.random.randn(num_samples)
    elif noise_type == 'pink':
        # Pink noise (1/f spectrum, equal energy per octave)
        # Approximation using filtering
        noise = np.random.randn(num_samples)
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        try:
            from scipy import signal
            noise = signal.lfilter(b, a, noise)
        except ImportError:
            # Fallback without scipy: simple first-order filter
            filtered = np.zeros(num_samples)
            filtered[0] = noise[0]
            for i in range(1, num_samples):
                filtered[i] = 0.99 * filtered[i-1] + noise[i]
            noise = filtered
    elif noise_type == 'brown':
        # Brown noise (1/f^2 spectrum, -6dB/octave)
        noise = np.random.randn(num_samples)
        # Integrate white noise
        noise = np.cumsum(noise)
        # Remove DC offset
        noise = noise - np.mean(noise)
        # Normalize
        noise = noise / np.max(np.abs(noise))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return amplitude * noise

def generate_rhythm(pattern, tempo, duration, sample_rate=48000, amplitude=0.8):
    """
    Generate a rhythm pattern with exponential decay on each hit
    
    Parameters:
    - pattern: List of 1s and 0s defining the rhythm pattern
    - tempo: Tempo in BPM
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - amplitude: Peak amplitude (0.0 to 1.0)
    
    Returns:
    - Numpy array containing the audio samples
    """
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples)
    
    # Calculate beat duration in samples
    beat_duration = int(sample_rate * 60 / tempo)
    pattern_duration = beat_duration * len(pattern)
    
    # Generate the pattern
    num_repeats = int(np.ceil(num_samples / pattern_duration))
    
    for repeat in range(num_repeats):
        for i, hit in enumerate(pattern):
            if hit > 0:
                start = repeat * pattern_duration + i * beat_duration
                if start >= num_samples:
                    break
                    
                # Calculate decay length (shorter than full beat for separation)
                decay_length = int(beat_duration * 0.8)
                end = min(start + decay_length, num_samples)
                
                # Create exponential decay envelope
                t = np.linspace(0, 10, end - start)
                envelope = np.exp(-t)
                
                # Apply envelope
                audio[start:end] = hit * amplitude * envelope
    
    return audio

def generate_instrument_like(instrument, note, duration, sample_rate=48000, amplitude=0.8):
    """
    Generate a synthesized instrument-like sound
    
    Parameters:
    - instrument: Instrument type ('piano', 'violin', 'bass', etc.)
    - note: MIDI note number (60 = C4)
    - duration: Duration in seconds
    - sample_rate: Sample rate in Hz
    - amplitude: Peak amplitude (0.0 to 1.0)
    
    Returns:
    - Numpy array containing the audio samples
    """
    # Convert MIDI note to frequency
    frequency = 440.0 * (2.0 ** ((note - 69) / 12.0))
    
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio = np.zeros(num_samples)
    
    if instrument == 'piano':
        # Piano-like (decaying harmonics)
        harmonics = [1, 2, 3, 4, 5]
        strengths = [1.0, 0.5, 0.25, 0.125, 0.0625]
        decay_rates = [3.0, 4.0, 5.0, 6.0, 7.0]
        
        for h, s, d in zip(harmonics, strengths, decay_rates):
            audio += s * np.sin(2 * np.pi * frequency * h * t) * np.exp(-t * d)
            
    elif instrument == 'violin':
        # Violin-like (with vibrato)
        vibrato_rate = 5.0  # Hz
        vibrato_depth = 0.03  # Fraction of frequency
        
        # Generate vibrato modulation
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        
        # Generate harmonics with different attack/decay
        harmonics = [1, 2, 3, 4]
        strengths = [1.0, 0.6, 0.3, 0.15]
        
        # Slow attack
        attack = np.minimum(t / 0.1, 1.0)
        
        for h, s in zip(harmonics, strengths):
            # Apply vibrato to frequency
            phase = 2 * np.pi * frequency * h * t * (1.0 + vibrato)
            audio += s * np.sin(phase) * attack
            
    elif instrument == 'bass':
        # Bass-like (strong fundamental, less harmonics)
        harmonics = [1, 2, 3]
        strengths = [1.0, 0.3, 0.1]
        
        for h, s in zip(harmonics, strengths):
            audio += s * np.sin(2 * np.pi * frequency * h * t)
            
        # Add a subtle envelope
        envelope = 1.0 - 0.3 * t/duration
        audio *= envelope
        
    elif instrument == 'drum':
        # Simple drum-like sound (noise with exponential decay)
        noise = np.random.randn(num_samples)
        envelope = np.exp(-t * 10)
        audio = noise * envelope
        
    else:
        # Default to a simple sine wave
        audio = np.sin(2 * np.pi * frequency * t)
        envelope = np.exp(-t/2)
        audio *= envelope
    
    # Normalize and apply requested amplitude
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * amplitude
        
    return audio

def save_wav(filename, audio_data, sample_rate=48000, bit_depth=16):
    """
    Save audio data to a WAV file
    
    Parameters:
    - filename: Output filename
    - audio_data: Numpy array containing audio samples (can be mono or stereo)
    - sample_rate: Sample rate in Hz
    - bit_depth: Bit depth (16 or 24)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Convert to the right shape
    if audio_data.ndim == 1:
        # Mono audio
        channels = 1
        num_samples = len(audio_data)
    else:
        # Assume audio_data shape is (channels, samples)
        channels, num_samples = audio_data.shape
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        
        # Convert to integer format
        if bit_depth == 16:
            # Scale to 16-bit range and convert to integers
            audio_int = (audio_data * 32767).astype(np.int16)
            
            # Write to file
            if channels == 1:
                wav_file.writeframes(audio_int.tobytes())
            else:
                # Interleave channels
                wav_file.writeframes(audio_int.T.tobytes())
        elif bit_depth == 24:
            # 24-bit WAV requires manual packing
            if channels == 1:
                # Mono
                audio_int = (audio_data * 8388607).astype(np.int32)
                for sample in audio_int:
                    # Pack as little-endian 24-bit
                    wav_file.writeframes(struct.pack('<i', sample)[:3])
            else:
                # Stereo or multichannel
                audio_int = (audio_data * 8388607).astype(np.int32)
                for i in range(num_samples):
                    for ch in range(channels):
                        wav_file.writeframes(struct.pack('<i', audio_int[ch, i])[:3])
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

def create_sample_audio_files():
    """
    Create a set of sample audio files for testing the SHAC system
    """
    print("Creating sample audio files...")
    
    # Create output directory
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    
    sample_rate = 48000
    duration = 10  # seconds
    
    # Create a variety of test sounds
    
    # Piano-like sound
    piano = generate_instrument_like('piano', 60, duration, sample_rate)
    save_wav(os.path.join(audio_dir, 'piano.wav'), piano, sample_rate)
    
    # Violin-like sound
    violin = generate_instrument_like('violin', 69, duration, sample_rate)
    save_wav(os.path.join(audio_dir, 'violin.wav'), violin, sample_rate)
    
    # Bass-like sound
    bass = generate_instrument_like('bass', 48, duration, sample_rate)
    save_wav(os.path.join(audio_dir, 'bass.wav'), bass, sample_rate)
    
    # Drum pattern
    pattern = [1, 0, 0.7, 0, 1, 0, 0.7, 0.5]
    drums = generate_rhythm(pattern, 120, duration, sample_rate)
    save_wav(os.path.join(audio_dir, 'drums.wav'), drums, sample_rate)
    
    # Ambient noise
    ambient = generate_noise('pink', duration, sample_rate, 0.3)
    save_wav(os.path.join(audio_dir, 'ambient.wav'), ambient, sample_rate)
    
    # Frequency sweep
    sweep = generate_sweep(100, 10000, duration, sample_rate)
    save_wav(os.path.join(audio_dir, 'sweep.wav'), sweep, sample_rate)
    
    print(f"Created sample audio files in {audio_dir}")


if __name__ == "__main__":
    create_sample_audio_files()