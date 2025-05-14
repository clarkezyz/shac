"""
Pytest configuration file for SHAC codec tests.
"""

import os
import pytest
import numpy as np
from shac.codec.config import SHACConfig


@pytest.fixture
def test_config():
    """Return a test configuration with predefined settings."""
    config = SHACConfig()
    config.processing.sample_rate = 44100
    config.processing.buffer_size = 512
    config.processing.order = 2
    return config


@pytest.fixture
def test_audio_mono():
    """Create a simple mono test signal."""
    # Create a 1-second sine wave at 440 Hz
    sr = 44100
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def test_audio_stereo():
    """Create a simple stereo test signal."""
    # Create a 1-second stereo signal with different frequencies in each channel
    sr = 44100
    t = np.linspace(0, 1, sr)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz in left channel
    right = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz in right channel
    return np.vstack([left, right])


@pytest.fixture
def fixtures_dir():
    """Return the path to the test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures')