"""
Adaptive Buffer Size Management for Streaming Audio

This module provides adaptive buffer size management for real-time audio processing,
dynamically adjusting buffer sizes based on system capabilities and load.
"""

import numpy as np
import time
import threading
import collections
from typing import Dict, List, Tuple, Optional, Union, Callable, Deque

from .streaming_optimized import SHACStreamProcessor, BufferPool


class AdaptiveSHACStreamProcessor(SHACStreamProcessor):
    """
    SHACStreamProcessor with adaptive buffer size management.
    
    This extension of SHACStreamProcessor can dynamically adjust its buffer size
    based on system capabilities and current load to maintain optimal performance.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, 
                 initial_buffer_size: int = 1024, max_sources: int = 32,
                 pool_size: int = 64, enable_adaptation: bool = True):
        """
        Initialize the adaptive SHAC stream processor.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            initial_buffer_size: Initial processing buffer size in samples
            max_sources: Maximum number of simultaneous sources
            pool_size: Size of the buffer memory pool
            enable_adaptation: Whether to enable adaptive buffer size management
        """
        super().__init__(order, sample_rate, initial_buffer_size, max_sources, pool_size)
        
        # Adaptive buffer size configuration
        self.enable_adaptation = enable_adaptation
        self.min_buffer_size = 128
        self.max_buffer_size = 4096
        self.current_buffer_size = initial_buffer_size
        self.target_cpu_usage = 0.7  # Target CPU usage (70%)
        self.adaptation_interval = 5.0  # Seconds between adaptation attempts
        self.last_adaptation_time = time.time()
        
        # Performance monitoring with more history for adaptation
        self.load_history = collections.deque(maxlen=100)  # CPU load history
        self.xrun_count = 0  # Count of buffer underruns/overruns
        
        # Resampling buffers for buffer size changes
        self.resampling_enabled = True
        self.resampling_quality = 'medium'  # 'low', 'medium', 'high'
        
        # Adaptation strategy
        self.adaptation_strategy = 'aggressive'  # 'conservative', 'moderate', 'aggressive'
        self.adaptation_thresholds = {
            'conservative': {'increase': 0.9, 'decrease': 0.5},
            'moderate': {'increase': 0.85, 'decrease': 0.6},
            'aggressive': {'increase': 0.8, 'decrease': 0.65}
        }
        
        # Create separate thread for monitoring and adaptation
        if self.enable_adaptation:
            self.adaptation_thread = threading.Thread(target=self._adaptation_loop)
            self.adaptation_thread.daemon = True
        else:
            self.adaptation_thread = None
    
    def start(self) -> None:
        """Start the real-time processing and adaptation threads."""
        super().start()
        
        # Start adaptation thread if enabled
        if self.enable_adaptation and self.adaptation_thread:
            self.adaptation_thread.start()
    
    def stop(self) -> None:
        """Stop the real-time processing and adaptation threads."""
        super().stop()
        
        # Adaptation thread will end since self.running will be False
    
    def _adaptation_loop(self) -> None:
        """Background thread for monitoring performance and adapting buffer size."""
        while self.running:
            try:
                # Wait for enough data to make decisions
                if len(self.process_times) < 30:
                    time.sleep(0.5)
                    continue
                
                # Only adapt every adaptation_interval seconds
                now = time.time()
                if now - self.last_adaptation_time < self.adaptation_interval:
                    time.sleep(0.1)
                    continue
                
                self.last_adaptation_time = now
                
                # Calculate current CPU load
                target_frame_time = self.current_buffer_size / self.sample_rate
                avg_process_time = sum(self.process_times) / len(self.process_times)
                current_load = avg_process_time / target_frame_time
                
                # Store in history
                self.load_history.append(current_load)
                
                # Calculate average load over history
                avg_load = sum(self.load_history) / len(self.load_history)
                
                # Get thresholds based on strategy
                thresholds = self.adaptation_thresholds[self.adaptation_strategy]
                
                # Decide if we need to adapt
                new_buffer_size = self.current_buffer_size
                
                if avg_load > thresholds['increase'] and self.xrun_count > 0:
                    # System is overloaded, increase buffer size
                    new_buffer_size = min(int(self.current_buffer_size * 1.5), self.max_buffer_size)
                    if new_buffer_size != self.current_buffer_size:
                        self._change_buffer_size(new_buffer_size)
                        print(f"Increased buffer size to {new_buffer_size} (load: {avg_load:.2f})")
                
                elif avg_load < thresholds['decrease'] and self.xrun_count == 0:
                    # System has excess capacity, decrease buffer size
                    new_buffer_size = max(int(self.current_buffer_size * 0.8), self.min_buffer_size)
                    if new_buffer_size != self.current_buffer_size:
                        self._change_buffer_size(new_buffer_size)
                        print(f"Decreased buffer size to {new_buffer_size} (load: {avg_load:.2f})")
                
                # Reset xrun counter after adaptation
                self.xrun_count = 0
                
                # Sleep for a bit before next check
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error in adaptation loop: {e}")
                time.sleep(1.0)
    
    def _change_buffer_size(self, new_size: int) -> None:
        """
        Change the buffer size, potentially resampling existing buffers.
        
        Args:
            new_size: New buffer size in samples
        """
        with self.lock:
            old_size = self.current_buffer_size
            
            # Store all source audio for resampling
            source_audio = {}
            for source_id, buffer in self.source_buffers.items():
                source_audio[source_id] = buffer.copy()
            
            # Update buffer size
            self.current_buffer_size = new_size
            
            # Create new buffer pools
            self.mono_buffer_pool = BufferPool((new_size,), self.mono_buffer_pool.max_buffers)
            self.ambi_buffer_pool = BufferPool((self.n_channels, new_size), self.ambi_buffer_pool.max_buffers)
            self.output_buffer_pool = BufferPool((2, new_size), self.output_buffer_pool.max_buffers)
            
            # Reset output buffer
            self.output_buffer = self.ambi_buffer_pool.get()
            
            # Recreate source buffers with new size
            new_source_buffers = {}
            for source_id, audio in source_audio.items():
                new_buffer = self.mono_buffer_pool.get()
                
                if self.resampling_enabled:
                    # Resample audio to new buffer size
                    # Simple approach: truncate or zero-pad
                    if new_size <= old_size:
                        new_buffer[:] = audio[:new_size]
                    else:
                        new_buffer[:old_size] = audio
                        new_buffer[old_size:] = 0.0
                    
                    # Advanced approach would use proper sample rate conversion
                    # if a resampling library like librosa is available
                
                new_source_buffers[source_id] = new_buffer
            
            # Replace the source buffers dictionary
            self.source_buffers = new_source_buffers
            
            # Reset performance monitoring
            self.process_times.clear()
            self.average_process_time = self.current_buffer_size / self.sample_rate * 0.5  # Estimate
    
    def process_block(self) -> np.ndarray:
        """
        Process a block of audio with xrun detection.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, buffer_size)
        """
        start_time = time.time()
        
        # Process audio using parent method
        output_buffer = super().process_block()
        
        # Track processing time
        process_time = time.time() - start_time
        
        # Check for xruns (buffer under/overruns)
        target_frame_time = self.current_buffer_size / self.sample_rate
        if process_time > target_frame_time:
            self.xrun_count += 1
        
        return output_buffer
    
    def get_performance_stats(self) -> Dict[str, Union[float, int, Dict]]:
        """
        Get performance statistics with adaptation information.
        
        Returns:
            Dictionary with processing and adaptation statistics
        """
        stats = super().get_performance_stats()
        
        # Add adaptation-specific stats
        adaptation_stats = {
            'adaptation_enabled': self.enable_adaptation,
            'current_buffer_size': self.current_buffer_size,
            'min_buffer_size': self.min_buffer_size,
            'max_buffer_size': self.max_buffer_size,
            'adaptation_strategy': self.adaptation_strategy,
            'xrun_count': self.xrun_count,
            'average_load': sum(self.load_history) / max(1, len(self.load_history)) if self.load_history else 0,
        }
        
        stats['adaptation'] = adaptation_stats
        return stats


class AdaptiveRealtimeDemo:
    """
    Demo for adaptive real-time spatial audio processing.
    
    This class extends the regular RealtimeDemo with adaptive buffer size management.
    """
    
    def __init__(self, sample_rate=48000, initial_buffer_size=1024, order=3,
                 adaptation_strategy='moderate', enable_adaptation=True):
        """
        Initialize the adaptive real-time demo.
        
        Args:
            sample_rate: Audio sample rate in Hz
            initial_buffer_size: Initial audio buffer size in samples
            order: Ambisonic order
            adaptation_strategy: 'conservative', 'moderate', or 'aggressive'
            enable_adaptation: Whether to enable adaptive buffer size management
        """
        self.sample_rate = sample_rate
        self.buffer_size = initial_buffer_size
        self.order = order
        
        # Create adaptive stream processor
        self.stream_processor = AdaptiveSHACStreamProcessor(
            order=order,
            sample_rate=sample_rate,
            initial_buffer_size=initial_buffer_size,
            enable_adaptation=enable_adaptation
        )
        
        # Set adaptation strategy
        self.stream_processor.adaptation_strategy = adaptation_strategy
        
        # Initialize other components (would connect to actual UI, controller, etc.)
        # This is a simplified demonstration
        self.running = False
    
    def start(self):
        """Start the adaptive demo."""
        if self.running:
            return
        
        self.running = True
        self.stream_processor.start()
        
        print(f"Started adaptive real-time demo with:")
        print(f"- Initial buffer size: {self.buffer_size} samples")
        print(f"- Adaptation strategy: {self.stream_processor.adaptation_strategy}")
        print(f"- Ambisonic order: {self.order}")
        print(f"- Sample rate: {self.sample_rate} Hz")
    
    def stop(self):
        """Stop the adaptive demo."""
        if not self.running:
            return
        
        self.running = False
        self.stream_processor.stop()
        
        print("Stopped adaptive real-time demo")
    
    def add_test_sources(self, num_sources=5):
        """
        Add test sources for benchmarking.
        
        Args:
            num_sources: Number of test sources to add
        """
        for i in range(num_sources):
            # Create a simple sine wave
            t = np.linspace(0, 2, self.buffer_size)
            freq = 220 * (i + 1)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Create a position
            position = (i / num_sources * 2 * np.pi, 0.0, 2.0 + i)
            
            # Add to stream processor
            self.stream_processor.add_source(f"test_source_{i}", position)
            self.stream_processor.update_source(f"test_source_{i}", audio)
    
    def get_performance_report(self) -> str:
        """
        Get a formatted performance report.
        
        Returns:
            Human-readable performance report
        """
        stats = self.stream_processor.get_performance_stats()
        
        report = [
            "=== Adaptive Streaming Performance Report ===",
            f"Current buffer size: {stats['adaptation']['current_buffer_size']} samples",
            f"Average CPU load: {stats['adaptation']['average_load']:.2f}",
            f"Adaptation strategy: {stats['adaptation']['adaptation_strategy']}",
            f"Buffer xruns: {stats['adaptation']['xrun_count']}",
            f"Active sources: {stats['active_sources']}",
            f"Average process time: {stats['average_process_time'] * 1000:.2f} ms",
            f"Target frame time: {stats['target_frame_time'] * 1000:.2f} ms",
            f"CPU usage: {stats['cpu_usage'] * 100:.1f}%",
        ]
        
        return "\n".join(report)
"""