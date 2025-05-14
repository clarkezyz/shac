#!/usr/bin/env python3
"""
SHAC Optimized Streaming Demo

This script demonstrates the optimized streaming capabilities of the SHAC spatial audio system.
It showcases the performance improvements from:
- Buffer pooling for efficient memory management
- Vectorized audio processing using NumPy
- Adaptive buffer size management

Author: Claude & Clarke Zyz
License: MIT License
"""

import numpy as np
import time
import argparse
import sys
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import sounddevice as sd
    from shac.codec.streaming_optimized import SHACStreamProcessor, BufferPool
    from shac.codec.adaptive_streaming import AdaptiveSHACStreamProcessor
    from shac.codec.vector_ops import precompute_sh_matrix, precompute_hrtf_matrix
except ImportError:
    print("Error importing required modules. Make sure you have installed all dependencies.")
    print("Try: pip install -r requirements.txt")
    sys.exit(1)


class PerformanceMetrics:
    """Track and display performance metrics for streaming audio."""
    
    def __init__(self, window_size=100):
        """
        Initialize performance tracking.
        
        Args:
            window_size: Number of samples to keep in history
        """
        self.cpu_usage = []
        self.buffer_usage = []
        self.process_times = []
        self.xruns = 0
        self.window_size = window_size
        self.start_time = time.time()
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
    
    def add_sample(self, cpu_usage, buffer_count, process_time):
        """
        Add a performance sample.
        
        Args:
            cpu_usage: CPU usage percentage (0-100)
            buffer_count: Number of buffers in use
            process_time: Processing time in seconds
        """
        self.cpu_usage.append(cpu_usage)
        self.buffer_usage.append(buffer_count)
        self.process_times.append(process_time)
        
        # Keep only the most recent samples
        if len(self.cpu_usage) > self.window_size:
            self.cpu_usage = self.cpu_usage[-self.window_size:]
            self.buffer_usage = self.buffer_usage[-self.window_size:]
            self.process_times = self.process_times[-self.window_size:]
        
        # Track peak values
        self.peak_cpu = max(self.peak_cpu, cpu_usage)
        self.peak_memory = max(self.peak_memory, buffer_count)
        
        # Check for xruns
        if process_time > 0.02:  # Assuming 20ms is the threshold for an xrun
            self.xruns += 1
    
    def get_summary(self):
        """
        Get a performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'avg_cpu': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'peak_cpu': self.peak_cpu,
            'avg_buffers': np.mean(self.buffer_usage) if self.buffer_usage else 0,
            'peak_memory': self.peak_memory,
            'avg_process_time': np.mean(self.process_times) if self.process_times else 0,
            'xruns': self.xruns,
            'uptime': time.time() - self.start_time
        }
    
    def reset(self):
        """Reset all metrics."""
        self.cpu_usage = []
        self.buffer_usage = []
        self.process_times = []
        self.xruns = 0
        self.start_time = time.time()
        self.peak_cpu = 0.0
        self.peak_memory = 0.0


class OptimizedStreamingDemo:
    """Demonstration for the optimized streaming features of SHAC."""
    
    def __init__(self, sample_rate=48000, buffer_size=1024, order=3, 
                 use_adaptive=True, visualization=True):
        """
        Initialize the demo.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Audio buffer size in samples
            order: Ambisonic order
            use_adaptive: Whether to use adaptive buffer size
            visualization: Whether to enable visualization
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.order = order
        self.use_adaptive = use_adaptive
        self.visualization = visualization
        
        # Create the appropriate stream processor
        if use_adaptive:
            self.processor = AdaptiveSHACStreamProcessor(
                order=order,
                sample_rate=sample_rate,
                initial_buffer_size=buffer_size,
                enable_adaptation=True
            )
        else:
            self.processor = SHACStreamProcessor(
                order=order,
                sample_rate=sample_rate,
                buffer_size=buffer_size
            )
        
        # Audio output stream
        self.audio_stream = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        
        # Control
        self.running = False
        self.threads = []
        
        # Test signal generation
        self.test_frequencies = [220, 440, 880, 1320, 1760]
        self.position_speeds = [0.2, 0.1, 0.3, 0.15, 0.25]  # Revolutions per second
        
        # For visualization
        if visualization:
            self.fig = None
            self.ax1 = None
            self.ax2 = None
            self.line_cpu = None
            self.line_buffers = None
            self.animation = None
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for sounddevice."""
        if status:
            print(f"Audio callback status: {status}")
        
        try:
            # Process audio through our processor
            output_block = self.processor.get_binaural_output()
            
            # Copy to output buffer
            if output_block.shape[1] >= frames:
                outdata[:] = output_block[:, :frames].T
            else:
                outdata[:] = np.zeros((frames, 2), dtype=np.float32)
                outdata[:output_block.shape[1], :] = output_block.T
            
            # Update performance metrics
            stats = self.processor.get_performance_stats()
            self.metrics.add_sample(
                stats['cpu_usage'] * 100,
                stats['buffer_pools']['ambi']['active_buffers'],
                stats['average_process_time']
            )
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
    
    def _source_update_thread(self):
        """Thread that updates test sound sources."""
        print("Source update thread started")
        
        try:
            # Create test sources once
            for i, freq in enumerate(self.test_frequencies):
                position = (0.0, 0.0, 2.0 + i * 0.5)
                self.processor.add_source(f"source_{i}", position)
            
            # Keep updating them while running
            sample_index = 0
            while self.running:
                # Update all sources
                for i, freq in enumerate(self.test_frequencies):
                    # Generate new audio chunk
                    t = np.linspace(
                        sample_index / self.sample_rate,
                        (sample_index + self.buffer_size) / self.sample_rate,
                        self.buffer_size
                    )
                    audio = 0.2 * np.sin(2 * np.pi * freq * t)
                    
                    # Update source position (circular motion)
                    speed = self.position_speeds[i]
                    angle = (sample_index / self.sample_rate) * speed * 2 * np.pi
                    position = (angle % (2 * np.pi), 0.0, 2.0 + i * 0.5)
                    
                    # Update the source
                    self.processor.update_source(f"source_{i}", audio)
                    self.processor.update_source_position(f"source_{i}", position)
                
                # Move to next chunk
                sample_index += self.buffer_size
                
                # Sleep to avoid hammering the CPU
                time.sleep(self.buffer_size / self.sample_rate / 2)
                
        except Exception as e:
            print(f"Error in source update thread: {e}")
    
    def _init_visualization(self):
        """Initialize the performance visualization."""
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # CPU usage plot
        self.ax1.set_ylim(0, 100)
        self.ax1.set_xlim(0, self.metrics.window_size)
        self.ax1.set_title('CPU Usage (%)')
        self.ax1.set_ylabel('CPU %')
        self.ax1.grid(True)
        self.line_cpu, = self.ax1.plot([], [], 'r-')
        
        # Buffer usage plot
        self.ax2.set_ylim(0, 100)
        self.ax2.set_xlim(0, self.metrics.window_size)
        self.ax2.set_title('Active Buffers')
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylabel('Buffer Count')
        self.ax2.grid(True)
        self.line_buffers, = self.ax2.plot([], [], 'b-')
        
        plt.tight_layout()
        
        # For FuncAnimation
        self.animation = FuncAnimation(
            self.fig, self._update_plot, interval=100, 
            blit=True, save_count=self.metrics.window_size
        )
    
    def _update_plot(self, frame):
        """Update the performance visualization."""
        # Update CPU usage plot
        x = range(len(self.metrics.cpu_usage))
        self.line_cpu.set_data(x, self.metrics.cpu_usage)
        
        # Update buffer usage plot
        x = range(len(self.metrics.buffer_usage))
        self.line_buffers.set_data(x, self.metrics.buffer_usage)
        
        # Adjust y limits if needed
        if self.metrics.peak_cpu > self.ax1.get_ylim()[1]:
            self.ax1.set_ylim(0, self.metrics.peak_cpu * 1.1)
        
        if self.metrics.peak_memory > self.ax2.get_ylim()[1]:
            self.ax2.set_ylim(0, self.metrics.peak_memory * 1.1)
        
        # Adjust x limits if needed
        if len(x) > self.ax1.get_xlim()[1]:
            self.ax1.set_xlim(0, len(x))
            self.ax2.set_xlim(0, len(x))
        
        return self.line_cpu, self.line_buffers
    
    def start(self):
        """Start the streaming demo."""
        if self.running:
            print("Demo is already running")
            return
        
        print("\n=== SHAC Optimized Streaming Demo ===")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Buffer size: {self.buffer_size} samples")
        print(f"Ambisonic order: {self.order}")
        print(f"Adaptive mode: {'Enabled' if self.use_adaptive else 'Disabled'}")
        print(f"Visualization: {'Enabled' if self.visualization else 'Disabled'}")
        print("\nStarting stream processor...")
        
        # Start processor
        self.processor.start()
        
        # Start source update thread
        self.running = True
        source_thread = threading.Thread(target=self._source_update_thread)
        source_thread.daemon = True
        source_thread.start()
        self.threads.append(source_thread)
        
        # Start audio stream
        print("Starting audio output...")
        self.audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=2,
            callback=self._audio_callback
        )
        self.audio_stream.start()
        
        # Start visualization if enabled
        if self.visualization:
            self._init_visualization()
        
        print("\nDemo is running! Press Ctrl+C to stop.")
        
        try:
            # Main loop - just keep running until interrupted
            while self.running:
                time.sleep(1.0)
                
                # Print stats periodically
                stats = self.metrics.get_summary()
                print(f"\rCPU: {stats['avg_cpu']:.1f}% | "
                      f"Buffers: {stats['avg_buffers']:.0f} | "
                      f"Xruns: {stats['xruns']} | "
                      f"Uptime: {stats['uptime']:.0f}s", end="")
                
                # If using adaptive mode, print current buffer size
                if self.use_adaptive:
                    adaptive_stats = self.processor.get_performance_stats()['adaptation']
                    print(f" | Buffer size: {adaptive_stats['current_buffer_size']}", end="")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.stop()
    
    def stop(self):
        """Stop the streaming demo."""
        print("\nStopping demo...")
        self.running = False
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Stop processor
        self.processor.stop()
        
        # Wait for threads to complete
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.threads.clear()
        
        # Print final stats
        self._print_final_stats()
        
        # Close visualization
        if self.visualization and self.fig:
            plt.close(self.fig)
        
        print("Demo stopped")
    
    def _print_final_stats(self):
        """Print final performance statistics."""
        stats = self.metrics.get_summary()
        
        print("\n=== Performance Summary ===")
        print(f"Average CPU usage: {stats['avg_cpu']:.1f}%")
        print(f"Peak CPU usage: {stats['peak_cpu']:.1f}%")
        print(f"Average buffer count: {stats['avg_buffers']:.1f}")
        print(f"Peak buffer count: {stats['peak_memory']:.0f}")
        print(f"Average processing time: {stats['avg_process_time'] * 1000:.2f} ms")
        print(f"Total xruns: {stats['xruns']}")
        print(f"Total runtime: {stats['uptime']:.1f} seconds")
        
        if self.use_adaptive:
            adaptive_stats = self.processor.get_performance_stats()['adaptation']
            print(f"Final buffer size: {adaptive_stats['current_buffer_size']} samples")
            print(f"Adaptation strategy: {adaptive_stats['adaptation_strategy']}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import numba
    except ImportError:
        missing.append("numba")
    
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Please install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SHAC Optimized Streaming Demo')
    parser.add_argument('--no-adapt', action='store_true', help='Disable adaptive buffer size')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--order', type=int, default=3, help='Ambisonic order (default: 3)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--buffer-size', type=int, default=1024, help='Buffer size in samples (default: 1024)')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create and start demo
    demo = OptimizedStreamingDemo(
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
        order=args.order,
        use_adaptive=not args.no_adapt,
        visualization=not args.no_vis
    )
    
    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        demo.stop()


if __name__ == "__main__":
    main()