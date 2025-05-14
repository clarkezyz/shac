"""
SHAC Sound Field Visualizer

This module provides real-time 3D visualization of spatial audio scenes,
showing sound sources, listener position, and interaction with the sound field.

Author: Claude
License: MIT License
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import threading

class SoundFieldVisualizer:
    """
    Provides real-time visualization of the sound field and layers
    """
    
    def __init__(self, layer_manager, controller_interface=None):
        """
        Initialize the visualizer
        
        Parameters:
        - layer_manager: SoundLayerManager instance
        - controller_interface: Optional ControllerInterface instance
        """
        self.layer_manager = layer_manager
        self.controller = controller_interface
        
        # Set up the plot
        self.fig = plt.figure(figsize=(12, 10))
        
        # Create 3D sound field visualization
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Store plot elements
        self.layer_points = {}
        self.labels = {}
        
        # Camera position marker
        self.camera_point = None
        self.camera_direction = None
        
        # Selected layer highlight
        self.selected_highlight = None
        
        # Configure the plot
        self.ax.set_xlabel('X (Right/Left)')
        self.ax.set_ylabel('Z (Front/Back)')
        self.ax.set_zlabel('Y (Up/Down)')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        self.ax.set_title('SHAC 3D Sound Field Visualization')
        
        # Color map for different layer types
        self.color_map = {
            'source': 'red',
            'extracted': 'blue',
            'frequency': 'green',
            'spatial': 'purple',
            'default': 'orange'
        }
        
        # Size map for layer gain
        self.min_size = 50
        self.max_size = 300
        
        # Animation properties
        self.animation = None
        self.is_running = False
        
        # For FPS calculation
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.fps_text = None
        
    def convert_to_cartesian(self, spherical_pos):
        """
        Convert spherical coordinates to Cartesian
        
        Parameters:
        - spherical_pos: (azimuth, elevation, distance) in radians and meters
        
        Returns:
        - Cartesian coordinates (x, y, z)
        """
        azimuth, elevation, distance = spherical_pos
        
        # Convert spherical to Cartesian
        x = distance * np.sin(azimuth) * np.cos(elevation)
        y = distance * np.sin(elevation)
        z = distance * np.cos(azimuth) * np.cos(elevation)
        
        return x, y, z
    
    def update(self, frame):
        """
        Update the visualization for animation
        
        Parameters:
        - frame: animation frame number
        
        Returns:
        - List of artists that were updated
        """
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        self.frame_count += 1
        
        if elapsed > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_frame_time = current_time
        
        updated_artists = []
        
        # Clear existing plot elements
        self.ax.cla()
        
        # Reset the plot
        self.ax.set_xlabel('X (Right/Left)')
        self.ax.set_ylabel('Z (Front/Back)')
        self.ax.set_zlabel('Y (Up/Down)')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        self.ax.set_title('SHAC 3D Sound Field Visualization')
        
        # Draw reference grid
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Draw a reference horizontal plane at y=0
        x_grid = np.linspace(-10, 10, 5)
        z_grid = np.linspace(-10, 10, 5)
        X, Z = np.meshgrid(x_grid, z_grid)
        Y = np.zeros_like(X)
        self.ax.plot_surface(X, Z, Y, alpha=0.1, color='gray')
        
        # Plot the listener position
        if self.controller:
            pos = self.controller.position
            # Plot the camera (user) position
            self.camera_point = self.ax.scatter(
                [pos[0]], [pos[2]], [pos[1]],  # Note the reordering to match our coordinate system
                color='blue', s=100, marker='^', label='Listener'
            )
            
            # Plot the listener direction
            yaw, pitch, roll = self.controller.orientation
            
            # Calculate direction vector
            dx = np.sin(yaw) * np.cos(pitch)  # X component (right/left)
            dy = np.sin(pitch)                # Y component (up/down)
            dz = np.cos(yaw) * np.cos(pitch)  # Z component (front/back)
            
            # Scale the direction vector
            scale = 2.0
            dx *= scale
            dy *= scale
            dz *= scale
            
            # Plot the direction arrow
            self.camera_direction = self.ax.quiver(
                pos[0], pos[2], pos[1],  # Start position
                dx, dz, dy,               # Direction vector
                color='blue', label='View Direction'
            )
        
        # Plot each layer
        selected_id = self.controller.selected_layer_id if self.controller else None
        
        for layer_id, layer_info in self.layer_manager.layers.items():
            # Get layer properties
            position = layer_info['position']
            name = layer_info['name']
            gain = layer_info['current_gain']
            muted = layer_info['muted']
            
            # Determine layer type based on properties
            layer_type = layer_info.get('properties', {}).get('type', 'default')
            
            # Convert spherical coordinates to Cartesian
            x, y, z = self.convert_to_cartesian(position)
            
            # Swap y and z for plotting consistency
            x, y, z = x, z, y
            
            # Determine color based on layer type
            color = self.color_map.get(layer_type, self.color_map['default'])
            
            # Determine size based on gain
            size = self.min_size + (self.max_size - self.min_size) * min(gain, 2.0) / 2.0
            
            # Apply muting
            alpha = 0.3 if muted else 0.8
            
            # Check if this is the selected layer
            is_selected = (layer_id == selected_id)
            
            # Plot the layer point with a border if selected
            if is_selected:
                # Add highlight ring around the selected layer
                self.ax.scatter(
                    [x], [y], [z],
                    color='yellow', s=size * 1.5, alpha=0.4,
                    marker='o', edgecolors='yellow', linewidth=2
                )
            
            # Plot the main point for this layer
            self.ax.scatter(
                [x], [y], [z],
                color=color, s=size, alpha=alpha,
                marker='o', edgecolors='black' if is_selected else None
            )
            
            # Add the label with a line connecting to the point
            self.ax.text(
                x, y, z + 0.5,  # Position slightly above the point
                name,
                color='white' if is_selected else 'black',
                fontsize=10,
                backgroundcolor='black' if is_selected else None,
                ha='center', va='bottom'
            )
        
        # Add mode display
        if self.controller:
            mode_text = f"Mode: {self.controller.mode.capitalize()}"
            self.ax.text2D(0.02, 0.98, mode_text, transform=self.ax.transAxes, 
                          fontsize=12, color='black', verticalalignment='top')
        
        # Add FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        self.ax.text2D(0.85, 0.98, fps_text, transform=self.ax.transAxes,
                      fontsize=10, color='gray', verticalalignment='top')
        
        # Add a legend
        self.ax.legend()
        
        # Position the camera for optimal viewing
        if self.controller and self.controller.mode == 'navigation':
            # Set the view to follow the listener orientation
            pos = self.controller.position
            yaw = self.controller.orientation[0]
            
            # Calculate viewing angle based on listener orientation
            azim = np.degrees(yaw) + 180  # Opposite direction to face listener
            elev = 30  # Fixed elevation angle
            
            self.ax.view_init(elev=elev, azim=azim)
            
            # Set the camera distance
            self.ax.dist = 12
        
        return self.ax.get_children()
    
    def start_animation(self, interval=100):
        """
        Start the real-time visualization
        
        Parameters:
        - interval: Update interval in milliseconds
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Create the animation
        self.animation = FuncAnimation(
            self.fig, self.update, interval=interval, 
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # After the window is closed
        self.is_running = False
    
    def run_in_thread(self, interval=100):
        """
        Run the visualization in a separate thread
        
        Parameters:
        - interval: Update interval in milliseconds
        
        Returns:
        - Thread object
        """
        thread = threading.Thread(target=self.start_animation, args=(interval,))
        thread.daemon = True  # Thread will close when main program exits
        thread.start()
        return thread
    
    def create_waveform_view(self, ax, layer_id=None):
        """
        Create a waveform visualization for a specific layer
        
        Parameters:
        - ax: matplotlib axes to plot on
        - layer_id: ID of the layer to visualize (None for mixed output)
        """
        # This would be implemented to show the waveform of a specific layer
        # or the final mixed output
        pass
    
    def create_spectrum_view(self, ax, layer_id=None):
        """
        Create a spectrum visualization for a specific layer
        
        Parameters:
        - ax: matplotlib axes to plot on
        - layer_id: ID of the layer to visualize (None for mixed output)
        """
        # This would be implemented to show the frequency spectrum of a 
        # specific layer or the final mixed output
        pass