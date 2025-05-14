"""
File Format and I/O Operations Module

This module contains functions for reading and writing SHAC files,
handling various audio file formats, and import/export utilities.
"""

import numpy as np
import struct
import os
import io
from typing import Dict, List, Tuple, Optional, Union

from .math_utils import AmbisonicNormalization


class SHACFileWriter:
    """Class for writing SHAC format files"""
    
    def __init__(self, order: int, sample_rate: int, 
                normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D):
        """
        Initialize a new SHAC file writer.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            normalization: Normalization convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.n_channels = (order + 1) ** 2
        self.normalization = normalization
        
        # Storage for layers
        self.layers = {}
        self.layer_metadata = {}
    
    def add_layer(self, layer_id: str, audio: np.ndarray, metadata: Dict) -> None:
        """
        Add a layer to the SHAC file.
        
        Args:
            layer_id: Unique identifier for the layer
            audio: Ambisonic audio data, shape (n_channels, n_samples)
            metadata: Dictionary of metadata for the layer
        """
        if audio.shape[0] != self.n_channels:
            raise ValueError(f"Audio has {audio.shape[0]} channels, expected {self.n_channels}")
        
        self.layers[layer_id] = audio
        self.layer_metadata[layer_id] = metadata
    
    def write_file(self, filename: str, bit_depth: int = 32) -> None:
        """
        Write SHAC data to a file.
        
        Args:
            filename: Output filename
            bit_depth: Bit depth (16 or 32)
        """
        if bit_depth not in [16, 32]:
            raise ValueError("Bit depth must be 16 or 32")
        
        # Find the maximum number of samples
        max_samples = 0
        for layer_id, audio in self.layers.items():
            max_samples = max(max_samples, audio.shape[1])
        
        # Prepare the file header
        header = {
            'magic': b'SHAC',
            'version': 1,
            'order': self.order,
            'n_channels': self.n_channels,
            'sample_rate': self.sample_rate,
            'bit_depth': bit_depth,
            'n_samples': max_samples,
            'n_layers': len(self.layers),
            'normalization': self.normalization.value
        }
        
        # Open the file for writing
        with open(filename, 'wb') as f:
            # Write the header
            f.write(header['magic'])
            f.write(struct.pack('<HHHIIIHH', 
                              header['version'],
                              header['order'],
                              header['n_channels'],
                              header['sample_rate'],
                              header['bit_depth'],
                              header['n_samples'],
                              header['n_layers'],
                              header['normalization']))
            
            # Write each layer
            for layer_id, audio in self.layers.items():
                # Pad audio to max_samples if needed
                if audio.shape[1] < max_samples:
                    padded_audio = np.zeros((self.n_channels, max_samples))
                    padded_audio[:, :audio.shape[1]] = audio
                    audio = padded_audio
                
                # Prepare layer header
                layer_header = {
                    'id_length': len(layer_id),
                    'metadata_length': len(str(self.layer_metadata[layer_id]))
                }
                
                # Write layer header
                f.write(struct.pack('<HI', 
                                  layer_header['id_length'],
                                  layer_header['metadata_length']))
                
                # Write layer ID and metadata
                f.write(layer_id.encode('utf-8'))
                f.write(str(self.layer_metadata[layer_id]).encode('utf-8'))
                
                # Write audio data
                if bit_depth == 16:
                    # Scale to 16-bit range and convert to int16
                    audio_int16 = (audio * 32767).astype(np.int16)
                    f.write(audio_int16.tobytes())
                else:  # 32-bit float
                    f.write(audio.astype(np.float32).tobytes())


class SHACFileReader:
    """Class for reading SHAC format files"""
    
    def __init__(self, filename: str):
        """
        Initialize a SHAC file reader.
        
        Args:
            filename: Path to SHAC file
        """
        self.filename = filename
        self.file_info = None
        self.layer_info = {}
        
        # Open and parse the file header
        with open(filename, 'rb') as f:
            # Read and check magic bytes
            magic = f.read(4)
            if magic != b'SHAC':
                raise ValueError(f"Not a valid SHAC file: {filename}")
            
            # Read header
            header_data = f.read(22)  # 2+2+2+4+4+4+2+2 = 22 bytes
            (version, order, n_channels, sample_rate, 
             bit_depth, n_samples, n_layers, normalization) = struct.unpack('<HHHIIIHH', header_data)
            
            # Store file info
            self.file_info = {
                'version': version,
                'order': order,
                'n_channels': n_channels,
                'sample_rate': sample_rate,
                'bit_depth': bit_depth,
                'n_samples': n_samples,
                'n_layers': n_layers,
                'normalization': AmbisonicNormalization(normalization)
            }
            
            # Read layer information
            for i in range(n_layers):
                # Read layer header
                layer_header_data = f.read(6)  # 2+4 = 6 bytes
                id_length, metadata_length = struct.unpack('<HI', layer_header_data)
                
                # Read layer ID and metadata
                layer_id = f.read(id_length).decode('utf-8')
                metadata_str = f.read(metadata_length).decode('utf-8')
                
                # Store layer offset and size for later reading
                data_size = n_channels * n_samples
                if bit_depth == 16:
                    data_size *= 2  # 2 bytes per sample
                else:
                    data_size *= 4  # 4 bytes per sample
                
                self.layer_info[layer_id] = {
                    'offset': f.tell(),
                    'data_size': data_size,
                    'metadata': eval(metadata_str)  # Convert string to dict (safe if from our writer)
                }
                
                # Skip the audio data for now
                f.seek(data_size, os.SEEK_CUR)
    
    def get_file_info(self) -> Dict:
        """
        Get information about the SHAC file.
        
        Returns:
            Dictionary with file information
        """
        return self.file_info
    
    def get_layer_names(self) -> List[str]:
        """
        Get a list of layer IDs in the file.
        
        Returns:
            List of layer IDs
        """
        return list(self.layer_info.keys())
    
    def get_layer_metadata(self, layer_id: str) -> Dict:
        """
        Get metadata for a specific layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Dictionary with layer metadata
        """
        if layer_id not in self.layer_info:
            raise ValueError(f"Layer not found: {layer_id}")
        
        return self.layer_info[layer_id]['metadata']
    
    def read_layer(self, layer_id: str) -> np.ndarray:
        """
        Read audio data for a specific layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Ambisonic audio data, shape (n_channels, n_samples)
        """
        if layer_id not in self.layer_info:
            raise ValueError(f"Layer not found: {layer_id}")
        
        # Get layer information
        layer = self.layer_info[layer_id]
        n_channels = self.file_info['n_channels']
        n_samples = self.file_info['n_samples']
        bit_depth = self.file_info['bit_depth']
        
        # Open file and seek to layer data
        with open(self.filename, 'rb') as f:
            f.seek(layer['offset'])
            
            # Read raw audio data
            if bit_depth == 16:
                # Read as int16 and convert to float
                raw_data = f.read(layer['data_size'])
                audio = np.frombuffer(raw_data, dtype=np.int16).reshape(n_channels, n_samples)
                audio = audio.astype(np.float32) / 32767.0
            else:  # 32-bit float
                raw_data = f.read(layer['data_size'])
                audio = np.frombuffer(raw_data, dtype=np.float32).reshape(n_channels, n_samples)
        
        return audio