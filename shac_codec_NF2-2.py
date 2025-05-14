

# =====================================================================================
# File Format Implementation
# =====================================================================================

class SHACFileWriter:
    """
    Writer for the Spherical Harmonic Audio Codec (SHAC) file format.
    """
    
    def __init__(self, order: int, sample_rate: int, normalize: AmbisonicNormalization = AmbisonicNormalization.SN3D):
        """
        Initialize the SHAC file writer.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            normalize: Normalization convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.n_channels = (order + 1) ** 2
        self.layers = {}
        self.layer_metadata = {}
    
    def add_layer(self, layer_id: str, ambi_signals: np.ndarray, metadata: Dict = None):
        """
        Add a layer to the SHAC file.
        
        Args:
            layer_id: Unique identifier for the layer
            ambi_signals: Ambisonic signals for this layer
            metadata: Optional metadata for the layer
        """
        if ambi_signals.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {ambi_signals.shape[0]}")
        
        self.layers[layer_id] = ambi_signals
        
        if metadata is None:
            metadata = {}
        
        self.layer_metadata[layer_id] = metadata
    
    def write_file(self, filename: str, bit_depth: int = 32) -> None:
        """
        Write the SHAC file to disk.
        
        Args:
            filename: Output filename
            bit_depth: Bit depth (16 or 32)
        """
        # Validate bit depth
        if bit_depth not in [16, 32]:
            raise ValueError("Bit depth must be 16 or 32")
        
        # Determine the sample format code
        sample_format = 1 if bit_depth == 16 else 2
        
        # Find the maximum number of frames across all layers
        max_frames = max(layer.shape[1] for layer in self.layers.values()) if self.layers else 0
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(b'SHAC')  # Magic number
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<I', self.order))  # Ambisonics order
            f.write(struct.pack('<I', self.sample_rate))  # Sample rate
            f.write(struct.pack('<I', self.n_channels))  # Number of channels
            f.write(struct.pack('<I', sample_format))  # Sample format
            f.write(struct.pack('<I', max_frames))  # Number of frames
            f.write(struct.pack('<I', 0))  # Reserved
            
            # Write channel metadata section
            channel_metadata_size = 16 * self.n_channels + 4
            f.write(struct.pack('<I', channel_metadata_size))  # Section size
            
            for ch in range(self.n_channels):
                l = math.floor(math.sqrt(ch))
                m = ch - l*l - l
                
                f.write(struct.pack('<I', ch))  # Channel index
                f.write(struct.pack('<I', l))  # Spherical harmonic degree
                f.write(struct.pack('<I', m))  # Spherical harmonic order
                f.write(struct.pack('<I', self.normalize.value))  # Normalization type
            
            # Write layer information section
            layer_section_size = 8  # Base size (section size + number of layers)
            for layer_id, metadata in self.layer_metadata.items():
                layer_section_size += 8  # Layer ID + name length
                layer_section_size += len(layer_id.encode('utf-8'))  # Layer name
                layer_section_size += 4 + (self.n_channels + 7) // 8  # Channel mask size + mask
            
            f.write(struct.pack('<I', layer_section_size))  # Section size
            f.write(struct.pack('<I', len(self.layers)))  # Number of layers
            
            # Write each layer's metadata
            for layer_id, metadata in self.layer_metadata.items():
                layer_id_bytes = layer_id.encode('utf-8')
                f.write(struct.pack('<I', hash(layer_id) & 0xFFFFFFFF))  # Layer ID (hashed)
                f.write(struct.pack('<I', len(layer_id_bytes)))  # Layer name length
                f.write(layer_id_bytes)  # Layer name
                
                # Write channel mask (which channels belong to this layer)
                mask_bytes = (self.n_channels + 7) // 8
                mask = bytearray(mask_bytes)
                for i in range(self.n_channels):
                    mask[i // 8] |= (1 << (i % 8))
                
                f.write(struct.pack('<I', mask_bytes))  # Mask size
                f.write(mask)  # Channel mask
            
            # Write audio data section
            # Mix all layers together
            mixed_audio = np.zeros((self.n_channels, max_frames))
            for layer_id, ambi_signals in self.layers.items():
                n_frames = ambi_signals.shape[1]
                mixed_audio[:, :n_frames] += ambi_signals
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.99:
                mixed_audio = mixed_audio * 0.99 / max_val
            
            # Convert to the appropriate format
            if bit_depth == 16:
                mixed_audio = (mixed_audio * 32767).astype(np.int16)
            else:
                mixed_audio = mixed_audio.astype(np.float32)
            
            # Write interleaved audio data
            for frame in range(max_frames):
                for ch in range(self.n_channels):
                    if bit_depth == 16:
                        f.write(struct.pack('<h', mixed_audio[ch, frame]))
                    else:
                        f.write(struct.pack('<f', mixed_audio[ch, frame]))


class SHACFileReader:
    """
    Reader for the Spherical Harmonic Audio Codec (SHAC) file format.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the SHAC file reader.
        
        Args:
            filename: Input filename
        """
        self.filename = filename
        self.file_info = {}
        self.layers = {}
        self.layer_metadata = {}
        self._read_header()
    
    def _read_header(self) -> None:
        """Read the SHAC file header and metadata."""
        with open(self.filename, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'SHAC':
                raise ValueError("Not a valid SHAC file")
            
            version = struct.unpack('<I', f.read(4))[0]
            order = struct.unpack('<I', f.read(4))[0]
            sample_rate = struct.unpack('<I', f.read(4))[0]
            n_channels = struct.unpack('<I', f.read(4))[0]
            sample_format = struct.unpack('<I', f.read(4))[0]
            n_frames = struct.unpack('<I', f.read(4))[0]
            _ = struct.unpack('<I', f.read(4))[0]  # Reserved
            
            # Store file info
            self.file_info = {
                'version': version,
                'order': order,
                'sample_rate': sample_rate,
                'n_channels': n_channels,
                'sample_format': sample_format,
                'n_frames': n_frames
            }
            
            # Read channel metadata section
            section_size = struct.unpack('<I', f.read(4))[0]
            channel_metadata = []
            
            for _ in range(n_channels):
                channel_idx = struct.unpack('<I', f.read(4))[0]
                sh_degree = struct.unpack('<I', f.read(4))[0]
                sh_order = struct.unpack('<I', f.read(4))[0]
                norm_type = struct.unpack('<I', f.read(4))[0]
                
                channel_metadata.append({
                    'channel_idx': channel_idx,
                    'sh_degree': sh_degree,
                    'sh_order': sh_order,
                    'normalization': AmbisonicNormalization(norm_type)
                })
            
            self.file_info['channel_metadata'] = channel_metadata
            
            # Read layer information section
            section_size = struct.unpack('<I', f.read(4))[0]
            n_layers = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(n_layers):
                layer_id = struct.unpack('<I', f.read(4))[0]
                name_length = struct.unpack('<I', f.read(4))[0]
                layer_name = f.read(name_length).decode('utf-8')
                
                mask_size = struct.unpack('<I', f.read(4))[0]
                channel_mask = f.read(mask_size)
                
                # Convert mask to list of channels
                channels = []
                for byte_idx, byte_val in enumerate(channel_mask):
                    for bit_idx in range(8):
                        if byte_val & (1 << bit_idx):
                            channel_idx = byte_idx * 8 + bit_idx
                            if channel_idx < n_channels:
                                channels.append(channel_idx)
                
                self.layer_metadata[layer_name] = {
                    'layer_id': layer_id,
                    'channels': channels
                }
            
            # Store the offset to the audio data
            self.audio_data_offset = f.tell()
    
    def read_audio_data(self) -> np.ndarray:
        """
        Read the complete audio data from the file.
        
        Returns:
            Ambisonic signals, shape (n_channels, n_frames)
        """
        n_channels = self.file_info['n_channels']
        n_frames = self.file_info['n_frames']
        sample_format = self.file_info['sample_format']
        
        # Determine data type and scaling
        if sample_format == 1:  # 16-bit PCM
            dtype = np.int16
            scale = 1.0 / 32768.0
        elif sample_format == 2:  # 32-bit float
            dtype = np.float32
            scale = 1.0
        else:
            raise ValueError(f"Unsupported sample format: {sample_format}")
        
        with open(self.filename, 'rb') as f:
            # Seek to audio data
            f.seek(self.audio_data_offset)
            
            # Read interleaved audio data
            audio_data = np.zeros((n_channels, n_frames), dtype=np.float32)
            
            # Read frame by frame
            for frame in range(n_frames):
                for ch in range(n_channels):
                    if sample_format == 1:  # 16-bit PCM
                        sample = struct.unpack('<h', f.read(2))[0]
                    else:  # 32-bit float
                        sample = struct.unpack('<f', f.read(4))[0]
                    
                    audio_data[ch, frame] = sample * scale
        
        return audio_data
    
    def read_layer(self, layer_name: str) -> Optional[np.ndarray]:
        """
        Read a specific layer from the file.
        
        Args:
            layer_name: Name of the layer to read
            
        Returns:
            Ambisonic signals for the specified layer, or None if not found
        """
        if layer_name not in self.layer_metadata:
            return None
        
        # Get the full audio data
        full_audio = self.read_audio_data()
        
        # Extract the channels for this layer
        channels = self.layer_metadata[layer_name]['channels']
        layer_audio = full_audio[channels]
        
        return layer_audio
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the file.
        
        Returns:
            List of layer names
        """
        return list(self.layer_metadata.keys())
    
    def get_file_info(self) -> Dict:
        """
        Get information about the SHAC file.
        
        Returns:
            Dictionary with file information
        """
        return self.file_info


# =====================================================================================
# SHAC Codec Implementation
# =====================================================================================

class SHACCodec:
    """
    Main class for the Spherical Harmonic Audio Codec (SHAC).
    
    This class provides a high-level interface for encoding, processing,
    and decoding spatial audio using spherical harmonics.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, 
                normalization: AmbisonicNormalization = AmbisonicNormalization.SN3D,
                ordering: AmbisonicOrdering = AmbisonicOrdering.ACN):
        """
        Initialize the SHAC codec.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            normalization: Normalization convention
            ordering: Channel ordering convention
        """
        self.order = order
        self.sample_rate = sample_rate
        self.normalization = normalization
        self.ordering = ordering
        self.n_channels = (order + 1) ** 2
        
        # Initialize processing state
        self.sources = {}
        self.layers = {}
        self.layer_metadata = {}
        
        # Initialize room model
        self.room = None
        
        # Initialize binaural renderer
        self.binaural_renderer = None
        
        # Load default HRTF if available
        self.hrtf_database = self._load_default_hrtf()
    
    def _load_default_hrtf(self) -> Optional[Dict]:
        """
        Load a default HRTF database if available.
        
        Returns:
            HRTF database or None if not available
        """
        # In a real implementation, this would load from a file
        # Here we'll use our synthetic HRTF generator
        try:
            return load_hrtf_database("")
        except:
            warnings.warn("Could not load default HRTF database")
            return None
    
    def add_mono_source(self, source_id: str, audio: np.ndarray, position: Tuple[float, float, float],
                       attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a mono audio source to the codec.
        
        Args:
            source_id: Unique identifier for the source
            audio: Mono audio signal
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if source_id in self.sources:
            warnings.warn(f"Source {source_id} already exists. Overwriting.")
        
        # Ensure audio is mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * 0.99 / max_val
        
        # Store source data
        self.sources[source_id] = {
            'audio': audio,
            'position': position,
            'attributes': attributes or SourceAttributes(position)
        }
        
        # Encode to ambisonic signals
        ambi_signals = encode_mono_source(audio, position, self.order, self.normalization)
        
        # Apply source directivity and other attributes if provided
        if attributes and (attributes.directivity > 0 or attributes.width > 0):
            source_directivity = {
                'pattern': 'cardioid',
                'order': attributes.directivity,
                'axis': attributes.directivity_axis,
                'frequency_dependent': True
            }
            
            ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                           position[2], attributes.air_absorption,
                                                           source_directivity)
        
        # Create a layer for this source
        self.layers[source_id] = ambi_signals
        
        # Store layer metadata
        self.layer_metadata[source_id] = {
            'type': 'source',
            'position': position,
            'original_gain': 1.0,
            'current_gain': 1.0,
            'muted': False
        }
    
    def add_stereo_source(self, source_id: str, left_audio: np.ndarray, right_audio: np.ndarray,
                         position: Tuple[float, float, float], width: float = 0.6,
                         attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a stereo audio source to the codec.
        
        Args:
            source_id: Unique identifier for the source
            left_audio: Left channel audio signal
            right_audio: Right channel audio signal
            position: (azimuth, elevation, distance) of center position
            width: Angular width in radians
            attributes: Optional source attributes
        """
        if source_id in self.sources:
            warnings.warn(f"Source {source_id} already exists. Overwriting.")
        
        # Ensure audio is mono for each channel
        if len(left_audio.shape) > 1 and left_audio.shape[1] > 1:
            left_audio = np.mean(left_audio, axis=1)
        if len(right_audio.shape) > 1 and right_audio.shape[1] > 1:
            right_audio = np.mean(right_audio, axis=1)
        
        # Ensure same length
        if len(left_audio) != len(right_audio):
            max_len = max(len(left_audio), len(right_audio))
            if len(left_audio) < max_len:
                left_audio = np.pad(left_audio, (0, max_len - len(left_audio)))
            if len(right_audio) < max_len:
                right_audio = np.pad(right_audio, (0, max_len - len(right_audio)))
        
        # Normalize audio to prevent clipping
        max_val = max(np.max(np.abs(left_audio)), np.max(np.abs(right_audio)))
        if max_val > 0.99:
            scale = 0.99 / max_val
            left_audio = left_audio * scale
            right_audio = right_audio * scale
        
        # Store source data
        self.sources[source_id] = {
            'left_audio': left_audio,
            'right_audio': right_audio,
            'position': position,
            'width': width,
            'attributes': attributes or SourceAttributes(position)
        }
        
        # Encode to ambisonic signals
        ambi_signals = encode_stereo_source(left_audio, right_audio, position, width,
                                          self.order, self.normalization)
        
        # Apply source attributes if provided
        if attributes:
            # For stereo sources, only apply distance-based effects
            ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                           position[2], attributes.air_absorption)
        
        # Create a layer for this source
        self.layers[source_id] = ambi_signals
        
        # Store layer metadata
        self.layer_metadata[source_id] = {
            'type': 'stereo_source',
            'position': position,
            'width': width,
            'original_gain': 1.0,
            'current_gain': 1.0,
            'muted': False
        }
    
    def set_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update position in source data
        self.sources[source_id]['position'] = position
        
        # Update position in layer metadata
        self.layer_metadata[source_id]['position'] = position
        
        # Re-encode the source with the new position
        source_data = self.sources[source_id]
        
        if 'audio' in source_data:
            # Mono source
            audio = source_data['audio']
            attributes = source_data['attributes']
            
            # Re-encode
            ambi_signals = encode_mono_source(audio, position, self.order, self.normalization)
            
            # Apply source attributes if needed
            if attributes and (attributes.directivity > 0 or attributes.width > 0):
                source_directivity = {
                    'pattern': 'cardioid',
                    'order': attributes.directivity,
                    'axis': attributes.directivity_axis,
                    'frequency_dependent': True
                }
                
                ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                               position[2], attributes.air_absorption,
                                                               source_directivity)
            
            # Update layer
            self.layers[source_id] = ambi_signals
        
        elif 'left_audio' in source_data:
            # Stereo source
            left_audio = source_data['left_audio']
            right_audio = source_data['right_audio']
            width = source_data['width']
            attributes = source_data['attributes']
            
            # Re-encode
            ambi_signals = encode_stereo_source(left_audio, right_audio, position, width,
                                              self.order, self.normalization)
            
            # Apply source attributes if needed
            if attributes:
                ambi_signals = apply_frequency_dependent_effects(ambi_signals, self.sample_rate,
                                                               position[2], attributes.air_absorption)
            
            # Update layer
            self.layers[source_id] = ambi_signals
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.layers:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update gain in layer metadata
        self.layer_metadata[source_id]['current_gain'] = gain
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source.
        
        Args:
            source_id: Identifier of the source to update
            muted: True to mute, False to unmute
        """
        if source_id not in self.layers:
            raise ValueError(f"Source {source_id} does not exist")
        
        # Update mute state in layer metadata
        self.layer_metadata[source_id]['muted'] = muted
    
    def set_room_model(self, room_dimensions: Tuple[float, float, float],
                      reflection_coefficients: Dict[str, float],
                      rt60: float) -> None:
        """
        Set a room model for reflections and reverberation.
        
        Args:
            room_dimensions: (width, height, length) in meters
            reflection_coefficients: Coefficients for each surface
            rt60: Reverberation time in seconds
        """
        # Calculate room volume
        width, height, length = room_dimensions
        volume = width * height * length
        
        # Store room model
        self.room = {
            'dimensions': room_dimensions,
            'reflection_coefficients': reflection_coefficients,
            'rt60': rt60,
            'volume': volume
        }
    
    def set_binaural_renderer(self, hrtf_database: Union[str, Dict],
                             interpolation_method: HRTFInterpolationMethod = HRTFInterpolationMethod.SPHERICAL) -> None:
        """
        Set the binaural renderer configuration.
        
        Args:
            hrtf_database: Path to HRTF database or dictionary with HRTF data
            interpolation_method: HRTF interpolation method
        """
        if isinstance(hrtf_database, str):
            self.hrtf_database = load_hrtf_database(hrtf_database)
        else:
            self.hrtf_database = hrtf_database
        
        self.binaural_renderer = {
            'interpolation_method': interpolation_method,
            'nearfield_compensation': True,
            'crossfade_time': 0.1
        }
    
    def process(self) -> np.ndarray:
        """
        Process all sources and layers to create the final ambisonic signals.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, n_samples)
        """
        # Determine the maximum number of samples across all layers
        max_samples = 0
        for layer_id, ambi_signals in self.layers.items():
            if not self.layer_metadata[layer_id]['muted']:
                max_samples = max(max_samples, ambi_signals.shape[1])
        
        if max_samples == 0:
            return np.zeros((self.n_channels, 0))
        
        # Initialize output signals
        output_signals = np.zeros((self.n_channels, max_samples))
        
        # Mix all active layers
        for layer_id, ambi_signals in self.layers.items():
            # Skip muted layers
            if self.layer_metadata[layer_id]['muted']:
                continue
            
            # Apply gain
            gain = self.layer_metadata[layer_id]['current_gain']
            scaled_signals = ambi_signals * gain
            
            # Mix into output
            n_samples = scaled_signals.shape[1]
            output_signals[:, :n_samples] += scaled_signals
        
        # Apply room model if available
        if self.room is not None:
            # Process each source for early reflections
            reflections = np.zeros_like(output_signals)
            
            for source_id, source_data in self.sources.items():
                # Skip muted sources
                if source_id not in self.layers or self.layer_metadata[source_id]['muted']:
                    continue
                
                # Get source position
                position = source_data['position']
                
                # Convert to Cartesian
                cart_pos = convert_to_cartesian(position)
                
                # Apply early reflections
                source_reflections = apply_early_reflections(
                    self.layers[source_id],
                    cart_pos,
                    self.room['dimensions'],
                    self.room['reflection_coefficients'],
                    self.sample_rate
                )
                
                # Apply gain
                gain = self.layer_metadata[source_id]['current_gain']
                reflections += source_reflections * gain
            
            # Mix in reflections
            output_signals = output_signals + reflections
            
            # Apply diffuse reverberation
            output_signals = apply_diffuse_reverberation(
                output_signals,
                self.room['rt60'],
                self.sample_rate,
                self.room['volume']
            )
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_signals))
        if max_val > 0.99:
            output_signals = output_signals * 0.99 / max_val
        
        return output_signals
    
    def rotate(self, ambi_signals: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        Rotate the ambisonic sound field.
        
        Args:
            ambi_signals: Ambisonic signals to rotate
            yaw: Rotation around vertical axis in radians
            pitch: Rotation around side axis in radians
            roll: Rotation around front axis in radians
            
        Returns:
            Rotated ambisonic signals
        """
        return rotate_ambisonics(ambi_signals, yaw, pitch, roll)
    
    def binauralize(self, ambi_signals: np.ndarray) -> np.ndarray:
        """
        Convert ambisonic signals to binaural stereo.
        
        Args:
            ambi_signals: Ambisonic signals to binauralize
            
        Returns:
            Binaural stereo signals, shape (2, n_samples)
        """
        if self.hrtf_database is None:
            raise ValueError("No HRTF database available for binauralization")
        
        return binauralize_ambisonics(ambi_signals, self.hrtf_database)
    
    def save_to_file(self, filename: str, bit_depth: int = 32) -> None:
        """
        Save the processed audio to a SHAC file.
        
        Args:
            filename: Output filename
            bit_depth: Bit depth (16 or 32)
        """
        # Process all sources and layers
        ambi_signals = self.process()
        
        # Create a SHAC file writer
        writer = SHACFileWriter(self.order, self.sample_rate, self.normalization)
        
        # Add a single layer with the processed audio
        writer.add_layer('main', ambi_signals, {'type': 'mixed'})
        
        # Add individual layers if available
        for layer_id, layer_signals in self.layers.items():
            if not self.layer_metadata[layer_id]['muted']:
                writer.add_layer(layer_id, layer_signals, self.layer_metadata[layer_id])
        
        # Write the file
        writer.write_file(filename, bit_depth)
    
    def load_from_file(self, filename: str) -> None:
        """
        Load audio from a SHAC file.
        
        Args:
            filename: Input filename
        """
        # Create a SHAC file reader
        reader = SHACFileReader(filename)
        
        # Get file info
        file_info = reader.get_file_info()
        
        # Update codec parameters
        self.order = file_info['order']
        self.sample_rate = file_info['sample_rate']
        self.n_channels = file_info['n_channels']
        
        # Clear existing layers
        self.layers = {}
        self.layer_metadata = {}
        
        # Load each layer
        for layer_name in reader.get_layer_names():
            layer_audio = reader.read_layer(layer_name)
            if layer_audio is not None:
                self.layers[layer_name] = layer_audio
                self.layer_metadata[layer_name] = {
                    'type': 'loaded',
                    'original_gain': 1.0,
                    'current_gain': 1.0,
                    'muted': False
                }


# =====================================================================================
# Real-time Processing Classes
# =====================================================================================

class SHACStreamProcessor:
    """
    Real-time streaming processor for SHAC audio.
    
    This class handles real-time processing of spatial audio streams
    with minimal latency and efficient CPU usage.
    """
    
    def __init__(self, order: int = 3, sample_rate: int = 48000, buffer_size: int = 1024,
                 max_sources: int = 32):
        """
        Initialize the SHAC stream processor.
        
        Args:
            order: Ambisonic order
            sample_rate: Sample rate in Hz
            buffer_size: Processing buffer size in samples
            max_sources: Maximum number of simultaneous sources
        """
        self.order = order
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        self.n_channels = (order + 1) ** 2
        
        # Create the main codec
        self.codec = SHACCodec(order, sample_rate)
        
        # Initialize streaming state
        self.sources = {}
        self.source_buffers = {}
        self.output_buffer = np.zeros((self.n_channels, buffer_size))
        
        # Initialize processing thread
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
    
    def start(self) -> None:
        """Start the real-time processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self) -> None:
        """Stop the real-time processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def add_source(self, source_id: str, position: Tuple[float, float, float],
                  attributes: Optional[SourceAttributes] = None) -> None:
        """
        Add a streaming source.
        
        Args:
            source_id: Unique identifier for the source
            position: (azimuth, elevation, distance) in radians and meters
            attributes: Optional source attributes
        """
        if len(self.sources) >= self.max_sources:
            raise ValueError(f"Maximum number of sources ({self.max_sources}) reached")
        
        self.sources[source_id] = {
            'position': position,
            'attributes': attributes or SourceAttributes(position),
            'gain': 1.0,
            'muted': False
        }
        
        # Initialize source buffer
        self.source_buffers[source_id] = np.zeros(self.buffer_size)
    
    def remove_source(self, source_id: str) -> None:
        """
        Remove a streaming source.
        
        Args:
            source_id: Identifier of the source to remove
        """
        if source_id in self.sources:
            del self.sources[source_id]
        
        if source_id in self.source_buffers:
            del self.source_buffers[source_id]
    
    def update_source(self, source_id: str, audio_chunk: np.ndarray) -> None:
        """
        Update a source with new audio data.
        
        Args:
            source_id: Identifier of the source to update
            audio_chunk: New audio data chunk
        """
        if source_id not in self.sources:
            return
        
        # Copy audio data to source buffer
        n_samples = min(len(audio_chunk), self.buffer_size)
        self.source_buffers[source_id][:n_samples] = audio_chunk[:n_samples]
        
        # If audio chunk is smaller than buffer, zero-pad
        if n_samples < self.buffer_size:
            self.source_buffers[source_id][n_samples:] = 0.0
    
    def update_source_position(self, source_id: str, position: Tuple[float, float, float]) -> None:
        """
        Update the position of a source.
        
        Args:
            source_id: Identifier of the source to update
            position: New (azimuth, elevation, distance) in radians and meters
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['position'] = position
    
    def set_source_gain(self, source_id: str, gain: float) -> None:
        """
        Set the gain for a source.
        
        Args:
            source_id: Identifier of the source to update
            gain: Gain factor (1.0 = unity gain)
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['gain'] = gain
    
    def mute_source(self, source_id: str, muted: bool = True) -> None:
        """
        Mute or unmute a source.
        
        Args:
            source_id: Identifier of the source to update
            muted: True to mute, False to unmute
        """
        if source_id not in self.sources:
            return
        
        self.sources[source_id]['muted'] = muted
    
    def set_listener_rotation(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Set the listener's head rotation.
        
        Args:
            yaw: Rotation around vertical axis in radians
            pitch: Rotation around side axis in radians
            roll: Rotation around front axis in radians
        """
        # Store for next processing cycle
        self.codec.listener_rotation = (yaw, pitch, roll)
    
    def process_block(self) -> np.ndarray:
        """
        Process a block of audio.
        
        Returns:
            Processed ambisonic signals, shape (n_channels, buffer_size)
        """
        # Initialize output buffer
        output_buffer = np.zeros((self.n_channels, self.buffer_size))
        
        # Process each source
        for source_id, source_info in self.sources.items():
            # Skip muted sources
            if source_info['muted']:
                continue
            
            # Get source parameters
            position = source_info['position']
            attributes = source_info['attributes']
            gain = source_info['gain']
            
            # Get audio data
            audio = self.source_buffers[source_id]
            
            # Encode to ambisonics
            ambi_source = encode_mono_source(audio, position, self.order, AmbisonicNormalization.SN3D)
            
            # Apply source attributes if needed
            if attributes and (attributes.directivity > 0 or attributes.width > 0):
                source_directivity = {
                    'pattern': 'cardioid',
                    'order': attributes.directivity,
                    'axis': attributes.directivity_axis,
                    'frequency_dependent': True
                }
                
                ambi_source = apply_frequency_dependent_effects(ambi_source, self.sample_rate,
                                                              position[2], attributes.air_absorption,
                                                              source_directivity)
            
            # Apply gain
            ambi_source *= gain
            
            # Mix into output buffer
            output_buffer += ambi_source
        
        # Apply room effects if configured
        if hasattr(self.codec, 'room') and self.codec.room is not None:
            # Add simple diffuse reverberation (simplified for real-time processing)
            # In a full implementation, this would use convolution with pre-computed IRs
            pass
        
        # Apply rotation if needed
        if hasattr(self.codec, 'listener_rotation'):
            yaw, pitch, roll = self.codec.listener_rotation
            output_buffer = rotate_ambisonics(output_buffer, yaw, pitch, roll)
        
        # Normalize if necessary
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0.99:
            output_buffer = output_buffer * 0.99 / max_val
        
        return output_buffer
    
    def get_binaural_output(self) -> np.ndarray:
        """
        Get the current block as binaural stereo.
        
        Returns:
            Binaural stereo signals, shape (2, buffer_size)
        """
        # Process the current block
        ambi_block = self.process_block()
        
        # Convert to binaural
        return self.codec.binauralize(ambi_block)
    
    def _processing_loop(self) -> None:
        """Main processing loop for the streaming thread."""
        while self.running:
            # Process the current block
            try:
                # Process the block
                ambi_block = self.process_block()
                
                # Binauralize if needed
                if hasattr(self.codec, 'binaural_renderer') and self.codec.binaural_renderer is not None:
                    binaural_block = self.codec.binauralize(ambi_block)
                    
                    # Add to output queue
                    self.output_queue.put(binaural_block)
                else:
                    # Add ambisonic block to output queue
                    self.output_queue.put(ambi_block)
                
                # Sleep for a bit to avoid burning CPU
                # In a real-time audio system, this would be synchronized with audio callbacks
                time.sleep(self.buffer_size / self.sample_rate / 2)
            
            except Exception as e:
                warnings.warn(f"Error in processing loop: {e}")
                time.sleep(0.1)


# =====================================================================================
# Example usage and demonstration
# =====================================================================================

def create_example_sound_scene():
    """
    Create and process an example 3D sound scene.
    
    This example demonstrates the core functionality of the SHAC codec.
    """
    print("Creating example 3D sound scene...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Create synthetic audio signals
    duration = 5.0  # seconds
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Piano sound (sine wave with harmonics and decay)
    piano_freq = 440.0  # A4
    piano_audio = 0.5 * np.sin(2 * np.pi * piano_freq * t) * np.exp(-t/2)
    piano_audio += 0.25 * np.sin(2 * np.pi * 2 * piano_freq * t) * np.exp(-t/1.5)
    piano_audio += 0.125 * np.sin(2 * np.pi * 3 * piano_freq * t) * np.exp(-t)
    
    # Drum sound (impulses with decay)
    drum_audio = np.zeros_like(t)
    for i in range(0, len(t), 12000):  # Four beats
        if i + 2000 < len(drum_audio):
            drum_audio[i:i+2000] = 0.8 * np.exp(-np.linspace(0, 10, 2000))
    
    # Ambient sound (filtered noise)
    np.random.seed(42)  # For reproducibility
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 0.1)
    ambient_audio = signal.filtfilt(b, a, noise) * 0.2
    
    # Add sources to the codec
    # Position format: (azimuth, elevation, distance) in radians and meters
    
    # Piano in front left
    piano_position = (-np.pi/4, 0.0, 3.0)
    piano_attributes = SourceAttributes(
        position=piano_position,
        directivity=0.7,
        directivity_axis=(0.0, 0.0, 1.0),
        width=0.2
    )
    codec.add_mono_source("piano", piano_audio, piano_position, piano_attributes)
    
    # Drum in front right
    drum_position = (np.pi/4, -0.1, 2.5)
    drum_attributes = SourceAttributes(
        position=drum_position,
        directivity=0.3,
        width=0.4
    )
    codec.add_mono_source("drum", drum_audio, drum_position, drum_attributes)
    
    # Ambient sound above
    ambient_position = (0.0, np.pi/3, 5.0)
    ambient_attributes = SourceAttributes(
        position=ambient_position,
        directivity=0.0,
        width=1.0
    )
    codec.add_mono_source("ambient", ambient_audio, ambient_position, ambient_attributes)
    
    # Set up a room model
    room_dimensions = (10.0, 3.0, 8.0)  # width, height, length in meters
    reflection_coefficients = {
        'left': 0.7,
        'right': 0.7,
        'floor': 0.4,
        'ceiling': 0.8,
        'front': 0.6,
        'back': 0.6
    }
    rt60 = 1.2  # seconds
    codec.set_room_model(room_dimensions, reflection_coefficients, rt60)
    
    # Process the scene
    print("Processing audio...")
    ambi_signals = codec.process()
    
    # Apply head rotation (as if user is looking to the left)
    print("Applying head rotation...")
    yaw = np.pi/3  # 60 degrees to the left
    rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
    
    # Convert to binaural
    print("Converting to binaural...")
    binaural_output = codec.binauralize(rotated_ambi)
    
    # Save outputs
    print("Saving outputs...")
    try:
        import soundfile as sf
        sf.write("shac_example_binaural.wav", binaural_output.T, sample_rate)
        print("Saved binaural output to shac_example_binaural.wav")
    except ImportError:
        print("Could not save audio files: soundfile module not available")
    
    # Save to SHAC file
    print("Saving to SHAC file...")
    codec.save_to_file("example_scene.shac")
    print("Saved SHAC file to example_scene.shac")
    
    print("Done!")
    return codec


def demonstrate_interactive_navigation():
    """
    Demonstrate interactive navigation through a 3D sound scene.
    
    This example creates a sequence of binaural renders while
    navigating through a sound scene.
    """
    print("Demonstrating interactive navigation...")
    
    # Create a SHAC codec
    codec = SHACCodec(order=3, sample_rate=48000)
    
    # Load scene from file if it exists, otherwise create a new one
    if os.path.exists("example_scene.shac"):
        print("Loading scene from file...")
        codec.load_from_file("example_scene.shac")
    else:
        print("Creating new scene...")
        create_example_sound_scene()
    
    # Define a navigation path
    yaw_angles = np.linspace(0, 2*np.pi, 8)  # Full 360° rotation in 8 steps
    
    # Process each step in the path
    for i, yaw in enumerate(yaw_angles):
        print(f"Step {i+1}/{len(yaw_angles)}: Yaw = {yaw:.2f} radians")
        
        # Process the scene
        ambi_signals = codec.process()
        
        # Apply rotation for this step
        rotated_ambi = codec.rotate(ambi_signals, yaw, 0.0, 0.0)
        
        # Convert to binaural
        binaural_output = codec.binauralize(rotated_ambi)
        
        # Save this step
        try:
            import soundfile as sf
            sf.write(f"navigation_step_{i+1}.wav", binaural_output.T, codec.sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
    
    print("Navigation demonstration complete!")


def demonstrate_streaming_processor():
    """
    Demonstrate the real-time streaming processor.
    
    This example shows how to use the SHAC stream processor for
    real-time audio processing.
    """
    print("Demonstrating streaming processor...")
    
    # Create a streaming processor
    processor = SHACStreamProcessor(order=3, sample_rate=48000, buffer_size=1024)
    
    # Create synthetic audio signals (single cycle of a sine wave)
    sample_rate = 48000
    buffer_size = 1024
    
    # Create three sources with different frequencies
    source1_freq = 440.0  # A4
    source1_audio = 0.5 * np.sin(2 * np.pi * source1_freq * np.arange(buffer_size) / sample_rate)
    
    source2_freq = 261.63  # C4
    source2_audio = 0.5 * np.sin(2 * np.pi * source2_freq * np.arange(buffer_size) / sample_rate)
    
    source3_freq = 329.63  # E4
    source3_audio = 0.5 * np.sin(2 * np.pi * source3_freq * np.arange(buffer_size) / sample_rate)
    
    # Add sources to the processor
    processor.add_source("source1", (-np.pi/4, 0.0, 2.0))
    processor.add_source("source2", (np.pi/4, 0.0, 2.0))
    processor.add_source("source3", (0.0, np.pi/4, 3.0))
    
    # Start the processor
    processor.start()
    
    # Simulate real-time processing for a few blocks
    for i in range(10):
        print(f"Processing block {i+1}/10")
        
        # Update sources with new audio data
        processor.update_source("source1", source1_audio)
        processor.update_source("source2", source2_audio)
        processor.update_source("source3", source3_audio)
        
        # Set listener rotation (changing over time)
        yaw = i * np.pi / 5  # Rotate gradually
        processor.set_listener_rotation(yaw, 0.0, 0.0)
        
        # Get binaural output
        binaural_output = processor.get_binaural_output()
        
        # Save this block
        try:
            import soundfile as sf
            sf.write(f"streaming_block_{i+1}.wav", binaural_output.T, sample_rate)
        except ImportError:
            print("Could not save audio file: soundfile module not available")
        
        # In a real application, this would feed audio to the sound card
        # For this demo, we sleep to simulate real-time processing
        time.sleep(buffer_size / sample_rate)
    
    # Stop the processor
    processor.stop()
    
    print("Streaming demonstration complete!")


def main():
    """Main function to demonstrate the SHAC codec."""
    print("SHAC Codec Demonstration")
    print("=======================")
    
    # Create and process an example sound scene
    create_example_sound_scene()
    
    # Demonstrate interactive navigation
    demonstrate_interactive_navigation()
    
    # Demonstrate streaming processor
    demonstrate_streaming_processor()


if __name__ == "__main__":
    main()
