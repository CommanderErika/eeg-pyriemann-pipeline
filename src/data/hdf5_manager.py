from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Iterable

from pathlib import Path
import logging
import h5py
import numpy as np

from .basic_type import (
    MOABBData, 
    CovarianceData, 
    TangentSpaceData
)

class DataManager(ABC):
    """Abstract class for data strategies"""
    @abstractmethod
    def save(self, file: h5py.File, data: Any, compression: str, chunk_shape: tuple):
        pass
    @abstractmethod
    def load(self, file: h5py.File) -> Any:
        pass

class MOABBDataManager(DataManager):
    """Data Manager for MOABB data type"""
    def save(self, file, data, compression, chunk_shape) -> None:
        # Saving data
        file.create_dataset(name='eeg_data', 
                            data=data.x, 
                            compression=compression, chunks=chunk_shape)
        # Processing and saving labels
        labels = np.array(data.y, dtype='S64') if data.y.dtype.kind in ['U', 'O'] else data.y
        file.create_dataset(name='labels',
                            data=labels,
                            compression=compression, chunks=chunk_shape)
        # Subjects
        file.create_dataset(name="subjects",
                            data=data.subjects,
                            compression=compression, chunks=chunk_shape)
        # Store metadata
        file.attrs.update({
            'data_type' : 'moabb',
            'freqr': data.freqr,
            'version': '1.0',
            'channel_names': np.array(data.channel_names, dtype='S64')
        })

    def load(self, file) -> MOABBData:
        # Load and convert labels
        labels = np.array(file['labels'])
        
        # Handle string decoding for labels
        if labels.dtype.kind in ['S', 'a']:  # String or bytes array
            labels = np.char.decode(labels, 'utf-8')
        
        # Handle channel_names safely
        channel_names_attr = file.attrs.get('channel_names', [])
        
        # --- FIXED LINE BELOW ---
        # Use len() > 0 instead of implicit boolean check
        if len(channel_names_attr) > 0:
            # Decode bytes to strings if necessary
            if isinstance(channel_names_attr[0], bytes):
                channel_names = [n.decode('utf-8') for n in channel_names_attr]
            else:
                channel_names = list(channel_names_attr)
        else:
            channel_names = None
        
        # Get frequency with safe handling
        freqr = file.attrs.get('freqr', None)
        
        # Loading
        return MOABBData(
            x=np.array(file['eeg_data']),
            y=labels,
            subjects=np.array(file['subjects']),
            dataset=None,   # TODO: Implement reconstruction
            paradigm=None,  # TODO: Implement reconstruction
            channel_names=channel_names,
            freqr=freqr
        )
    
class CovarianceDataManager(DataManager):
    """Data Manager for Covariance data type"""
    def save(self, file, data, compression, chunk_shape) -> None:
        # Saving data
        file.create_dataset(name='covariances', 
                            data=data.x, 
                            compression=compression, chunks=chunk_shape)
        
        # Processing and saving labels
        labels = np.array(data.y, dtype='S64') if data.y.dtype.kind in ['U', 'O'] else data.y
        file.create_dataset(name='labels',
                            data=labels,
                            compression=compression, chunks=chunk_shape)
        # Subjects
        file.create_dataset(name="subjects",
                            data=data.subjects,
                            compression=compression, chunks=chunk_shape)
        # Store metadata
        file.attrs.update({
            'data_type' : 'covariances',
            'version': '1.0',
            'channel_names': np.array(data.channel_names, dtype='S64')
        })

    def is_valid_iterable(self, obj: Any) -> bool:
        """Type guard para verificar se é um iterável válido."""
        if obj is None:
            return False
        if isinstance(obj, (str, bytes)):
            return False
        return hasattr(obj, '__iter__')

    def load(self, file) -> CovarianceData:
        # Load covariance data
        for key in ['covariances', 'x', 'convariances']:
            if key in file:
                cov_dataset = file[key]
                break
        else:
            raise KeyError("No valid dataset found in HDF5 file")
        
        # Load labels
        labels = np.array(file['labels'])
        if labels.dtype.kind in ['S', 'a']:
            labels = np.char.decode(labels, 'utf-8')
        
        # Convert to expected dtype
        cov_data = np.array(cov_dataset, dtype=np.float32)
        
        # Process channel names safely
        channel_names: list[str] = []
        if 'channel_names' in file.attrs:
            channel_attr = file.attrs['channel_names']
            
            if self.is_valid_iterable(channel_attr):
                channel_list = list(channel_attr)
                if channel_list:
                    channel_names = [
                        n.decode('utf-8') if isinstance(n, bytes) else str(n)
                        for n in channel_list
                    ]
        # Return loaded data
        return CovarianceData(
                    x=cov_data,
                    y=labels,
                    subjects=np.array(file['subjects']),
                    channel_names=channel_names,
        )
    
class TangentSpaceManager(DataManager):
    """Data Manager for Tangent Space data type"""
    def save(self, file, data, compression, chunk_shape) -> None:
        # Saving data
        file.create_dataset(name='tangent', 
                            data=data.x, 
                            compression=compression, chunks=chunk_shape)
        # Processing and saving labels
        labels = np.array(data.y, dtype='S64') if data.y.dtype.kind in ['U', 'O'] else data.y
        file.create_dataset(name='labels',
                            data=labels,
                            compression=compression, chunks=chunk_shape)
        # Subjects
        file.create_dataset(name="subjects",
                            data=data.subjects,
                            compression=compression, chunks=chunk_shape)
        # Store metadata
        file.attrs.update({
            'data_type' : 'tangent',
            'version': '1.0',
            'channel_names': np.array(data.channel_names, dtype='S64')
        })

    def load(self, file) -> TangentSpaceData:
        # Load tangent space data
        tangent_dataset = file.get('x') or file.get('tangent')
        if tangent_dataset is None:
            raise KeyError("Neither 'x' nor 'tangent' dataset found in HDF5 file")
        
        # Verify and convert data type if necessary
        expected_dtype = np.float32
        if tangent_dataset.dtype != expected_dtype:
            print(f"WARNING: Converting from {tangent_dataset.dtype} to {expected_dtype}")
            tangent_data = np.array(tangent_dataset, dtype=expected_dtype)
        else:
            tangent_data = np.array(tangent_dataset)
        
        # Loading Labels
        labels_ds = file.get('labels') or file.get('y')
        if labels_ds is None:
            raise KeyError("Labels dataset not found")
        
        # Handle both fixed-width and variable-length strings
        if hasattr(labels_ds, 'asstr') and callable(labels_ds.asstr):
            # Modern method for variable-length strings (h5py >= 2.10)
            labels = labels_ds.asstr()[:]
        else:
            # Fallback for fixed-width strings
            labels = np.array(labels_ds)
            if labels.dtype.kind == 'S':  # byte strings
                labels = np.char.decode(labels, 'utf-8')
            # Remove any null padding if present
            labels = np.array([label.strip('\x00') if isinstance(label, str) else label 
                            for label in labels])
        
        # Load channel names
        channel_names = []
        if 'channel_names' in file.attrs:
            chan_attr = file.attrs['channel_names']
            if hasattr(chan_attr, 'asstr') and callable(chan_attr.asstr):
                channel_names = chan_attr.asstr()[:]
            else:
                # Handle fixed-width strings in attributes
                channel_names = [name.decode('utf-8').strip('\x00') 
                            if isinstance(name, bytes) else str(name).strip('\x00')
                            for name in chan_attr]
        
        # Checking data dimension
        if 'feature_dimension' in file.attrs:
            expected_feature_dim = file.attrs['feature_dimension']
            if tangent_data.ndim == 2 and tangent_data.shape[1] != expected_feature_dim:
                print(f"WARNING: Expected feature dimension {expected_feature_dim}, " 
                    f"got {tangent_data.shape[1]}")
        
        # Return loaded data
        return TangentSpaceData(
            x=tangent_data,
            y=labels,
            subjects=np.array(file['subjects']),
            channel_names=channel_names
        )


@dataclass
class HDF5Manager:
    """HDF5 read/write manager"""
    verbose: bool                   = field(default=True, repr=False)
    compression: str                = field(default='gzip')  # 'gzip', 'lzf', or None
    chunk_shape: Optional[tuple]    = field(default=None)

    def __post_init__(self):
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Datasets strategies
        self.strategies = {
            'moabb': MOABBDataManager(),
            'covariances': CovarianceDataManager(),
            'tangent' : TangentSpaceManager(),
            'riemann_tangent_space' : TangentSpaceManager()
        }
        # Validation
        if self.verbose:
            self.logger.info(f"HDF5 manager initialized.")
        
    def save(self, data: any, filename: str, data_type: str = 'moabb') -> None:
        """Save with compression and chunking"""
        strategy = self.strategies.get(data_type)
        if not strategy:
            raise ValueError(f"Unknown data type: {data_type}")
        # Getting file
        file_path = Path(filename).with_suffix(".h5")
        with h5py.File(file_path, 'w') as hf_file:
            # Dynamic chunking if not set
            #optimal_chunks = self.chunk_shape or self._guess_chunk_shape(data.x)
            # Saving file
            strategy.save(
                file=hf_file,
                data=data,
                compression=self.compression,
                chunk_shape=self.chunk_shape
            )

    def load(self, filename: str) -> any:
        """Memory-mapped loading with validation"""
        # Getting file
        file_path = Path(filename).with_suffix(".h5")
        with h5py.File(file_path, 'r') as hf_file:
            data_type = hf_file.attrs.get('data_type')
            strategy = self.strategies.get(data_type)
            if not strategy:
                raise ValueError(f"Unknown data type in file: {data_type}")
            
            return strategy.load(hf_file)
        
    # Helper methods
    def _guess_chunk_shape(self, array: np.ndarray) -> tuple:
        """Auto-determine optimal chunks for EEG data"""
        return (min(100, array.shape[0]), *array.shape[1:])