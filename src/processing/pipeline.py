import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple, Any

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from ..data.basic_type import MOABBData, CovarianceData, TangentSpaceData
from ..data.hdf5_manager import HDF5Manager

# TODO: Add parallel processing
# TODO: handle del procedures

@dataclass
class ProcessingPipeline:
    """
    A pipeline for processing EEG data through covariance estimation and tangent space mapping.
    
    This class handles the complete processing workflow from raw EEG data to 
    Riemannian tangent space features, including covariance matrix computation 
    and tangent space projection.
    
    Attributes:
        raw_dir (str): Directory path containing raw input data files
        ts_dir (str): Directory path for saving tangent space processed data
        cov_dir (str): Directory path for saving covariance matrices
        max_workers (int): Maximum number of parallel workers (default: 4)
        verbose (bool): Enable verbose logging (default: True)
    """
    
    raw_dir: str
    ts_dir: str
    cov_dir: str
    max_workers: int = 4
    verbose: bool = True

    def __post_init__(self):
        """
        Initialize the processing pipeline after dataclass instantiation.
        
        Sets up logging, HDF5 manager, validates directories, and initializes
        covariance and tangent space estimators.
        """
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        # Initialize HDF5 manager for data storage
        self.hdf_manager = HDF5Manager()
        # Validate and create necessary directories
        self._validate_dirs()
        # Initialize covariance estimator
        self.cov_estimator = Covariances(estimator='cov')
        # Initialize tangent space estimator
        self.ts_estimator = TangentSpace(metric='riemann')

    def process_all(self) -> None:
        """
        Process all files in the raw directory.
        
        Iterates through all files in the raw data directory and processes each
        file through the complete pipeline including covariance computation and
        tangent space projection.
        """
        files = os.listdir(self.raw_dir)
        print(files)
        for file in files:
            self._process(file=Path(file).stem)

    def _process(self, file: str) -> None:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file (str): Filename (without extension) to process
            
        Steps:
            1. Load raw data from HDF5 file
            2. Compute covariance matrices
            3. Save covariance matrices
            4. Project to tangent space
            5. Save tangent space features and covariance reference
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If loaded data is invalid or empty
            Exception: For any other processing errors
        """
        try:
            # Load raw data from HDF5 file
            raw_file_path = Path(self.raw_dir) / file
            if not Path(self.raw_dir).exists():
                raise FileNotFoundError(f"Raw data file not found: {Path(self.raw_dir)}")
            
            self.logger.info(f"Loading data from: {raw_file_path}")
            data = self.hdf_manager.load(filename=str(raw_file_path))
            
            if data is None:
                raise ValueError(f"Failed to load data or empty data from: {raw_file_path}")

            # Compute covariance matrices
            covs = self._covariances(data=data)
            
            # Save covariance matrices
            cov_file_path = Path(self.cov_dir) / f"{file}_covs"
            self.hdf_manager.save(data=covs, filename=str(cov_file_path), data_type='covariances')

            # Project to tangent space
            ts, cov_ref = self._tangent_space(data=covs)
            
            # Save covariance reference matrix
            cov_ref_path = Path(self.cov_dir) / f"../cov_ref/{file}_cov_ref.bin"
            cov_ref.tofile(cov_ref_path)
            
            # Save tangent space features
            ts_file_path = Path(self.ts_dir) / f"{file}_ts"
            self.hdf_manager.save(data=ts, filename=str(ts_file_path), data_type='tangent')
            
            self.logger.info(f"Successfully processed file: {file}")
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Data validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing file {file}: {e}")
            raise

    def _covariances(self, data: MOABBData) -> CovarianceData:
        """
        Compute covariance matrices from input data.
        
        Args:
            data: Input data object containing EEG data and metadata
            Must have attributes: x (EEG data), y (labels), subjects, channel_names
            
        Returns:
            CovarianceData: Object containing covariance matrices and metadata
            
        Raises:
            Exception: If covariance computation fails
        """
        try:
            # Compute covariance matrices
            x_cov = self.cov_estimator.fit_transform(data.x)
            # Return structured covariance data
            return CovarianceData(
                x=x_cov,
                y=data.y,
                subjects=data.subjects,
                channel_names=data.channel_names
            )
        except Exception as e:
            self.logger.error(f"Error in covariance calculation: {str(e)}")
            raise

    def _tangent_space(self, data: CovarianceData) -> Tuple[TangentSpaceData, Any]:
        """
        Project covariance matrices to Riemannian tangent space.
        
        Args:
            data: Input data containing covariance matrices and metadata
            Must have attributes: x (covariance matrices), y (labels), 
            subjects, channel_names
            
        Returns:
            tuple: (TangentSpaceData, reference_covariance)
            - TangentSpaceData: Object containing tangent space features and metadata
            - reference_covariance: Reference covariance matrix used for projection
            
        Raises:
            Exception: If tangent space projection fails
        """
        try:
            # Project to tangent space
            x_ts = self.ts_estimator.fit_transform(data.x)
            # Get reference covariance matrix
            cov_ref = self.ts_estimator.reference_
            
            # Return structured tangent space data
            ts = TangentSpaceData(
                x=x_ts,
                y=data.y,
                subjects=data.subjects,
                channel_names=data.channel_names
            )
            return ts, cov_ref
        except Exception as e:
            self.logger.error(f"Error in tangent space transformation: {str(e)}")
            raise

    def _validate_dirs(self) -> None:
        """
        Validate and create necessary directories if they don't exist.
        
        Ensures that raw, tangent space, and covariance directories are available
        for reading and writing operations.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.ts_dir, exist_ok=True)
        os.makedirs(self.cov_dir, exist_ok=True)
        os.makedirs(self.cov_dir + "/../cov_ref", exist_ok=True)