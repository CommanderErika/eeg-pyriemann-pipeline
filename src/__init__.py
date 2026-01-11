# src/__init__.py

# Version of your package
__version__ = '0.1.0'

# Package documentation
__doc__ = """
    EEG Processing Pipeline - A complete toolbox for EEG feature extraction
    including covariance matrix computation and Riemannian tangent space projection.
"""

# Import data extraction components
from .data import ( 
    DataExtractor,
    HDF5Manager,
)

# Import processing components
from .processing import (
    ProcessingPipeline,
)

from .models import (
    BaseModel,
    SklearnModel,
    LogisticModel,
    SVMModel,
    LDAModel,
    RidgeModel
)

from .tracking import (
    MLflowTracker,
    ExperimentOrchestrator
)

from .visualization import (
    generate_boxplot,
    generate_density_plot,
    generate_histogram,
    confusion_matrix_plot
)

from .utils import (
    read_file_cov_c,
    read_file_ts_c,
    handle_datasets,
    get_paradigm,
    MOABB_DATASETS,
    MOABB_PARADIGMS
)

# Explicitly define public API
__all__ = [
    # Data extraction components
    'DataExtractor',
    'HDF5Manager',
    # Processing components
    'ProcessingPipeline',
    # Models components
    'BaseModel',
    'SklearnModel',
    'LogisticModel',
    'SVMModel',
    'LDAModel',
    'RidgeModel',
    # Tracking components
    'MLflowTracker',
    'ExperimentOrchestrator',
    # Generate plots
    'generate_boxplot', 
    'generate_density_plot', 
    'generate_histogram',
    'confusion_matrix_plot',
    # Utils
    'read_file_cov_c',
    'read_file_ts_c',
    'handle_datasets',
    'get_paradigm',
    'MOABB_DATASETS',
    'MOABB_PARADIGMS',
    # Package metadata
    '__version__',
    '__doc__'
]