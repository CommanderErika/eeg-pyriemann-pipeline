from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import warnings

import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm


@dataclass
class MOABBData:
    """
    Data struct to MOABB datasets.
    """
    # Data
    x: np.ndarray
    y: np.ndarray
    # Metadata
    subjects: np.ndarray
    dataset: BaseDataset | Any
    paradigm: BaseParadigm | Any
    freqr: float | int
    channel_names:  Optional[list[str]] = None

@dataclass
class CovarianceData:
    """Data struct for Covariance Matrix"""
    # Data
    x: np.ndarray
    y: np.ndarray
    # Metadata
    subjects: np.ndarray
    channel_names: Optional[list[str]] = None

@dataclass
class TangentSpaceData:
    """Data struct for TangentSpace array"""
    # Data
    x: np.ndarray
    y: np.ndarray
    # Metadata
    subjects: np.ndarray
    channel_names: Optional[list[str]] = None


    


    

