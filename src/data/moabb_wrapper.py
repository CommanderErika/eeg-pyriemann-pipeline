from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import warnings

from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

from .basic_type import MOABBData

# TODO: Enable resample
# TODO: Verify if the dataset is for the specific Paradigm, and if there is enough subjects

@dataclass
class MOABBwrapper:
    """
    Robust dataclass wrapper for MOABB datasets with proper initialization and data handling.
    
    Args:
        dataset: Dataset class
        paradigm: Paradigm class
        subjects: Number of subjects to load (default: 1, minimum 1)
        freqr: Optional resampling frequency (must be > 0 if provided)
    """
    dataset: BaseDataset
    paradigm: BaseParadigm
    subjects: int | None        = field(default=None)
    freqr: Optional[float]      = field(default=None)

    def __post_init__(self):
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Validation
        self._validate()
        self._dataset   = self.dataset()
        self._paradigm  = self.paradigm(resample=self.freqr)

    def _validate(self) -> None:

        """Validate parameters and initialize dataset/paradigm instances"""
        if not self.subjects is None:
            if self.subjects < 1:
                raise ValueError("Number of subjects must be at least 1")

        if self.freqr is not None and self.freqr <= 0:
            raise ValueError("Resampling frequency must be positive if provided")
        if self.freqr is None:
            warnings.warn(
                "Using default sampling frequency. Consider specifying freqr for resampling.",
                UserWarning
            )

    def get_data(self) -> MOABBData:
        """
        Get data from dataset.
        """
        # Load data for specified number of subjects
        if not self.subjects is None:
            subject_ids = list(range(1, self.subjects + 1))
        else:
            subject_ids = None

        x, y, meta = self._paradigm.get_data(dataset=self._dataset,
                                          subjects=subject_ids,
                                         )
        
        self.logger.info("Data extracted from MOABB.")

        return MOABBData(
                x=x,
                y=y,
                subjects=meta['subject'].to_numpy(),
                paradigm=self.paradigm,
                dataset=self.dataset,
                channel_names=[], # self.paradigm.channels, TODO: CHANGE THIS LATER
                freqr=self.freqr
            )