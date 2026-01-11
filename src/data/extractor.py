from typing import Any, Optional
from dataclasses import dataclass, field
import concurrent.futures
import logging

from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

from .basic_type import MOABBData
from .moabb_wrapper import MOABBwrapper


@dataclass
class DataExtractor:
    dataset_names: list[BaseDataset]
    paradigm: Any
    n_subjects: int | None              = field(default=None)
    resample: bool                      = field(default=False)
    freqr: Optional[float]              = field(default=None)
    _data: dict[str, Any]               = field(init=False, default_factory=dict)

    def __post_init__(self,):
        # Initialize logger with class-specific name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("DataExtractor initialized with %d datasets", len(self.dataset_names))

        # Validating configurations
        self._validate()
        # Download all datasets
        self._get_data()

    @property
    def data(self) -> dict[str, Any]:
        """Return a copy of the downloaded data"""
        return self._data.copy()
    
    def _validate(self):
        """Validate configuration parameters"""
        self.logger.debug("Validating configuration parameters")
        
        if self.resample and self.freqr is None:
            error_msg = "freqr must be provided when resample=True"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not self.resample and self.freqr is not None:
            warning_msg = "freqr is provided but resample=False - this parameter will be ignored"
            self.logger.warning(warning_msg)
        
        if not self.n_subjects is None:
            if self.n_subjects < 1:
                error_msg = f"n_subjects must be at least 1, got {self.n_subjects}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
        
    def _process_dataset(self, name: BaseDataset) -> MOABBData:
        """Download one dataset"""
        return MOABBwrapper(
            dataset=name,
            paradigm=self.paradigm,
            subjects=self.n_subjects,
            freqr=self.freqr
        ).get_data()

    def _get_data(self):

        self.logger.info(f"Downloading Datasets {self.dataset_names}")

        for name in self.dataset_names:
            # Tries to download each data
            try:
                print(name.__name__)
                self._data[name.__name__] = self._process_dataset(name)
                self.logger.info(f"The dataset {name.__name__} was download successfully.")
            except:
                self.logger.warning(f"It was not possible to download {name.__name__} dataset.")
        