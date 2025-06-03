from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm


@dataclass
class MOABBData:
    """
    Data struct to MOABB datasets.
    """
    x: np.ndarray
    y: np.ndarray
    dataset: BaseDataset
    paradigm: BaseParadigm
    channel_names: list[str]
    sfreq: float | int

class MOABBwrapper:
    """
    Wrapper to MOABB datasets.
    """
    def __init__(
        self,
        dataset: BaseDataset, # list[base.BaseDataset] | str, # TODO Later let this list also be list of strings
        paradigm: BaseParadigm,
        config: Optional[dict] = None
    ):

        self.dataset = dataset() # [dataset() for dataset in datasets]
        self.paradigm = paradigm()
        self.config = config or {}
        self._validate_config()

    def _validate_config(self):
        """
        Validate dict configurations.
        """
        if "n_subjects" not in self.config:
            Warning("n_subject were not defined")
        
    def get_data(self) -> MOABBData:
        """
        Get data from dataset.
        """

        data = self.paradigm.get_data(dataset=self.dataset, subjects=[1])

        x, y, metadata = data[0], data[1], data[2]

        return MOABBData(
                x=x,
                y=y,
                paradigm=self.paradigm,
                dataset=self.dataset,
                channel_names=self.paradigm.channels,
                sfreq=512 # self.dataset.fs # self.paradigm.fs
            )
    
@dataclass
class DataGetter:
    dataset_names: list[BaseDataset]
    paradigm: BaseParadigm
    _data: dict[str, any] = field(init=False, default_factory=dict)

    def __post_init__(self,):
        # Logger
        self.logger = logging.getLogger(__name__)
        # Download all datasets
        self.__get_all_data()

    @property
    def data(self) -> dict[str, any]:
        return self._data

    def __get_all_data(self,):
        
        for name in self.dataset_names:
            # Tries to download each data
            try:
                self._data[name.__name__] = MOABBwrapper(dataset=name, paradigm=self.paradigm).get_data()
                self.logger.info(f"The dataset {name.__name__} was download successfully.")
            except:
                self.logger.warning(f"It was not possible to download {name.__name__} dataset.")

    

