from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from moabb.datasets import base
from moabb.paradigms import Paradigm


@dataclass
class MOABBData:
    """
    Data struct to MOABB datasets.
    """
    x: np.ndarray
    y: np.ndarray
    dataset: base.BaseDataset
    paradigm: Paradigm
    channel_names: list[str]
    sfreq: float | int

class MOABBwrapper:
    """
    Wrapper to MOABB datasets.
    """
    def __init__(
        self,
        datasets: list[base.BaseDataset] | str, # TODO Later let this list also be list of strings
        paradigm: Paradigm,
        config: Optional[dict] = None
    ):

        self.datasets = [dataset() for dataset in datasets]
        self.paradigm = paradigm
        self.config = config or {}
        self._validate_config()

    def _validate_config(self):
        """
        Validate dict configurations.
        """
        if "n_subjects" not in self.config:
            raise Warning("n_subject were not defined")
        
    def get_data(self) -> list[MOABBData] | dict[MOABBData]:
        """
        Get data from all datasets.
        """
        results = {}

        for dataset in self.datasets:
            data = self.paradigm.get_data(dataset=dataset, subjects=[1])

            x, y, metadata = data.X, data.y, data.metadata

            results[dataset.code] = MOABBData(
                x=x,
                y=y,
                paradigm=self.paradigm,
                channel_names=self.paradigm.channels,
                sfreq=self.paradigm.fs
            )
    
        return results if len(results) > 1 else next(iter(results.values()))