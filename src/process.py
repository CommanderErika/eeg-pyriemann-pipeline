import os
import logging
from dataclasses import dataclass, field

import numpy
import dask
import pyriemann


@dataclass
class ProcessData:
    data_by_datasets: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        # Logger
        self.logger = logging.getLogger(__name__)

    def __process(self):
        pass