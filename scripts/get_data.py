import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import logging

from src.data import DataExtractor
from src.data import HDF5Manager

from src.utils import handle_datasets, get_paradigm

# Configure logging once at application start
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

if __name__ == "__main__":

    # Configuration variables
    # That works: PhysionetMI, Cho2017, Liu2024, Zhou2016, Lee2019_MI, Zhou2016
    DATASETS            = ["Cho2017", "Zhou2016", "BNCI2014_004", "BNCI2014_001",
                           "BNCI2014_002", "AlexMI", "Liu2024", "Lee2019_MI", 
                           "PhysionetMI"]
    PARADIGM            = "LeftRightImagery"
    FREQR               = 500               # Must be equal for all
    N_SUBJ              = None              # If set None, will get all data
    FORCE_SAVE          = False
    DATA_RAW            = "./data/raw/"

    # Setting Logger
    logger = logging.getLogger(__name__)
    # Manager
    save_manager = HDF5Manager()
    # Getting data
    # TODO: Put the handler inside DataExtractor
    datasets = handle_datasets(DATASETS)
    paradigm = get_paradigm(PARADIGM)
    # TODO: Concurrent Download
    downloader = DataExtractor(dataset_names=datasets, 
                               paradigm=paradigm, 
                               n_subjects=N_SUBJ,
                               freqr=FREQR, 
                               resample=True)
    data = downloader.data
    # Saving each dataset
    for dt in data.keys():
        filename: str = f'{DATA_RAW}/{dt}'
        save_manager.save(data=data[dt], filename=filename, data_type='moabb')