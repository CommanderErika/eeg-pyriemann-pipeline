import os
import logging
import logging.config

import numpy
import mne
import moabb
from moabb.datasets import FakeDataset, BNCI2014_004, BNCI2014_001, Cho2017
from moabb.paradigms import LeftRightImagery

from src.moabb_wrapper import DataGetter
from src import utils

# Global Variables
CWD: str = os.getcwd()
LOGGING_CONFIG: dict = { 
    'version': 1.0,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            # 'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file' : {
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/test.log'
        }
    },
    'loggers': { 
        #'': {  # root logger
        #    'handlers': ['default'],
        #    'level': 'WARNING',
        #    'propagate': False
        #},
        'src.moabb_wrapper': { 
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}

logging.config.dictConfig(LOGGING_CONFIG)

if __name__ == "__main__":

    # Setting Logger
    logger = logging.getLogger(__name__)
    logger.info("Configured Logger")
    # Example Datasets to be downloaded
    datasets = [Cho2017, BNCI2014_001]
    paradigm = LeftRightImagery
    # Getting data
    downloader = DataGetter(dataset_names=datasets, paradigm=paradigm)
    data = downloader.data
    print(data.keys())
    # Processing Data




    