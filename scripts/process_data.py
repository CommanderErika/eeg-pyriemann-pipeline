import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import yaml

from src.processing import ProcessingPipeline

# Configure logging once at application start
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

if __name__ == "__main__":

    # Configuration variables
    DATA_RAW            = "./data/raw/"
    DATA_COV            = "./data/processed/cov/"
    DATA_TS             = "./data/processed/ts/"

    # Setting Logger
    logger = logging.getLogger(__name__)
    
    # Processing Data
    processer = ProcessingPipeline(raw_dir=DATA_RAW, cov_dir=DATA_COV, ts_dir=DATA_TS)
    processer.process_all()
    




    