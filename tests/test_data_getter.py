import pytest
from unittest.mock import Mock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moabb_wrapper import DataGetter
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm

class MockDataset(BaseDataset):
    def __init__(self):
        super().__init__(subjects=[1], code="MockDataset")

def test_successful_download():
    # Mock MOABBwrapper to avoid real downloads
    with patch("moabb_wrapper.MOABBwrapper") as mock_wrapper:
        # Setup mock to return dummy data
        mock_wrapper.return_value.get_data.return_value = [1, 2, 3] # {"data": [1, 2, 3]}
        
        # Initialize DataGetter
        getter = DataGetter(
            dataset_names=[MockDataset],
            paradigm=BaseParadigm,
        )
        
        # Verify data was stored
        assert "MockDataset" in getter.data
        assert getter.data["MockDataset"] == [1, 2, 3] # {"data": [1, 2, 3]}

def test_failed_download():
    with patch("moabb_wrapper.MOABBwrapper") as mock_wrapper:
        # Force an exception
        mock_wrapper.return_value.get_data.side_effect = Exception("Download failed")
        
        getter = DataGetter(
            dataset_names=[MockDataset],
            paradigm=BaseParadigm
        )
        
        # Verify the dataset was marked as failed
        assert "MockDataset" not in getter.data.keys()