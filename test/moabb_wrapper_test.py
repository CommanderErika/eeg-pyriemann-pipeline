import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moabb_wrapper import MOABBData, MOABBwrapper
