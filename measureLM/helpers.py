"""
helper functions
"""

import os
import matplotlib
from pathlib import Path

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
DEVICE="GPU"


if __name__ == "__main__":
    pass