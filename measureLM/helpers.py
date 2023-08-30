"""
helper functions
"""

import os
import matplotlib

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DEVICE="GPU"