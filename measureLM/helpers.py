"""
helper functions
"""

import os
import matplotlib
from pathlib import Path
import collections, gc, resource, torch

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
DEVICE="GPU"


def empty_gpu_cache():
    if torch.cuda.is_available():
        select_device = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        select_device = f"mps:0"
    else:
        select_device = f"cpu"
    gc.collect()
    print("empty cache at: " + str(select_device))


if __name__ == "__main__":
    pass