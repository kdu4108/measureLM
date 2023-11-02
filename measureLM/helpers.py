"""
helper functions
"""

import os
import numpy as np
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


class LDA:

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )
        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]

            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T

        idxs = np.argsort(abs(eigenvalues))[::-1]
        self.eigenvals = eigenvalues[idxs]
        self.eigenvecs = eigenvectors[idxs]
        self.inverse_eigenvecs = np.linalg.inv(eigenvectors[idxs])


if __name__ == "__main__":
    pass