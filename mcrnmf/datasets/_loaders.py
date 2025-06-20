import os

import numpy as np


def load_rxn_spectra():
    """
    Loads the raman spectral dataset of a reaction.

    Returns
    -------
    X : np.ndarray of shape (n_wavenumbers, n_samples)
        The spectral data.
    wv : np.ndarray of shape (n_wavenumbers,)
        The wavenumbers corresponding to the spectra.
    time : np.ndarray of shape (n_samples,)
        The time at which the samples were acquired.

    """
    DATA_DIR_PATH = os.path.join(os.path.dirname(__file__))

    fpath = os.path.join(DATA_DIR_PATH, "raman.csv")
    data = np.genfromtxt(fpath, delimiter=",", skip_header=1)

    time = np.genfromtxt(
        fpath, delimiter=",", max_rows=1, missing_values="", filling_values=np.nan
    )[1:]
    wv = data[:, 0].astype(dtype=np.float64)
    X = data[:, 1:].astype(dtype=np.float64)

    return X, wv, time
