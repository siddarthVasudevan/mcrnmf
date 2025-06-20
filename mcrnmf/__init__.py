"""
MCR-NMF: Multivariate Curve Resolution with Non-negative Matrix Factorization

A Python package implementing various non-negative matrix factorization (NMF) algorithms
for multivariate curve resolution (MCR) applications.
"""

# To get sub-modules
from .models import FroALS, FroFPGM, MinVol, SNPA
from .nnls import FPGM

__version__ = "0.1.0"

__all__ = ["FroALS", "FroFPGM", "MinVol", "SNPA", "FPGM"]
