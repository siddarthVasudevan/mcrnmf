API Reference
=============

The ``mcrnmf`` package exposes two primary modules:

1. ``models`` - contains core decomposition algorithms
2. ``nnls`` - provides a standalone solver for estimating concentration profiles :math:`H`
   given fixed spectra :math:`W`

models
------

The ``models`` module provides implementations of various Nonnegative Matrix
Factorization (NMF) algorithms used in Multivariate Curve Resolution (MCR).

.. autosummary::
   :toctree: generated
   :signatures: none
   :caption: Models

   mcrnmf.models.FroALS
   mcrnmf.models.FroFPGM
   mcrnmf.models.MinVol
   mcrnmf.models.SNPA

nnls
----

The ``nnls`` module offers a fast projected gradient solver to estimate :math:`H`
from new data :math:`X` when :math:`W` is fixed and known.

.. autosummary::
   :toctree: generated
   :signatures: none
   :caption: NNLS

   mcrnmf.nnls.FPGM
