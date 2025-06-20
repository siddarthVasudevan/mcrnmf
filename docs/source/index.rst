McrNmf Documentation
====================

Welcome to the documentation for **McrNmf**, a Python package for Multivariate Curve
Resolution (MCR) using different Nonnegative Matrix Factorization (NMF) algorithms.

MCR is widely used in chemometrics to decompose mixture spectra into their pure
component spectra and associated concentration profiles.

Key Features
------------

- **Multiple solvers** -- unified interface for three NMF variants:
  classic Alternating Least Squares (:class:`~mcrnmf.models.FroALS`),
  Fast Projected Gradient (:class:`~mcrnmf.models.FroFPGM`), and
  Minimum-Volume formulation (:class:`~mcrnmf.models.MinVol`)
- **Built-in constraints** -- supports closure, normalization, equality, and per-component
  unimodality constraints
- **Robust initialisation** -- includes :class:`~mcrnmf.models.SNPA` algorithm for generating
  reliable starting estimates of spectra and concentrations
- **Lightweight implementation** -- written almost entirely in `NumPy <https://numpy.org/doc/stable/>`__,
  with `Numba <https://numba.readthedocs.io/en/stable/>`__ used only to speed up unimodal regression


Installation
------------

.. code-block:: bash

   pip install mcrnmf

Basic Usage
-----------

.. code-block:: python

   from mcrnmf import MinVol, SNPA
   from mcrnmf.datasets import load_rxn_spectra

   # Load example data
   X, wv, time = load_rxn_spectra()

   # Get initial guess using SNPA
   snpa = SNPA(rank=4)
   snpa.fit(X)
   Wi = snpa.W.copy()
   Hi = snpa.H.copy()

   # Initialize MinVol with SNPA results and apply constraints
   model = MinVol(rank=4, constraint_kind=1)
   model.fit(X=X, Wi=Wi, Hi=Hi)

   # Access results
   W = model.W  # Pure component spectra
   H = model.H  # Concentration profiles

Next Steps
----------

- See the :doc:`usage/index` page for worked examples
- Browse the :doc:`api` for full API documentation
- Check the :doc:`installation` page for detailed setup info

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   usage/index
   api
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`