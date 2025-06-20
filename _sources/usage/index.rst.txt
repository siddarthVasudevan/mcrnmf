Usage Guide
===========

MCR-NMF Fundamentals and Matrix Decomposition
---------------------------------------------

MCR-NMF decomposes a data matrix :math:`X` into two non-negative matrices :math:`W` and
:math:`H`, where :math:`W` represents the pure component spectra and :math:`H` represents
the concentration profiles over time.

.. figure:: ../_static/figures/decomp_visual.svg
   :align: center
   :alt: NMF matrix decomposition in chemometrics
   :width: 80%

.. note::
   Throughout this documentation, we use certain terms interchangeably:

   * :math:`W` matrix = pure component spectra (one spectrum per column)
   * :math:`H` matrix = concentration profiles (one profile per row)

   This interpretation assumes the data matrix :math:`X` is organized with
   wavelengths/wavenumbers as rows and time points as columns. If your input is
   transposed, adjust accordingly.

Understanding Constraints
-------------------------
The package offers three core models for NMF decomposition: :class:`~mcrnmf.models.FroALS`,
:class:`~mcrnmf.models.FroFPGM`, and :class:`~mcrnmf.models.MinVol`. All of them support
physical and chemical constraints, including **equality**, **closure**, **normalization**,
and **unimodality**.

Several types of constraints can be applied using the ``constraint_kind`` parameter.
These constraints act on either the spectra matrix :math:`W` or the concentration matrix
:math:`H`, and enforce physically meaningful structure in the decomposition.

.. list-table:: Available Constraint Options via ``constraint_kind``
   :widths: 10 30 60
   :header-rows: 1

   * - Value
     - Mathematical Form
     - Interpretation
   * - 0
     - :math:`W \geq 0`, :math:`H \geq 0`
     - Basic non-negativity (default)
   * - 1
     - :math:`H^T e \leq e`, non-negativity
     - **Relaxed closure**: sum of concentrations ≤ 1 at each time point
       (allows unmodeled components)
   * - 2
     - :math:`H e = e`, non-negativity
     - **Row normalization**: each component's total abundance is scaled to 1
   * - 3
     - :math:`W^T e = e`, non-negativity
     - **Spectral normalization**: each pure component spectrum sums to 1
   * - 4
     - :math:`H^T e = e`, non-negativity
     - **Strict closure**: sum of concentrations = 1 at each time point

These constraints influence the chemical interpretability of the results. For example,
``constraint_kind=1`` or ``constraint_kind=4`` are typically appropriate for
reaction monitoring data where the major components are captured.

.. note::
   Constraints are applied in a fixed sequence during optimization:

   1. Any known values of :math:`H` and :math:`W` are imposed first.
   2. Then, unimodality constraints (if any) are enforced on the rows of :math:`H` and/or
      columns of :math:`W`.
   3. Finally, closure or normalization constraints (as specified by ``constraint_kind``)
      are applied.

   Because these operations are applied sequentially, some combinations of constraints
   may conflict or override others. For example, applying both unimodality and strict
   closure may result in only partial satisfaction of unimodal constraint.

   See the :doc:`advanced` section for a detailed example of constraint conflicts and
   practical workarounds.

Detailed Guides
---------------
The following guides walk you through core and advanced workflows:

- **Basic Workflow** guide shows step-by-step tutorial from raw spectra to component
  resolution using SNPA and MinVol.
- **Advanced Topics** guide explains constraint conflicts, incorporating component
  specific unimodality constraint and known concentrations.
- **Predict Concentration Profiles from Fixed Spectra** guide demonstrates how to
  reuse a previously estimated :math:`W` matrix to predict :math:`H` for new datasets.

.. toctree::
   :maxdepth: 1

   workflow
   advanced
   reuse_w.rst