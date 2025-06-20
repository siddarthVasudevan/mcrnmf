Predicting Concentration Profiles from Fixed Spectra
====================================================

After fitting an MCR-NMF model using :class:`~mcrnmf.models.MinVol`,
:class:`~mcrnmf.models.FroALS`, or :class:`~mcrnmf.models.FroFPGM`, you may want to reuse
the estimated :math:`W` matrix (pure component spectra) to estimate :math:`H`
(concentration profiles) for new measurements :math:`X`.

The :class:`~mcrnmf.nnls.FPGM` class provides a fast projected gradient method to solve
the following problem:

.. math::

    \min_{H \geq 0} \; \|X - WH\|_F^2

It supports the same chemical constraints on :math:`H` used in the main models:
closure, normalization, unimodality, and fixed known values.

This makes it ideal for downstream applications like:

* Reaction monitoring where new data arrives over time
* High-throughput screening where spectra from new mixtures are analyzed
* Scenarios where pure components are fixed and known from prior runs

Basic Usage Example
-------------------

Here's how to reuse a previously estimated :math:`W` to estimate :math:`H` for new data.

.. code-block:: python

    from mcrnmf.datasets import load_rxn_spectra
    from mcrnmf import SNPA, MinVol, FPGM

    # Load the reference dataset
    X, wv, time = load_rxn_spectra()
    # splitting the X into two subsets for illustration
    X_train, X_new = X[:, :-5], X[:, -5:]

    num_components = 4
    # Use SNPA to generate initial estimate for W and H from X_train
    snpa = SNPA(rank=num_components)
    snpa.fit(X_train)
    Wi, Hi = snpa.W, snpa.H

    # estimate W from X_train using MinVol
    mvol = MinVol(
        rank=num_components,
        constraint_kind=1,
        unimodal={"H": True},
        lambdaa=1e-4,
        iter_max=2000
    )
    mvol.fit(X=X_train, Wi=Wi, Hi=Hi)
    # the value of W we will use for predicting H for X_new
    W = mvol.W

    # Create FPGM solver with same constraints on H as specified for MinVol
    fpgm = FPGM(constraint_kind=1, tol=1e-4, iter_max=500)

    # Solve for H on new data with unimodality
    H_new, converged = fpgm.solve(X=X_new, W=W, unimodal_H=True)

    print(f"Converged: {converged}")
