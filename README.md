# McrNmf

**McrNmf** is a Python package for Multivariate Curve Resolution (MCR) using a range of
Nonnegative Matrix Factorization (NMF) algorithms. MCR is widely used in chemometrics,
for instance, to decompose mixture spectra into their pure component spectra and
associated concentration profiles.

## Key Features

- **Multiple solvers** – unified interface for three NMF variants: classic Alternating
  Least Squares (FroALS), Fast Projected Gradient (FroFPGM), and Minimum-Volume
  formulation (MinVol).
- **Built-in constraints** – supports closure, normalization, equality, and per-component
  unimodality constraints.
- **Robust initialisation** – includes Successive Nonnegative Projection (SNPA) algorithm
  for generating reliable starting estimates of spectra and concentrations
- **Lightweight implementation** – written almost entirely in NumPy, with Numba used
  only to speed up unimodal regression.

For more details, see the [documentation](https://siddarthvasudevan.github.io/mcrnmf/).

---

## Installation

With pip:

```bash
pip install mcrnmf
```

With conda:

```
conda install -c conda-forge mcrnmf
```

`mcrnmf` requires:
- Python ≥ 3.11.9, < 3.14
- NumPy ≥ 1.24, < 2.3
- Numba ≥ 0.61.2

---
## Quick Start

```python
from mcrnmf.datasets import load_rxn_spectra
from mcrnmf import SNPA, MinVol

# load a Raman spectroscopy data (wv × time)
X, wv, t = load_rxn_spectra()

# specify the number of components
rank = 4

# generate initial guess for W and H using SNPA
snpa = SNPA(rank=rank, iter_max=1000)
snpa.fit(X)
Wi = snpa.W  # intial guess for W (pure spectra)
Hi = snpa.H  # intial guess for H (conc. profiles)


mvol = MinVol(rank=rank,
              constraint_kind=1,
              unimodality={"H": True},
              iter_max=2000,
              tol=1e-4)
# fit the Minimum Volume NMF model
mvol.fit(X=X, Wi=Wi, Hi=Hi)
# access the decomposed factors
W = mvol.W
H = mvol.H
# access relative reconstruction error at each iter.
rel_recon_err = mvol.rel_reconstruction_error_ls

```

More worked examples can be found in the [Usage](https://siddarthvasudevan.github.io/mcrnmf/usage/index.html) section of the documentation.

---
## License

MIT – see [LICENSE](LICENSE).
