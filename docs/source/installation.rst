Installation
============

MCR-NMF can be installed using either `pip` or `conda`. Both methods will install the
package along with its dependencies.

With pip
--------

To install the latest release from PyPI:

.. code-block:: bash

   pip install mcrnmf

With conda
----------

To install via `conda` from the `conda-forge` channel:

.. code-block:: bash

   conda install -c conda-forge mcrnmf

From Source
-----------

To install from the latest source:

.. code-block:: bash

   git clone https://github.com/siddarthVasudevan/mcrnmf.git
   cd mcrnmf
   pip install .

Requirements
------------

The package requires the following:

* Python ≥ 3.11.9, < 3.14
* NumPy ≥ 1.24, < 2.3
* Numba ≥ 0.61.2

.. note::

   It is recommended to install in a clean virtual environment (via `venv` or `conda`)
   to avoid dependency conflicts.
