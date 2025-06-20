from abc import ABC, abstractmethod
import warnings

import numpy as np
from numpy.typing import NDArray

from ._utils import unimodal_reg_l2, simplex_proj, simplex_proj_using_axis

_EPSILON = np.finfo(np.float64).eps


class _BaseNMF(ABC):
    """
    Base class for Non-negative Matrix Factorization (NMF) algorithms.

    This abstract class provides the foundation for various NMF implementations.
    It handles common functionality such as convergence checking, error
    computation, and projection operations for enforcing constraints.

    NMF decomposes a non-negative matrix X into two lower-rank non-negative
    matrices: X ≈ W * H, where W and H are the factor matrices.

    Parameters
    ----------
    rank : int
        The rank of the factorization (number of components).
    constraint_kind : int, optional
        The type of constraint to apply (default is 0):
        - 0: Default, Non-negativity constraints only (W ≥ 0, H ≥ 0)
        - 1: Simplex constraint (H^T e ≤ e)
        - 2: Row simplex constraint (H e = e)
        - 3: Column simplex constraint on W (W^T e = e)
        - 4: Column simplex constraint on H (H^T e = e)
    unimodal : dict or None, default=None
        Specifies unimodality constraints for W and H matrices.
        If None, no unimodality constraints are applied.

        If dict, Format: {'W': [bool list or bool], 'H': [bool list or bool]}
        * Must contain at least one key ('W' or 'H')
        * No other keys besides 'W' and 'H' are allowed
        * For 'W': Controls unimodality of columns in W
        * For 'H': Controls unimodality of rows in H

        Each value can be:

        * A boolean: applies to all components
        * A list of booleans: selectively applies to specific components

        Examples:

        * {'W': True, 'H': True} - All :math:`W` and :math:`H` components are unimodal
        * {'H': [True, False, True]} - Only components 0 and 2 of :math:`H` have
          unimodal behavior
        * {'W': [False, True, False]} - Only component 1 of :math:`W` has unimodal
          behavior
    iter_max : int, optional
        Maximum number of outer iterations for the algorithm (default is 500).
    tol : float, optional
        Convergence tolerance for the relative loss (default is 1e-4).

    Methods
    -------
    fit(X, Wi, Hi, known_W=None, known_H=None)
        Fit the NMF model to the provided data matrix.

    Raises
    ------
    TypeError
        If input parameters are not of the correct type.
    ValueError
        If input parameters have invalid values.

    Notes
    -----
    This is an abstract base class. Concrete implementations should override
    the `fit`, `_update_H`, and `_update_W` methods.
    """

    def __init__(
        self,
        rank: int,
        constraint_kind: int = 0,
        unimodal: dict | None = None,
        iter_max: int = 500,
        tol: float = 1e-4,
    ):
        # validate type
        if not isinstance(rank, (int, np.integer)):
            raise TypeError(
                f"`rank` must be of type {int}, currrent type is {type(rank)}"
            )
        if not isinstance(constraint_kind, (int, np.integer)):
            raise TypeError(
                f"`constraint_kind` must be of type {int}, current type is "
                f"{type(constraint_kind)}"
            )
        if unimodal is not None:
            if not isinstance(unimodal, dict):
                raise TypeError(f"`unimodal` must be None or a dictionary")
        if not isinstance(iter_max, (int, np.integer)):
            raise TypeError(
                f"`iter_max` must be of type {int}, current type is {type(iter_max)}"
            )
        if not isinstance(tol, float):
            raise TypeError(
                f"`tol` must be of type {float}, current type is {type(tol)}"
            )

        # validate values
        if rank <= 0:
            raise ValueError("`rank` must be > 0")
        if constraint_kind not in [0, 1, 2, 3, 4]:
            raise ValueError("`constraint_kind` must be 0, 1, 2, 3, or 4")
        # default unimodal constraints if None
        unimodal_W = [False] * rank
        unimodal_H = [False] * rank
        if unimodal is not None:
            # empty dictionary check
            if not unimodal:
                raise ValueError(
                    "If not None, unimodal dictionary cannot be empty; must contain "
                    "'W' and/or 'H' keys."
                )
            # check for required keys
            valid_keys = ["W", "H"]
            provided_keys = list(unimodal.keys())
            for provided_key in provided_keys:
                if provided_key not in valid_keys:
                    raise KeyError(
                        f"`unimodal` dictionary can contain only keys in: {valid_keys} "
                        f"got {provided_key}."
                    )
            if "W" in unimodal:
                unimodal_W = self._process_unimodal_arg(unimodal["W"], rank)
            if "H" in unimodal:
                unimodal_H = self._process_unimodal_arg(unimodal["H"], rank)
        if iter_max < 10:
            raise ValueError("`iter_max` must be >= 10")
        if not (0 < tol < 1):
            raise ValueError("`tol` must be in the open interval (0, 1)")

        self.rank_ = rank
        self.constraint_kind_ = constraint_kind
        self.unimodal_W_ = unimodal_W
        self.unimodal_H_ = unimodal_H
        self.iter_max_ = iter_max
        self.tol_ = tol

        # initialize attributes
        self._W = None
        self._H = None
        self._rel_loss_ls = None
        self._is_converged = None

    @property
    def W(self):
        """
        The basis matrix :math:`W` obtained after fitting the model.

        Returns
        -------
        ndarray of shape (n_features, rank)
            The obtained :math:`W` after fitting the model.
        """
        if self._W is None:
            raise AttributeError(
                "`W` is not available. Model has not been fitted yet. Call 'fit' first."
            )
        return self._W

    @property
    def H(self):
        """
        The coefficient matrix :math:`H` obtained after fitting the model.

        Returns
        -------
        ndarray of shape (rank, n_samples)
            The obtained :math:`H` after fitting the model.
        """
        if self._H is None:
            raise AttributeError(
                "`H` is not available. Model has not been fitted yet. Call 'fit' first."
            )
        return self._H

    @property
    def rel_loss_ls(self):
        """
        List of relative loss values from each iteration during model fitting.

        It is defined as:

        .. math::

            \\dfrac{\\sqrt{f_{\\textrm{obj}}^{i}}}{||X||_F}\\ \\textrm{,}

        where:

        * :math:`||\\cdot||_F` denotes the Frobenius norm
        * :math:`X` is the original data matrix
        * :math:`f_{\\textrm{obj}}^{i}` is the value of objective function after
          iteration :math:`i`

        Returns
        -------
        list of float
            Relative loss value from each iteration.
        """
        if self._rel_loss_ls is None:
            raise AttributeError(
                "`rel_loss_ls` is not available. Model has not been fitted yet. Call "
                "`fit` first."
            )
        return self._rel_loss_ls

    @property
    def is_converged(self):
        """
        The convergence status.

        Returns
        -------
        bool
            Whether the algorithm converged within ``iter_max`` iterations.

            * ``True`` if convergence was reached based on ``tol`` criterion
            * ``False`` if maximum iterations were reached without convergence
        """
        if self._is_converged is None:
            raise AttributeError(
                "`is_converged` is not available. Model has not been fitted yet. Call "
                "`fit` first."
            )
        return self._is_converged

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float64],
        Wi: NDArray[np.float64],
        Hi: NDArray[np.float64],
    ) -> None:
        """
        Fit the NMF model to the provided data matrix.

        This method must be implemented by derived classes. Different NMF
        algorithms implement this method differently.
        """
        self._W = Wi.copy()
        self._H = Hi.copy()
        self._rel_loss_ls = []
        self._is_converged = False

    @abstractmethod
    def _update_H(self) -> None:
        """
        Update the H matrix based on the current W matrix.

        This method must be implemented by derived classes. Different NMF
        algorithms implement this method differently.
        """
        pass

    @abstractmethod
    def _update_W(self) -> None:
        """
        Update the W matrix based on the current H matrix.

        This method must be implemented by derived classes. Different NMF
        algorithms implement this method differently.
        """
        pass

    def _process_unimodal_arg(self, constraint: bool | list[bool], rank: int):
        """
        Process unimodal constraint for either H or W key into a boolean list of length
        rank.
        """
        if isinstance(constraint, bool):
            return [constraint] * rank
        elif isinstance(constraint, list):
            if len(constraint) != rank:
                raise ValueError(f"Unimodal constraint list must have length {rank}.")
            if not all(isinstance(c, bool) for c in constraint):
                raise TypeError("Unimodal constraint list must contain only booleans.")
            return constraint
        else:
            raise TypeError("Unimodal constraint must be bool or list of booleans")

    def _apply_unimodal_H_rows(self) -> None:
        """Apply unimodal constraints to rows of H."""
        nrows, ncols = self._H.shape
        w = np.ones(ncols)
        for row in range(nrows):
            if self.unimodal_H_[row] is True:
                self._H[row, :] = unimodal_reg_l2(y=self._H[row, :], w=w)

    def _apply_unimodal_W_cols(self) -> None:
        """Apply unimodal constraints to columns of W."""
        nrows, ncols = self._W.shape
        w = np.ones(nrows)
        for col in range(ncols):
            if self.unimodal_W_[col]:
                self._W[:, col] = unimodal_reg_l2(y=self._W[:, col], w=w)

    def _check_convergence(self, iter: int) -> bool:
        """
        Check if the algorithm has converged.

        Parameters
        ----------
        iter : int
            Current iteration number.

        Returns
        -------
        bool
            True if the algorithm has converged, False otherwise.
        """
        if iter < 10:
            is_converged = False
        else:
            is_converged = (
                np.abs(self._rel_loss_ls[iter - 10] - self._rel_loss_ls[iter])
                / np.maximum(_EPSILON, self._rel_loss_ls[iter])
                < self.tol_
            )

        return is_converged

    def _fro_err_sq_after_W_update(
        self, norm_X: float, HHt: NDArray[np.float64], XHt: NDArray[np.float64]
    ) -> float:
        """
        Calculate the squared Frobenius error after W update.

        Computes ||X - W*H||_F^2 efficiently using precomputed matrices.

        Parameters
        ----------
        norm_X : float
            Frobenius norm of X.
        HHt : ndarray
            Precomputed H * H^T matrix.
        XHt : ndarray
            Precomputed X * H^T matrix.

        Returns
        -------
        float
            Squared Frobenius error.
        """
        fro_error_sq = np.maximum(
            _EPSILON,
            norm_X**2 - 2 * np.sum(XHt * self._W) + np.sum(HHt * (self._W.T @ self._W)),
        )

        return fro_error_sq

    def _fro_err_sq_after_H_update(
        self, norm_X: float, WtW: NDArray[np.float64], WtX: NDArray[np.float64]
    ) -> float:
        """
        Calculate the squared Frobenius error after H update.

        Computes ||X - W*H||_F^2 efficiently using precomputed matrices.

        Parameters
        ----------
        norm_X : float
            Frobenius norm of X.
        WtW : ndarray
            Precomputed W^T * W matrix.
        WtX : ndarray
            Precomputed W^T * X matrix.

        Returns
        -------
        float
            Squared Frobenius error.
        """
        fro_error_sq = np.maximum(
            _EPSILON,
            norm_X**2 - 2 * np.sum(WtX * self._H) + np.sum(WtW * (self._H @ self._H.T)),
        )

        return fro_error_sq

    def _project_H(self):
        """
        Project the coefficient matrix H onto the constraint set.

        Applies different projections to the H matrix based on the specified
        constraint type. The projection ensures that H satisfies the constraints
        required for the non-negative matrix factorization problem.

        The following constraints are implemented:
        - constraint_kind = 0: Non-negativity constraint (H >= 0)
        - constraint_kind = 1: Simplex constraint (H^T e <= e)
        - constraint_kind = 2: Row simplex constraint (H e = e)
        - constraint_kind = 3: Non-negativity constraint (H >= 0, same as 0)
        - constraint_kind = 4: Column simplex constraint (H^T e = e)
        """
        if self.constraint_kind_ in [0, 3]:
            self._H[:] = np.maximum(self._H, 0.0)
        elif self.constraint_kind_ == 1:
            simplex_proj(self._H)
        elif self.constraint_kind_ == 2:
            self._H[:] = simplex_proj_using_axis(B=self._H, is_col=False)
        elif self.constraint_kind_ == 4:
            self._H[:] = simplex_proj_using_axis(B=self._H, is_col=True)

    def _project_W(self):
        """
        Project the basis matrix W onto the constraint set.

        Applies different projections to the W matrix based on the specified
        constraint type. The projection ensures that W satisfies the constraints
        required for the non-negative matrix factorization problem.

        The following constraints are implemented:
        - constraint_kind = 3: Column simplex constraint (W >= 0, W^T e = e)
        - Any other value: Non-negativity constraint (W >= 0)
        """
        if self.constraint_kind_ == 3:
            self._W[:] = simplex_proj_using_axis(B=self._W, is_col=True)
        else:
            self._W[:] = np.maximum(self._W, 0.0)

    def _validate_fit_args(
        self,
        X: NDArray,
        Wi: NDArray,
        Hi: NDArray,
        known_W: NDArray | None,
        known_H: NDArray | None,
    ):
        """
        Validate input arguments for matrix factorization.

        This function checks the dimensions and values of the input matrices
        to ensure they meet the requirements for non-negative matrix factorization.

        Parameters
        ----------
        X : ndarray
            The input data matrix to be factorized.
        Wi : ndarray
            Initial guess for the W matrix.
        Hi : ndarray
            Initial guess for the H matrix.
        known_W : ndarray or None
            Matrix with known values of W.
        known_H : ndarray or None
            Matrix with known values of H.

        Raises
        ------
        TypeError
            If any input matrix is not a numpy array or has incorrect dtype.
        ValueError
            If any dimension compatibility check fails or if any matrix
            contains negative values.

        Warns
        -----
        UserWarning
            If known_W or known_H contains no finite values.
        """
        # check type
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` must be a numpy array.")
        if not isinstance(Wi, np.ndarray):
            raise TypeError("`Wi` must be a numpy array.")
        if not isinstance(Hi, np.ndarray):
            raise TypeError("`Hi` must be a numpy array.")
        if not ((known_W is None) or (isinstance(known_W, np.ndarray))):
            raise TypeError("`known_W` must be None or a numpy array.")
        if not ((known_H is None) or (isinstance(known_H, np.ndarray))):
            raise TypeError("`known_H` must be None or a numpy array.")

        # check shape matches
        rows_X, cols_X = X.shape
        if Wi.shape[0] != rows_X:
            raise ValueError(
                "Shape mismatch: number of rows of `Wi` not equal to number of rows of "
                "`X`"
            )
        if Wi.shape[1] != self.rank_:
            raise ValueError(
                "Shape mismatch: number of columns of `Wi` not equal to `rank`"
            )
        if Hi.shape[1] != cols_X:
            raise ValueError(
                "Shape mismatch: number of columns of `Hi` not equal to number of "
                "columns of `X`"
            )
        if Hi.shape[0] != self.rank_:
            raise ValueError(
                f"Shape mismatch: number of rows of `Hi` not equal to `rank`"
            )
        # check shape matches between known_W, known_H, initial_W, initial_H
        if (known_W is not None) and (known_W.shape != Wi.shape):
            raise ValueError(f"Shape mismatch: `known_W` shape not equal to `Wi` shape")
        if (known_H is not None) and (known_H.shape != Hi.shape):
            raise ValueError(f"Shape mismatch: `known_H` shape not equal to `Hi` shape")

        # check dtype is np.float64
        if X.dtype != np.float64:
            raise TypeError(
                f"The dtype of `X` elements should be {np.float64}, current dtype is "
                f"{X.dtype}"
            )
        if Wi.dtype != np.float64:
            raise TypeError(
                f"The dtype of `Wi` elements should be {np.float64}, current dtype is "
                f"{Wi.dtype}"
            )
        if Hi.dtype != np.float64:
            raise TypeError(
                f"The dtype of `Hi` elements should be {np.float64}, current dtype is "
                f"{Hi.dtype}"
            )
        # check if non-negative
        if X.min() < 0:
            raise ValueError("All elements of `X` must be >= 0")
        if Wi.min() < 0:
            raise ValueError("All elements of `Wi` must be >= 0")
        if Hi.min() < 0:
            raise ValueError("All elements of `Hi` must be >= 0")
        # check the specific validation arguments
        if known_W is not None:
            mask_fvals_knw_W = np.isfinite(known_W)
            tot_fvals_knw_W = np.sum(mask_fvals_knw_W)
            if tot_fvals_knw_W == 0:  # if all elements are NaN
                warnings.warn(
                    "`known_W` contains no finite values, setting it to None."
                )
                known_W = None
            elif tot_fvals_knw_W == known_W.size:  # if all elements are finite
                raise ValueError(
                    "All values of `W` are known, can't perform NMF. Try `NNLS solver` "
                    "to compute H."
                )
            else:
                # finite elems are non-neg
                if known_W[mask_fvals_knw_W].min() < 0:
                    raise ValueError("All finite elements of `known_W` must be >= 0.")
        if known_H is not None:
            mask_fvals_knw_H = np.isfinite(known_H)
            tot_fvals_knw_H = np.sum(mask_fvals_knw_H)
            if tot_fvals_knw_H == 0:
                warnings.warn(
                    "`known_H` contains no finite values, setting it to None."
                )
                known_H = None
            elif tot_fvals_knw_H == known_H.size:
                raise ValueError(
                    "All values of `H` are known, can't perform NMF. Try `NNLS solver` "
                    "to compute W."
                )
            else:
                # finite elems are non-neg
                if known_H[mask_fvals_knw_H].min() < 0:
                    raise ValueError("All finite elements of `known_H` must be >= 0.")


class FroALS(_BaseNMF):
    """
    Frobenius norm-based Nonnegative Matrix Factorization (NMF) using Alternating Least
    Squares (ALS) method.

    Factorizes a nonnegative matrix :math:`X` into two nonnegative matrices :math:`W`
    and :math:`H` by minimizing the squared Frobenius norm between :math:`X` and product
    :math:`WH` using FPGM algorithm. The following objective function is minimized
    subject to nonnegativity and other optional constraints on :math:`W` and :math:`H`:

    .. math::
        f_{\\textrm{obj}} = ||X - WH||_F^2

    where :math:`||\\cdot||_{F}` is the Frobenius norm.

    Parameters
    ----------
    rank : int
        The number of components for the factorization.
    constraint_kind : integer-like {0, 1, 2, 3, 4}, default=0
        The following constraints are applied based on the integer value specified:

        * If ``0``: Only :math:`W \\geq 0`, :math:`H \\geq 0`.
        * If ``1``: Closure constraint :math:`H^T e ≤ e`.
        * If ``2``: Closure constraint :math:`H e = e`.
        * If ``3``: Constraint :math:`W^T e = e`.
        * If ``4``: Closure constraint :math:`H^T e = e`.

        Note, for 1, 2, 3, and 4 values of ``constraint_kind`` nonnegativity constraints
        are also applied along with the additional constraint specified above.
    unimodal : dict or None, default=None
        Specifies unimodality constraints for :math:`W` and :math:`H` matrices. If
        ``None``, no unimodality constraints are applied.

        If ``dict``, Format: {'W': bool | list of bool, 'H': bool | list of bool}:

            * Must contain at least one key ('W' or 'H')
            * No other keys besides 'W' and 'H' are allowed
            * For 'W': Controls unimodality of columns in W
            * For 'H': Controls unimodality of rows in H

        Each value can be:

            * A boolean: applies to all components
            * A list of booleans: selectively applies to specific components

        Examples:

            * ``{'H': True}``: All :math:`H` components are unimodal
            * ``{'W': True, 'H': True}``: All :math:`W` and :math:`H` components are
              unimodal
            * ``{'H': [True, False, True]}``: Only components 0 and 2 of :math:`H` have
              unimodal behavior
            * ``{'W': [False, True, False]}``: Only component 1 of :math:`W` has
              unimodal behavior
    iter_max : int, default=500
        Maximum number of iterations. It must be greater :math:`\\geq 10`.
    tol : float, default=1e-4
        Tolerance for convergence. Must be in the interval :math:`(0, 1)`.

            Convergence is reached when:

            .. math::

                {|e[i] - e[i-10]| \\over e[i]} \\leq \\textrm{tol}

            where:

            * iteration :math:`i \\geq 10`
            * :math:`e[i]` is the squared relative loss after iteration :math:`i`, which
              is defined as

                .. math::

                    e[i] = {||X - W^{i}H^{i}||_{F}^2 \\over ||X||_{F}^2}

                where :math:`W^{i}` and :math:`H^{i}` is the value of :math:`W` and
                :math:`H`, respectively, after iteration :math:`i`.

    References
    ----------
    .. [1] Gillis, Nicolas. Nonnegative matrix factorization. Society for Industrial and
           Applied Mathematics, 2020.
    .. [2] Van Benthem, Mark H., and Michael R. Keenan. "Fast algorithm for the solution
           of large‐scale non‐negativity‐constrained least squares problems." Journal of
           Chemometrics: A Journal of the Chemometrics Society 18.10 (2004): 441-450.

    Examples
    --------
    >>> from mcrnmf.models import FroALS, SNPA
    >>> from mcrnmf.datasets import load_rxn_spectra
    >>>
    >>> # load the example dataset from mcrnmf
    >>> X, wv, time  = load_rxn_spectra()
    >>>
    >>> # generate initial guess using SNPA
    >>> snpa = SNPA(rank=4)
    >>> snpa.fit(X)
    >>> Wi = snpa.W  # Initial estimate for W
    >>> Hi = snpa.H  # Initial estimate for H
    >>>
    >>> # create an instance of FroALS and fit the model
    >>> model = FroALS(rank=4, constraint_kind=1, iter_max=2000, tol=1e-4)
    >>> model.fit(X, Wi, Hi)
    >>> # access decomposed factors
    >>> W, H = model.W, model.H
    >>> # check convergence status
    >>> converged = model.is_converged
    >>> # access rel. reconstruction error after each iterations
    >>> rel_recon_err = model.rel_reconstruction_error_ls
    """

    def __init__(
        self,
        rank: int,
        constraint_kind: int = 0,
        unimodal: dict | None = None,
        iter_max: int = 500,
        tol: int = 1e-4,
    ):
        super().__init__(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal,
            iter_max=iter_max,
            tol=tol,
        )

    @property
    def rel_reconstruction_error_ls(self):
        """
        List of relative reconstruction errors from each iteration during model fitting.

        The relative reconstruction error measures how well the current factors
        approximate the original data. It is the ratio:

        .. math::
            \\dfrac{||X - W^{i}H^{i}||_F}{||X||_F}\\ \\textrm{,}

        where:

        * :math:`||\\cdot||_F` denotes the Frobenius norm
        * :math:`X` is the original data matrix
        * :math:`W^{i}` and :math:`H^{i}` are values of :math:`W` and :math:`H` after
          iteration :math:`i`

        Returns
        -------
        list of float
            Relative reconstruction error from each iteration.
        """
        if self._rel_loss_ls is None:
            raise AttributeError(
                "`rel_reconstruction_error_ls` is not available. Model has not been "
                "fitted yet. Call 'fit' first."
            )
        return self._rel_loss_ls

    def fit(
        self,
        X: NDArray[np.float64],
        Wi: NDArray[np.float64],
        Hi: NDArray[np.float64],
        known_W: NDArray[np.float64] | None = None,
        known_H: NDArray[np.float64] | None = None,
        preprocess_scale_WH: bool = False,
    ):
        """
        Fit the FroALS model to the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Data array to be factorized.
        Wi : ndarray of shape (n_features, rank)
            Initial guess for the factor :math:`W`.
        Hi : ndarray of shape (rank, n_samples)
            Initial guess for the factor :math:`H`.
        known_W : ndarray of shape (n_features, rank), default=None
            Array containing known values of :math:`W`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`W` which do not
              correspond ``np.nan`` entries in ``known_W``.
        known_H : ndarray of shape (rank, n_samples), default=None
            Array containing known values of :math:`H`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`H` which do not
              correspond ``np.nan`` entries in ``known_H``.
        preprocess_scale_WH : bool, default=False
            If ``True``,  ``Wi`` and ``Hi`` are scaled before optimization.
        """
        self._validate_fit_args(X, Wi, Hi, known_W, known_H)
        self._W = Wi.copy()
        self._H = Hi.copy()
        self._rel_loss_ls = []
        self._is_converged = False

        # Check if known values are passed, if so compute a mask.
        if known_W is not None:
            mask_known_W = np.isfinite(known_W)
        else:
            mask_known_W = None
        if known_H is not None:
            mask_known_H = np.isfinite(known_H)
        else:
            mask_known_H = None
        # preallocate arrays for H updates
        WtW = np.zeros((self.rank_, self.rank_))
        WtX = np.zeros((self.rank_, X.shape[1]))

        # preallocate arrays for W updates
        XHt = X @ self._H.T
        HHt = self._H @ self._H.T
        if preprocess_scale_WH is True:
            self._W[:] = self._W * (
                np.sum(XHt * self._W) / (np.sum((self._W.T @ self._W) * HHt))
            )
            norm_Wcol = np.linalg.norm(self._W, axis=0) + _EPSILON
            norm_Hrow = np.linalg.norm(self._H, axis=1) + _EPSILON
            scaling_vec = np.sqrt(norm_Wcol / norm_Hrow)
            self._W[:] = self._W / scaling_vec
            self._H[:] = self._H * scaling_vec[:, np.newaxis]
            HHt[:] = HHt * scaling_vec[np.newaxis, :]
            HHt[:] = scaling_vec[:, np.newaxis] * HHt
            XHt[:] = XHt * scaling_vec[np.newaxis, :]

        norm_X = np.linalg.norm(X, ord="fro")
        rel_loss = np.sqrt(self._fro_err_sq_after_W_update(norm_X, HHt, XHt)) / norm_X
        self._rel_loss_ls.append(rel_loss)

        iter = 0
        while iter < self.iter_max_:
            self._update_H(X, WtW, WtX, known_H, mask_known_H)
            self._update_W(X, HHt, XHt, known_W, mask_known_W)
            rel_loss = (
                np.sqrt(self._fro_err_sq_after_W_update(norm_X, HHt, XHt)) / norm_X
            )
            self._rel_loss_ls.append(rel_loss)
            self._is_converged = self._check_convergence(iter)
            if self._is_converged:
                break
            iter += 1

    def _update_H(
        self,
        X: NDArray[np.float64],
        WtW: NDArray[np.float64],
        WtX: NDArray[np.float64],
        known_H: NDArray[np.float64] | None,
        mask_known_H: NDArray[np.bool_] | None,
    ):
        """
        Update the H matrix based on the current W matrix using ALS.

        Solves the least squares problem H = argmin ||X - WH||_F^2 subject to
        constraints. Uses efficient matrix operations and handles potential
        ill-conditioning.

        Parameters
        ----------
        X : ndarray
            The input matrix to be factorized.
        WtW : ndarray
            Precomputed W^T * W matrix for efficiency.
        WtX : ndarray
            Precomputed W^T * X matrix for efficiency.
        known_H : ndarray or None
            Matrix of known values in H (NaN elements are treated as unknown).
        mask_known_H : ndarray or None
            Boolean mask indicating which values in H are known.

        Returns
        -------
        None
            The H matrix is updated in-place.
        """
        WtW[:] = self._W.T @ self._W
        WtX[:] = self._W.T @ X
        if np.linalg.cond(WtW) > 1e6:
            delta = np.trace(WtW) / self.rank_
            WtW[:] += 1e-6 * delta * np.eye(self.rank_)
            self._H[:] = np.linalg.solve(WtW, WtX)
        else:
            self._H[:] = np.linalg.solve(WtW, WtX)
        # apply known values
        if mask_known_H is not None:
            self._H[mask_known_H] = known_H[mask_known_H]
        self._apply_unimodal_H_rows()
        self._project_H()

    def _update_W(
        self,
        X: NDArray[np.float64],
        HHt: NDArray[np.float64],
        XHt: NDArray[np.float64],
        known_W: NDArray[np.float64] | None,
        mask_known_W: NDArray[np.bool_] | None,
    ):
        """
        Update the W matrix based on the current H matrix using ALS.

        Solves the least squares problem W = argmin ||X - WH||_F^2 subject to
        constraints. Uses efficient matrix operations and handles potential
        ill-conditioning.

        Parameters
        ----------
        X : ndarray
            The input matrix to be factorized.
        HHt : ndarray
            Precomputed H * H^T matrix for efficiency.
        XHt : ndarray
            Precomputed X * H^T matrix for efficiency.
        known_W : ndarray or None
            Matrix of known values in W (NaN elements are treated as unknown).
        mask_known_W : ndarray or None
            Boolean mask indicating which values in W are known.

        Returns
        -------
        None
            The W matrix is updated in-place.
        """
        HHt[:] = self._H @ self._H.T
        XHt[:] = X @ self._H.T
        if np.linalg.cond(HHt) > 1e6:
            delta = np.trace(HHt) / self.rank_
            HHt[:] += 1e-6 * delta * np.eye(self.rank_)
            self._W[:] = np.linalg.solve(HHt, XHt.T).T
        else:
            self._W[:] = np.linalg.solve(HHt, XHt.T).T
        # apply known values
        if mask_known_W is not None:
            self._W[mask_known_W] = known_W[mask_known_W]
        self._apply_unimodal_W_cols()
        self._project_W()


class FroFPGM(_BaseNMF):
    """
    Frobenius norm-based Nonnegative Matrix Factorization (NMF) using Fast Projected
    Gradient Method (FPGM).

    Factorizes a nonnegative matrix :math:`X` into two nonnegative matrices :math:`W`
    and :math:`H` by minimizing the squared Frobenius norm between :math:`X` and product
    :math:`WH` using FPGM algorithm. The following objective function is minimized
    subject to nonnegativity and other optional constraints on :math:`W` and :math:`H`:

    .. math::
        f_{\\textrm{obj}} = ||X - WH||_F^2

    where :math:`||\\cdot||_{F}` is the Frobenius norm.

    Parameters
    ----------
    rank : int
        The number of components for the factorization.
    constraint_kind : integer-like {0, 1, 2, 3, 4}, default=0
        The following constraints are applied based on the integer value specified:

        * If ``0``: Only :math:`W \\geq 0`, :math:`H \\geq 0`.
        * If ``1``: Closure constraint :math:`H^T e ≤ e`.
        * If ``2``: Closure constraint :math:`H e = e`.
        * If ``3``: Constraint :math:`W^T e = e`.
        * If ``4``: Closure constraint :math:`H^T e = e`.

        Note, for 1, 2, 3, and 4 values of ``constraint_kind`` nonnegativity constraints
        are also applied along with the additional constraint specified above.
    unimodal : dict or None, default=None
        Specifies unimodality constraints for :math:`W` and :math:`H` matrices. If
        ``None``, no unimodality constraints are applied.

        If ``dict``, Format: {'W': bool | list of bool, 'H': bool | list of bool}:

            * Must contain at least one key ('W' or 'H')
            * No other keys besides 'W' and 'H' are allowed
            * For 'W': Controls unimodality of columns in W
            * For 'H': Controls unimodality of rows in H

        Each value can be:

            * A boolean: applies to all components
            * A list of booleans: selectively applies to specific components

        Examples:

            * ``{'H': True}``: All :math:`H` components are unimodal
            * ``{'W': True, 'H': True}``: All :math:`W` and :math:`H` components are
              unimodal
            * ``{'H': [True, False, True]}``: Only components 0 and 2 of :math:`H` have
              unimodal behavior
            * ``{'W': [False, True, False]}``: Only component 1 of :math:`W` has
              unimodal behavior
    iter_max : int, default=500
        Maximum number of iterations. It must be greater :math:`\\geq 10`.
    tol : float, default=1e-4
        Tolerance for convergence. Must be in the interval :math:`(0, 1)`.

        Convergence is reached when:

            .. math::

                {|e[i] - e[i-10]| \\over e[i]} \\leq \\textrm{tol}

            where:

            * iteration :math:`i \\geq 10`
            * :math:`e[i]` is the squared relative loss after iteration :math:`i`, which
              is defined as

                .. math::

                    e[i] = {||X - W^{i}H^{i}||_{F}^2 \\over ||X||_{F}^2}

                where :math:`W^{i}` and :math:`H^{i}` is the value of :math:`W` and
                :math:`H`, respectively, after iteration :math:`i`.
    inner_iter_max : int, default=20
        Maximum number of inner iterations performed during each single update of either
        :math:`W` or :math:`H` while the other is held fixed.
    inner_iter_tol : float, default=0.1
        Tolerance for the convergence of the inner loop. The inner loop convergence is
        reached when:

        .. math::

            {||A^{k} - A^{k-1}||_{F} \\over ||A^{1} - A^{0}||_{F}} \\leq
            \\textrm{inner_iter_tol}

        where:

        * :math:`A` is either :math:`H` or :math:`W`.
        * :math:`A^{k}` is value of :math:`A` after inner-iteration :math:`k`.
        * :math:`A^{0}` is the value of :math:`A` before the first inner-iteration.

    References
    ----------
    .. [1] Gillis, N. (2020). Nonnegative matrix factorization. Society for Industrial
           and Applied Mathematics.
    .. [2] Lin, Chih-Jen. "Projected gradient methods for nonnegative matrix
           factorization." Neural computation 19.10 (2007): 2756-2779.

    Examples
    --------
    >>> from mcrnmf.models import FroFPGM, SNPA
    >>> from mcrnmf.datasets import load_rxn_spectra
    >>>
    >>> # load the example dataset from mcrnmf
    >>> X, wv, time  = load_rxn_spectra()
    >>>
    >>> # generate initial guess using SNPA
    >>> snpa = SNPA(rank=4)
    >>> snpa.fit(X)
    >>> Wi = snpa.W  # Initial estimate for W
    >>> Hi = snpa.H  # Initial estimate for H
    >>>
    >>> # create an instance of FroALS and fit the model
    >>> model = FroFPGM(rank=4, constraint_kind=1, iter_max=2000, tol=1e-4)
    >>> model.fit(X, Wi, Hi)
    >>> # access decomposed factors
    >>> W, H = model.W, model.H
    >>> # check convergence status
    >>> converged = model.is_converged
    >>> # access rel. reconstruction error after each iterations
    >>> rel_recon_err = model.rel_reconstruction_error_ls
    """

    def __init__(
        self,
        rank: int,
        constraint_kind: int = 0,
        unimodal: dict | None = None,
        iter_max: int = 500,
        tol: int = 0.0001,
        inner_iter_max: int = 20,
        inner_iter_tol: float = 0.1,
    ):
        super().__init__(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal,
            iter_max=iter_max,
            tol=tol,
        )

        if not isinstance(inner_iter_max, (int, np.integer)):
            raise TypeError(
                f"`inner_iter_max` must be of type {int}, current type is "
                f"{type(inner_iter_max)}"
            )
        if not isinstance(inner_iter_tol, float):
            raise TypeError(
                f"`tol` must be of type {float}, current type is {type(inner_iter_tol)}"
            )

        if inner_iter_max <= 0:
            raise ValueError("`inner_iter_max` must be > 0")
        if not (0 < inner_iter_tol < 1):
            raise ValueError("`inner_iter_tol` must be in the open interval (0, 1)")

        self.inner_iter_max_ = inner_iter_max
        self.inner_iter_tol_ = inner_iter_tol

    @property
    def rel_reconstruction_error_ls(self):
        """
        List of relative reconstruction errors from each iteration during model fitting.

        It is defined as:

        .. math::
            \\dfrac{||X - W^{i}H^{i}||_F}{||X||_F}\\ \\textrm{,}

        where:

        * :math:`||\\cdot||_F` denotes the Frobenius norm
        * :math:`X` is the original data matrix
        * :math:`W^{i}` and :math:`H^{i}` are values of :math:`W` and :math:`H` after
          iteration :math:`i`

        Returns
        -------
        list of float
            Relative reconstruction error from each iteration.
        """
        if self._rel_loss_ls is None:
            raise AttributeError(
                "`rel_reconstruction_error_ls` is not available. Model has not been "
                "fitted yet. Call `fit` first."
            )
        return self._rel_loss_ls

    def fit(
        self,
        X: NDArray[np.float64],
        Wi: NDArray[np.float64],
        Hi: NDArray[np.float64],
        known_W: NDArray[np.float64] | None = None,
        known_H: NDArray[np.float64] | None = None,
        preprocess_scale_WH: bool = False,
    ):
        """
        Fit the FroFPGM model to the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Data array to be factorized.
        Wi : ndarray of shape (n_features, rank)
            Initial guess for the factor :math:`W`.
        Hi : ndarray of shape (rank, n_samples)
            Initial guess for the factor :math:`H`.
        known_W : ndarray of shape (n_features, rank), default=None
            Array containing known values of :math:`W`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`W` which do not
              correspond ``np.nan`` entries in ``known_W``.
        known_H : ndarray of shape (rank, n_samples), default=None
            Array containing known values of :math:`H`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`H` which do not
              correspond ``np.nan`` entries in ``known_H``.
        preprocess_scale_WH : bool, default=False
            If ``True``,  ``Wi`` and ``Hi`` are scaled before optimization.
        """
        self._validate_fit_args(X, Wi, Hi, known_W, known_H)
        self._W = Wi.copy()
        self._H = Hi.copy()
        self._rel_loss_ls = []
        self._is_converged = False

        # Check if known values are passed, if so compute a mask.
        if known_W is not None:
            mask_known_W = np.isfinite(known_W)
        else:
            mask_known_W = None
        if known_H is not None:
            mask_known_H = np.isfinite(known_H)
        else:
            mask_known_H = None

        # preallocate arrays for H updates
        WtW = np.zeros((self.rank_, self.rank_))
        WtX = np.zeros((self.rank_, X.shape[1]))
        Y_H = np.zeros_like(self._H)
        H_diff = np.zeros_like(self._H)
        H_prev = np.zeros_like(self._H)

        # preallocate arrays for W updates
        XHt = X @ self._H.T
        HHt = self._H @ self._H.T
        Y_W = np.zeros_like(self._W)
        W_diff = np.zeros_like(self._W)
        W_prev = np.zeros_like(self._W)

        if preprocess_scale_WH is True:
            self._W[:] = self._W * (
                np.sum(XHt * self._W) / (np.sum((self._W.T @ self._W) * HHt))
            )
            norm_Wcol = np.linalg.norm(self._W, axis=0) + _EPSILON
            norm_Hrow = np.linalg.norm(self._H, axis=1) + _EPSILON
            scaling_vec = np.sqrt(norm_Wcol / norm_Hrow)
            self._W[:] = self._W / scaling_vec
            self._H[:] = self._H * scaling_vec[:, np.newaxis]
            HHt[:] = HHt * scaling_vec[np.newaxis, :]
            HHt[:] = scaling_vec[:, np.newaxis] * HHt
            XHt[:] = XHt * scaling_vec[np.newaxis, :]

        norm_X = np.linalg.norm(X, ord="fro")
        rel_loss = np.sqrt(self._fro_err_sq_after_W_update(norm_X, HHt, XHt)) / norm_X
        self._rel_loss_ls.append(rel_loss)

        iter = 0
        while iter < self.iter_max_:
            self._update_H(X, WtW, WtX, Y_H, H_diff, H_prev, known_H, mask_known_H)
            self._update_W(X, HHt, XHt, Y_W, W_diff, W_prev, known_W, mask_known_W)
            rel_loss = (
                np.sqrt(self._fro_err_sq_after_W_update(norm_X, HHt, XHt)) / norm_X
            )
            self._rel_loss_ls.append(rel_loss)
            self._is_converged = self._check_convergence(iter)
            if self._is_converged:
                break
            iter += 1

    def _update_H(
        self,
        X: NDArray[np.float64],
        WtW: NDArray[np.float64],
        WtX: NDArray[np.float64],
        Y_H: NDArray[np.float64],
        H_diff: NDArray[np.float64],
        H_prev: NDArray[np.float64],
        known_H: None | NDArray[np.float64],
        mask_known_H: None | NDArray[np.float64],
    ) -> None:
        """
        Update the H matrix based on the current W matrix.

        This method implements the FPGM update for the H matrix.

        Parameters
        ----------
        X : ndarray
            The input matrix to be factorized.
        WtW : ndarray
            Precomputed W^T * W matrix.
        WtX : ndarray
            Precomputed W^T * X matrix.
        Y_H : ndarray
            Auxiliary variable for momentum updates.
        H_diff : ndarray
            Difference between consecutive H updates.
        H_prev : ndarray
            Previous H matrix for momentum calculation.
        known_H : ndarray or None
            Matrix with known values of H.
        mask_known_H : ndarray or None
            Mask indicating which values in H are known.

        Returns
        -------
        None
            The H matrix is updated in-place.
        """
        self._project_H()
        WtW[:] = self._W.T @ self._W
        WtX[:] = self._W.T @ X

        # hyperparameters for optimization
        l = np.linalg.norm(WtW, 2)  # lipschitz constant
        theta_prev = 0.05  # first value of theta, heuristic

        Y_H = self._H.copy()  # look ahead variable
        inner_iter = 0
        while inner_iter < self.inner_iter_max_:
            H_prev[:] = self._H.copy()
            self._H[:] = Y_H - ((WtW @ Y_H) - WtX) / l
            # apply known values
            if mask_known_H is not None:
                self._H[mask_known_H] = known_H[mask_known_H]
            self._apply_unimodal_H_rows()
            self._project_H()
            H_diff[:] = self._H - H_prev
            norm_H_diff = np.linalg.norm(H_diff, "fro")
            if inner_iter == 0:
                norm_H_diff_0 = norm_H_diff
            # convergence check
            if norm_H_diff <= (self.inner_iter_tol_ * norm_H_diff_0):
                break
            # do the momentum update only if convergence criteria is not satisfied
            theta_prev_sq = theta_prev**2
            # theta and beta computation based on scheme 2.2.19 in Introductory lectures
            # on convex optimization by Nesterov
            theta = (np.sqrt(theta_prev_sq**2 + 4 * theta_prev_sq) - theta_prev_sq) / 2
            beta = theta_prev * (1 - theta_prev) / (theta_prev_sq + theta)
            # momentum step
            H_diff[:] = H_diff * beta
            Y_H[:] = self._H + H_diff
            theta_prev = theta
            inner_iter += 1

    def _update_W(
        self,
        X: NDArray[np.float64],
        HHt: NDArray[np.float64],
        XHt: NDArray[np.float64],
        Y_W: NDArray[np.float64],
        W_diff: NDArray[np.float64],
        W_prev: NDArray[np.float64],
        known_W: None | NDArray[np.float64],
        mask_known_W: None | NDArray[np.float64],
    ) -> None:
        """
        Update the W matrix based on the current H matrix.

        This method implements the FPGM update for the W matrix.

        Parameters
        ----------
        X : ndarray
            The input matrix to be factorized.
        HHt : ndarray
            Precomputed H * H^T matrix.
        XHt : ndarray
            Precomputed X * H^T matrix.
        Y_W : ndarray
            Auxiliary variable for momentum updates.
        W_diff : ndarray
            Difference between consecutive W updates.
        W_prev : ndarray
            Previous W matrix for momentum calculation.
        known_W : ndarray or None
            Matrix with known values of W.
        mask_known_W : ndarray or None
            Mask indicating which values in W are known.

        Returns
        -------
        None
            The W matrix is updated in-place.
        """
        self._project_W()
        HHt[:] = self._H @ self._H.T
        XHt[:] = X @ self._H.T

        # hyperparameters for optimization
        l = np.linalg.norm(HHt, 2)  # lipschitz constant
        theta_prev = 0.05  # first value of theta, heuristic

        Y_W = self._W.copy()  # look ahead variable
        inner_iter = 0
        while inner_iter < self.inner_iter_max_:
            W_prev[:] = self._W.copy()
            self._W[:] = Y_W - ((Y_W @ HHt) - XHt) / l
            # apply known values
            if mask_known_W is not None:
                self._W[mask_known_W] = known_W[mask_known_W]
            self._apply_unimodal_W_cols()
            self._project_W()
            W_diff[:] = self._W - W_prev
            norm_W_diff = np.linalg.norm(W_diff, "fro")
            if inner_iter == 0:
                norm_W_diff_0 = norm_W_diff
            # convergence check
            if norm_W_diff <= (self.inner_iter_tol_ * norm_W_diff_0):
                break
            # do the momentum update only if convergence criteria is not satisfied
            theta_prev_sq = theta_prev**2
            # theta and beta computation based on scheme 2.2.19 in Introductory lectures
            # on convex optimization by Nesterov
            theta = (np.sqrt(theta_prev_sq**2 + 4 * theta_prev_sq) - theta_prev_sq) / 2
            beta = theta_prev * (1 - theta_prev) / (theta_prev_sq + theta)
            # momentum step
            W_diff[:] = W_diff * beta
            Y_W[:] = self._W + W_diff
            theta_prev = theta
            inner_iter += 1


class MinVol(FroFPGM):
    """
    Minimum-Volume Nonnegative Matrix Factorization (NMF) implementation using Fast
    Projected Gradient Method.

    This class implements the FPGM to perform Minimum-Volume NMF, minimizing the
    following objective function subject to non-negativity and other optional
    constraints.

    .. math::
        f_{\\textrm{obj}} = ||X - WH||_F^2 + \\lambda \\times \\log(\\det(W^TW + \\delta
        \\times I))

    where:

    * :math:`||\\cdot||_{F}` is the Frobenius norm.
    * :math:`X` the nonnegative matrix to be factorized
    * :math:`W` and :math:`H` are the nonnegative factors
    * :math:`\\lambda` and :math:`\\delta` are model parameters. See definition in the
      Parameters section
    * :math:`I` is an identity matrix

    Parameters
    ----------
    rank : int
        The number of components for the factorization.
    constraint_kind : integer-like {0, 1, 2, 3, 4}, default=0
        The following constraints are applied based on the integer value specified:

        * If ``0``: Only :math:`W \\geq 0`, :math:`H \\geq 0`.
        * If ``1``: Closure constraint :math:`H^T e ≤ e`.
        * If ``2``: Closure constraint :math:`H e = e`.
        * If ``3``: Constraint :math:`W^T e = e`.
        * If ``4``: Closure constraint :math:`H^T e = e`.

        Note, for 1, 2, 3, and 4 values of ``constraint_kind`` nonnegativity constraints
        are also applied along with the additional constraint specified above.
    unimodal : dict or None, default=None
        Specifies unimodality constraints for :math:`W` and :math:`H` matrices. If
        ``None``, no unimodality constraints are applied.

        If ``dict``, Format: {'W': bool | list of bool, 'H': bool | list of bool}:

            * Must contain at least one key ('W' or 'H')
            * No other keys besides 'W' and 'H' are allowed
            * For 'W': Controls unimodality of columns in W
            * For 'H': Controls unimodality of rows in H

        Each value can be:

            * A boolean: applies to all components
            * A list of booleans: selectively applies to specific components

        Examples:

            * ``{'H': True}``: All :math:`H` components are unimodal
            * ``{'W': True, 'H': True}``: All :math:`W` and :math:`H` components are
              unimodal
            * ``{'H': [True, False, True]}``: Only components 0 and 2 of :math:`H` have
              unimodal behavior
            * ``{'W': [False, True, False]}``: Only component 1 of :math:`W` has
              unimodal behavior
    iter_max : int, default=500
        Maximum number of iterations. It must be greater :math:`\\geq 10`.
    tol : float, default=1e-4
        Tolerance for convergence. Must be in the interval :math:`(0, 1)`.

            Convergence is reached when:

            .. math::

                {|e[i] - e[i-10]| \\over e[i]} \\leq \\textrm{tol}

            where:

            * iteration :math:`i \\geq 10`
            * :math:`e[i]` is the squared relative loss after iteration :math:`i`, which
              is defined as

                .. math::

                    e[i] = {||X - W^{i}H^{i}||_{F}^2 \\over ||X||_{F}^2}

                where :math:`W^{i}` and :math:`H^{i}` is the value of :math:`W` and
                :math:`H`, respectively, after iteration :math:`i`.
    inner_iter_max : int, default=20
        Maximum number of inner iterations performed during each single update of either
        :math:`W` or :math:`H` while the other is held fixed.
    inner_iter_tol : float, default=0.1
        Tolerance for the convergence of the inner loop while updating :math:`H`.

            The inner loop convergence is reached when:

            .. math::

                {||H^{k} - H^{k-1}||_{F} \\over ||H^{1} - H^{0}||_{F}} \\leq
                \\textrm{inner_iter_tol}

            where:

            * :math:`H^{k}` is value of :math:`H` after inner-iteration :math:`k`.
            * :math:`H^{0}` is the value of :math:`H` before the first inner-iteration.
    delta : float, default=0.1
        Value of parameter :math:`\\delta`. It ensures numerical stability when
        :math:`W` is rank-deficient.
    lambdaa : float, default=1e-3
        The intial weight, :math:`\\lambda`, of the volume-regularization term.

    References
    ----------
    .. [1] Gillis, N. (2020). Nonnegative matrix factorization. Society for Industrial
           and Applied Mathematics.
    .. [2] Leplat, Valentin, Andersen MS Ang, and Nicolas Gillis. "Minimum-volume rank-
           deficient nonnegative matrix factorizations." ICASSP 2019-2019 IEEE
           International Conference on Acoustics, Speech and Signal Processing (ICASSP).
           IEEE, 2019.

    Examples
    --------
    >>> from mcrnmf.models import MinVol, SNPA
    >>> from mcrnmf.datasets import load_rxn_spectra
    >>>
    >>> # load the example dataset from mcrnmf
    >>> X, wv, time  = load_rxn_spectra()
    >>>
    >>> # generate initial guess using SNPA
    >>> snpa = SNPA(rank=4)
    >>> snpa.fit(X)
    >>> Wi = snpa.W  # Initial estimate for W
    >>> Hi = snpa.H  # Initial estimate for H
    >>>
    >>> # create an instance of FroALS and fit the model
    >>> model = MinVol(rank=4, constraint_kind=1, iter_max=2000, tol=1e-4, lambdaa=1e-3)
    >>> model.fit(X, Wi, Hi)
    >>> # access decomposed factors
    >>> W, H = model.W, model.H
    >>> # check convergence status
    >>> converged = model.is_converged
    >>> # access rel. reconstruction error after each iterations
    >>> rel_recon_err = model.rel_reconstruction_error_ls
    """

    def __init__(
        self,
        rank: int,
        constraint_kind: int = 0,
        unimodal: dict | None = None,
        iter_max: int = 500,
        tol: float = 0.0001,
        inner_iter_max: int = 20,
        inner_iter_tol: float = 0.1,
        delta: float | int = 0.1,
        lambdaa: float = 0.001,
    ):
        super().__init__(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal,
            iter_max=iter_max,
            tol=tol,
            inner_iter_max=inner_iter_max,
            inner_iter_tol=inner_iter_tol,
        )
        if not isinstance(delta, (int, np.integer, float)):
            raise TypeError(
                f"`delta` must be of type {float} or {int}, current type is "
                f"{type(delta)}"
            )
        if not isinstance(lambdaa, (float, int, np.integer)):
            raise TypeError(
                f"`lambdaa` must be of type {float} or {int}, current type is "
                f"{type(lambdaa)}"
            )
        if delta <= 0:
            raise ValueError("`delta` must be > 0")
        if lambdaa <= 0:
            raise ValueError("`lambdaa` must be > 0")

        self.delta_ = delta
        self.lambdaa_ = lambdaa

        # initialize the additional attribute specific to MinVol
        self._rel_reconstruction_error_ls = None

    @property
    def rel_reconstruction_error_ls(self):
        """
        List of relative reconstruction errors from each iteration during model fitting.

        The relative reconstruction error measures how well the current factors
        approximate the original data. It is the ratio:

        .. math::
            \\dfrac{||X - W^{i}H^{i}||_F}{||X||_F}\\ \\textrm{,}

        where:

        * :math:`||\\cdot||_F` denotes the Frobenius norm
        * :math:`X` is the original data matrix
        * :math:`W^{i}` and :math:`H^{i}` are values of :math:`W` and :math:`H` after
          iteration :math:`i`

        Returns
        -------
        list of float
            Relative reconstruction error from each iteration.
        """
        if self._rel_reconstruction_error_ls is None:
            raise AttributeError(
                "`rel_reconstruction_error_ls` is not available. Model has not been "
                "fitted yet. Call `fit` first."
            )
        return self._rel_reconstruction_error_ls

    def fit(
        self,
        X: NDArray[np.floating],
        Wi: NDArray[np.floating],
        Hi: NDArray[np.floating],
        known_W: None | NDArray[np.floating] = None,
        known_H: None | NDArray[np.floating] = None,
        preprocess_scale_WH: bool = True,
    ):
        """
        Fit the MinVol model to the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Data array to be factorized.
        Wi : ndarray of shape (n_features, rank)
            Initial guess for the factor :math:`W`.
        Hi : ndarray of shape (rank, n_samples)
            Initial guess for the factor :math:`H`.
        known_W : ndarray of shape (n_features, rank), default=None
            Array containing known values of :math:`W`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`W` which do not
              correspond ``np.nan`` entries in ``known_W``.
        known_H : ndarray of shape (rank, n_samples), default=None
            Array containing known values of :math:`H`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`H` which do not
              correspond ``np.nan`` entries in ``known_H``.
        preprocess_scale_WH : bool, default=True
            If ``True``,  ``Wi`` and ``Hi`` are scaled before optimization.
        """
        self._validate_fit_args(X, Wi, Hi, known_W, known_H)
        self._W = Wi.copy()
        self._H = Hi.copy()
        self._rel_loss_ls = []
        self._rel_reconstruction_error_ls = []
        self._is_converged = False

        # Check if known values are passed, if so compute a mask.
        if known_W is not None:
            mask_known_W = np.isfinite(known_W)
        else:
            mask_known_W = None
        if known_H is not None:
            mask_known_H = np.isfinite(known_H)
        else:
            mask_known_H = None

        # pre-allocate matrices used in update H
        Y_H = np.zeros_like(self._H)
        H_diff = np.zeros_like(self._H)
        H_prev = np.zeros_like(self._H)
        WtW = np.zeros((self.rank_, self.rank_))
        WtX = np.zeros((self.rank_, X.shape[1]))

        # preallocate matrices used in update W
        Y_W = np.zeros_like(self._W)
        W_diff = np.zeros_like(self._W)
        W_prev = np.zeros_like(self._W)
        HHt = self._H @ self._H.T
        XHt = X @ self._H.T

        # scale W and H based on the constraint
        if (self.constraint_kind_ != 0) and preprocess_scale_WH:
            self._scale_WH(
                X, H_prev, Y_W, W_diff, W_prev, HHt, XHt, known_W, mask_known_W
            )

        norm_X = np.sqrt(np.sum(X**2))
        WtW[:] = self._W.T @ self._W
        WtX[:] = self._W.T @ X
        I = np.eye(self.rank_, dtype=X.dtype)
        fro_error_sq = self._fro_err_sq_after_H_update(norm_X, WtW, WtX)
        self._rel_reconstruction_error_ls.append(np.sqrt(fro_error_sq) / norm_X)
        vol_error = self._calc_vol_error(WtW=WtW, I=I)
        lambdaa_updt = self.lambdaa_ * max(1e-6, fro_error_sq) / np.abs(vol_error)
        rel_loss = np.sqrt(fro_error_sq + lambdaa_updt * vol_error) / norm_X
        self._rel_loss_ls.append(rel_loss)

        # pre-allocate memory for frequently used matrices
        A = np.zeros((self.rank_, self.rank_))
        Y = np.zeros((self.rank_, self.rank_))
        iter = 0
        while iter < self.iter_max_:
            XHt[:] = X @ self._H.T
            HHt[:] = self._H @ self._H.T
            Y[:] = np.linalg.inv(WtW + self.delta_ * I)
            A[:] = lambdaa_updt * Y + HHt
            self._update_W_minvol(A, XHt, Y_W, W_diff, W_prev, known_W, mask_known_W)
            self._update_H(X, WtW, WtX, Y_H, H_diff, H_prev, known_H, mask_known_H)
            fro_error_sq = self._fro_err_sq_after_H_update(norm_X, WtW, WtX)
            self._rel_reconstruction_error_ls.append(np.sqrt(fro_error_sq) / norm_X)
            vol_error = self._calc_vol_error(WtW=WtW, I=I)
            rel_loss = np.sqrt(fro_error_sq + lambdaa_updt * vol_error) / norm_X
            self._rel_loss_ls.append(rel_loss)
            self._is_converged = self._check_convergence(iter)
            if self._is_converged:
                break
            iter += 1

    def _calc_vol_error(
        self, WtW: NDArray[np.float64], I: NDArray[np.float64]
    ) -> np.float64:
        """
        Calculate the volume error term in the objective function.

        Computes log(det(W^T * W + delta * I)) which quantifies the volume of the
        simplex defined by the columns of W.

        Parameters
        ----------
        WtW : ndarray
            Precomputed W^T·W matrix.
        I : ndarray
            Identity matrix of appropriate size.

        Returns
        -------
        float
            The volume error term.
        """
        return np.log(np.linalg.det(WtW + self.delta_ * I))

    def _scale_WH(
        self,
        X: NDArray[np.float64],
        H_prev: NDArray[np.float64],
        Y_W: NDArray[np.float64],
        W_diff: NDArray[np.float64],
        W_prev: NDArray[np.float64],
        HHt: NDArray[np.float64],
        XHt: NDArray[np.float64],
        known_W: None | NDArray[np.float64],
        mask_known_W: None | NDArray[np.float64],
    ) -> None:
        """
        Scale W and H matrices to maintain constraints while improving conditioning.

        Depending on the constraint type, this method applies appropriate scaling
        transformations to W and H that preserve their product.

        Parameters
        ----------
        X : ndarray
            The input matrix.
        H_prev : ndarray
            Previous H matrix for comparison.
        Y_W : ndarray
            Auxiliary variable for W updates.
        W_diff : ndarray
            Difference between consecutive W updates.
        W_prev : ndarray
            Previous W matrix.
        HHt : ndarray
            Precomputed H·H^T matrix.
        XHt : ndarray
            Precomputed X·H^T matrix.
        known_W : ndarray or None
            Matrix with known values of W.
        mask_known_W : ndarray or None
            Mask indicating which values in W are known.

        Returns
        -------
        None
            The W and H matrices are updated in-place.
        """
        if self.constraint_kind_ in [1, 4]:
            H_prev = self._H.copy()
            if self.constraint_kind_ == 1:
                simplex_proj(self._H)
            else:
                simplex_proj_using_axis(self._H, True)
            # update W if the H projection has changed H significantly
            if np.linalg.norm(self._H - H_prev, 2) > (
                1e-3 * np.linalg.norm(self._H, 2)
            ):
                self._update_W(X, HHt, XHt, Y_W, W_diff, W_prev, known_W, mask_known_W)
        elif self.constraint_kind_ == 2:
            H_row_sum = np.sum(self._H, axis=1)
            self._H[:] = self._H / H_row_sum[:, np.newaxis]
            self._W[:] = self._W * H_row_sum
        elif self.constraint_kind_ == 3:
            W_col_sum = np.sum(self._W, axis=0)
            self._H[:] = self._H * W_col_sum[:, np.newaxis]
            self._W[:] = self._W / W_col_sum

    def _update_W_minvol(
        self,
        A: NDArray[np.float64],
        XHt: NDArray[np.float64],
        Y_W: NDArray[np.float64],
        W_diff: NDArray[np.float64],
        W_prev: NDArray[np.float64],
        known_W: NDArray[np.float64] | None,
        mask_known_W: NDArray[np.float64] | None,
    ) -> None:
        """
        Update W with minimum volume regularization.

        Performs gradient descent on W with the objective function that includes
        both data fidelity and volume regularization terms. The update optimizes:
        ||X - WH||_F^2 + lambdaa * log(det(W^T*W + delta*I))

        Parameters
        ----------
        A : ndarray
            Precomputed regularized Hessian matrix.
        XHt : ndarray
            Precomputed X*H^T matrix.
        Y_W : ndarray
            Look-ahead matrix for momentum updates.
        W_diff : ndarray
            Matrix to store differences between iterations.
        W_prev : ndarray
            Matrix to store previous W values.
        known_W : ndarray or None
            Matrix containing known values for W.
        mask_known_W : ndarray or None
            Boolean mask indicating which W values are known.

        Returns
        -------
        None
            The W matrix is updated in-place.
        """
        self._project_W()
        # hyperparamters for optimization
        s = np.linalg.svd(A, compute_uv=False)
        l = s[0]  # lipschitz constant
        kappa_inv = s[-1] / s[0]  # inverse of the condition number
        beta = (1 - np.sqrt(kappa_inv)) / (1 + np.sqrt(kappa_inv))

        Y_W = self._W.copy()  # look ahead variable
        inner_iter = 0
        error, error_prev = 1, 0
        while inner_iter < self.inner_iter_max_:
            W_prev[:] = self._W.copy()
            self._W[:] = Y_W - ((Y_W @ A) - XHt) / l
            # apply known values
            if mask_known_W is not None:
                self._W[mask_known_W] = known_W[mask_known_W]
            self._apply_unimodal_W_cols()
            self._project_W()
            W_diff[:] = self._W - W_prev
            norm_W_diff = np.linalg.norm(W_diff, "fro")
            if inner_iter == 0:
                norm_W_diff_0 = norm_W_diff
            # convergence check
            if norm_W_diff <= (1e-6 * norm_W_diff_0):
                break
            error = np.sum((self._W.T @ self._W) * A) - 2 * np.sum(self._W * XHt)
            if (inner_iter == 0) or (error <= error_prev):
                W_diff[:] = W_diff * beta
                Y_W[:] = self._W + W_diff
            else:
                # If current error is greater than previous error, replace look ahead
                # variable with current value of W
                Y_W[:] = self._W.copy()
            error_prev = error
            inner_iter += 1


class SNPA:
    """
    Successive Nonnegative Projection Algorithm (SNPA) for NMF initialization.

    The algorithm sequentially selects columns from :math:`X` with maximum residual norm
    after projecting onto the convex cone generated by previously selected columns,
    providing a robust initialization for :math:`W` which can be used by other NMF
    models. The corresponding :math:`H` matrix is then computed by minimizing
    :math:`||X - WH||_F^2` subject to nonnegativity and simplex constraints on
    :math:`H`.

    Parameters
    ----------
    rank : int
        The number of components to extract.
    iter_max : int, default=500
        Maximum number of iterations for updating :math:`H` after extraction of
        :math:`W`.
    tol : float, default=1e-6
        Tolerance for early stopping. The algorithm stops extracting basis compnents
        when:

        .. math::
            \\dfrac{||X - WH||_F}{||X||_{F}} \\leq \\textrm{tol}

        where :math:`||\\cdot||_{F}` is the Frobenius norm.

    Examples
    --------
    >>> from mcrnmf.models import SNPA
    >>> from mcrnmf.datasets import load_rxn_spectra
    >>>
    >>> # load the example dataset from mcrnmf
    >>> X, wv, time  = load_rxn_spectra()
    >>>
    >>> # create an instance of SNPA and fit to the data
    >>> snpa = SNPA(rank=4)
    >>> snpa.fit(X)
    >>> Wi = snpa.W  # Estimate for W
    >>> Hi = snpa.H  # Estimate for H
    >>> selected_indices = model.col_indices_ls
    """

    def __init__(
        self,
        rank: int,
        iter_max: int = 500,
        tol: float = 1e-6,
    ):
        # validate types of input arguments
        if not isinstance(rank, (int, np.integer)):
            raise TypeError(
                f"`rank` must be of type {int}, currrent type is {type(rank)}"
            )
        if not isinstance(iter_max, (np.integer, int)):
            raise TypeError(
                f"`iter_max` must be of type {int}, current type is {type(iter_max)}"
            )
        if not isinstance(tol, float):
            raise TypeError(
                f"`tol` must be of type {float}, current type is {type(tol)}"
            )
        # validate values of input arguments
        if rank < 1:
            raise ValueError("`rank` must be > 0")
        if iter_max < 1:
            raise ValueError("`iter_max` must be > 0")
        if not (0 < tol < 1):
            raise ValueError("`tol` must be in the open interval (0, 1)")

        self.rank_ = rank
        self.iter_max_ = iter_max
        self.tol_ = tol

        # Initialize the attributes
        self._W = None
        self._H = None
        self._col_indices_ls = None

    @property
    def W(self):
        """
        The basis matrix containing selected columns from :math:`X`.

        Returns
        -------
        ndarray of shape (n_features, rank)
            The basis matrix containing selected columns from :math:`X`.
        """
        if self._W is None:
            raise AttributeError(
                "'W' is not available. Model has not been fitted yet. Call 'fit' first."
            )
        return self._W

    @property
    def H(self):
        """
        The coefficient matrix derived from :math:`W`.

        Returns
        -------
        ndarray of shape (rank, n_samples)
            The coefficient matrix derived from :math:`W`.
        """
        if self._H is None:
            raise AttributeError(
                "'H' is not available. Model has not been fitted yet. Call 'fit' first."
            )
        return self._H

    @property
    def col_indices_ls(self):
        """
        The indices of columns selected from :math:`X` for :math:`W`.

        Returns
        -------
        list of int
            The indices of columns selected from :math:`X` for :math:`W` after fitting
            the model.
        """
        if self._col_indices_ls is None:
            raise AttributeError(
                "'col_indices_ls' is not available. Model has not been fitted yet. "
                "Call 'fit' first."
            )
        return self._col_indices_ls

    def fit(
        self,
        X: NDArray[np.float64],
    ):
        """
        Fit the SNPA model to the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Data array to be factorized.

        Warns
        -----
        UserWarning
            If the desired rank is not reached due to early convergence. If this happens
            try specifying a lower value for ``tol``.
        """
        # validate X
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` must be a numpy array")
        if X.dtype != np.float64:
            raise TypeError(
                f"The dtype of `X` elements should be {np.float64}, current dtype is "
                f"{X.dtype}"
            )
        if self.rank_ > X.shape[1]:
            raise ValueError(
                f"`rank` cannot be greater than columns in `X` {X.shape[1]}."
            )
        if X.min() < 0:
            raise ValueError("All elements of `X` must be >= 0")

        X_col_norm = np.sum(X**2, axis=0)
        X_col_norm_max = np.max(X_col_norm)

        # initialize Residuals column norm
        R_col_norm = X_col_norm.copy()

        self._col_indices_ls = []  # a list to store the column indices
        XtW = np.empty(0)
        self._W = np.empty(0)
        self._H = np.empty(0)
        idx_count = 0
        while idx_count < self.rank_:
            R_col_norm_max = np.max(R_col_norm)
            # check if there are multiples columns with max value
            idx = np.where((R_col_norm_max - R_col_norm) / R_col_norm_max <= 1e-6)[0]
            if idx.size > 1:
                # in case of ties select the one of with largest col norm
                self._col_indices_ls.append(idx[np.argmax(X_col_norm[idx])])
            else:
                self._col_indices_ls.append(idx[0])

            self._W = X[:, self._col_indices_ls]
            Wi = self._W[:, idx_count].reshape(-1, 1)
            if XtW.size == 0:
                XtW = X.T @ self._W
            else:
                XtW = np.hstack((XtW, X.T @ Wi))

            if idx_count == 0:
                WiTWi = self._W.T @ self._W
            else:
                WtWi = self._W[:, :idx_count].T @ Wi
                WiTWi = np.vstack(
                    (np.hstack((WiTWi, WtWi)), np.hstack((WtWi.T, Wi.T @ Wi)))
                )
            if idx_count == 0:
                self._update_H(X)
            else:
                self._H[:, self._col_indices_ls[-1]] = 0
                h = np.zeros((1, X.shape[1]))
                h[0, self._col_indices_ls[-1]] = 1
                self._H = np.vstack([self._H, h])
                self._update_H(X)

            if idx_count == 0:
                R_col_norm[:] = (
                    X_col_norm - 2 * (XtW.T * self._H) + (self._H * (WiTWi @ self._H))
                )
            else:
                R_col_norm[:] = (
                    X_col_norm
                    - 2 * np.sum(XtW.T * self._H, axis=0)
                    + np.sum(self._H * (WiTWi @ self._H), axis=0)
                )
            np.maximum(R_col_norm, 0, out=R_col_norm)

            if np.sqrt(np.max(R_col_norm) / X_col_norm_max) <= self.tol_:
                if len(self._col_indices_ls) != self.rank_:
                    warnings.warn(
                        f"Extracted only {len(self._col_indices_ls)} of {self.rank_} "
                        "indices"
                    )
                break
            idx_count += 1

    def _update_H(self, X: NDArray[np.float64]):
        """
        Update the coefficient matrix H given data X and basis W.

        This method solves the optimization problem to find H that minimizes
        ||X - WH||_F subject to simplex constraints on H. It uses an accelerated
        projected gradient descent approach with momentum updates for faster
        convergence.

        Parameters
        ----------
        X : ndarray
            The input data matrix of shape (n_features, n_samples).

        Notes
        -----
        This is an internal method used by the `fit` method and should not be
        called directly by users.
        """
        # initialization of H while doing rank-1 factorization
        if self._H.size == 0:
            if np.linalg.cond(self._W) > 1e6:
                warnings.warn(
                    "`W` has condition number > 1e6. Initial guess of H generated might"
                    " be poor."
                )
            self._H = np.linalg.lstsq(self._W, X, rcond=None)[0]
            self._H[:] = np.maximum(self._H, 0.0)
            alpha = np.sum(self._H * (self._W.T @ X)) / np.sum(
                (self._W.T @ self._W) * (self._H @ self._H.T)
            )
            self._H[:] = self._H * alpha

            # replaces rows which sum to zero with small random values
            zero_sum_rows = np.where(np.sum(self._H, axis=1) == 0)[0]
            if len(zero_sum_rows) > 0:
                self._H[zero_sum_rows, :] = (
                    0.001
                    * np.max(self._H)
                    * np.random.rand(len(zero_sum_rows), self._H.shape[1])
                )
        simplex_proj(self._H)

        # precompute WtW, and WtX since it doesn't change in the loop
        WtW = self._W.T @ self._W
        WtX = self._W.T @ X

        # hyperparameters for optimization
        l = np.linalg.norm(WtW, 2)  # lipschitz constant
        alpha_prev = 0.05  # first value of alpha, heuristic
        delta = 1e-6

        # pre-allocate memory for matrices computed or used repeatedly in the loop
        Y = self._H.copy()  # look ahead variable
        H_diff = np.zeros(self._H.shape)
        H_prev = np.zeros(self._H.shape)

        iter = 0
        while iter < self.iter_max_:
            H_prev[:] = self._H.copy()
            self._H[:] = Y - ((WtW @ Y) - WtX) / l
            simplex_proj(self._H)
            H_diff[:] = self._H - H_prev
            norm_H_diff = np.linalg.norm(H_diff, "fro")
            if iter == 0:
                norm_H_diff_0 = norm_H_diff
            # convergence check
            if norm_H_diff <= (delta * norm_H_diff_0):
                break
            # do the momentum update only if convergence criteria is not satisfied
            alpha_prev_sq = alpha_prev**2
            alpha = (np.sqrt(alpha_prev_sq**2 + 4 * alpha_prev_sq) - alpha_prev_sq) / 2
            beta = alpha_prev * (1 - alpha_prev) / (alpha_prev_sq + alpha)
            # momentum step
            H_diff[:] = H_diff * beta
            Y[:] = self._H + H_diff
            alpha_prev = alpha
            iter += 1
