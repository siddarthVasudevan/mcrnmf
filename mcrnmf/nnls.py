import warnings
import numpy as np
from numpy.typing import NDArray

from ._utils import simplex_proj, simplex_proj_using_axis, unimodal_reg_l2


class FPGM:
    """
    An implementation of Non-negative Least Squares solver using Fast Projected Gradient
    Method.

    Solves for :math:`H` given :math:`X` and :math:`W` by minimizing the following
    objective function with other optional constraints on :math:`H`

    .. math::
        ||X - WH||_{F}^2

    where :math:`||\\cdot||_F` is the Frobenius norm.

    Parameters
    ----------
    constraint_kind : integer-like {0, 1, 2, 3}, default=0
        The following constraints are applied based on the integer value specified

        * If ``0``: Only :math:`H \\geq 0`.
        * If ``1``: Closure constraint :math:`H^T e ≤ e`.
        * If ``2``: Closure constraint :math:`H e = e`.
        * If ``3``: Closure constraint :math:`H^T e = e`.
    iter_max : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence. Must be in the interval :math:`(0, 1)`.

            Convergence is reached when:

            .. math::

                {||H^{i} - H^{i-1}||_{F} \\over ||H^{1} - H^{0}||_{F}} \\leq
                \\textrm{tol}

            where :math:`H^{i}` is :math:`H` after iteration :math:`i` and :math:`H^{0}` is
            initial :math:`H`.

    References
    ----------
    .. [1] Gillis, N. (2020). Nonnegative matrix factorization. Society for Industrial
           and Applied Mathematics.
    .. [2] Lin, Chih-Jen. "Projected gradient methods for nonnegative matrix
           factorization." Neural computation 19.10 (2007): 2756-2779.

    Examples
    --------
    >>> from mcrnmf.nnls import FPGM
    >>> from mcrnmf import SNPA
    >>> from mcrnmf.datasets import load_rxn_spectra
    >>>
    >>> # load the dataset available in the package
    >>> X, wv, time = load_rxn_spectra()
    >>> # Get an estimate for W from SNPA
    >>> snpa = SNPA(rank=4)
    >>> snpa.fit(X=X)
    >>> W = snpa.W
    >>>
    >>> # create an instance of FPGM
    >>> fpgm = FPGM(constraint_kind=1, iter_max=400, tol=1e-4)
    >>> # solve for H
    >>> H, is_converged = fpgm.solve(X=X, W=W, unimodal_H=True)

    """

    def __init__(self, constraint_kind=0, iter_max=500, tol=1e-6):
        # validate types
        if not isinstance(constraint_kind, (int, np.integer)):
            raise TypeError(
                f"`constraint_kind` must be of type {int}, current type is "
                f"{type(constraint_kind)}"
            )
        if not isinstance(iter_max, (int, np.integer)):
            raise TypeError(
                f"`iter_max` must be of type {int}, current type is {type(iter_max)}"
            )
        if not isinstance(tol, float):
            raise TypeError(
                f"`tol` must be of type {float}, current type is {type(tol)}"
            )

        # validate values
        if constraint_kind not in [0, 1, 2, 3]:
            raise ValueError("`constraint_kind` must be in 0, 1, 2, or 3.")
        if iter_max <= 0:
            raise ValueError("`iter_max` must be > 0")
        if not (0 < tol < 1):
            raise ValueError("`tol` must be in the open interval (0, 1)")

        self.constraint_kind_ = constraint_kind
        self.iter_max_ = iter_max
        self.tol_ = tol

    def solve(
        self,
        X: NDArray[np.float64],
        W: NDArray[np.float64],
        H: NDArray[np.float64] | None = None,
        known_H: NDArray[np.float64] | None = None,
        unimodal_H: bool | list[bool] = False,
    ):
        """
        Solves for :math:`H` given :math:`X` and :math:`W`.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Data array.

        W : ndarray of shape (m, k)
            Coefficient array.

        H : ndarray of shape (k, n) or None, defualt=None
            Initial guess for the solution array. If ``None``, an initial guess is
            generated using least squares solution of :math:`WH \\approx X`.

        known_H : ndarray of shape (rank, n_samples), default=None
            Array containing known values of :math:`H`.

            * The ``np.nan`` elements of the array are treated as unknown.
            * Equality constraint is applied at those indices of :math:`H` which do not
              correspond ``np.nan`` entries in ``known_H``.

        unimodal_H : bool or list of bool, default=False
            Applies unimodality constraint to rows of H.

                * If ``bool`` and ``True`` applies unimodality constraint to all rows
                  of `H`
                * If ``list[bool]``:

                    * Length of list must be equal to number of columns in `W`.
                    * Applies unimodality constraint only those rows of `H` which are
                      marked as ``True``. Example: [True, False, True] applies
                      unimodality constraint to rows 1 and 3 of `H`.

        Returns
        -------
        H : ndarray of shape (k, n)
            Solution array.

        is_converged : bool
            Whether algorithm converged within ``iter_max`` iterations.

            * ``True`` if convergence was reached based on ``tol`` criterion
            * ``False`` if maximum iterations were reached without convergence

        """

        X, W, H = _setup_input_args(X=X, W=W, H=H)

        # check known_H
        if not ((known_H is None) or (isinstance(known_H, np.ndarray))):
            raise TypeError("`known_H` must be None or a numpy array.")
        elif isinstance(known_H, np.ndarray):
            if known_H.shape[0] != W.shape[1]:
                raise ValueError(
                    "Shape mismatch: number of columns of `known_H` not equal to "
                    "number of columns of `W`"
                )
        if known_H is not None:
            mask_known_H = np.isfinite(known_H)
            tot_fvals_knw_H = np.sum(mask_known_H)
            if tot_fvals_knw_H == 0:
                warnings.warn(
                    "`known_H` contains no finite values, setting it to None."
                )
                known_H = None
            elif tot_fvals_knw_H == known_H.size:
                raise ValueError(
                    "All values of `H` are specified as known, can't solve."
                )
            else:
                # finite elems are non-neg
                if known_H[mask_known_H].min() < 0:
                    raise ValueError("All finite elements of `known_H` must be >= 0.")
        # using if statement instead of else with previous because known_H is set to
        # None if none of the values are finite.
        if known_H is None:
            mask_known_H = None
        # check unimodal arg
        if not isinstance(unimodal_H, (bool, list)):
            raise TypeError(f"`unimodal_H` must be of type bool or list of bool")
        elif isinstance(unimodal_H, list):
            if not all(isinstance(elem, bool) for elem in unimodal_H):
                raise TypeError(
                    f"If `unimodal_H` is a list, it must contain only boolean elements"
                )
        if isinstance(unimodal_H, list):
            if len(unimodal_H) != W.shape[1]:
                raise ValueError(
                    "If `unimodal_H` is specified as list, then the length of list "
                    "must be equal to number of columns of W"
                )
        else:
            unimodal_H = [unimodal_H] * W.shape[1]

        # precompute WtW, and WtX since it doesn't change in the loop
        WtW = W.T @ W
        WtX = W.T @ X

        # hyperparameters for optimization
        l = np.linalg.norm(WtW, 2)  # lipschitz constant
        theta_prev = 0.05  # first value of theta, heuristic

        # pre-allocate memory for matrices computed or used repeatedly in the loop
        Y = H.copy()  # look ahead variable
        Y_grad = np.zeros(H.shape, dtype=H.dtype)
        H_diff = np.zeros(H.shape, dtype=H.dtype)
        H_prev = np.zeros(H.shape, dtype=H.dtype)

        i = 0
        is_converged = False
        while i < self.iter_max_:
            np.copyto(H_prev, H)
            # compute the gradient using look ahead point
            np.matmul(WtW, Y, out=Y_grad)
            np.subtract(Y_grad, WtX, out=Y_grad)
            np.divide(Y_grad, l, out=Y_grad)
            # gradient step: H = Y - ((W.T @ W @ H) - (W.T @ X))) / l
            np.subtract(Y, Y_grad, out=H)

            if mask_known_H is not None:
                H[mask_known_H] = known_H[mask_known_H]
            self._apply_unimodal_H_rows(H=H, unimodal_H=unimodal_H)
            self._apply_projection(A=H, constraint_kind=self.constraint_kind_)

            np.subtract(H, H_prev, out=H_diff)
            norm_H_diff = np.linalg.norm(H_diff, "fro")
            if i == 0:
                norm_H_diff_0 = norm_H_diff

            # convergence check
            if norm_H_diff <= (self.tol_ * norm_H_diff_0):
                is_converged = True
                break

            # do the momentum update only if convergence criteria is not satisfied
            theta_prev_sq = theta_prev**2
            # theta and beta computation based on scheme 2.2.19 in Introductory lectures
            # on convex optimization by Nesterov
            theta = (np.sqrt(theta_prev_sq**2 + 4 * theta_prev_sq) - theta_prev_sq) / 2
            beta = theta_prev * (1 - theta_prev) / (theta_prev_sq + theta)
            # momentum step
            np.multiply(H_diff, beta, out=H_diff)
            np.add(H, H_diff, out=Y)

            theta_prev = theta
            i += 1

        return H, is_converged

    def _apply_unimodal_H_rows(self, H, unimodal_H):
        """
        Apply unimodal constraints to rows of H in-place.

        Note: This is private method intended for internal use only.

        Parameters
        ----------
        H : NDArray[np.float64]
            The current value of H
        unimodal_H : list of bool
            A list with boolean values specifying to which rows of H unimodal
            constraints should be applied.
        """
        nrows, ncols = H.shape
        w = np.ones(ncols)
        for row in range(nrows):
            if unimodal_H[row] is True:
                H[row, :] = unimodal_reg_l2(y=H[row, :], w=w)

    def _apply_projection(self, A, constraint_kind):
        """
        Perform in-place projection of A based on the constraint.

        Note: This is private method intended for internal use only.

        Parameters
        ----------
        A : NDArray[np.float64]
            H matrix to be projected.

        constraint_kind: int | None
            The kind of constraint to use during projection.

            If constraint_kind=0, then A >= 0

            If constraint_kind=1 then A^Te <= e and A >= 0

            If constraint_kind=2 then Ae = e and A >= 0

            If constraint_kind=3 then A^Te = e and A >= 0
        """
        if constraint_kind == 0:
            # performs projection such that H >= 0
            np.maximum(A, 0, out=A)
        elif constraint_kind == 1:
            # performs projection such that H^T*e <= e and H >= 0
            simplex_proj(A)
        elif constraint_kind == 2:
            # performs projection such that H*e = e and H >= 0
            A[:] = simplex_proj_using_axis(A, is_col=False)
        elif constraint_kind == 3:
            # performs projection such that H^T*e = e and H >= 0
            A[:] = simplex_proj_using_axis(A, is_col=True)


def _setup_input_args(X, W, H):
    """
    Validate the shapes of matrices used in NNLS and generates an initial guess for H
    if it is None, else validates it

    Note: This is a private method intended for internal use only.

    Parameters
    ----------
    X : ndarray
        Input data matrix of shape (m, n).

    W : ndarray
        Left factor matrix of shape (m, r).

    H : ndarray | None, optional
        Initial matrix of shape (r, n) to be updated. Default is None.

        If None, then an initial guess of `H` is generated.

    """
    # check type
    if not isinstance(X, np.ndarray):
        raise TypeError("`X` must be a numpy array.")
    if not isinstance(W, np.ndarray):
        raise TypeError("`W` must be a numpy array.")

    is_H_guessed = False
    if H is None:
        H = _initialize_H(X=X, W=W)
        is_H_guessed = True
    else:
        if not isinstance(H, np.ndarray):
            raise TypeError("`H` must be a numpy array or None.")
    # check shape matches
    m, n = X.shape
    rank = W.shape[1]

    if W.shape[0] != m:
        raise ValueError(
            f"Shape mismatch: rows of `W`: {W.shape[0]} not equal to rows of X: {m}"
        )
    if not is_H_guessed:
        if H.shape[0] != rank:
            raise ValueError(
                f"Shape mismatch: rows of `H`: {H.shape[0]} not equal to columns of W: "
                f"{rank}"
            )
        if H.shape[1] != n:
            raise ValueError(
                f"Shape mismatch: columns of `H`: {H.shape[1]} not equal to columns of "
                f"X: {n}."
            )

    if not np.all(H >= 0):
        raise ValueError("All elements of `H` must be >= 0.")

    # ensure all arrays are of type float64 and C-contiguous
    X = np.ascontiguousarray(X.copy(), dtype=np.float64)
    W = np.ascontiguousarray(W.copy(), dtype=np.float64)
    H = np.ascontiguousarray(H.copy(), dtype=np.float64)

    return X, W, H


def _initialize_H(X, W):
    """
    Generated initial guess for H in the NNLS problem.

        min_{H>=0} ||X - WH||_F^2

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (m, n)

    W : np.ndarray
        Matrix of shape (m, k)
    """

    if np.linalg.cond(W) > 1e6:
        warnings.warn(
            "`W` has very high condition number. Initial guess of H generated might be "
            "poor."
        )
    H = np.linalg.lstsq(W, X, rcond=None)[0]
    np.maximum(H, 0, out=H)
    alpha = np.sum(H * (W.T @ X)) / np.sum((W.T @ W) * (H @ H.T))
    H *= alpha  # scale H

    # replaces rows which sum to zero with small random values
    zero_sum_rows = np.where(np.sum(H, axis=1) == 0)[0]
    if len(zero_sum_rows) > 0:
        H[zero_sum_rows, :] = (
            0.001 * np.max(H) * np.random.rand(len(zero_sum_rows), H.shape[1])
        )

    return H
