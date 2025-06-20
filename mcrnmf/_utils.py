import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def prefix_isotonic_regression(
    y: npt.NDArray[np.float64], w: npt.NDArray[np.float64], is_increasing: bool
) -> npt.NDArray[np.float64]:
    """
    Perform prefix isotonic regression on a given sequence with weights.

    This function computes the prefix isotonic regression of a given sequence `y`
    with corresponding weights `w`. The regression can be either increasing or
    decreasing based on the `is_increasing` flag. The function returns an array
    of errors associated with the isotonic regression at each prefix of the
    sequence.

    Parameters
    ----------
    y : NDArray[np.floating]
        The input 1D array of values to be isotonic-regressed.
    w : NDArray[np.floating]
        The 1D array of weights corresponding to the values in `y`.
    is_increasing : bool
        A flag indicating the direction of the isotonic regression.

            - If True, the regression is performed in an increasing manner.

            - If False, the regression is performed in a decreasing manner.

    Returns
    -------
    np.ndarray
        An array of errors for each prefix of the sequence after performing
        isotonic regression.
    """
    n = y.size
    if is_increasing:
        mean = y.copy()
        sum_wy2 = y**2 * w**2
        sum_wy = w * y
        sum_w = w.copy()
    else:
        mean = y[::-1].copy()
        sum_wy = w[::-1] * y[::-1]
        sum_wy2 = (w[::-1] ** 2) * (y[::-1] ** 2)
        sum_w = w[::-1].copy()

    error = np.zeros(n)
    lvl_error = np.zeros(n)
    vec_back = 0
    for j in range(1, n):
        error[j] = error[j - 1]
        while (vec_back >= 0) and (mean[j] <= mean[vec_back]):
            sum_wy[j] += sum_wy[vec_back]
            sum_wy2[j] += sum_wy2[vec_back]
            sum_w[j] += sum_w[vec_back]
            mean[j] = sum_wy[j] / sum_w[j]
            lvl_error[j] = sum_wy2[j] - (sum_wy[j] ** 2 / sum_w[j])
            error[j] -= lvl_error[vec_back]
            vec_back -= 1
        vec_back += 1
        sum_wy[vec_back] = sum_wy[j]
        sum_wy2[vec_back] = sum_wy2[j]
        sum_w[vec_back] = sum_w[j]
        mean[vec_back] = mean[j]
        lvl_error[vec_back] = lvl_error[j]
        error[j] += lvl_error[vec_back]

    return error


def simplex_proj(Y: npt.NDArray[np.float64]):
    """
    In-place projection of matrix Y onto the simplex.

    Solves:

        min_{x>=0} ||x - y(:, j)||_2  for all j

            s.t:   x^T e <= e

    If Y is a 2D array of shape [d, n] then the n points in d

    Parameters
    ----------
    Y : ndarray
        Matrix or vector to be projected.

        If Y is a 2D array of shape `(d, n)`, then the `n` columns are project to a
        `d` dimensional simplex

        If Y is a 1D array of shape `(n,)` then each element of the array is
        projected to a 1-dimensional simplex, i.e., a line segment.

    """
    np.maximum(Y, 0, out=Y)
    if Y.ndim == 1:
        # This case would arise only while using rank = 1 during factorization.
        np.minimum(Y, 1, out=Y)
    else:
        col_idx_arr = np.where(np.sum(Y, axis=0) > 1)[0]
        # although _simplex_col_proj does the projection in-place we need the
        # reassignment because Y[:, col_idx_arr] slicing creates a copy. Hence the
        # in-place operation is done on the copy.
        if col_idx_arr.size > 0:
            Y[:, col_idx_arr] = simplex_proj_using_axis(Y[:, col_idx_arr], is_col=True)


def simplex_proj_using_axis(B: npt.NDArray[np.float64], is_col: bool):
    """
    Performs in-place projection of matrix B's columns or rows to simplex and
    non-negative orthant.

    Parameters
    ----------
    B : NDArray[np.float64]
        Matrix to be projected.

    is_col : bool
        If True, project columns; if False, project rows.

    Returns
    -------
    NDArray[np.float64]
        Projected matrix B

    """
    if is_col:
        d, n = B.shape
        B_sorted = np.sort(B, axis=0)[::-1, :]
        B_temp = (np.cumsum(B_sorted, axis=0) - 1) / (
            np.arange(1, d + 1)[:, np.newaxis]
        )
        rho = np.sum(B_sorted > B_temp, axis=0)
        lambdaa = B_temp[rho - 1, np.arange(n)]
        np.maximum(B - lambdaa, 0, out=B)
    else:
        n, d = B.shape
        B_sorted = np.sort(B, axis=1)[:, ::-1]
        B_temp = (np.cumsum(B_sorted, axis=1) - 1) / np.arange(1, d + 1)
        rho = np.sum(B_sorted > B_temp, axis=1)
        lambdaa = B_temp[np.arange(n), rho - 1]
        np.maximum(B - lambdaa[:, np.newaxis], 0, out=B)

    return B


@njit
def unimodal_reg_l2(
    y: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Perform unimodal regression with L2 loss on a given sequence `y`.

    This function computes the unimodal regression of a given sequence `y`
    with corresponding weights `w`. It finds the mode of the sequence and
    performs isotonic regression on both the ascending and descending parts.

    Based on the cpp implementation in UniIsoRegression package (see [1]_)

    Parameters
    ----------
    y : np.ndarray
        The input 1D array of values to be unimodal-regressed.
    w : np.ndarray
        The 1D array of weights corresponding to the values in `y`.

    Returns
    -------
    np.ndarray
        The unimodal-regressed sequence.

    Notes
    -----
    The function uses a helper function `prefix_isotonic_regression` to
    perform isotonic regression on prefixes of the sequence.

    References
    ----------
    .. [1] https://github.com/xzp1995/UniIsoRegression

    """

    # compute errors for ascending and descending isotonic regression
    error_asc = prefix_isotonic_regression(y=y, w=w, is_increasing=True)
    error_des = prefix_isotonic_regression(y=y, w=w, is_increasing=False)
    # find the mode by minimizing total error
    n = y.size
    total_error = np.zeros(n + 1)
    total_error[0] = error_des[-1]
    total_error[-1] = error_asc[-1]
    total_error[1:-1] = error_asc[:-1] + np.flip(error_des[1:])
    # there could be multiple modes with same minima. In those cases it picks the first
    # minima idx as mode
    mode_idx = np.argmin(total_error)

    # find the unimodal fit
    left = np.arange(n, dtype=np.int64)
    right = np.arange(n, dtype=np.int64)
    mean = np.zeros(n)
    sum_wy = np.zeros(n)
    sum_w = np.zeros(n)
    yfit = np.zeros(n)

    # perform regression on ascending part
    if mode_idx != 0:
        mean[:mode_idx] = y[:mode_idx]
        sum_wy[:mode_idx] = y[:mode_idx] * w[:mode_idx]
        sum_w[:mode_idx] = w[:mode_idx]
        vec_back = 0
        for j in range(1, mode_idx):
            while (vec_back >= 0) and (mean[j] <= mean[vec_back]):
                left[j] = left[vec_back]
                sum_wy[j] += sum_wy[vec_back]
                sum_w[j] += sum_w[vec_back]
                mean[j] = sum_wy[j] / sum_w[j]
                vec_back -= 1
            vec_back += 1
            left[vec_back] = left[j]
            right[vec_back] = right[j]
            sum_wy[vec_back] = sum_wy[j]
            sum_w[vec_back] = sum_w[j]
            mean[vec_back] = mean[j]
        for k in range(vec_back + 1):
            yfit[left[k] : right[k] + 1] = mean[k]

    # perform regression on descending part
    if mode_idx != n:
        mean[: n - mode_idx] = np.flip(y[mode_idx:])
        left[: n - mode_idx] = np.arange(n - mode_idx, dtype=np.int64)
        right[: n - mode_idx] = np.arange(n - mode_idx, dtype=np.int64)
        sum_w[: n - mode_idx] = np.flip(w[mode_idx:])
        sum_wy[: n - mode_idx] = np.flip(w[mode_idx:] * y[mode_idx:])

        vec_back = 0
        for j in range(1, n - mode_idx):
            while (vec_back >= 0) and (mean[j] <= mean[vec_back]):
                left[j] = left[vec_back]
                sum_wy[j] += sum_wy[vec_back]
                sum_w[j] += sum_w[vec_back]
                mean[j] = sum_wy[j] / sum_w[j]
                vec_back -= 1
            vec_back += 1
            left[vec_back] = left[j]
            right[vec_back] = right[j]
            sum_wy[vec_back] = sum_wy[j]
            sum_w[vec_back] = sum_w[j]
            mean[vec_back] = mean[j]
        for k in range(vec_back + 1):
            yfit[n - right[k] - 1 : n - left[k]] = mean[k]

    # mode index w.r.t to input array since `total_error` has n + 1 elements.
    mode_idx -= 1

    return yfit
