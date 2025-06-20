import numpy as np
from numpy.typing import NDArray
import pytest

from mcrnmf.nnls import FPGM, _initialize_H, _setup_input_args


def is_unimodal(arr_1d: NDArray) -> bool:
    """Verifies that the array has at most one peak"""
    increasing = False
    decreasing = False
    for i in range(1, arr_1d.size):
        if arr_1d[i] > arr_1d[i - 1]:
            increasing = True
            if decreasing:
                return False
        elif arr_1d[i] < arr_1d[i - 1]:
            decreasing = True
    return True


class TestFPGMValidation:
    """Test initialization parameter validation in FPGM."""

    def test_constraint_kind_validation(self):
        """Test constraint_kind parameter validation."""
        # Type validation
        msg = r"`constraint_kind` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            FPGM(constraint_kind="1")
        with pytest.raises(TypeError, match=msg):
            FPGM(constraint_kind=1.0)

        # Value validation
        msg = r"`constraint_kind` must be in 0, 1, 2, or 3."
        with pytest.raises(ValueError, match=msg):
            FPGM(constraint_kind=4)
        with pytest.raises(ValueError, match=msg):
            FPGM(constraint_kind=-1)

    def test_iter_max_validation(self):
        """Test iter_max parameter validation."""
        # Type validation
        msg = r"`iter_max` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            FPGM(iter_max="200")
        with pytest.raises(TypeError, match=msg):
            FPGM(iter_max=200.0)

        # Value validation
        msg = r"`iter_max` must be > 0"
        with pytest.raises(ValueError, match=msg):
            FPGM(iter_max=0)
        with pytest.raises(ValueError, match=msg):
            FPGM(iter_max=-10)

    def test_tol_validation(self):
        """Test tol parameter validation."""
        # Type validation
        msg = r"`tol` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            FPGM(tol=1)
        with pytest.raises(TypeError, match=msg):
            FPGM(tol="0.001")

        # Value validation
        msg = r"`tol` must be in the open interval \(0, 1\)"
        with pytest.raises(ValueError, match=msg):
            FPGM(tol=0.0)
        with pytest.raises(ValueError, match=msg):
            FPGM(tol=1.0)
        with pytest.raises(ValueError, match=msg):
            FPGM(tol=2.0)


class TestSetupInputArgs:
    """Test the validation in _setup_input_args function."""

    @pytest.fixture
    def valid_matrices(self):
        """Create valid matrices for testing."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 10)).astype(np.float64)
        W = rng.random((20, 5)).astype(np.float64)
        H = rng.random((5, 10)).astype(np.float64)
        return X, W, H

    def test_input_type_validation(self, valid_matrices):
        """Test validation of input types."""
        X, W, H = valid_matrices

        # X must be ndarray
        with pytest.raises(TypeError, match=r"`X` must be a numpy array."):
            _setup_input_args(X.tolist(), W, H)

        # W must be ndarray
        with pytest.raises(TypeError, match=r"`W` must be a numpy array."):
            _setup_input_args(X, W.tolist(), H)

        # H must be ndarray or None
        with pytest.raises(TypeError, match=r"`H` must be a numpy array or None."):
            _setup_input_args(X, W, "not_array")

    def test_shape_validation(self, valid_matrices):
        """Test validation of matrix shapes."""
        X, W, H = valid_matrices

        # W rows must match X rows
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: rows of `W`: \d+ not equal to rows of X: \d+",
        ):
            _setup_input_args(X, W[1:, :], H)

        # H rows must match W columns
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: rows of `H`: \d+ not equal to columns of W: \d+",
        ):
            _setup_input_args(X, W, H[1:, :])

        # H cols must match X cols
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: columns of `H`: \d+ not equal to columns of X: \d+",
        ):
            _setup_input_args(X, W, H[:, 1:])

    def test_nonnegativity_validation(self, valid_matrices):
        """Test validation of non-negativity constraints."""
        X, W, H = valid_matrices

        # H must be non-negative
        H_neg = H.copy()
        H_neg[0, 0] = -0.1
        with pytest.raises(ValueError, match=r"All elements of `H` must be >= 0."):
            _setup_input_args(X, W, H_neg)

    def test_H_initialization(self, valid_matrices):
        """Test H initialization when None is provided."""
        X, W, _ = valid_matrices

        # When H is None, it should be initialized
        X_out, W_out, H_out = _setup_input_args(X, W, None)

        # Check that shapes are correct
        assert H_out.shape == (W.shape[1], X.shape[1])

        # Verify contiguous arrays with float64 dtype
        assert X_out.flags.c_contiguous
        assert W_out.flags.c_contiguous
        assert H_out.flags.c_contiguous
        assert X_out.dtype == np.float64
        assert W_out.dtype == np.float64
        assert H_out.dtype == np.float64

        # Check that H is non-negative
        assert np.all(H_out >= 0)

    def test_valid_inputs_pass_validation(self, valid_matrices):
        """Test that valid inputs pass validation without errors."""
        X, W, H = valid_matrices
        X_out, W_out, H_out = _setup_input_args(X, W, H)

        # Check that shapes are preserved
        assert X_out.shape == X.shape
        assert W_out.shape == W.shape
        assert H_out.shape == H.shape


class TestInitializeH:
    """Test the H initialization function."""

    @pytest.fixture
    def matrices(self):
        """Create matrices for testing initialization."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 10)).astype(np.float64)
        W = rng.random((20, 5)).astype(np.float64)
        return X, W

    def test_initialize_H_shape(self, matrices):
        """Test that initialized H has the correct shape."""
        X, W = matrices
        H = _initialize_H(X, W)
        assert H.shape == (W.shape[1], X.shape[1])

    def test_initialize_H_nonnegativity(self, matrices):
        """Test that initialized H is non-negative."""
        X, W = matrices
        H = _initialize_H(X, W)
        assert np.all(H >= 0)

    def test_initialize_H_warning(self):
        """Test warning for ill-conditioned W."""
        # Create a matrix W with high condition number
        X = np.ones((5, 3))
        W = np.ones((5, 2))  # Rank deficient

        with pytest.warns(UserWarning, match=r"`W` has very high condition number.*"):
            H = _initialize_H(X, W)


class TestApplyProjection:
    """Test the _apply_projection method of FPGM."""

    @pytest.fixture
    def test_matrix(self):
        """Create a test matrix for projection."""
        rng = np.random.default_rng(42)
        A = rng.random((3, 10)).astype(np.float64)
        A[0, 0] = -0.5  # Add a negative value
        return A

    def test_projection_constraint_0(self, test_matrix):
        """Test projection with constraint_kind=0."""
        fpgm = FPGM(constraint_kind=0)
        A = test_matrix.copy()
        fpgm._apply_projection(A, 0)

        # Check that all values are non-negative
        assert np.all(A >= 0)
        # Check that only negative values were modified
        assert np.isclose(A[0, 0], 0)

    def test_projection_constraint_1(self, test_matrix):
        """Test projection with constraint_kind=1."""
        fpgm = FPGM(constraint_kind=1)
        A = test_matrix.copy()
        fpgm._apply_projection(A, 1)

        # Check that all values are non-negative
        assert np.all(A >= 0)
        # Check that column sums are <= 1
        assert np.all(np.sum(A, axis=0) <= 1.0 + 1e-10)

    def test_projection_constraint_2(self, test_matrix):
        """Test projection with constraint_kind=2."""
        fpgm = FPGM(constraint_kind=2)
        A = test_matrix.copy()
        fpgm._apply_projection(A, 2)

        # Check that all values are non-negative
        assert np.all(A >= 0)
        # Check that row sums equal 1
        assert np.allclose(np.sum(A, axis=1), 1.0, atol=1e-10)

    def test_projection_constraint_3(self, test_matrix):
        """Test projection with constraint_kind=3."""
        fpgm = FPGM(constraint_kind=3)
        A = test_matrix.copy()
        fpgm._apply_projection(A, 3)

        # Check that all values are non-negative
        assert np.all(A >= 0)
        # Check that column sums equal 1
        assert np.allclose(np.sum(A, axis=0), 1.0, atol=1e-10)


class TestSolve:
    """Test the solve method of FPGM."""

    @pytest.fixture
    def test_data(self):
        """Create test data for solve."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 10)).astype(np.float64)
        W = rng.random((20, 5)).astype(np.float64)
        H = rng.random((5, 10)).astype(np.float64)
        return X, W, H

    @pytest.mark.parametrize("constraint_kind", [0, 1, 2, 3])
    def test_solve_constraints(self, test_data, constraint_kind):
        """Test that solve respects different constraints."""
        X, W, _ = test_data
        fpgm = FPGM(constraint_kind=constraint_kind, iter_max=10)
        H, _ = fpgm.solve(X, W)

        # Check that solution is non-negative
        assert np.all(H >= 0)

        if constraint_kind == 1:
            # Check that column sums are <= 1
            assert np.all(np.sum(H, axis=0) <= 1.0 + 1e-10)
        elif constraint_kind == 2:
            # Check that row sums equal 1
            assert np.allclose(np.sum(H, axis=1), 1.0, atol=1e-10)
        elif constraint_kind == 3:
            # Check that column sums equal 1
            assert np.allclose(np.sum(H, axis=0), 1.0, atol=1e-10)

    def test_solve_with_H(self, test_data):
        """Test solve with provided initial H."""
        X, W, H = test_data
        fpgm = FPGM(iter_max=10)
        H_out, _ = fpgm.solve(X, W, H)

        # Check that output has same shape as input
        assert H_out.shape == H.shape
        # Check that H was modified (not identical to input)
        assert not np.array_equal(H_out, H)

    def test_solve_convergence(self, test_data):
        """Test solve convergence flag."""
        X, W, _ = test_data

        # With strict tolerance and few iterations, should not converge
        fpgm_no_converge = FPGM(iter_max=1, tol=1e-10)
        _, converged_1 = fpgm_no_converge.solve(X, W)
        assert converged_1 is False

        # With relaxed tolerance and more iterations, should converge
        fpgm_converge = FPGM(iter_max=100, tol=1e-2)
        _, converged_2 = fpgm_converge.solve(X, W)
        assert converged_2 is True

    def test_solve_objective_decreases(self, test_data):
        """Test that the objective function decreases during solve."""
        X, W, H_init = test_data

        # Calculate initial error
        initial_err = np.linalg.norm(X - W @ H_init, "fro")

        # Solve with initial H
        fpgm = FPGM(iter_max=50)
        H_final, _ = fpgm.solve(X, W, H_init.copy())

        # Calculate final error
        final_err = np.linalg.norm(X - W @ H_final, "fro")

        # Check that error decreased
        assert final_err <= initial_err


class TestSolveKnownH:
    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng(0)
        X = rng.random((8, 6)).astype(np.float64)
        W = rng.random((8, 4)).astype(np.float64)
        return X, W

    def test_known_H_type_and_shape(self, test_data):
        X, W = test_data
        fpgm = FPGM()
        with pytest.raises(
            TypeError, match=r"`known_H` must be None or a numpy array."
        ):
            fpgm.solve(X, W, known_H=123)
        bad_shape = np.empty((W.shape[1] + 1, X.shape[1]))
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: number of columns of `known_H` not equal to number "
            "of columns of `W`",
        ):
            fpgm.solve(X, W, known_H=bad_shape)

    def test_known_H_all_specified(self, test_data):
        X, W = test_data
        full = np.zeros((W.shape[1], X.shape[1]))
        with pytest.raises(
            ValueError, match=r"All values of `H` are specified as known, can't solve"
        ):
            fpgm = FPGM()
            fpgm.solve(X, W, known_H=full)

    def test_known_H_negative_finite(self, test_data):
        X, W = test_data
        known = np.full((W.shape[1], X.shape[1]), np.nan)
        known[0, 0] = -0.1
        fpgm = FPGM()
        with pytest.raises(
            ValueError, match=r"All finite elements of `known_H` must be >= 0"
        ):
            fpgm.solve(X, W, known_H=known)

    def test_known_H_no_finite_warns_and_ignores(self, test_data):
        X, W = test_data
        known = np.full((W.shape[1], X.shape[1]), np.nan)
        fpgm = FPGM(iter_max=5)
        with pytest.warns(
            UserWarning,
            match=r"`known_H` contains no finite values, setting it to None.",
        ):
            H_out, _ = fpgm.solve(X, W, known_H=known)
        # should behave like no known_H
        assert H_out.shape == (W.shape[1], X.shape[1])
        assert np.all(H_out >= 0)

    def test_solve_with_known_H_values_fixed(self, test_data):
        X, W = test_data
        H_init = np.zeros((W.shape[1], X.shape[1]))
        known = np.full_like(H_init, np.nan)
        known[1, 2] = 0.5
        known[3, 4] = 1.0
        fpgm = FPGM(iter_max=20)
        H_out, _ = fpgm.solve(X, W, known_H=known)
        assert np.isclose(H_out[1, 2], 0.5)
        assert np.isclose(H_out[3, 4], 1.0)
        # other entries free and non-negative
        mask = np.isnan(known)
        assert np.all(H_out[mask] >= 0)


class TestSolveUnimodalH:
    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng(1)
        X = rng.random((7, 9)).astype(np.float64)
        W = rng.random((7, 3)).astype(np.float64)
        return X, W

    def test_unimodal_H_type_and_length(self, test_data):
        X, W = test_data
        fpgm = FPGM()
        with pytest.raises(
            TypeError, match=r"`unimodal_H` must be of type bool or list of bool"
        ):
            fpgm.solve(X, W, unimodal_H=123)
        with pytest.raises(
            TypeError,
            match=r"If `unimodal_H` is a list, it must contain only boolean elements",
        ):
            fpgm.solve(X, W, unimodal_H=[True, 1, False])
        with pytest.raises(
            ValueError,
            match="If `unimodal_H` is specified as list, then the length of list must "
            "be equal to number of columns of W",
        ):
            fpgm.solve(X, W, unimodal_H=[True] * (W.shape[1] + 1))

    def test_solve_unimodal_all_rows(self, test_data):
        X, W = test_data
        fpgm = FPGM(iter_max=30)
        H_out, _ = fpgm.solve(X, W, unimodal_H=True)
        for row in range(W.shape[1]):
            assert is_unimodal(H_out[row, :])

    def test_solve_unimodal_selected_rows(self, test_data):
        X, W = test_data
        mask = [True, False, True]
        fpgm = FPGM(iter_max=30)
        H_out, _ = fpgm.solve(X, W, unimodal_H=mask)
        assert is_unimodal(H_out[0, :])
        assert is_unimodal(H_out[2, :])
        assert H_out.shape == (W.shape[1], X.shape[1])
