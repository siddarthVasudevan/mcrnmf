import numpy as np
import pytest

from mcrnmf.models import FroALS, FroFPGM, MinVol, SNPA


class TestBaseNMFValidation:
    """Test initialization parameter validation in _BaseNMF."""

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_rank_validation(self, model_class):
        """Test rank parameter validation."""
        # type validation
        msg = r".*`rank` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            model_class(rank="2")
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2.0)

        # value validation
        msg = "`rank` must be > 0"
        with pytest.raises(ValueError, match=msg):
            model_class(rank=0)
        with pytest.raises(ValueError, match=msg):
            model_class(rank=-3)

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_constraint_kind_validation(self, model_class):
        """Test constraint_kind parameter validation"""
        # type validation
        msg = r".*`constraint_kind` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, constraint_kind="1")
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, constraint_kind=1.0)

        # value validation
        msg = r"`constraint_kind` must be 0, 1, 2, 3, or 4"
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, constraint_kind=5)

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_unimodal_constraints_validation(self, model_class):
        """Test unimodal_constraints parameter validation."""
        # type validation
        msg = r"`unimodal` must be None or a dictionary"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, unimodal="dict")
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, unimodal=1)

        # empty dictionary validation
        msg = r"If not None, unimodal dictionary cannot be empty; must contain 'W' "
        "and/or 'H' keys."
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, unimodal={})

        # invalid key validation
        msg = r"`unimodal` dictionary can contain only keys in: \['W', 'H'\]"
        with pytest.raises(KeyError, match=msg):
            model_class(rank=2, unimodal={"W": True, "invalid_key": False})

        # value type validation
        msg = r"Unimodal constraint must be bool or list of booleans"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, unimodal={"W": "True"})
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, unimodal={"H": 1})

        # list validation
        msg = r"Unimodal constraint list must contain only booleans"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, unimodal={"W": [True, 1]})

        # list length validation
        msg = r"Unimodal constraint list must have length 2"
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, unimodal={"H": [True, False, True]})

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_iter_max_validation(self, model_class):
        """Test iter_max parameter validation."""
        # type validation
        msg = r".*`iter_max` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, iter_max="200")
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, iter_max=200.0)

        # value validation
        msg = r"`iter_max` must be >= 10"
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, iter_max=0)
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, iter_max=-200)

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_tol_validation(self, model_class):
        """Test tol parameter validation."""
        # Type validation
        msg = r".*`tol` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, tol=1)
        with pytest.raises(TypeError, match=msg):
            model_class(rank=2, tol="0.001")

        # Value validation
        msg = r"`tol` must be in the open interval \(0, 1\)"
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, tol=0.0)
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, tol=1.0)
        with pytest.raises(ValueError, match=msg):
            model_class(rank=2, tol=2.0)


class TestFitArgumentsValidation:
    """Test validation of fit arguments in _BaseNMF._validate_fit_args."""

    @pytest.fixture(params=[FroALS, FroFPGM, MinVol])
    def test_model(self, request):
        """Create model instances for testing validation."""
        model_class = request.param
        return model_class(rank=3)

    @pytest.fixture
    def valid_matrices(self):
        """Create valid matrices for fit."""
        X = np.random.rand(20, 10).astype(np.float64)
        Wi = np.random.rand(20, 3).astype(np.float64)
        Hi = np.random.rand(3, 10).astype(np.float64)
        return X, Wi, Hi

    def test_input_type_validation(self, test_model, valid_matrices):
        """Test validation of input types."""
        X, Wi, Hi = valid_matrices

        # X must be ndarray
        with pytest.raises(TypeError, match=r"`X` must be a numpy array"):
            test_model._validate_fit_args(X.tolist(), Wi, Hi, None, None)

        # Wi must be ndarray
        with pytest.raises(TypeError, match=r"`Wi` must be a numpy array"):
            test_model._validate_fit_args(X, Wi.tolist(), Hi, None, None)

        # Hi must be ndarray
        with pytest.raises(TypeError, match=r"`Hi` must be a numpy array"):
            test_model._validate_fit_args(X, Wi, Hi.tolist(), None, None)

        # known_W must be ndarray or None
        with pytest.raises(TypeError, match=r"`known_W` must be None or a numpy array"):
            test_model._validate_fit_args(X, Wi, Hi, "not_array", None)

        # known_H must be ndarray or None
        with pytest.raises(TypeError, match=r"`known_H` must be None or a numpy array"):
            test_model._validate_fit_args(X, Wi, Hi, None, "not_array")

    def test_shape_validation(self, test_model, valid_matrices):
        """Test validation of matrix shapes."""
        X, Wi, Hi = valid_matrices

        # Wi rows must match X rows
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: number of rows of `Wi` not equal to number of rows "
            "of `X`",
        ):
            test_model._validate_fit_args(X, Wi[1:, :], Hi, None, None)

        # Wi cols must match rank
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: number of columns of `Wi` not equal to `rank`",
        ):
            test_model._validate_fit_args(X, Wi[:, :2], Hi, None, None)

        # Hi rows must match rank
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: number of rows of `Hi` not equal to `rank`",
        ):
            test_model._validate_fit_args(X, Wi, Hi[:2, :], None, None)

        # Hi cols must match X cols
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch: number of columns of `Hi` not equal to number of "
            "columns of `X`",
        ):
            test_model._validate_fit_args(X, Wi, Hi[:, :9], None, None)

        # known_W shape must match Wi shape
        known_W_wrong_shape = np.full((Wi.shape[0], Wi.shape[1] - 1), np.nan)
        with pytest.raises(
            ValueError, match=r"Shape mismatch: `known_W` shape not equal to `Wi` shape"
        ):
            test_model._validate_fit_args(X, Wi, Hi, known_W_wrong_shape, None)

        # known_H shape must match Hi shape
        known_H_wrong_shape = np.full((Hi.shape[0] - 1, Hi.shape[1]), np.nan)
        with pytest.raises(
            ValueError, match=r"Shape mismatch: `known_H` shape not equal to `Hi` shape"
        ):
            test_model._validate_fit_args(X, Wi, Hi, None, known_H_wrong_shape)

    def test_dtype_validation(self, test_model, valid_matrices):
        """Test validation of data types."""
        X, Wi, Hi = valid_matrices

        # X must be float64
        with pytest.raises(TypeError, match=r"The dtype of `X` elements should be"):
            test_model._validate_fit_args(X.astype(np.float32), Wi, Hi, None, None)

        # Wi must be float64
        with pytest.raises(TypeError, match=r"The dtype of `Wi` elements should be"):
            test_model._validate_fit_args(X, Wi.astype(np.float32), Hi, None, None)

        # Hi must be float64
        with pytest.raises(TypeError, match=r"The dtype of `Hi` elements should be"):
            test_model._validate_fit_args(X, Wi, Hi.astype(np.float32), None, None)

    def test_nonnegativity_validation(self, test_model, valid_matrices):
        """Test validation of non-negativity constraints."""
        X, Wi, Hi = valid_matrices

        # X must be non-negative
        X_neg = X.copy()
        X_neg[0, 0] = -0.1
        with pytest.raises(ValueError, match=r"All elements of `X` must be >= 0"):
            test_model._validate_fit_args(X_neg, Wi, Hi, None, None)

        # Wi must be non-negative
        Wi_neg = Wi.copy()
        Wi_neg[0, 0] = -0.1
        with pytest.raises(ValueError, match=r"All elements of `Wi` must be >= 0"):
            test_model._validate_fit_args(X, Wi_neg, Hi, None, None)

        # Hi must be non-negative
        Hi_neg = Hi.copy()
        Hi_neg[0, 0] = -0.1
        with pytest.raises(ValueError, match=r"All elements of `Hi` must be >= 0"):
            test_model._validate_fit_args(X, Wi, Hi_neg, None, None)

    def test_known_W_validation(self, test_model, valid_matrices):
        """Test validation of known_W."""
        X, Wi, Hi = valid_matrices

        # All NaNs in known_W should trigger warning
        known_W_all_nan = np.full(Wi.shape, np.nan)
        with pytest.warns(
            UserWarning,
            match=r"`known_W` contains no finite values, setting it to None",
        ):
            test_model._validate_fit_args(X, Wi, Hi, known_W_all_nan, None)

        # All finite values in known_W should raise error
        known_W_all_finite = np.ones_like(Wi)
        with pytest.raises(ValueError, match=r"All values of `W` are known.*"):
            test_model._validate_fit_args(X, Wi, Hi, known_W_all_finite, None)

        # Negative finite values in known_W should raise error
        known_W_with_neg = np.full(Wi.shape, np.nan)
        known_W_with_neg[0, 0] = -0.1
        with pytest.raises(
            ValueError, match=r"All finite elements of `known_W` must be >= 0"
        ):
            test_model._validate_fit_args(X, Wi, Hi, known_W_with_neg, None)

    def test_known_H_validation(self, test_model, valid_matrices):
        """Test validation of known_H."""
        X, Wi, Hi = valid_matrices

        # All NaNs in known_H should trigger warning
        known_H_all_nan = np.full(Hi.shape, np.nan)
        with pytest.warns(
            UserWarning,
            match=r"`known_H` contains no finite values, setting it to None",
        ):
            test_model._validate_fit_args(X, Wi, Hi, None, known_H_all_nan)

        # All finite values in known_H should raise error
        known_H_all_finite = np.ones_like(Hi)
        with pytest.raises(ValueError, match=r"All values of `H` are known.*"):
            test_model._validate_fit_args(X, Wi, Hi, None, known_H_all_finite)

        # Negative finite values in known_H should raise error
        known_H_with_neg = np.full(Hi.shape, np.nan)
        known_H_with_neg[0, 0] = -0.1
        with pytest.raises(
            ValueError, match=r"All finite elements of `known_H` must be >= 0"
        ):
            test_model._validate_fit_args(X, Wi, Hi, None, known_H_with_neg)

    def test_valid_inputs_pass_validation(self, test_model, valid_matrices):
        """Test that valid inputs pass validation without errors."""
        X, Wi, Hi = valid_matrices

        # Valid inputs with no known_W or known_H
        test_model._validate_fit_args(X, Wi, Hi, None, None)

        # Valid inputs with partially specified known_W and known_H
        known_W = np.full(Wi.shape, np.nan)
        known_W[0, 0] = 0.5

        known_H = np.full(Hi.shape, np.nan)
        known_H[0, 0] = 0.5

        test_model._validate_fit_args(X, Wi, Hi, known_W, known_H)


class TestBaseNMFFunctionality:
    """Test common fit functionality across NMF implementations."""

    @pytest.fixture
    def test_data(self):
        """Create test data that can be used across tests."""
        rng = np.random.default_rng(seed=42)
        X = rng.random((50, 30))
        rank = 3
        Wi = rng.random((50, rank))
        Hi = rng.random((rank, 30))
        return {"X": X, "Wi": Wi, "Hi": Hi, "rank": rank}

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_property_validation(self, test_data, model_class):
        """Test that validation of properties common to all NMF implementations."""
        model: FroALS | FroFPGM | MinVol = model_class(rank=test_data["rank"])

        with pytest.raises(AttributeError, match=r"`W` is not available.*"):
            _ = model.W

        with pytest.raises(AttributeError, match=r"`H` is not available.*"):
            _ = model.H

        with pytest.raises(AttributeError, match=r"`rel_loss_ls` is not available.*"):
            _ = model.rel_loss_ls

        with pytest.raises(AttributeError, match=r"`is_converged` is not available.*"):
            _ = model.is_converged

        # not truly a base nmf functionality but including it here for convenience
        with pytest.raises(
            AttributeError, match=r"`rel_reconstruction_error_ls` is not available.*"
        ):
            _ = model.rel_reconstruction_error_ls

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_basic_fit(self, test_data, model_class):
        """Test the basic fit method works and produces the expected attributes."""
        model: FroALS | FroFPGM | MinVol = model_class(rank=test_data["rank"])
        model.fit(X=test_data["X"], Wi=test_data["Wi"], Hi=test_data["Hi"])

        # check dimensions of resulting matrices
        assert model._W.shape == test_data["Wi"].shape
        assert model._H.shape == test_data["Hi"].shape

        # check that error decreases or stays the same
        assert model._rel_loss_ls[-1] <= model._rel_loss_ls[0]

    @pytest.mark.parametrize(
        "model_class, constraint_kind",
        [
            (FroALS, 0),
            (FroALS, 1),
            (FroALS, 2),
            (FroALS, 3),
            (FroALS, 4),
            (FroFPGM, 0),
            (FroFPGM, 1),
            (FroFPGM, 2),
            (FroFPGM, 3),
            (FroFPGM, 4),
            (MinVol, 0),
            (MinVol, 1),
            (MinVol, 2),
            (MinVol, 3),
            (MinVol, 4),
        ],
    )
    def test_constraints(self, test_data, model_class, constraint_kind):
        """Test that constraints are enforced correctly."""
        model: FroALS | FroFPGM | MinVol = model_class(
            rank=test_data["rank"], constraint_kind=constraint_kind
        )
        model.fit(X=test_data["X"], Wi=test_data["Wi"], Hi=test_data["Hi"])

        # check non-negativity constraint is always enforced
        assert np.all(model.W >= 0)
        assert np.all(model.H >= 0)

        if constraint_kind == 1:
            # simplex constraint on H
            assert np.all(np.sum(model.H, axis=0) <= 1.0 + 1e-10)
        elif constraint_kind == 2:
            # row simplex constraint on H
            assert np.allclose(np.sum(model.H, axis=1), 1.0, atol=1e-10)
        elif constraint_kind == 3:
            # column simplex constraint on W
            assert np.allclose(np.sum(model.W, axis=0), 1.0, atol=1e-10)
        elif constraint_kind == 4:
            # column simplex constraint on H
            assert np.allclose(np.sum(model.H, axis=0), 1.0, atol=1e-10)

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_unimodal_constraints(self, test_data, model_class):
        """Test that unimodal constraints are enforced correctly."""
        model: FroALS | FroFPGM | MinVol = model_class(
            rank=test_data["rank"], unimodal={"H": [True, False, True]}
        )
        model.fit(X=test_data["X"], Wi=test_data["Wi"], Hi=test_data["Hi"])

        def is_unimodal(arr_1d: np.ndarray):
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

        # assert first and third row are unimodal
        assert is_unimodal(model.H[0, :])
        assert is_unimodal(model.H[2, :])

    @pytest.mark.parametrize("model_class", [FroALS, FroFPGM, MinVol])
    def test_known_values_preserved(self, test_data, model_class):
        """Test that known values in W and H are preserved during updates."""

        # Create known_W and known_H with some specified values
        known_W = np.full_like(test_data["Wi"].copy(), np.nan)
        known_W[0, 0] = 0.5

        known_H = np.full_like(test_data["Hi"].copy(), np.nan)
        known_H[0, 0] = 0.7

        model = model_class(rank=test_data["rank"])
        model.fit(
            X=test_data["X"],
            Wi=test_data["Wi"],
            Hi=test_data["Hi"],
            known_W=known_W,
            known_H=known_H,
        )

        # Check that the known values are preserved
        assert model.W[0, 0] == 0.5
        assert model.H[0, 0] == 0.7


class TestSNPA:
    """Test all aspects of SNPA class."""

    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng(seed=42)
        X = rng.random((50, 30))
        rank = 3
        return {"X": X, "rank": rank}

    def test_rank_validation(self):
        """Test rank parameter validation."""
        # type validation
        msg = r".*`rank` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            SNPA(rank="2")
        with pytest.raises(TypeError, match=msg):
            SNPA(rank=2.0)

        # value validation
        msg = "`rank` must be > 0"
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=0)
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=-3)

    def test_iter_max_validation(self):
        """Test iter_max parameter validation."""
        # type validation
        msg = r".*`iter_max` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            SNPA(rank=2, iter_max="200")
        with pytest.raises(TypeError, match=msg):
            SNPA(rank=2, iter_max=200.0)

        # value validation
        msg = r"`iter_max` must be > 0"
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=2, iter_max=0)
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=2, iter_max=-200)

    def test_tol_validation(self):
        """Test tol parameter validation."""
        # Type validation
        msg = r".*`tol` must be of type.*"
        with pytest.raises(TypeError, match=msg):
            SNPA(rank=2, tol=1)
        with pytest.raises(TypeError, match=msg):
            SNPA(rank=2, tol="0.001")

        # Value validation
        msg = r"`tol` must be in the open interval \(0, 1\)"
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=2, tol=0.0)
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=2, tol=1.0)
        with pytest.raises(ValueError, match=msg):
            SNPA(rank=2, tol=2.0)

    def test_fit_X_validation(self, test_data):
        """Test X array validation."""
        # X must be ndarray
        with pytest.raises(TypeError, match=r"`X` must be a numpy array"):
            snpa = SNPA(rank=test_data["rank"])
            snpa.fit(X=test_data["X"].tolist())

        # Each element of X must of np.float64
        with pytest.raises(TypeError, match=r"The dtype of `X` elements should be.*"):
            snpa = SNPA(rank=test_data["rank"])
            snpa.fit(X=test_data["X"].astype(np.float32))

        # test nonnegativity
        with pytest.raises(ValueError, match=r"All elements of `X` must be >= 0"):
            snpa = SNPA(rank=test_data["rank"])
            temp_X = test_data["X"].copy()
            temp_X[0, 0] = -0.1
            snpa.fit(X=temp_X)

        # The number of columns in X cannot be lesser than rank
        with pytest.raises(
            ValueError, match=r"`rank` cannot be greater than columns in `X`*"
        ):
            snpa = SNPA(rank=test_data["rank"])
            snpa.fit(X=test_data["X"][:, :2])

    def test_tol_warning(self, test_data):
        """Test if tolerance warning is raised."""
        with pytest.warns(UserWarning, match=r"Extracted only.*"):
            snpa = SNPA(rank=test_data["rank"], tol=0.9)
            snpa.fit(X=test_data["X"])

    def test_property_validation(self, test_data):
        """Test that validation of properties."""
        snpa = SNPA(rank=test_data["rank"])

        with pytest.raises(AttributeError, match=r"'W' is not available.*"):
            _ = snpa.W

        with pytest.raises(AttributeError, match=r"'H' is not available.*"):
            _ = snpa.H

        with pytest.raises(AttributeError, match=r"'col_indices_ls' is not available."):
            _ = snpa.col_indices_ls

    def test_fit_functionality(self, test_data):
        """Test common SNPA fit functionality."""

        snpa = SNPA(rank=test_data["rank"])
        snpa.fit(X=test_data["X"])

        # check dimensions of resulting matrices
        assert snpa.W.shape == (test_data["X"].shape[0], test_data["rank"])
        assert snpa.H.shape == (test_data["rank"], test_data["X"].shape[1])
        assert len(snpa.col_indices_ls) == test_data["rank"]

        # check non-negativity
        assert np.min(snpa.W) >= 0
        assert np.min(snpa.H) >= 0
