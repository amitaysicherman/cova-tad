import numpy as np
import pytest

from covatad import CoVATabularDetector


def test_rejects_one_feature_input_before_loading_model():
    with pytest.raises(ValueError, match="at least two features"):
        CoVATabularDetector().fit(np.ones((4, 1), dtype=np.float32))


def test_rejects_nonfinite_input_before_loading_model():
    X = np.ones((4, 2), dtype=np.float32)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        CoVATabularDetector().fit(X)


def test_rejects_invalid_hyperparameters_before_loading_model():
    X = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="n_projections"):
        CoVATabularDetector(n_projections=0).fit(X)
    with pytest.raises(ValueError, match="projection_strategy"):
        CoVATabularDetector(projection_strategy="unknown").fit(X)


def test_projection_selection_strategies_are_deterministic():
    X = np.arange(60, dtype=np.float32).reshape(10, 6)
    X[:, 4] *= 10

    random_a = CoVATabularDetector(
        n_projections=3, projection_strategy="random", random_state=7
    )
    random_a.regressor_ = object()
    random_a.fit(X)
    random_b = CoVATabularDetector(
        n_projections=3, projection_strategy="random", random_state=7
    )
    random_b.regressor_ = object()
    random_b.fit(X)
    assert random_a.selected_cols_ == random_b.selected_cols_

    ranked = CoVATabularDetector(
        n_projections=2, projection_strategy="variance_ranked"
    )
    ranked.regressor_ = object()
    ranked.fit(X)
    assert 4 in ranked.selected_cols_


def test_rejects_invalid_backbone():
    X = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="backbone"):
        CoVATabularDetector(backbone="invalid").fit(X)


def test_tabpfn_backbone_mock_predict():
    class DummyTabPFNRegressor:
        def fit(self, X, y):
            pass

        def predict(self, X, output_type="quantiles", quantiles=None):
            # return shape (len(X), 2)
            n = len(X)
            return np.ones((n, 2), dtype=np.float32) * [0.0, 1.0]

    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    detector = CoVATabularDetector(n_projections=1, backbone="tabpfn")
    detector.regressor_ = DummyTabPFNRegressor()
    detector.fit(X)
    scores = detector.decision_function(X)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))

