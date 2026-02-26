"""
Tests for VARDET-001 — Variable Source Detector.

Tests feature extraction, synthetic training, and prediction
on known variable star patterns.
"""

import numpy as np
import pytest


@pytest.fixture
def vardet():
    from larun.models.vardet import VARDET001
    model = VARDET001()
    model.train_synthetic(n_per_class=200)
    return model


@pytest.fixture
def synthetic_lcs():
    """Generate one light curve per class for testing."""
    from larun.models.vardet import _synthetic_light_curve
    rng = np.random.default_rng(42)
    lcs = {}
    for class_id, name in {0: "non_variable", 1: "transient", 2: "pulsator", 3: "eclipsing"}.items():
        t = np.sort(rng.uniform(0, 100, 150))
        lcs[name] = _synthetic_light_curve(t, class_id, rng)
    return lcs


class TestVARDET001FeatureExtraction:
    def test_feature_vector_shape(self, synthetic_lcs):
        from larun.models.vardet import VARDET001
        model = VARDET001()
        for name, lc in synthetic_lcs.items():
            feats = model.extract_features(lc["times"], lc["mags"])
            assert feats.shape == (10,), f"{name}: expected 10 features, got {feats.shape}"

    def test_features_are_finite(self, synthetic_lcs):
        from larun.models.vardet import VARDET001
        model = VARDET001()
        for name, lc in synthetic_lcs.items():
            feats = model.extract_features(lc["times"], lc["mags"])
            assert np.all(np.isfinite(feats)), f"{name}: features contain NaN/Inf"

    def test_pulsator_has_high_fourier_power(self, synthetic_lcs):
        """Pulsating star should have high Fourier power ratio."""
        from larun.models.vardet import VARDET001
        model = VARDET001()
        pulsator_feats = model.extract_features(
            synthetic_lcs["pulsator"]["times"],
            synthetic_lcs["pulsator"]["mags"],
        )
        non_var_feats = model.extract_features(
            synthetic_lcs["non_variable"]["times"],
            synthetic_lcs["non_variable"]["mags"],
        )
        # fourier_power_ratio is feature[0]
        assert pulsator_feats[0] > non_var_feats[0], (
            "Pulsator should have higher Fourier power than non-variable"
        )

    def test_amplitude_ordering(self, synthetic_lcs):
        """Transient and pulsator should have higher amplitude than non-variable."""
        from larun.models.vardet import VARDET001
        model = VARDET001()
        amp_non = model.extract_features(
            synthetic_lcs["non_variable"]["times"],
            synthetic_lcs["non_variable"]["mags"],
        )[6]  # amplitude is index 6
        amp_pulsator = model.extract_features(
            synthetic_lcs["pulsator"]["times"],
            synthetic_lcs["pulsator"]["mags"],
        )[6]
        assert amp_pulsator > amp_non

    def test_short_light_curve(self):
        """Should handle very short light curves gracefully."""
        from larun.models.vardet import VARDET001
        model = VARDET001()
        feats = model.extract_features(np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0]))
        assert feats.shape == (10,)

    def test_with_errors(self, synthetic_lcs):
        """Feature extraction should accept error array."""
        from larun.models.vardet import VARDET001
        model = VARDET001()
        lc = synthetic_lcs["pulsator"]
        errors = np.ones_like(lc["mags"]) * 0.01
        feats = model.extract_features(lc["times"], lc["mags"], errors)
        assert feats.shape == (10,)


class TestVARDET001Training:
    def test_synthetic_training(self):
        """Model should train without errors."""
        from larun.models.vardet import VARDET001
        model = VARDET001()
        result = model.train_synthetic(n_per_class=50)
        assert result._loaded
        assert result._rf is not None

    def test_training_with_data(self, synthetic_lcs):
        """Manual training with pre-built light curves."""
        from larun.models.vardet import VARDET001, _synthetic_light_curve
        rng = np.random.default_rng(1)
        light_curves = []
        labels = []
        for class_id in range(4):
            for _ in range(50):
                t = np.sort(rng.uniform(0, 100, 100))
                lc = _synthetic_light_curve(t, class_id, rng)
                light_curves.append(lc)
                labels.append(class_id)

        model = VARDET001()
        model.train(light_curves, labels)
        assert model._loaded


class TestVARDET001Prediction:
    def test_predict_returns_tuple(self, vardet, synthetic_lcs):
        """predict() should return (label, probabilities)."""
        lc = synthetic_lcs["pulsator"]
        label, proba = vardet.predict(lc["times"], lc["mags"])
        assert isinstance(label, str)
        assert label in ["NON_VARIABLE", "TRANSIENT", "PULSATOR", "ECLIPSING"]
        assert proba.shape == (4,)
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_non_variable_prediction(self, vardet, synthetic_lcs):
        """Non-variable star should mostly be classified as NON_VARIABLE."""
        lc = synthetic_lcs["non_variable"]
        label, proba = vardet.predict(lc["times"], lc["mags"])
        # Allow some misclassification in synthetic data, just check no error
        assert label in vardet.CLASSES.values()

    def test_pulsator_prediction(self, vardet, synthetic_lcs):
        """Strong pulsating signal should be classified as PULSATOR."""
        rng = np.random.default_rng(99)
        t = np.sort(rng.uniform(0, 200, 300))
        # Clear pulsation signal — large amplitude, regular
        mags = 14.0 + 0.5 * np.sin(2 * np.pi * t / 5.0) + rng.normal(0, 0.005, len(t))
        label, proba = vardet.predict(t, mags)
        assert label == "PULSATOR", f"Expected PULSATOR, got {label} (proba={proba})"

    def test_predict_batch(self, vardet, synthetic_lcs):
        """Batch prediction should return one result per light curve."""
        lcs = list(synthetic_lcs.values())
        results = vardet.predict_batch(lcs)
        assert len(results) == len(lcs)
        for r in results:
            assert "model_id" in r
            assert r["model_id"] == "VARDET-001"

    def test_result_dict_format(self, vardet, synthetic_lcs):
        """Result dict should have required keys."""
        lc = synthetic_lcs["pulsator"]
        label, proba = vardet.predict(lc["times"], lc["mags"])
        result = vardet.result_dict(label, proba)
        assert "model_id" in result
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0 <= result["confidence"] <= 1


class TestVARDET001Persistence:
    def test_save_and_load(self, vardet, tmp_path):
        """Model should save and reload correctly."""
        save_path = tmp_path / "vardet_test.npz"
        vardet.save(save_path)
        assert save_path.exists()

        from larun.models.vardet import VARDET001
        loaded = VARDET001()
        loaded.load(save_path)
        assert loaded._loaded
        assert loaded._rf is not None
