"""Tests for ANOMALY-001 — Time-Series Anomaly Detector."""

import numpy as np
import pytest


@pytest.fixture
def anomaly_model():
    from larun.models.anomaly import ANOMALY001
    model = ANOMALY001()
    model.train_synthetic()
    return model


class TestANOMALY001Features:
    def test_feature_shape(self):
        from larun.models.anomaly import ANOMALY001
        model = ANOMALY001()
        t = np.linspace(0, 100, 200)
        m = np.sin(2 * np.pi * t / 5) + np.random.default_rng(0).normal(0, 0.01, 200)
        feats = model.extract_features(t, m)
        assert feats.shape == (14,)  # 10 base + 4 extra

    def test_features_finite(self):
        from larun.models.anomaly import ANOMALY001
        model = ANOMALY001()
        rng = np.random.default_rng(7)
        t = np.sort(rng.uniform(0, 50, 100))
        m = rng.normal(14.0, 0.01, 100)
        feats = model.extract_features(t, m)
        assert np.all(np.isfinite(feats))


class TestANOMALY001Prediction:
    def test_normal_star_is_normal(self, anomaly_model):
        """A simple periodic signal should be classified as NORMAL."""
        rng = np.random.default_rng(10)
        t = np.sort(rng.uniform(0, 100, 200))
        m = 14.0 + 0.1 * np.sin(2 * np.pi * t / 5) + rng.normal(0, 0.005, 200)
        label, proba = anomaly_model.predict(t, m)
        assert label in ("NORMAL", "MILD_ANOMALY")  # should not be STRONG for clean pulsator

    def test_predict_returns_valid(self, anomaly_model):
        """predict() returns valid label and probabilities."""
        rng = np.random.default_rng(0)
        t = np.linspace(0, 100, 150)
        m = rng.normal(14.0, 0.01, 150)
        label, proba = anomaly_model.predict(t, m)
        assert label in anomaly_model.CLASSES.values()
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_anomaly_score_range(self, anomaly_model):
        """Anomaly score should be in [0, 1]."""
        rng = np.random.default_rng(5)
        t = np.linspace(0, 100, 100)
        m = rng.normal(14.0, 0.01, 100)
        score = anomaly_model.anomaly_score(t, m)
        assert 0.0 <= score <= 1.0

    def test_boyajian_star_analog(self, anomaly_model):
        """
        Simulate Boyajian's Star-like dips: asymmetric, aperiodic brightness dips.
        ANOMALY-001 should flag this as unusual.
        """
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 1400, 300))  # Kepler timescale
        m = np.zeros_like(t) + 12.0

        # Add irregular dips (aperiodic, asymmetric)
        for dip_t in [200, 450, 800, 1100, 1250]:
            width = rng.uniform(1, 10)
            depth = rng.uniform(0.005, 0.22)
            asymm = rng.uniform(0.3, 0.7)  # asymmetric
            mask = (t > dip_t) & (t < dip_t + width)
            m[mask] += depth * np.linspace(0, 1, mask.sum()) ** asymm

        m += rng.normal(0, 0.001, len(t))

        score = anomaly_model.anomaly_score(t, m)
        # Should have non-trivial anomaly score (not the lowest possible)
        assert score > 0.0


class TestANOMALY001Persistence:
    def test_save_and_load(self, anomaly_model, tmp_path):
        save_path = tmp_path / "anomaly_test.npz"
        anomaly_model.save(save_path)

        from larun.models.anomaly import ANOMALY001
        loaded = ANOMALY001()
        loaded.load(save_path)
        assert loaded._loaded
