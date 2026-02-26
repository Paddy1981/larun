"""Tests for PERIODOGRAM-001 — Multi-Method Period Finder."""

import numpy as np
import pytest


@pytest.fixture
def periodogram():
    from larun.models.periodogram import PERIODOGRAM001
    return PERIODOGRAM001()


def make_pulsator(period: float, n: int = 200, noise: float = 0.01) -> tuple:
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 10 * period, n))
    m = 14.0 + 0.3 * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)
    return t, m


class TestPERIODOGRAM001:
    def test_find_period_returns_dict(self, periodogram):
        t, m = make_pulsator(5.0)
        result = periodogram.find_period(t, m)
        assert isinstance(result, dict)
        assert "best_period" in result
        assert "confidence" in result
        assert "period_type" in result
        assert "all_methods" in result
        assert result["model_id"] == "PERIODOGRAM-001"

    def test_finds_correct_period(self, periodogram):
        """Should recover known synthetic period within 10%."""
        true_period = 5.0
        t, m = make_pulsator(true_period, n=300, noise=0.005)
        result = periodogram.find_period(t, m)
        recovered = result["best_period"]
        error = abs(recovered - true_period) / true_period
        assert error < 0.15, f"Period error too large: {error:.1%} (expected {true_period}, got {recovered:.3f})"

    def test_period_confidence_range(self, periodogram):
        t, m = make_pulsator(3.0, n=200)
        result = periodogram.find_period(t, m)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_short_time_series(self, periodogram):
        """Should handle very short time series gracefully."""
        t = np.array([0.0, 1.0, 2.0])
        m = np.array([14.0, 14.1, 14.0])
        result = periodogram.find_period(t, m)
        assert result["best_period"] == 0.0
        assert result["period_type"] == "NO_PERIOD"

    def test_individual_methods_present(self, periodogram):
        """All 4 methods should appear in all_methods dict."""
        t, m = make_pulsator(7.0, n=200)
        result = periodogram.find_period(t, m)
        for method in ["lomb_scargle", "bls", "pdm", "acf"]:
            assert method in result["all_methods"], f"Method {method} missing"

    def test_period_type_classification(self, periodogram):
        """Period type should be a valid class."""
        t, m = make_pulsator(2.5, n=200)
        result = periodogram.find_period(t, m)
        assert result["period_type"] in periodogram.CLASSES.values()

    def test_predict_interface(self, periodogram):
        """predict() should conform to BaseModel interface."""
        t, m = make_pulsator(4.0, n=150)
        label, proba = periodogram.predict(t, m)
        assert isinstance(label, str)
        assert proba.shape == (len(periodogram.CLASSES),)
