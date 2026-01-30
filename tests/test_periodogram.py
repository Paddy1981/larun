"""
Tests for LARUN Periodogram Skills
==================================

Tests for BLS and Lomb-Scargle periodogram implementations.

Run with: pytest tests/test_periodogram.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skills.periodogram import (
    BLSPeriodogram,
    LombScarglePeriodogram,
    TransitCandidate,
    phase_fold,
    bin_phase_curve
)


class TestBLSPeriodogram:
    """Tests for BLS periodogram."""

    def test_synthetic_transit_detection(self):
        """Test detection of synthetic transit signal."""
        # Generate synthetic transit data
        np.random.seed(42)
        true_period = 5.0  # days
        t0 = 1.0
        depth = 0.01  # 1% depth
        duration_frac = 0.05  # 5% of period

        # Time array (like TESS 27-day sector)
        time = np.linspace(0, 27, 5000)

        # Base flux with realistic noise
        noise_level = 0.001
        flux = np.ones_like(time) + np.random.normal(0, noise_level, len(time))

        # Add box-shaped transits
        phase = ((time - t0) % true_period) / true_period
        in_transit = phase < duration_frac
        flux[in_transit] -= depth

        # Run BLS
        bls = BLSPeriodogram(min_period=1, max_period=15, n_periods=2000)
        result = bls.compute(time, flux, min_snr=5.0)

        # Check period recovery
        assert abs(result.best_period - true_period) < 0.1, \
            f"Period error too large: {abs(result.best_period - true_period)}"

        # Should have at least one candidate
        assert len(result.candidates) >= 1, "No transit candidate found"

        # Check candidate properties
        if result.candidates:
            c = result.candidates[0]
            assert abs(c.period - true_period) < 0.1
            assert c.snr > 5.0

    def test_no_signal(self):
        """Test behavior on noise-only data."""
        np.random.seed(123)
        time = np.linspace(0, 27, 3000)
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

        bls = BLSPeriodogram(min_period=1, max_period=10, n_periods=500)
        result = bls.compute(time, flux, min_snr=15.0)  # Higher threshold for noise

        # Should have high FAP (no real signal) or few candidates
        # In pure noise, BLS may still find spurious peaks, but FAP should be higher
        assert result.fap > 0.001 or len(result.candidates) == 0, \
            f"FAP={result.fap}, candidates={len(result.candidates)}"

    def test_input_validation(self):
        """Test input validation."""
        bls = BLSPeriodogram()

        # Mismatched lengths
        with pytest.raises(ValueError):
            bls.compute(np.array([1, 2, 3]), np.array([1, 2]))

        # Too few points
        with pytest.raises(ValueError):
            bls.compute(np.arange(50), np.ones(50))

    def test_nan_handling(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        time = np.linspace(0, 27, 1000)
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

        # Add some NaN values
        flux[100:110] = np.nan
        flux[500:505] = np.nan

        bls = BLSPeriodogram(min_period=1, max_period=10, n_periods=200)
        result = bls.compute(time, flux)

        # Should complete without error
        assert result.best_period > 0
        assert np.isfinite(result.best_power)

    def test_period_range(self):
        """Test different period ranges."""
        np.random.seed(42)
        time = np.linspace(0, 100, 5000)  # 100 days
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

        # Add transit at P=20 days
        phase = (time % 20) / 20
        flux[phase < 0.02] -= 0.01

        bls = BLSPeriodogram(min_period=10, max_period=30, n_periods=1000)
        result = bls.compute(time, flux, min_snr=3.0)

        assert abs(result.best_period - 20.0) < 0.5


class TestLombScarglePeriodogram:
    """Tests for Lomb-Scargle periodogram."""

    def test_sinusoidal_detection(self):
        """Test detection of sinusoidal signal."""
        np.random.seed(42)
        true_period = 5.0  # days
        amplitude = 0.01

        # Unevenly sampled time
        time = np.sort(np.random.uniform(0, 50, 500))
        flux = 1.0 + amplitude * np.sin(2 * np.pi * time / true_period)
        flux += np.random.normal(0, 0.002, len(time))

        lsp = LombScarglePeriodogram(min_period=1, max_period=20)
        result = lsp.compute(time, flux)

        # Check period recovery
        assert abs(result.best_period - true_period) < 0.2, \
            f"Period error: {abs(result.best_period - true_period)}"

    def test_no_signal(self):
        """Test behavior on pure noise."""
        np.random.seed(42)
        time = np.linspace(0, 100, 500)
        flux = 1.0 + np.random.normal(0, 0.01, len(time))

        lsp = LombScarglePeriodogram()
        result = lsp.compute(time, flux)

        # FAP should be high
        assert result.fap > 0.01

    def test_input_validation(self):
        """Test input validation."""
        lsp = LombScarglePeriodogram()

        # Too few points
        with pytest.raises(ValueError):
            lsp.compute(np.array([1, 2, 3]), np.array([1, 2, 3]))


class TestPhaseFolding:
    """Tests for phase folding functions."""

    def test_basic_folding(self):
        """Test basic phase folding."""
        time = np.linspace(0, 10, 100)
        flux = np.ones_like(time)
        period = 2.5

        phase, flux_folded = phase_fold(time, flux, period)

        # Phase should be in [-0.5, 0.5]
        assert phase.min() >= -0.5
        assert phase.max() <= 0.5

        # Should be sorted
        assert np.all(np.diff(phase) >= 0)

    def test_transit_centering(self):
        """Test that transit is centered at phase 0."""
        period = 5.0
        t0 = 2.5
        time = np.linspace(0, 20, 1000)
        flux = np.ones_like(time)

        # Add transit at t0
        flux[np.abs(time - t0) < 0.1] = 0.99
        flux[np.abs(time - t0 - period) < 0.1] = 0.99
        flux[np.abs(time - t0 - 2*period) < 0.1] = 0.99

        phase, flux_folded = phase_fold(time, flux, period, t0)

        # Transit should be near phase 0
        transit_mask = flux_folded < 0.995
        transit_phases = phase[transit_mask]
        assert np.abs(np.mean(transit_phases)) < 0.05

    def test_binning(self):
        """Test phase curve binning."""
        phase = np.random.uniform(-0.5, 0.5, 1000)
        flux = np.ones_like(phase) + np.random.normal(0, 0.01, len(phase))

        bin_centers, bin_flux, bin_err = bin_phase_curve(phase, flux, n_bins=50)

        # Check output shapes
        assert len(bin_centers) == 50
        assert len(bin_flux) == 50
        assert len(bin_err) == 50

        # Bin centers should span [-0.5, 0.5]
        assert bin_centers[0] > -0.5
        assert bin_centers[-1] < 0.5


class TestTransitCandidate:
    """Tests for TransitCandidate dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        candidate = TransitCandidate(
            period=3.5,
            t0=2458000.0,
            depth=0.001,
            duration=0.1,
            snr=15.5,
            power=0.25,
            fap=1e-10
        )

        d = candidate.to_dict()

        assert d['period_days'] == 3.5
        assert d['depth_ppm'] == 1000.0  # 0.001 * 1e6
        assert d['duration_hours'] == 2.4  # 0.1 * 24
        assert d['snr'] == 15.5

    def test_str(self):
        """Test string representation."""
        candidate = TransitCandidate(
            period=3.5, t0=0, depth=0.001, duration=0.1,
            snr=15.5, power=0.25, fap=1e-10
        )

        s = str(candidate)
        assert '3.5' in s
        assert '1000' in s or 'ppm' in s.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
