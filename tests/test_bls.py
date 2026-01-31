"""
LARUN BLS Periodogram Unit Tests
=================================
Comprehensive tests for the Box Least Squares transit detection algorithm.

Tests include:
- Synthetic transit injection and recovery
- Period accuracy verification
- SNR threshold testing
- Edge cases (no transit, multi-planet)

Run with: python tests/test_bls.py
       or: python -m pytest tests/test_bls.py -v
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestBLSCore(unittest.TestCase):
    """Core BLS functionality tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic time series (30 days, 2 minute cadence)
        self.time = np.linspace(0, 30, 21600)  # ~30 days
        self.true_period = 3.5  # days
        self.true_t0 = 0.5
        self.true_depth = 0.01  # 1% depth
        self.true_duration = 0.15  # days
    
    def _inject_transit(
        self,
        time: np.ndarray,
        period: float,
        t0: float,
        depth: float,
        duration: float,
        noise_level: float = 0.001
    ) -> np.ndarray:
        """Inject a synthetic box transit into flux data."""
        flux = np.ones_like(time)
        
        # Calculate phase
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Transit region (centered at phase 0)
        duration_phase = duration / period
        in_transit = np.abs(phase) < duration_phase / 2
        
        flux[in_transit] = 1.0 - depth
        
        # Add white noise
        flux += np.random.normal(0, noise_level, len(time))
        
        return flux
    
    def test_transit_injection(self):
        """Test that transit injection works correctly."""
        flux = self._inject_transit(
            self.time, self.true_period, self.true_t0,
            self.true_depth, self.true_duration, noise_level=0.0
        )
        
        # Check that transit is present
        min_flux = np.min(flux)
        self.assertAlmostEqual(min_flux, 1.0 - self.true_depth, places=4)
        
        # Check that out-of-transit is ~1
        out_of_transit = np.abs(((self.time - self.true_t0) % self.true_period) / 
                                self.true_period - 0.5) < 0.3
        self.assertAlmostEqual(np.median(flux[out_of_transit]), 1.0, places=4)
    
    def test_period_recovery(self):
        """Test that BLS can recover the injected period."""
        try:
            from skills.bls import BLSPeriodSearch
            
            flux = self._inject_transit(
                self.time, self.true_period, self.true_t0,
                self.true_depth, self.true_duration
            )
            
            bls = BLSPeriodSearch()
            result = bls.search(self.time, flux, period_range=(1, 10))
            
            # Period should be within 1% of true value
            period_error = abs(result.best_period - self.true_period) / self.true_period
            self.assertLess(period_error, 0.01,
                           f"Period error too large: {period_error:.1%}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_depth_recovery(self):
        """Test that BLS recovers the correct depth."""
        try:
            from skills.bls import BLSPeriodSearch
            
            flux = self._inject_transit(
                self.time, self.true_period, self.true_t0,
                self.true_depth, self.true_duration
            )
            
            bls = BLSPeriodSearch()
            result = bls.search(self.time, flux, period_range=(1, 10))
            
            # Depth should be within 20% of true value
            depth_error = abs(result.depth - self.true_depth) / self.true_depth
            self.assertLess(depth_error, 0.2,
                           f"Depth error too large: {depth_error:.1%}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_snr_threshold(self):
        """Test that SNR properly distinguishes real transits from noise."""
        try:
            from skills.bls import BLSPeriodSearch
            
            bls = BLSPeriodSearch()
            
            # With transit
            flux_transit = self._inject_transit(
                self.time, self.true_period, self.true_t0,
                0.01, self.true_duration
            )
            result_transit = bls.search(self.time, flux_transit, period_range=(1, 10))
            
            # Without transit (pure noise)
            flux_noise = 1.0 + np.random.normal(0, 0.001, len(self.time))
            result_noise = bls.search(self.time, flux_noise, period_range=(1, 10))
            
            # Transit should have higher SNR
            self.assertGreater(result_transit.snr, result_noise.snr,
                              "Transit SNR should exceed noise SNR")
            
            # Transit SNR should be significant
            self.assertGreater(result_transit.snr, 5.0,
                              f"Transit SNR too low: {result_transit.snr}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_no_transit(self):
        """Test BLS behavior when no transit is present."""
        try:
            from skills.bls import BLSPeriodSearch
            
            # Pure noise
            flux = 1.0 + np.random.normal(0, 0.002, len(self.time))
            
            bls = BLSPeriodSearch()
            result = bls.search(self.time, flux, period_range=(1, 10))
            
            # SNR should be low
            self.assertLess(result.snr, 7.0,
                           f"SNR too high for no-transit case: {result.snr}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_shallow_transit(self):
        """Test detection of shallow transits (500 ppm)."""
        try:
            from skills.bls import BLSPeriodSearch
            
            shallow_depth = 0.0005  # 500 ppm
            
            flux = self._inject_transit(
                self.time, self.true_period, self.true_t0,
                shallow_depth, self.true_duration,
                noise_level=0.0002  # Low noise
            )
            
            bls = BLSPeriodSearch()
            result = bls.search(self.time, flux, period_range=(1, 10))
            
            # Should still find the period
            period_error = abs(result.best_period - self.true_period) / self.true_period
            self.assertLess(period_error, 0.05,
                           f"Shallow transit period error: {period_error:.1%}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_long_period(self):
        """Test detection of long-period transits."""
        try:
            from skills.bls import BLSPeriodSearch
            
            long_period = 15.0  # days
            time_extended = np.linspace(0, 60, 43200)  # 60 days
            
            flux = self._inject_transit(
                time_extended, long_period, 1.0,
                0.01, 0.3
            )
            
            bls = BLSPeriodSearch()
            result = bls.search(time_extended, flux, period_range=(5, 30))
            
            period_error = abs(result.best_period - long_period) / long_period
            self.assertLess(period_error, 0.02,
                           f"Long period error: {period_error:.1%}")
            
        except ImportError:
            self.skipTest("BLS module not available")


class TestBLSEdgeCases(unittest.TestCase):
    """Edge cases and error handling for BLS."""
    
    def test_short_time_series(self):
        """Test BLS with very short time series."""
        try:
            from skills.bls import BLSPeriodSearch
            
            # Only 100 points
            time = np.linspace(0, 2, 100)
            flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
            
            bls = BLSPeriodSearch()
            
            # Should handle gracefully
            try:
                result = bls.search(time, flux, period_range=(0.5, 1.5))
                self.assertIsNotNone(result)
            except ValueError:
                pass  # Also acceptable to raise error for too-short data
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_gapped_data(self):
        """Test BLS with gaps in the time series."""
        try:
            from skills.bls import BLSPeriodSearch
            
            time = np.linspace(0, 30, 21600)
            flux = 1.0 + np.random.normal(0, 0.001, len(time))
            
            # Add transit
            period = 3.5
            for epoch in np.arange(0.5, 30, period):
                in_transit = np.abs(time - epoch) < 0.05
                flux[in_transit] = 0.99
            
            # Create gap (simulate data gap)
            gap_start, gap_end = 10, 15
            mask = (time < gap_start) | (time > gap_end)
            time_gapped = time[mask]
            flux_gapped = flux[mask]
            
            bls = BLSPeriodSearch()
            result = bls.search(time_gapped, flux_gapped, period_range=(1, 10))
            
            # Should still find period reasonably
            period_error = abs(result.best_period - period) / period
            self.assertLess(period_error, 0.05,
                           f"Period error with gaps: {period_error:.1%}")
            
        except ImportError:
            self.skipTest("BLS module not available")
    
    def test_nan_handling(self):
        """Test that BLS handles NaN values."""
        try:
            from skills.bls import BLSPeriodSearch
            
            time = np.linspace(0, 10, 5000)
            flux = np.ones_like(time)
            
            # Add some NaNs
            flux[100:110] = np.nan
            flux[500:510] = np.nan
            
            bls = BLSPeriodSearch()
            
            # Should handle without crashing
            try:
                result = bls.search(time, flux, period_range=(1, 5))
                self.assertIsNotNone(result)
            except (ValueError, RuntimeError):
                pass  # Also acceptable to raise error
            
        except ImportError:
            self.skipTest("BLS module not available")


class TestBLSMultiPlanet(unittest.TestCase):
    """Multi-planet system tests."""
    
    def test_two_planets(self):
        """Test detection in a two-planet system."""
        try:
            from skills.bls import BLSPeriodSearch
            
            time = np.linspace(0, 60, 43200)
            flux = np.ones_like(time)
            
            # Planet 1: P = 3 days, depth = 1%
            period1 = 3.0
            phase1 = ((time - 0.5) % period1) / period1
            transit1 = np.abs(phase1) < 0.01
            flux[transit1] = 0.99
            
            # Planet 2: P = 7 days, depth = 0.5%
            period2 = 7.0
            phase2 = ((time - 1.0) % period2) / period2
            transit2 = np.abs(phase2) < 0.01
            flux[transit2] = 0.995
            
            flux += np.random.normal(0, 0.0005, len(time))
            
            bls = BLSPeriodSearch()
            
            # First search should find strongest signal (planet 1)
            result1 = bls.search(time, flux, period_range=(1, 15))
            
            # Should find period close to 3 days
            self.assertLess(abs(result1.best_period - period1), 0.2,
                           f"First planet not found: got {result1.best_period}")
            
        except ImportError:
            self.skipTest("BLS module not available")


def run_tests():
    """Run all BLS tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBLSCore))
    suite.addTests(loader.loadTestsFromTestCase(TestBLSEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestBLSMultiPlanet))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("LARUN BLS Periodogram Unit Tests")
    print("=" * 70)
    success = run_tests()
    print("\n" + "=" * 70)
    print(f"Overall: {'PASSED' if success else 'FAILED'}")
    print("=" * 70)
    sys.exit(0 if success else 1)
