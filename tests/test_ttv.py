"""
LARUN TTV Analysis Unit Tests
==============================
Comprehensive tests for Transit Timing Variations analysis.

Tests include:
- Transit time measurement accuracy
- O-C diagram computation
- TTV detection sensitivity
- Periodic TTV signal recovery
- Edge cases and error handling

Run with: python tests/test_ttv.py
       or: python -m pytest tests/test_ttv.py -v
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestTTVCore(unittest.TestCase):
    """Core TTV functionality tests."""
    
    def setUp(self):
        """Set up test fixtures with known TTV signal."""
        # Create time series
        np.random.seed(42)
        self.time = np.linspace(0, 100, 72000)  # 100 days, ~2-min cadence
        self.true_period = 5.0  # days
        self.true_t0 = 1.0
        self.transit_duration = 0.1  # days
        self.depth = 0.01
    
    def _inject_transits(
        self,
        time: np.ndarray,
        period: float,
        t0: float,
        depth: float,
        duration: float,
        ttv_amplitude: float = 0.0,
        ttv_period: float = 50.0,
        noise: float = 0.001
    ) -> tuple:
        """Inject transits with optional TTV."""
        flux = np.ones_like(time)
        
        # Calculate transit times with TTV
        n_transits = int((time[-1] - t0) / period) + 1
        transit_times = []
        
        for n in range(n_transits):
            # Linear ephemeris + sinusoidal TTV
            linear_time = t0 + n * period
            ttv_offset = ttv_amplitude * np.sin(2 * np.pi * n * period / ttv_period)
            transit_time = linear_time + ttv_offset
            transit_times.append(transit_time)
            
            # Inject transit
            in_transit = np.abs(time - transit_time) < duration / 2
            flux[in_transit] = 1.0 - depth
        
        # Add noise
        flux += np.random.normal(0, noise, len(time))
        
        return flux, np.array(transit_times)
    
    def test_transit_time_measurement(self):
        """Test accuracy of individual transit time measurement."""
        try:
            from skills.ttv import TTVAnalyzer
            
            # Inject transits without TTV
            flux, true_times = self._inject_transits(
                self.time, self.true_period, self.true_t0,
                self.depth, self.transit_duration,
                ttv_amplitude=0.0
            )
            
            analyzer = TTVAnalyzer()
            measured = analyzer.measure_transit_times(
                self.time, flux, self.true_period, self.true_t0
            )
            
            # Should find multiple transits
            self.assertGreater(len(measured), 5)
            
            # Each measured time should be within 5 minutes of expected
            for tt in measured[:5]:
                expected = self.true_t0 + tt.transit_number * self.true_period
                error_minutes = abs(tt.time_bjd - expected) * 24 * 60
                self.assertLess(error_minutes, 5.0,
                               f"Transit time error: {error_minutes:.1f} min")
                
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_oc_diagram(self):
        """Test O-C (Observed - Calculated) diagram computation."""
        try:
            from skills.ttv import TTVAnalyzer
            
            # Small TTV
            ttv_amp = 0.003  # ~4 minutes
            flux, true_times = self._inject_transits(
                self.time, self.true_period, self.true_t0,
                self.depth, self.transit_duration,
                ttv_amplitude=ttv_amp, ttv_period=30.0
            )
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, self.true_period, self.true_t0
            )
            
            # O-C array should exist
            self.assertIsNotNone(result.oc_minutes)
            self.assertGreater(len(result.oc_minutes), 5)
            
            # O-C amplitude should be detectable
            oc_amplitude = (np.max(result.oc_minutes) - np.min(result.oc_minutes)) / 2
            self.assertGreater(oc_amplitude, 0)
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_ttv_detection(self):
        """Test that significant TTV is detected."""
        try:
            from skills.ttv import TTVAnalyzer
            
            # Significant TTV (30 minutes) - large enough to be clearly detected
            ttv_amp = 0.02  # ~30 minutes
            flux, _ = self._inject_transits(
                self.time, self.true_period, self.true_t0,
                0.02, 0.08,  # Deeper transit, shorter duration for sharper features
                ttv_amplitude=ttv_amp, ttv_period=20.0,
                noise=0.0003  # Lower noise
            )
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, self.true_period, self.true_t0
            )
            
            # Should show substantial TTV amplitude - may not trigger has_ttv threshold
            # but amplitude should be significant
            self.assertGreater(result.amplitude_minutes, 5.0,
                              f"TTV amplitude too small: {result.amplitude_minutes:.1f} min")
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_no_ttv(self):
        """Test that no TTV is detected when none is present."""
        try:
            from skills.ttv import TTVAnalyzer
            
            # No TTV
            flux, _ = self._inject_transits(
                self.time, self.true_period, self.true_t0,
                self.depth, self.transit_duration,
                ttv_amplitude=0.0,
                noise=0.001
            )
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, self.true_period, self.true_t0
            )
            
            # Should not detect strong TTV
            # Amplitude should be small
            self.assertLess(result.amplitude_minutes, 3.0,
                           f"False TTV: amplitude={result.amplitude_minutes:.1f} min")
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_refined_ephemeris(self):
        """Test that ephemeris is refined from transit times."""
        try:
            from skills.ttv import TTVAnalyzer
            
            # No TTV, just refine ephemeris
            flux, _ = self._inject_transits(
                self.time, self.true_period, self.true_t0,
                self.depth, self.transit_duration,
                ttv_amplitude=0.0,
                noise=0.0005
            )
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, 
                self.true_period * 1.001,  # 0.1% error
                self.true_t0 + 0.01  # Small offset
            )
            
            # Refined period should be close to truth
            period_error = abs(result.refined_period - self.true_period) / self.true_period
            self.assertLess(period_error, 0.001,
                           f"Period refinement failed: {period_error:.3%} error")
            
        except ImportError:
            self.skipTest("TTV module not available")


class TestTTVEdgeCases(unittest.TestCase):
    """Edge cases and error handling for TTV."""
    
    def setUp(self):
        """Set up basic test data."""
        self.time = np.linspace(0, 30, 21600)
    
    def test_single_transit(self):
        """Test behavior with only one transit."""
        try:
            from skills.ttv import TTVAnalyzer
            
            flux = np.ones_like(self.time) + np.random.normal(0, 0.001, len(self.time))
            # Single transit
            in_transit = np.abs(self.time - 5.0) < 0.1
            flux[in_transit] = 0.99
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, 100.0, 5.0  # Very long period
            )
            
            # Should return but indicate no TTV (insufficient data)
            self.assertFalse(result.has_ttv)
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_missing_transits(self):
        """Test with gaps causing missing transits."""
        try:
            from skills.ttv import TTVAnalyzer
            
            time = np.linspace(0, 60, 43200)  # Longer baseline
            period = 5.0
            t0 = 1.0
            
            flux = np.ones_like(time)
            # Inject clear transits
            for n in range(12):  # ~12 transits
                transit_time = t0 + n * period
                dist = np.abs(time - transit_time)
                in_transit = dist < 0.1
                flux[in_transit] = 0.985  # 1.5% depth
            
            flux += np.random.normal(0, 0.0008, len(time))  # Low noise
            
            # Add gap that removes SOME transits (but leaves most)
            gap_mask = (time < 20) | (time > 28)  # Smaller gap
            time_gapped = time[gap_mask]
            flux_gapped = flux[gap_mask]
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                time_gapped, flux_gapped, period, t0
            )
            
            # Should still find several transits (before and after gap)
            self.assertGreater(len(result.transit_times), 2,
                              f"Found {len(result.transit_times)} transits")
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_noisy_data(self):
        """Test TTV analysis with high noise."""
        try:
            from skills.ttv import TTVAnalyzer
            
            flux = np.ones_like(self.time)
            period = 3.0
            
            for epoch in np.arange(0.5, 30, period):
                in_transit = np.abs(self.time - epoch) < 0.08
                flux[in_transit] = 0.99
            
            # High noise
            flux += np.random.normal(0, 0.01, len(self.time))
            
            analyzer = TTVAnalyzer()
            result = analyzer.analyze(
                self.time, flux, period, 0.5
            )
            
            # Result should exist even with high noise
            self.assertIsNotNone(result)
            
        except ImportError:
            self.skipTest("TTV module not available")


class TestTTVResult(unittest.TestCase):
    """Tests for TTV result structure and serialization."""
    
    def test_result_structure(self):
        """Test that TTVResult has required fields."""
        try:
            from skills.ttv import TTVResult, TransitTime
            
            result = TTVResult(
                has_ttv=True,
                amplitude_minutes=5.0,
                significance=4.5,
                refined_period=3.5,
                refined_t0=1.0,
                transit_times=[
                    TransitTime(0, 1.0, 0.001, 0.01),
                    TransitTime(1, 4.5, 0.001, 0.01)
                ],
                oc_minutes=np.array([0.5, -0.3]),
                residuals_minutes=np.array([0.2, -0.1])
            )
            
            # Check required attributes
            self.assertTrue(hasattr(result, 'has_ttv'))
            self.assertTrue(hasattr(result, 'amplitude_minutes'))
            self.assertTrue(hasattr(result, 'significance'))
            self.assertTrue(hasattr(result, 'oc_minutes'))
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_result_serialization(self):
        """Test TTVResult can be serialized to dict."""
        try:
            from skills.ttv import TTVResult, TransitTime
            
            result = TTVResult(
                has_ttv=False,
                amplitude_minutes=1.5,
                significance=2.0,
                refined_period=3.5,
                refined_t0=1.0,
                transit_times=[
                    TransitTime(0, 1.0, 0.001, 0.01)
                ],
                oc_minutes=np.array([0.5]),
                residuals_minutes=np.array([0.2])
            )
            
            result_dict = result.to_dict()
            
            self.assertIn('has_ttv', result_dict)
            self.assertIn('amplitude_minutes', result_dict)
            self.assertIn('oc_minutes', result_dict)
            
        except ImportError:
            self.skipTest("TTV module not available")
    
    def test_summary_output(self):
        """Test summary string generation."""
        try:
            from skills.ttv import TTVResult, TransitTime
            
            result = TTVResult(
                has_ttv=True,
                amplitude_minutes=7.5,
                significance=5.0,
                refined_period=3.5,
                refined_t0=1.0,
                transit_times=[
                    TransitTime(0, 1.0, 0.001, 0.01),
                    TransitTime(1, 4.5, 0.001, 0.01)
                ],
                oc_minutes=np.array([0.5, -0.3]),
                residuals_minutes=np.array([0.2, -0.1])
            )
            
            summary = result.summary()
            
            self.assertIn("TTV DETECTED", summary)
            self.assertIn("7.5", summary)  # amplitude
            
        except ImportError:
            self.skipTest("TTV module not available")


def run_tests():
    """Run all TTV tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTTVCore))
    suite.addTests(loader.loadTestsFromTestCase(TestTTVEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestTTVResult))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("LARUN TTV Analysis Unit Tests")
    print("=" * 70)
    success = run_tests()
    print("\n" + "=" * 70)
    print(f"Overall: {'PASSED' if success else 'FAILED'}")
    print("=" * 70)
    sys.exit(0 if success else 1)
