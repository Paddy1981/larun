"""
LARUN Skill: Transit Vetting
============================
False positive identification tests for exoplanet transit candidates.

Skill ID: VET-001
Commands: larun vet, /vet

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: docs/research/EXOPLANET_DETECTION.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestResult:
    """Result of a single vetting test."""
    test_name: str
    passed: bool
    confidence: float  # 0-1, how confident we are in the result
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'confidence': round(self.confidence, 3),
            'message': self.message,
            **self.details
        }


@dataclass
class VettingResult:
    """Combined result of all vetting tests."""
    is_likely_planet: bool
    confidence: float
    tests: List[TestResult]
    target_name: str = ""
    period: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target_name,
            'period_days': round(self.period, 6),
            'is_likely_planet': self.is_likely_planet,
            'confidence': round(self.confidence, 3),
            'n_tests_passed': sum(1 for t in self.tests if t.passed),
            'n_tests_total': len(self.tests),
            'tests': [t.to_dict() for t in self.tests]
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        status = "PLANET CANDIDATE" if self.is_likely_planet else "LIKELY FALSE POSITIVE"
        lines = [
            f"Vetting Result: {status} (confidence: {self.confidence:.0%})",
            f"Period: {self.period:.4f} days",
            "",
            "Test Results:"
        ]
        for test in self.tests:
            icon = "✓" if test.passed else "✗"
            lines.append(f"  {icon} {test.test_name}: {test.message}")
        return "\n".join(lines)


# ============================================================================
# Transit Vetter
# ============================================================================

class TransitVetter:
    """
    Vetting suite for transit candidates.
    
    Performs multiple tests to distinguish true planets from false positives
    like eclipsing binaries, blended sources, and systematic artifacts.
    
    Based on: docs/research/EXOPLANET_DETECTION.md (Section 6)
    
    Example:
        >>> vetter = TransitVetter()
        >>> result = vetter.run_all(time, flux, period=3.5, t0=2458000.0)
        >>> print(result.summary())
    """

    def __init__(self, significance_threshold: float = 3.0):
        """
        Initialize the vetter.
        
        Args:
            significance_threshold: Sigma threshold for flagging issues (default 3σ)
        """
        self.significance_threshold = significance_threshold

    def run_all(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        flux_err: Optional[np.ndarray] = None,
        target_name: str = ""
    ) -> VettingResult:
        """
        Run all vetting tests.
        
        Args:
            time: Time array (days, BJD)
            flux: Normalized flux array
            period: Orbital period (days)
            t0: Mid-transit time (BJD)
            flux_err: Optional flux uncertainties
            target_name: Optional target identifier
            
        Returns:
            VettingResult with all test outcomes
        """
        logger.info(f"Running vetting suite for P={period:.4f}d, t0={t0:.4f}")
        
        tests = []
        
        # 1. Odd-Even Depth Test
        odd_even = self.odd_even_test(time, flux, period, t0)
        tests.append(odd_even)
        
        # 2. Secondary Eclipse Search
        secondary = self.secondary_eclipse_test(time, flux, period, t0)
        tests.append(secondary)
        
        # 3. V-Shape Test (grazing binary)
        v_shape = self.v_shape_test(time, flux, period, t0)
        tests.append(v_shape)
        
        # 4. Transit Duration Test
        duration = self.duration_test(time, flux, period, t0)
        tests.append(duration)
        
        # Combine results
        n_passed = sum(1 for t in tests if t.passed)
        n_total = len(tests)
        
        # Simple confidence calculation
        confidence = n_passed / n_total
        is_likely_planet = n_passed >= n_total - 1  # Allow 1 failed test
        
        return VettingResult(
            is_likely_planet=is_likely_planet,
            confidence=confidence,
            tests=tests,
            target_name=target_name,
            period=period
        )

    def odd_even_test(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> TestResult:
        """
        Check for depth difference between odd and even transits.
        
        Eclipsing binaries often show different depths for primary and
        secondary eclipses at the same period. A significant difference
        suggests the signal is from a binary, not a planet.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period (days)
            t0: Mid-transit time
            
        Returns:
            TestResult indicating if odd/even depths are consistent
        """
        logger.debug("Running odd-even depth test...")
        
        # Calculate transit number for each point
        transit_num = np.floor((time - t0) / period)
        
        # Identify odd and even transits
        is_odd = transit_num % 2 == 1
        is_even = transit_num % 2 == 0
        
        # Phase fold each set
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Define in-transit mask (within 5% of phase 0)
        in_transit_mask = np.abs(phase) < 0.05
        
        # Get in-transit points for odd and even
        odd_in_transit = in_transit_mask & is_odd
        even_in_transit = in_transit_mask & is_even
        
        n_odd = np.sum(odd_in_transit)
        n_even = np.sum(even_in_transit)
        
        if n_odd < 3 or n_even < 3:
            return TestResult(
                test_name="Odd-Even Depth",
                passed=True,  # Inconclusive, assume OK
                confidence=0.3,
                message="Insufficient data for odd-even test",
                details={'n_odd': int(n_odd), 'n_even': int(n_even)}
            )
        
        # Calculate depths
        out_of_transit = ~in_transit_mask
        baseline = np.median(flux[out_of_transit]) if np.sum(out_of_transit) > 0 else 1.0
        
        depth_odd = baseline - np.mean(flux[odd_in_transit])
        depth_even = baseline - np.mean(flux[even_in_transit])
        
        # Calculate uncertainty
        std_odd = np.std(flux[odd_in_transit]) / np.sqrt(n_odd)
        std_even = np.std(flux[even_in_transit]) / np.sqrt(n_even)
        combined_err = np.sqrt(std_odd**2 + std_even**2)
        
        # Significance of difference
        depth_diff = np.abs(depth_odd - depth_even)
        if combined_err > 0:
            significance = depth_diff / combined_err
        else:
            significance = 0.0
        
        # Test passes if difference is not significant
        passed = significance < self.significance_threshold
        
        return TestResult(
            test_name="Odd-Even Depth",
            passed=passed,
            confidence=min(0.9, 1.0 - significance / 10),
            message=f"Depth difference: {significance:.1f}σ" + 
                    (" (consistent)" if passed else " (INCONSISTENT - possible EB)"),
            details={
                'depth_odd_ppm': round(depth_odd * 1e6, 1),
                'depth_even_ppm': round(depth_even * 1e6, 1),
                'difference_sigma': round(significance, 2),
                'n_odd_transits': int(n_odd),
                'n_even_transits': int(n_even)
            }
        )

    def secondary_eclipse_test(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> TestResult:
        """
        Search for secondary eclipse at phase 0.5.
        
        A detectable secondary eclipse at phase 0.5 indicates an
        eclipsing binary with two comparable stars. Planets typically
        have no detectable secondary eclipse in optical wavelengths.
        
        Args:
            time: Time array
            flux: Flux array  
            period: Orbital period
            t0: Mid-transit time
            
        Returns:
            TestResult indicating if secondary eclipse is detected
        """
        logger.debug("Searching for secondary eclipse...")
        
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Define regions
        # Primary transit: phase near 0
        # Secondary eclipse: phase near 0.5 (or -0.5)
        primary_mask = np.abs(phase) < 0.05
        secondary_mask = np.abs(np.abs(phase) - 0.5) < 0.05
        out_of_eclipse = ~primary_mask & ~secondary_mask
        
        n_secondary = np.sum(secondary_mask)
        n_out = np.sum(out_of_eclipse)
        
        if n_secondary < 3 or n_out < 10:
            return TestResult(
                test_name="Secondary Eclipse",
                passed=True,
                confidence=0.3,
                message="Insufficient data for secondary eclipse search",
                details={'n_secondary_points': int(n_secondary)}
            )
        
        # Calculate depths
        baseline = np.median(flux[out_of_eclipse])
        primary_depth = baseline - np.mean(flux[primary_mask]) if np.sum(primary_mask) > 0 else 0
        secondary_depth = baseline - np.mean(flux[secondary_mask])
        
        # Uncertainty
        noise = np.std(flux[out_of_eclipse])
        secondary_err = noise / np.sqrt(n_secondary)
        
        # Significance of secondary
        if secondary_err > 0:
            secondary_significance = secondary_depth / secondary_err
        else:
            secondary_significance = 0.0
        
        # Test passes if no significant secondary eclipse
        # A small secondary might be planetary thermal emission, so we're generous
        has_secondary = secondary_significance > self.significance_threshold
        
        # Also check depth ratio - binaries often have comparable depths
        if primary_depth > 0:
            depth_ratio = secondary_depth / primary_depth
        else:
            depth_ratio = 0.0
        
        passed = not has_secondary or depth_ratio < 0.1  # Allow up to 10% secondary
        
        return TestResult(
            test_name="Secondary Eclipse",
            passed=passed,
            confidence=0.8 if passed else 0.6,
            message=f"Secondary: {secondary_significance:.1f}σ, ratio: {depth_ratio:.1%}" +
                    (" (none detected)" if passed else " (DETECTED - possible EB)"),
            details={
                'secondary_depth_ppm': round(secondary_depth * 1e6, 1),
                'primary_depth_ppm': round(primary_depth * 1e6, 1),
                'depth_ratio': round(depth_ratio, 3),
                'secondary_significance': round(secondary_significance, 2)
            }
        )

    def v_shape_test(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> TestResult:
        """
        Test for V-shaped transit (grazing binary indicator).
        
        Planet transits have flat bottoms (box-like shape) while grazing
        eclipsing binaries have V-shaped transits with no flat bottom.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            t0: Mid-transit time
            
        Returns:
            TestResult indicating transit shape
        """
        logger.debug("Checking transit shape...")
        
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Bin the phase-folded data
        n_bins = 50
        bin_edges = np.linspace(-0.15, 0.15, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_flux = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_flux[i] = np.mean(flux[mask])
                bin_counts[i] = np.sum(mask)
            else:
                bin_flux[i] = np.nan
        
        # Find transit region
        valid = ~np.isnan(bin_flux)
        if np.sum(valid) < 10:
            return TestResult(
                test_name="V-Shape (Grazing)",
                passed=True,
                confidence=0.3,
                message="Insufficient data for shape analysis"
            )
        
        bin_flux_valid = bin_flux[valid]
        bin_centers_valid = bin_centers[valid]
        
        # Find minimum and check for flat bottom
        min_idx = np.argmin(bin_flux_valid)
        min_flux = bin_flux_valid[min_idx]
        baseline = np.median(bin_flux_valid[np.abs(bin_centers_valid) > 0.1])
        
        if baseline - min_flux < 0.0001:  # No significant dip
            return TestResult(
                test_name="V-Shape (Grazing)",
                passed=True,
                confidence=0.5,
                message="No significant transit detected"
            )
        
        # Check flatness: compare flux at center vs slightly off-center
        center_mask = np.abs(bin_centers_valid) < 0.02
        near_center_mask = (np.abs(bin_centers_valid) >= 0.02) & (np.abs(bin_centers_valid) < 0.04)
        
        if np.sum(center_mask) < 2 or np.sum(near_center_mask) < 2:
            return TestResult(
                test_name="V-Shape (Grazing)",
                passed=True,
                confidence=0.4,
                message="Insufficient resolution for shape test"
            )
        
        center_flux = np.mean(bin_flux_valid[center_mask])
        near_center_flux = np.mean(bin_flux_valid[near_center_mask])
        
        # For a flat bottom, center and near-center should be similar
        # For V-shape, center is lower than near-center
        depth = baseline - min_flux
        flatness_ratio = (near_center_flux - center_flux) / depth if depth > 0 else 0
        
        # Flatness ratio near 0 = flat bottom (planet-like)
        # Flatness ratio > 0.3 = V-shaped (binary-like)
        is_v_shaped = flatness_ratio > 0.3
        passed = not is_v_shaped
        
        return TestResult(
            test_name="V-Shape (Grazing)",
            passed=passed,
            confidence=0.7,
            message=f"Flatness ratio: {flatness_ratio:.2f}" +
                    (" (flat bottom)" if passed else " (V-SHAPED - possible grazing EB)"),
            details={
                'flatness_ratio': round(flatness_ratio, 3),
                'center_depth_ppm': round((baseline - center_flux) * 1e6, 1),
                'is_v_shaped': is_v_shaped
            }
        )

    def duration_test(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> TestResult:
        """
        Check if transit duration is physically plausible.
        
        Transit duration should be less than ~25% of the orbital period
        for circular orbits. Longer durations suggest systematic issues.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            t0: Mid-transit time
            
        Returns:
            TestResult indicating if duration is plausible
        """
        logger.debug("Checking transit duration...")
        
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Estimate duration from phase-folded curve
        out_mask = np.abs(phase) > 0.2
        if np.sum(out_mask) < 10:
            return TestResult(
                test_name="Duration Check",
                passed=True,
                confidence=0.3,
                message="Insufficient out-of-transit data"
            )
        
        baseline = np.median(flux[out_mask])
        noise = np.std(flux[out_mask])
        
        # Find in-transit points (below baseline - 3σ)
        threshold = baseline - 3 * noise
        in_transit = flux < threshold
        
        if np.sum(in_transit) < 3:
            return TestResult(
                test_name="Duration Check", 
                passed=True,
                confidence=0.5,
                message="Transit not clearly detected"
            )
        
        # Estimate duration from phase range
        transit_phases = phase[in_transit]
        duration_phase = np.max(transit_phases) - np.min(transit_phases)
        duration_days = duration_phase * period
        duration_hours = duration_days * 24
        
        # Physical limit: duration/period < 0.25 for circular orbit
        duration_ratio = duration_phase
        
        # Check if plausible
        is_plausible = duration_ratio < 0.25 and duration_hours > 0.1
        
        return TestResult(
            test_name="Duration Check",
            passed=is_plausible,
            confidence=0.8 if is_plausible else 0.6,
            message=f"Duration: {duration_hours:.1f}h ({duration_ratio:.1%} of period)" +
                    (" (plausible)" if is_plausible else " (IMPLAUSIBLE)"),
            details={
                'duration_hours': round(duration_hours, 2),
                'duration_fraction': round(duration_ratio, 4),
                'period_days': round(period, 4)
            }
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def vet_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    **kwargs
) -> VettingResult:
    """
    Convenience function to run all vetting tests.
    
    Args:
        time: Time array
        flux: Flux array
        period: Orbital period
        t0: Mid-transit time
        **kwargs: Additional arguments for TransitVetter
        
    Returns:
        VettingResult
    """
    vetter = TransitVetter(**kwargs)
    return vetter.run_all(time, flux, period, t0)


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'VET-001': {
        'id': 'VET-001',
        'name': 'Transit Vetting',
        'command': 'vet',
        'class': TransitVetter,
        'description': 'False positive identification for transit candidates'
    }
}


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Transit Vetter...")
    
    np.random.seed(42)
    
    # Generate synthetic planet transit
    period = 3.5
    t0 = 0.5
    depth = 0.01
    
    time = np.linspace(0, 30, 20000)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    
    # Add transits
    phase = ((time - t0) % period) / period
    in_transit = phase < 0.03
    flux[in_transit] -= depth
    
    # Run vetting
    vetter = TransitVetter()
    result = vetter.run_all(time, flux, period, t0, target_name="Synthetic Planet")
    
    print(result.summary())
    print(f"\nRaw result: {result.to_dict()}")
