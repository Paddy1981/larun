"""
LARUN Skill: Transit Timing Variations (TTV) Analysis
======================================================
Detect additional planets through transit timing variations.

Skill ID: PLANET-009
Commands: larun ttv, /ttv

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: docs/research/EXOPLANET_DETECTION.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TransitTime:
    """Individual transit measurement."""
    transit_number: int
    time_bjd: float
    uncertainty: float
    depth: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transit_number': self.transit_number,
            'time_bjd': round(self.time_bjd, 6),
            'uncertainty_minutes': round(self.uncertainty * 24 * 60, 2),
            'depth_ppm': round(self.depth * 1e6, 1)
        }


@dataclass
class TTVResult:
    """Result of TTV analysis."""
    has_ttv: bool
    amplitude_minutes: float
    significance: float
    refined_period: float
    refined_t0: float
    transit_times: List[TransitTime]
    oc_minutes: np.ndarray
    residuals_minutes: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_ttv': self.has_ttv,
            'amplitude_minutes': round(self.amplitude_minutes, 2),
            'significance_sigma': round(self.significance, 1),
            'refined_period_days': round(self.refined_period, 8),
            'refined_t0_bjd': round(self.refined_t0, 6),
            'n_transits': len(self.transit_times),
            'transits': [t.to_dict() for t in self.transit_times],
            'oc_minutes': [round(x, 2) for x in self.oc_minutes.tolist()],
            'residuals_minutes': [round(x, 2) for x in self.residuals_minutes.tolist()]
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        status = "TTV DETECTED" if self.has_ttv else "No significant TTV"
        lines = [
            f"TTV Analysis: {status}",
            f"Amplitude: {self.amplitude_minutes:.2f} minutes ({self.significance:.1f}σ)",
            f"Refined Period: {self.refined_period:.6f} days",
            f"Transits Measured: {len(self.transit_times)}"
        ]
        return "\n".join(lines)


@dataclass 
class AdditionalPlanetCandidate:
    """Candidate for additional planet from residual search."""
    period: float
    snr: float
    depth: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_days': round(self.period, 4),
            'snr': round(self.snr, 1),
            'depth_ppm': round(self.depth * 1e6, 1)
        }


# ============================================================================
# TTV Analyzer
# ============================================================================

class TTVAnalyzer:
    """
    Transit Timing Variation analysis for multi-planet detection.
    
    TTV occurs when gravitational interactions between planets cause
    deviations from strict periodicity in transit times.
    
    Based on: Holman & Murray (2005)
    Reference: docs/research/EXOPLANET_DETECTION.md (Section 7)
    
    Example:
        >>> analyzer = TTVAnalyzer()
        >>> result = analyzer.analyze(time, flux, period=3.5, t0=2458000.0)
        >>> print(result.summary())
        >>> if result.has_ttv:
        ...     print("Possible additional planet!")
    """

    def __init__(self, transit_window: float = 0.1):
        """
        Initialize TTV analyzer.
        
        Args:
            transit_window: Phase window around transit to search (default 0.1)
        """
        self.transit_window = transit_window

    def analyze(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        flux_err: Optional[np.ndarray] = None
    ) -> TTVResult:
        """
        Perform complete TTV analysis.
        
        Args:
            time: Time array (BJD)
            flux: Normalized flux array
            period: Known orbital period (days)
            t0: Reference mid-transit time (BJD)
            flux_err: Optional flux uncertainties
            
        Returns:
            TTVResult with TTV detection and O-C diagram
        """
        logger.info(f"Starting TTV analysis: P={period:.4f}d, t0={t0:.4f}")
        
        # Measure individual transit times
        transit_times = self.measure_transit_times(time, flux, period, t0, flux_err)
        
        if len(transit_times) < 3:
            logger.warning(f"Only {len(transit_times)} transits found, need at least 3")
            return TTVResult(
                has_ttv=False,
                amplitude_minutes=0.0,
                significance=0.0,
                refined_period=period,
                refined_t0=t0,
                transit_times=transit_times,
                oc_minutes=np.array([]),
                residuals_minutes=np.array([])
            )
        
        # Compute O-C diagram
        return self.compute_oc(transit_times, period, t0)

    def measure_transit_times(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        flux_err: Optional[np.ndarray] = None
    ) -> List[TransitTime]:
        """
        Measure individual transit mid-times.
        
        Uses simple centroid method for each transit event.
        
        Args:
            time: Time array
            flux: Flux array
            period: Orbital period
            t0: Reference epoch
            flux_err: Optional uncertainties
            
        Returns:
            List of TransitTime objects
        """
        logger.debug("Measuring individual transit times...")
        
        # Calculate expected transit numbers
        time_span = time.max() - time.min()
        n_expected = int(time_span / period) + 1
        
        transit_times = []
        
        # Estimate noise level
        if flux_err is not None:
            noise = np.median(flux_err)
        else:
            noise = np.std(flux) * 1.4826  # MAD estimate
        
        for n in range(-1, n_expected + 1):
            # Expected transit time
            expected_time = t0 + n * period
            
            # Check if transit is in data range
            if expected_time < time.min() - 0.5 * period or expected_time > time.max() + 0.5 * period:
                continue
            
            # Extract data around transit
            window_half = self.transit_window * period
            mask = np.abs(time - expected_time) < window_half
            
            if np.sum(mask) < 10:
                continue
            
            t_transit = time[mask]
            f_transit = flux[mask]
            
            # Measure transit time using flux-weighted centroid
            measured_time, uncertainty, depth = self._measure_single_transit(
                t_transit, f_transit, expected_time, noise
            )
            
            if measured_time is not None and depth > 0:
                transit_times.append(TransitTime(
                    transit_number=n,
                    time_bjd=measured_time,
                    uncertainty=uncertainty,
                    depth=depth
                ))
        
        logger.info(f"Measured {len(transit_times)} transit times")
        return transit_times

    def _measure_single_transit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        expected_time: float,
        noise: float
    ) -> Tuple[Optional[float], float, float]:
        """
        Measure mid-transit time for a single transit event.
        
        Uses inverse-flux weighted centroid method.
        
        Returns:
            (measured_time, uncertainty, depth) or (None, 0, 0) if failed
        """
        # Normalize to baseline
        baseline = np.median(flux)
        if baseline <= 0:
            return None, 0.0, 0.0
        
        flux_norm = flux / baseline
        
        # Find minimum (deepest point)
        min_idx = np.argmin(flux_norm)
        depth = 1.0 - flux_norm[min_idx]
        
        if depth < 3 * noise / baseline:  # Not significant
            return None, 0.0, 0.0
        
        # Define in-transit region (flux < baseline - depth/2)
        threshold = 1.0 - depth / 2
        in_transit = flux_norm < threshold
        
        if np.sum(in_transit) < 3:
            return time[min_idx], 0.01, depth
        
        # Flux-weighted centroid
        weights = 1.0 - flux_norm[in_transit]
        weights = np.maximum(weights, 0)
        
        if np.sum(weights) > 0:
            measured_time = np.average(time[in_transit], weights=weights)
        else:
            measured_time = time[min_idx]
        
        # Estimate uncertainty from scatter
        uncertainty = np.std(time[in_transit]) / np.sqrt(np.sum(in_transit))
        
        return measured_time, uncertainty, depth

    def compute_oc(
        self,
        transit_times: List[TransitTime],
        period: float,
        t0: float
    ) -> TTVResult:
        """
        Compute O-C (Observed - Calculated) diagram.
        
        Args:
            transit_times: List of measured transit times
            period: Initial period estimate
            t0: Initial epoch
            
        Returns:
            TTVResult with O-C analysis
        """
        logger.debug("Computing O-C diagram...")
        
        # Extract arrays
        n_arr = np.array([t.transit_number for t in transit_times])
        t_arr = np.array([t.time_bjd for t in transit_times])
        err_arr = np.array([t.uncertainty for t in transit_times])
        
        # Expected times using initial ephemeris
        expected = t0 + n_arr * period
        oc = t_arr - expected
        oc_minutes = oc * 24 * 60
        
        # Refine ephemeris with linear fit
        if len(n_arr) >= 3:
            # Weighted linear fit: t = t0 + n * P
            weights = 1.0 / (err_arr**2 + 1e-10)
            coeffs = np.polyfit(n_arr, t_arr, 1, w=weights)
            refined_period = coeffs[0]
            refined_t0 = coeffs[1]
        else:
            refined_period = period
            refined_t0 = t0
        
        # Residuals from refined ephemeris
        refined_expected = refined_t0 + n_arr * refined_period
        residuals = t_arr - refined_expected
        residuals_minutes = residuals * 24 * 60
        
        # TTV amplitude (RMS of residuals)
        amplitude_minutes = np.std(residuals_minutes)
        
        # Significance: compare to timing uncertainties
        mean_uncertainty_minutes = np.mean(err_arr) * 24 * 60
        if mean_uncertainty_minutes > 0:
            significance = amplitude_minutes / mean_uncertainty_minutes
        else:
            significance = 0.0
        
        # TTV detected if amplitude > 1 minute and significant
        has_ttv = amplitude_minutes > 1.0 and significance > 3.0
        
        logger.info(f"TTV analysis: amplitude={amplitude_minutes:.2f}min, "
                   f"significance={significance:.1f}σ")
        
        return TTVResult(
            has_ttv=has_ttv,
            amplitude_minutes=amplitude_minutes,
            significance=significance,
            refined_period=refined_period,
            refined_t0=refined_t0,
            transit_times=transit_times,
            oc_minutes=oc_minutes,
            residuals_minutes=residuals_minutes
        )

    def search_additional_planets(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        known_periods: List[Tuple[float, float]],
        min_snr: float = 5.0
    ) -> List[AdditionalPlanetCandidate]:
        """
        Search for additional planets after removing known transits.
        
        Args:
            time: Time array
            flux: Flux array
            known_periods: List of (period, t0) tuples for known planets
            min_snr: Minimum SNR for detection
            
        Returns:
            List of additional planet candidates
        """
        logger.info(f"Searching for additional planets (removing {len(known_periods)} known)")
        
        # Create residual flux
        residual_flux = flux.copy()
        
        for period, t0 in known_periods:
            residual_flux = self._remove_transit(time, residual_flux, period, t0)
        
        # Run BLS on residuals
        try:
            from skills.periodogram import BLSPeriodogram
            
            bls = BLSPeriodogram(
                min_period=0.5,
                max_period=min(50, (time.max() - time.min()) / 2),
                n_periods=5000
            )
            
            result = bls.compute(time, residual_flux, min_snr=min_snr)
            
            candidates = []
            for c in result.candidates:
                # Check it's not a harmonic of known periods
                is_harmonic = False
                for known_p, _ in known_periods:
                    ratio = c.period / known_p
                    if abs(ratio - round(ratio)) < 0.05:  # Within 5% of harmonic
                        is_harmonic = True
                        break
                
                if not is_harmonic:
                    candidates.append(AdditionalPlanetCandidate(
                        period=c.period,
                        snr=c.snr,
                        depth=c.depth
                    ))
            
            return candidates
            
        except ImportError:
            logger.warning("BLS not available for additional planet search")
            return []

    def _remove_transit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> np.ndarray:
        """
        Remove transit signal by replacing in-transit points with baseline.
        
        Simple masking approach - more sophisticated would fit and subtract.
        """
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        in_transit = np.abs(phase) < 0.05
        out_transit = ~in_transit
        
        baseline = np.median(flux[out_transit]) if np.sum(out_transit) > 10 else 1.0
        
        cleaned = flux.copy()
        cleaned[in_transit] = baseline
        
        return cleaned


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_ttv(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    **kwargs
) -> TTVResult:
    """
    Convenience function for TTV analysis.
    
    Args:
        time: Time array
        flux: Flux array
        period: Orbital period
        t0: Reference epoch
        
    Returns:
        TTVResult
    """
    analyzer = TTVAnalyzer(**kwargs)
    return analyzer.analyze(time, flux, period, t0)


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'PLANET-009': {
        'id': 'PLANET-009',
        'name': 'TTV Analysis',
        'command': 'ttv',
        'class': TTVAnalyzer,
        'description': 'Transit Timing Variations for multi-planet detection'
    }
}


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing TTV Analyzer...")
    
    np.random.seed(42)
    
    # Generate synthetic light curve with TTV
    period = 3.5
    t0 = 1.0
    n_transits = 10
    
    # Time array covering multiple transits
    time = np.linspace(0, period * (n_transits + 1), 50000)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    
    # Add transits with small TTV
    for n in range(n_transits):
        # Add TTV: sinusoidal variation with 5-minute amplitude
        ttv = 5.0 / (24 * 60) * np.sin(2 * np.pi * n / 5)  # 5-transit period
        transit_time = t0 + n * period + ttv
        
        # Add transit
        dist = np.abs(time - transit_time)
        in_transit = dist < 0.05 * period
        flux[in_transit] -= 0.01  # 1% depth
    
    # Run analysis
    analyzer = TTVAnalyzer()
    result = analyzer.analyze(time, flux, period, t0)
    
    print(result.summary())
    print(f"\nMeasured {len(result.transit_times)} transits")
    print(f"O-C values (minutes): {result.oc_minutes}")
