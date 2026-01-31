"""
LARUN Skill: Multi-Planet Detection
===================================
Detect multiple planets by iterative BLS analysis.

Skill ID: DISC-001
Command: larun discover multiplanet

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlanetCandidate:
    """A detected planet candidate."""
    planet_number: int        # 1, 2, 3, etc.
    period: float             # Orbital period (days)
    t0: float                 # Mid-transit time
    depth: float              # Transit depth (fractional)
    depth_ppm: float          # Transit depth (ppm)
    duration: float           # Transit duration (days)
    snr: float                # Detection SNR
    power: float              # BLS power
    fap: float                # False alarm probability

    def to_dict(self) -> Dict[str, Any]:
        return {
            'planet': self.planet_number,
            'period_days': round(self.period, 6),
            't0': round(self.t0, 6),
            'depth_ppm': round(self.depth_ppm, 1),
            'duration_hours': round(self.duration * 24, 2),
            'snr': round(self.snr, 2),
            'power': round(self.power, 4),
            'fap': self.fap
        }

    def __str__(self) -> str:
        return f"Planet {self.planet_number}: P={self.period:.4f}d, depth={self.depth_ppm:.0f}ppm, SNR={self.snr:.1f}"


@dataclass
class MultiPlanetResult:
    """Result of multi-planet detection."""
    target: str
    n_planets: int
    candidates: List[PlanetCandidate]
    baseline: float           # Data baseline (days)
    n_points: int             # Number of data points
    residual_rms: float       # Final residual RMS

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target,
            'n_planets_detected': self.n_planets,
            'candidates': [c.to_dict() for c in self.candidates],
            'baseline_days': round(self.baseline, 1),
            'n_points': self.n_points,
            'residual_rms_ppm': round(self.residual_rms * 1e6, 1)
        }

    def __str__(self) -> str:
        planets_str = ", ".join([f"P{c.planet_number}={c.period:.3f}d" for c in self.candidates])
        return f"MultiPlanet: {self.n_planets} planets detected [{planets_str}]"


# ============================================================================
# Multi-Planet Detector
# ============================================================================

class MultiPlanetDetector:
    """
    Detect multiple planets using iterative BLS.

    Algorithm:
    1. Run BLS to find strongest periodic signal
    2. If SNR > threshold, record as planet candidate
    3. Remove transit signal from data
    4. Repeat until no more significant signals found

    Example:
        >>> detector = MultiPlanetDetector(max_planets=5, min_snr=7.0)
        >>> result = detector.detect(time, flux, target="TOI-700")
        >>> for planet in result.candidates:
        ...     print(planet)
    """

    def __init__(
        self,
        max_planets: int = 5,
        min_snr: float = 7.0,
        min_period: float = 0.5,
        max_period: float = 50.0,
        n_periods: int = 10000,
        period_tolerance: float = 0.01
    ):
        """
        Initialize multi-planet detector.

        Args:
            max_planets: Maximum number of planets to search for
            min_snr: Minimum SNR for detection
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            n_periods: Number of period samples
            period_tolerance: Fractional tolerance for alias rejection
        """
        self.max_planets = max_planets
        self.min_snr = min_snr
        self.min_period = min_period
        self.max_period = max_period
        self.n_periods = n_periods
        self.period_tolerance = period_tolerance

        logger.info(f"MultiPlanetDetector initialized: max={max_planets}, min_snr={min_snr}")

    def detect(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        target: str = "Unknown"
    ) -> MultiPlanetResult:
        """
        Detect multiple planets in a light curve.

        Args:
            time: Time array (days)
            flux: Normalized flux array
            flux_err: Optional flux uncertainties
            target: Target name for reporting

        Returns:
            MultiPlanetResult with all detected candidates
        """
        try:
            from .periodogram import BLSPeriodogram
        except ImportError:
            from periodogram import BLSPeriodogram

        logger.info(f"Starting multi-planet search for {target}")

        # Clean input data
        mask = np.isfinite(time) & np.isfinite(flux)
        time = np.asarray(time[mask], dtype=np.float64)
        flux = np.asarray(flux[mask], dtype=np.float64)

        if flux_err is not None:
            flux_err = np.asarray(flux_err[mask], dtype=np.float64)

        # Normalize
        flux = flux / np.median(flux)

        baseline = time.max() - time.min()
        n_points = len(time)

        candidates = []
        detected_periods = []
        residual_flux = flux.copy()

        # Iterative detection
        for planet_num in range(1, self.max_planets + 1):
            logger.info(f"Searching for planet {planet_num}...")

            # Run BLS on residuals
            bls = BLSPeriodogram(
                min_period=self.min_period,
                max_period=min(self.max_period, baseline / 2),
                n_periods=self.n_periods
            )

            result = bls.compute(time, residual_flux, flux_err, min_snr=self.min_snr)

            # Check if significant detection
            if not result.candidates:
                logger.info(f"No more significant signals found after {planet_num - 1} planets")
                break

            best = result.candidates[0]

            # Check for aliases/harmonics of previously detected planets
            if self._is_alias(best.period, detected_periods):
                logger.info(f"Period {best.period:.4f}d is alias of previous detection, skipping")
                continue

            # Record candidate
            candidate = PlanetCandidate(
                planet_number=planet_num,
                period=best.period,
                t0=best.t0,
                depth=best.depth,
                depth_ppm=best.depth * 1e6,
                duration=best.duration,
                snr=best.snr,
                power=best.power,
                fap=best.fap
            )

            candidates.append(candidate)
            detected_periods.append(best.period)

            logger.info(f"Detected: {candidate}")

            # Remove transit from residuals
            residual_flux = self._remove_transit(
                time, residual_flux, best.period, best.t0, best.depth, best.duration
            )

        # Calculate final residual RMS
        residual_rms = np.std(residual_flux - 1.0)

        result = MultiPlanetResult(
            target=target,
            n_planets=len(candidates),
            candidates=candidates,
            baseline=baseline,
            n_points=n_points,
            residual_rms=residual_rms
        )

        logger.info(f"Detection complete: {result}")
        return result

    def _is_alias(self, period: float, detected_periods: List[float]) -> bool:
        """
        Check if period is an alias of previously detected periods.

        Checks for harmonics (2P, P/2) and common aliases.
        """
        for p in detected_periods:
            # Check harmonics
            for harmonic in [0.5, 1.0, 2.0, 3.0, 1/3]:
                if abs(period - p * harmonic) / p < self.period_tolerance:
                    return True

            # Check day aliases (for ground-based)
            for alias in [1.0, 0.5]:
                alias_period = 1.0 / (1.0/p + alias) if p > alias else None
                if alias_period and abs(period - alias_period) / p < self.period_tolerance:
                    return True

        return False

    def _remove_transit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        depth: float,
        duration: float
    ) -> np.ndarray:
        """
        Remove transit signal from flux.

        Uses a simple box model to mask and interpolate over transits.
        """
        # Calculate phase
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0

        # Identify in-transit points
        phase_duration = duration / period
        in_transit = np.abs(phase) < phase_duration / 2

        # Create copy and fix transits
        flux_cleaned = flux.copy()

        # Option 1: Add back the depth
        flux_cleaned[in_transit] += depth

        # Option 2: Interpolate (alternative, more conservative)
        # Could implement linear interpolation across transits

        return flux_cleaned

    def search_ttv(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        duration: float
    ) -> Dict[str, Any]:
        """
        Search for Transit Timing Variations (TTV).

        TTVs can indicate gravitational interactions between planets.

        Args:
            time: Time array
            flux: Flux array
            period: Nominal orbital period
            t0: Reference mid-transit time
            duration: Transit duration

        Returns:
            Dictionary with TTV measurements
        """
        logger.info(f"Searching for TTVs at P={period:.4f}d")

        # Find individual transits
        n_transits = int((time.max() - time.min()) / period) + 1
        transit_times = []
        o_c = []  # Observed - Calculated

        for n in range(n_transits):
            expected_t0 = t0 + n * period

            # Find data near this transit
            near_transit = np.abs(time - expected_t0) < duration * 2

            if np.sum(near_transit) < 10:
                continue

            t_near = time[near_transit]
            f_near = flux[near_transit]

            # Find actual minimum (simple approach)
            min_idx = np.argmin(f_near)
            observed_t0 = t_near[min_idx]

            transit_times.append(observed_t0)
            o_c.append((observed_t0 - expected_t0) * 24 * 60)  # Convert to minutes

        transit_times = np.array(transit_times)
        o_c = np.array(o_c)

        # TTV statistics
        ttv_rms = np.std(o_c) if len(o_c) > 1 else 0
        ttv_amplitude = (np.max(o_c) - np.min(o_c)) / 2 if len(o_c) > 1 else 0

        return {
            'n_transits': len(transit_times),
            'transit_times': transit_times.tolist(),
            'o_c_minutes': o_c.tolist(),
            'ttv_rms_minutes': round(ttv_rms, 2),
            'ttv_amplitude_minutes': round(ttv_amplitude, 2),
            'significant_ttv': ttv_rms > 1.0  # > 1 minute RMS
        }


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'DISC-001': {
        'id': 'DISC-001',
        'name': 'Multi-Planet Detection',
        'command': 'discover multiplanet',
        'class': MultiPlanetDetector,
        'description': 'Detect multiple planets via iterative BLS'
    }
}


# ============================================================================
# CLI Functions
# ============================================================================

def detect_multiplanet(
    time: np.ndarray,
    flux: np.ndarray,
    target: str = "Unknown",
    **kwargs
) -> MultiPlanetResult:
    """Detect multiple planets (convenience function)."""
    detector = MultiPlanetDetector(**{k: v for k, v in kwargs.items()
                                      if k in ['max_planets', 'min_snr', 'min_period', 'max_period']})
    return detector.detect(time, flux, target=target)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Multi-Planet Detection...")
    print("=" * 50)

    # Generate synthetic multi-planet system
    np.random.seed(42)

    # Three planets with different periods
    planets = [
        {'period': 3.5, 't0': 0.5, 'depth': 0.01},
        {'period': 7.2, 't0': 1.0, 'depth': 0.005},
        {'period': 15.0, 't0': 2.0, 'depth': 0.003},
    ]

    # Generate time array (like TESS extended mission)
    time = np.linspace(0, 100, 20000)

    # Base flux with noise
    noise_level = 0.001
    flux = np.ones_like(time) + np.random.normal(0, noise_level, len(time))

    # Add transits for each planet
    for p in planets:
        phase = ((time - p['t0']) % p['period']) / p['period']
        in_transit = phase < 0.02
        flux[in_transit] -= p['depth']

    print(f"Injected {len(planets)} planets:")
    for i, p in enumerate(planets, 1):
        print(f"  Planet {i}: P={p['period']:.1f}d, depth={p['depth']*1e6:.0f}ppm")

    # Run detection
    print("\nRunning multi-planet detection...")
    detector = MultiPlanetDetector(max_planets=5, min_snr=5.0)
    result = detector.detect(time, flux, target="Synthetic System")

    print(f"\n{result}")
    print("\nDetected planets:")
    for c in result.candidates:
        print(f"  {c}")

    # Compare with injected
    print("\nComparison:")
    for i, (injected, detected) in enumerate(zip(planets, result.candidates), 1):
        period_err = abs(detected.period - injected['period'])
        print(f"  Planet {i}: P_true={injected['period']:.2f}d, "
              f"P_detected={detected.period:.2f}d, error={period_err:.4f}d")
