"""
LARUN Skill: Periodogram Analysis
==================================
Compute BLS and Lomb-Scargle periodograms for transit and variability detection.

Skill IDs: ANAL-001 (BLS), ANAL-010 (Lomb-Scargle)
Commands: larun analyze bls, larun analyze lomb-scargle

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: docs/research/EXOPLANET_DETECTION.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TransitCandidate:
    """Represents a potential transit signal."""
    period: float           # Orbital period (days)
    t0: float              # Mid-transit time (BJD)
    depth: float           # Transit depth (fractional)
    duration: float        # Transit duration (days)
    snr: float             # Signal-to-noise ratio
    power: float           # BLS power
    fap: float             # False alarm probability

    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_days': round(self.period, 6),
            't0_bjd': round(self.t0, 6),
            'depth_ppm': round(self.depth * 1e6, 1),
            'duration_hours': round(self.duration * 24, 2),
            'snr': round(self.snr, 2),
            'power': round(self.power, 4),
            'fap': self.fap
        }

    def __str__(self) -> str:
        return (f"Transit Candidate: P={self.period:.4f}d, "
                f"depth={self.depth*1e6:.0f}ppm, SNR={self.snr:.1f}")


@dataclass
class PeriodogramResult:
    """Result of periodogram analysis."""
    periods: np.ndarray
    power: np.ndarray
    best_period: float
    best_power: float
    fap: float
    method: str
    candidates: List[TransitCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'best_period': round(self.best_period, 6),
            'best_power': round(self.best_power, 4),
            'fap': self.fap,
            'n_candidates': len(self.candidates),
            'candidates': [c.to_dict() for c in self.candidates],
            'periods': self.periods.tolist(),
            'power': self.power.tolist()
        }


# ============================================================================
# BLS Periodogram
# ============================================================================

class BLSPeriodogram:
    """
    Box Least Squares Periodogram for transit detection.

    Based on: Kovács, Zucker, & Mazeh (2002)
    Reference: docs/research/EXOPLANET_DETECTION.md

    The BLS algorithm models transits as box-shaped dips in the light curve
    and searches for periodic signals.

    Example:
        >>> bls = BLSPeriodogram(min_period=0.5, max_period=20)
        >>> result = bls.compute(time, flux)
        >>> print(f"Best period: {result.best_period:.4f} days")
        >>> for candidate in result.candidates:
        ...     print(candidate)
    """

    def __init__(
        self,
        min_period: float = 0.5,
        max_period: float = 50.0,
        min_duration: float = 0.01,
        max_duration: float = 0.15,
        n_periods: int = 10000,
        n_durations: int = 10,
        oversample: int = 5
    ):
        """
        Initialize BLS periodogram parameters.

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            min_duration: Minimum transit duration (fraction of period)
            max_duration: Maximum transit duration (fraction of period)
            n_periods: Number of period samples
            n_durations: Number of duration samples
            oversample: Frequency oversampling factor
        """
        self.min_period = min_period
        self.max_period = max_period
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_periods = n_periods
        self.n_durations = n_durations
        self.oversample = oversample

        logger.info(f"BLS initialized: P=[{min_period}, {max_period}] days")

    def compute(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        min_snr: float = 7.0
    ) -> PeriodogramResult:
        """
        Compute BLS periodogram.

        Args:
            time: Time array (days, BJD recommended)
            flux: Normalized flux array (median ~1.0)
            flux_err: Optional flux uncertainties
            min_snr: Minimum SNR for candidate detection

        Returns:
            PeriodogramResult with periods, power, and transit candidates
        """
        logger.info("Computing BLS periodogram...")

        # Input validation
        time, flux, flux_err = self._validate_input(time, flux, flux_err)

        # Try to use astropy's optimized BLS
        try:
            return self._compute_astropy(time, flux, flux_err, min_snr)
        except ImportError:
            logger.warning("astropy.timeseries not available, using fallback")
            return self._compute_fallback(time, flux, flux_err, min_snr)

    def _validate_input(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and clean input arrays."""

        if len(time) != len(flux):
            raise ValueError("Time and flux arrays must have same length")

        if len(time) < 100:
            raise ValueError("Need at least 100 data points for BLS")

        # Remove NaN/Inf values
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err)

        time = np.asarray(time[mask], dtype=np.float64)
        flux = np.asarray(flux[mask], dtype=np.float64)

        if flux_err is not None:
            flux_err = np.asarray(flux_err[mask], dtype=np.float64)
        else:
            # Estimate uncertainty from scatter
            flux_err = np.full_like(flux, np.std(flux) * 1.4826)  # MAD estimate

        # Normalize flux to median = 1
        median_flux = np.median(flux)
        if median_flux > 0:
            flux = flux / median_flux
            flux_err = flux_err / median_flux

        logger.info(f"Input validated: {len(time)} points, "
                   f"baseline={time.max()-time.min():.1f} days")

        return time, flux, flux_err

    def _compute_astropy(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        min_snr: float
    ) -> PeriodogramResult:
        """Compute BLS using astropy (optimized)."""
        from astropy.timeseries import BoxLeastSquares
        from astropy import units as u

        # Create BLS model
        bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)

        # Define period grid
        periods = np.linspace(self.min_period, self.max_period, self.n_periods)

        # Define duration grid (must be shorter than min period)
        # Duration is in days, not fraction of period for astropy
        max_duration_days = min(self.max_duration * self.min_period, self.min_period * 0.5)
        min_duration_days = self.min_duration * self.min_period
        durations = np.linspace(min_duration_days, max_duration_days, self.n_durations)

        # Compute periodogram
        logger.info(f"Computing BLS over {len(periods)} periods...")
        periodogram = bls.power(
            periods * u.day,
            duration=durations * u.day,
            oversample=self.oversample
        )

        power = periodogram.power.value if hasattr(periodogram.power, 'value') else periodogram.power
        periods_out = periodogram.period.value if hasattr(periodogram.period, 'value') else periodogram.period

        # Find best period
        best_idx = np.argmax(power)
        best_period = periods_out[best_idx]
        best_power = power[best_idx]

        # Get transit parameters for best period
        best_duration = periodogram.duration[best_idx]
        if hasattr(best_duration, 'value'):
            best_duration = best_duration.value

        best_t0 = periodogram.transit_time[best_idx]
        if hasattr(best_t0, 'value'):
            best_t0 = best_t0.value

        # Compute statistics
        stats = bls.compute_stats(
            best_period * u.day,
            best_duration * u.day,
            best_t0 * u.day
        )

        depth = stats['depth'][0] if hasattr(stats['depth'], '__len__') else stats['depth']

        # False alarm probability
        fap = self._estimate_fap(best_power, len(periods))

        # Create candidate list
        candidates = []
        snr = best_power / np.std(power)  # Approximate SNR

        if snr >= min_snr:
            candidate = TransitCandidate(
                period=float(best_period),
                t0=float(best_t0),
                depth=float(depth),
                duration=float(best_duration),
                snr=float(snr),
                power=float(best_power),
                fap=float(fap)
            )
            candidates.append(candidate)
            logger.info(f"Found candidate: {candidate}")

        # Search for additional candidates (secondary peaks)
        candidates.extend(self._find_secondary_peaks(
            periods_out, power, candidates, min_snr
        ))

        logger.info(f"BLS complete: best_period={best_period:.4f}d, "
                   f"SNR={snr:.1f}, FAP={fap:.2e}")

        return PeriodogramResult(
            periods=periods_out,
            power=power,
            best_period=float(best_period),
            best_power=float(best_power),
            fap=float(fap),
            method='BLS (astropy)',
            candidates=candidates
        )

    def _compute_fallback(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        min_snr: float
    ) -> PeriodogramResult:
        """Compute BLS using pure numpy (slower but no dependencies)."""
        logger.info("Using fallback BLS implementation...")

        # Period grid
        periods = np.linspace(self.min_period, self.max_period, self.n_periods)
        power = np.zeros(len(periods))

        # Duration grid (as fraction)
        durations = np.linspace(self.min_duration, self.max_duration, self.n_durations)

        # Compute BLS for each period
        for i, period in enumerate(periods):
            if i % 1000 == 0:
                logger.debug(f"Processing period {i}/{len(periods)}")

            # Phase fold
            phase = (time % period) / period

            best_sr = 0
            for q in durations:
                # Slide box across all phases
                n_phases = max(10, int(1.0 / q))
                for phi0 in np.linspace(0, 1 - q, n_phases):
                    # Points in transit
                    in_transit = (phase >= phi0) & (phase < phi0 + q)
                    n_in = np.sum(in_transit)

                    if n_in < 3:
                        continue

                    # BLS statistic (signal residue)
                    r = n_in / len(flux)
                    s = np.sum(flux[in_transit] - 1.0)

                    # Avoid division by zero
                    denom = r * (1 - r) * len(flux)
                    if denom > 0:
                        sr = np.abs(s) / np.sqrt(denom)
                        if sr > best_sr:
                            best_sr = sr

            power[i] = best_sr

        # Find best period
        best_idx = np.argmax(power)
        best_period = periods[best_idx]
        best_power = power[best_idx]

        # Estimate FAP
        fap = self._estimate_fap(best_power, len(periods))

        # SNR estimate
        snr = best_power / np.std(power)

        # Create candidate
        candidates = []
        if snr >= min_snr:
            # Estimate depth and duration at best period
            phase = (time % best_period) / best_period
            depth, duration, t0 = self._estimate_transit_params(
                time, flux, phase, best_period
            )

            candidates.append(TransitCandidate(
                period=best_period,
                t0=t0,
                depth=depth,
                duration=duration,
                snr=snr,
                power=best_power,
                fap=fap
            ))

        return PeriodogramResult(
            periods=periods,
            power=power,
            best_period=best_period,
            best_power=best_power,
            fap=fap,
            method='BLS (fallback)',
            candidates=candidates
        )

    def _estimate_transit_params(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        phase: np.ndarray,
        period: float
    ) -> Tuple[float, float, float]:
        """Estimate transit depth, duration, and t0."""
        # Bin the phase-folded light curve
        n_bins = 100
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_flux = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_flux[i] = np.mean(flux[mask])
            else:
                bin_flux[i] = 1.0

        # Find minimum (transit)
        min_idx = np.argmin(bin_flux)
        depth = 1.0 - bin_flux[min_idx]

        # Estimate duration (where flux < median - 0.5*depth)
        threshold = 1.0 - 0.5 * depth
        in_transit = bin_flux < threshold
        duration = np.sum(in_transit) / n_bins * period

        # Estimate t0
        t0_phase = (bin_edges[min_idx] + bin_edges[min_idx + 1]) / 2
        t0 = time.min() + t0_phase * period

        return depth, duration, t0

    def _find_secondary_peaks(
        self,
        periods: np.ndarray,
        power: np.ndarray,
        existing: List[TransitCandidate],
        min_snr: float
    ) -> List[TransitCandidate]:
        """Find secondary peaks that might indicate additional planets."""
        candidates = []

        # Mask out regions around existing candidates
        masked_power = power.copy()
        for c in existing:
            # Mask harmonics and aliases
            for harmonic in [0.5, 1.0, 2.0]:
                mask = np.abs(periods - c.period * harmonic) < c.period * 0.05
                masked_power[mask] = 0

        # Find next highest peak
        noise = np.std(masked_power[masked_power > 0])
        if noise > 0:
            secondary_idx = np.argmax(masked_power)
            secondary_snr = masked_power[secondary_idx] / noise

            if secondary_snr >= min_snr * 0.7:  # Slightly lower threshold
                logger.info(f"Secondary peak at P={periods[secondary_idx]:.4f}d, "
                           f"SNR={secondary_snr:.1f}")

        return candidates

    def _estimate_fap(self, max_power: float, n_trials: int) -> float:
        """
        Estimate false alarm probability.

        Uses approximation from Kovács et al. (2002).
        """
        # Simplified FAP estimate
        fap = 1.0 - (1.0 - np.exp(-max_power))**n_trials
        return min(fap, 1.0)


# ============================================================================
# Lomb-Scargle Periodogram
# ============================================================================

class LombScarglePeriodogram:
    """
    Lomb-Scargle Periodogram for variability detection.

    Based on: Lomb (1976), Scargle (1982)

    Best for detecting sinusoidal variability (pulsations, rotation, binaries).
    For box-shaped transits, use BLSPeriodogram instead.

    Example:
        >>> lsp = LombScarglePeriodogram(min_period=0.1, max_period=100)
        >>> result = lsp.compute(time, flux)
        >>> print(f"Best period: {result.best_period:.4f} days")
    """

    def __init__(
        self,
        min_period: float = 0.1,
        max_period: float = 100.0,
        samples_per_peak: int = 5
    ):
        """
        Initialize Lomb-Scargle parameters.

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            samples_per_peak: Frequency resolution
        """
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak

    def compute(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None
    ) -> PeriodogramResult:
        """
        Compute Lomb-Scargle periodogram.

        Args:
            time: Time array (days)
            flux: Flux array (normalized)
            flux_err: Optional flux uncertainties

        Returns:
            PeriodogramResult with periods and power
        """
        logger.info("Computing Lomb-Scargle periodogram...")

        # Remove NaN values
        mask = np.isfinite(time) & np.isfinite(flux)
        time = np.asarray(time[mask], dtype=np.float64)
        flux = np.asarray(flux[mask], dtype=np.float64)

        if len(time) < 10:
            raise ValueError("Need at least 10 data points")

        # Frequency grid
        baseline = time.max() - time.min()
        df = 1.0 / (baseline * self.samples_per_peak)

        f_min = 1.0 / self.max_period
        f_max = 1.0 / self.min_period

        frequency = np.arange(f_min, f_max, df)

        # Try astropy first
        try:
            from astropy.timeseries import LombScargle

            ls = LombScargle(time, flux, flux_err)
            power = ls.power(frequency)
            fap = ls.false_alarm_probability(power.max())

        except ImportError:
            # Fallback to scipy
            from scipy.signal import lombscargle

            angular_freq = 2 * np.pi * frequency
            flux_centered = flux - np.mean(flux)
            power = lombscargle(time, flux_centered, angular_freq)
            power = power / (0.5 * len(flux) * np.var(flux))
            fap = self._estimate_fap(power.max(), len(frequency))

        # Convert to period
        periods = 1.0 / frequency

        # Find best period
        best_idx = np.argmax(power)
        best_period = periods[best_idx]
        best_power = power[best_idx]

        logger.info(f"LS complete: best_period={best_period:.4f}d, FAP={fap:.2e}")

        return PeriodogramResult(
            periods=periods,
            power=power,
            best_period=best_period,
            best_power=best_power,
            fap=fap,
            method='Lomb-Scargle',
            candidates=[]
        )

    def _estimate_fap(self, max_power: float, n_freq: int) -> float:
        """Estimate false alarm probability."""
        fap = 1 - (1 - np.exp(-max_power))**n_freq
        return min(fap, 1.0)


# ============================================================================
# Phase Folding
# ============================================================================

def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase fold a light curve.

    Args:
        time: Time array (days)
        flux: Flux array
        period: Orbital period (days)
        t0: Reference epoch (mid-transit time)

    Returns:
        phase: Phase array (-0.5 to 0.5, transit at 0)
        flux: Flux sorted by phase

    Example:
        >>> phase, flux_folded = phase_fold(time, flux, period=3.5, t0=2458000.0)
        >>> plt.scatter(phase, flux_folded, s=1)
    """
    # Calculate phase
    phase = ((time - t0) % period) / period

    # Center transit at phase 0
    phase[phase > 0.5] -= 1.0

    # Sort by phase
    sort_idx = np.argsort(phase)

    return phase[sort_idx], flux[sort_idx]


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a phase-folded light curve.

    Args:
        phase: Phase array (-0.5 to 0.5)
        flux: Flux array
        n_bins: Number of bins

    Returns:
        bin_centers: Center of each bin
        bin_flux: Mean flux in each bin
        bin_err: Standard error in each bin
    """
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_flux = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_flux[i] = np.mean(flux[mask])
            bin_err[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
        else:
            bin_flux[i] = np.nan
            bin_err[i] = np.nan

    return bin_centers, bin_flux, bin_err


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'ANAL-001': {
        'id': 'ANAL-001',
        'name': 'BLS Periodogram',
        'command': 'analyze bls',
        'class': BLSPeriodogram,
        'description': 'Box Least Squares periodogram for transit detection'
    },
    'ANAL-010': {
        'id': 'ANAL-010',
        'name': 'Lomb-Scargle Periodogram',
        'command': 'analyze lomb-scargle',
        'class': LombScarglePeriodogram,
        'description': 'Lomb-Scargle periodogram for variability detection'
    }
}


# ============================================================================
# CLI Functions
# ============================================================================

def run_bls(time: np.ndarray, flux: np.ndarray, **kwargs) -> PeriodogramResult:
    """Run BLS analysis (convenience function)."""
    bls = BLSPeriodogram(**{k: v for k, v in kwargs.items()
                           if k in ['min_period', 'max_period', 'n_periods']})
    return bls.compute(time, flux, min_snr=kwargs.get('min_snr', 7.0))


def run_lomb_scargle(time: np.ndarray, flux: np.ndarray, **kwargs) -> PeriodogramResult:
    """Run Lomb-Scargle analysis (convenience function)."""
    lsp = LombScarglePeriodogram(**{k: v for k, v in kwargs.items()
                                   if k in ['min_period', 'max_period', 'samples_per_peak']})
    return lsp.compute(time, flux)


if __name__ == '__main__':
    # Quick test
    print("Testing BLS Periodogram...")

    # Generate synthetic transit data
    np.random.seed(42)
    period = 3.5  # days
    t0 = 0.5
    depth = 0.01  # 1% transit
    duration = 0.1  # 10% of period

    # Time array (like TESS)
    time = np.linspace(0, 27, 10000)  # 27 days, 2-min cadence approx

    # Base flux with noise
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

    # Add transits
    phase = ((time - t0) % period) / period
    in_transit = phase < duration
    flux[in_transit] -= depth

    # Run BLS
    bls = BLSPeriodogram(min_period=1, max_period=10, n_periods=1000)
    result = bls.compute(time, flux)

    print(f"True period: {period:.4f} days")
    print(f"Found period: {result.best_period:.4f} days")
    print(f"Error: {abs(result.best_period - period):.4f} days")
    print(f"Candidates: {len(result.candidates)}")

    if result.candidates:
        print(f"Best candidate: {result.candidates[0]}")
