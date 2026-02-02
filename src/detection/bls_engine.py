"""
BLS Engine
==========
Wrapper around the existing BLS periodogram implementation.

This module provides a clean interface for BLS analysis, wrapping
the existing src/skills/periodogram.py implementation.

Author: Agent ALPHA
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

from .models import PeriodogramData

# Import existing BLS implementation
from src.skills.periodogram import (
    BLSPeriodogram,
    PeriodogramResult as SkillPeriodogramResult,
    TransitCandidate,
)

logger = logging.getLogger(__name__)


class BLSEngine:
    """
    Box Least Squares periodogram engine.

    Wraps the existing BLSPeriodogram class from src/skills/periodogram.py
    and provides a simplified interface for the detection service.

    Example:
        >>> engine = BLSEngine(min_period=0.5, max_period=20.0)
        >>> result = engine.compute(time, flux)
        >>> print(f"Best period: {result.best_period:.4f} days")
    """

    def __init__(
        self,
        min_period: float = 0.5,
        max_period: float = 50.0,
        min_duration_fraction: float = 0.01,
        max_duration_fraction: float = 0.15,
        n_periods: int = 10000,
        n_durations: int = 10,
        oversample: int = 5,
        min_snr: float = 7.0,
    ):
        """
        Initialize the BLS engine.

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            min_duration_fraction: Minimum transit duration as fraction of period
            max_duration_fraction: Maximum transit duration as fraction of period
            n_periods: Number of period samples
            n_durations: Number of duration samples
            oversample: Frequency oversampling factor
            min_snr: Minimum SNR for detection threshold
        """
        self.min_period = min_period
        self.max_period = max_period
        self.min_snr = min_snr

        # Initialize underlying BLS
        self._bls = BLSPeriodogram(
            min_period=min_period,
            max_period=max_period,
            min_duration=min_duration_fraction,
            max_duration=max_duration_fraction,
            n_periods=n_periods,
            n_durations=n_durations,
            oversample=oversample,
        )

        logger.info(
            f"BLSEngine initialized: period=[{min_period}, {max_period}] days, "
            f"min_snr={min_snr}"
        )

    def compute(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
    ) -> Tuple[PeriodogramData, Optional[TransitCandidate]]:
        """
        Compute BLS periodogram.

        Args:
            time: Time array (days, BJD/BTJD)
            flux: Normalized flux array (median ~1.0)
            flux_err: Optional flux uncertainties

        Returns:
            Tuple of (PeriodogramData, best TransitCandidate or None)
        """
        logger.info(f"Computing BLS on {len(time)} data points")

        # Run the underlying BLS computation
        skill_result = self._bls.compute(
            time=time,
            flux=flux,
            flux_err=flux_err,
            min_snr=self.min_snr,
        )

        # Convert to our PeriodogramData model
        periodogram_data = self._convert_to_periodogram_data(skill_result)

        # Get best candidate
        best_candidate = None
        if skill_result.candidates:
            best_candidate = skill_result.candidates[0]

        logger.info(
            f"BLS complete: best_period={periodogram_data.best_period:.4f}d, "
            f"candidates={len(skill_result.candidates)}"
        )

        return periodogram_data, best_candidate

    def _convert_to_periodogram_data(
        self, result: SkillPeriodogramResult
    ) -> PeriodogramData:
        """Convert skill result to PeriodogramData model."""
        # Get top N periods (up to 3)
        n_top = min(3, len(result.periods))

        # Sort by power to get top periods
        sorted_idx = np.argsort(result.power)[::-1][:n_top]
        top_periods = result.periods[sorted_idx].tolist()
        top_powers = result.power[sorted_idx].tolist()

        return PeriodogramData(
            periods=result.periods.tolist(),
            powers=result.power.tolist(),
            best_period=float(result.best_period),
            best_power=float(result.best_power),
            top_periods=top_periods,
            top_powers=top_powers,
        )

    def estimate_transit_parameters(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
    ) -> dict:
        """
        Estimate transit parameters at a given period.

        Args:
            time: Time array
            flux: Flux array
            period: Orbital period (days)
            t0: Mid-transit epoch

        Returns:
            Dictionary with depth, duration, snr estimates
        """
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0

        # Find in-transit points
        in_transit = np.abs(phase) < 0.05
        out_transit = np.abs(phase) > 0.15

        if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
            return {
                "depth": 0.0,
                "depth_ppm": 0.0,
                "duration_hours": 0.0,
                "snr": 0.0,
            }

        # Calculate depth
        baseline = np.median(flux[out_transit])
        in_transit_flux = np.mean(flux[in_transit])
        depth = baseline - in_transit_flux

        # Estimate duration
        threshold = baseline - 0.5 * depth
        below_threshold = flux < threshold
        duration_fraction = np.sum(below_threshold & (np.abs(phase) < 0.2)) / len(flux)
        duration_days = duration_fraction * period
        duration_hours = duration_days * 24

        # Estimate SNR
        noise = np.std(flux[out_transit])
        snr = depth / noise if noise > 0 else 0.0

        return {
            "depth": float(depth),
            "depth_ppm": float(depth * 1e6),
            "duration_hours": float(duration_hours),
            "snr": float(snr),
        }

    @staticmethod
    def validate_light_curve(
        time: np.ndarray,
        flux: np.ndarray,
        min_points: int = 100,
    ) -> Tuple[bool, str]:
        """
        Validate light curve data before BLS analysis.

        Args:
            time: Time array
            flux: Flux array
            min_points: Minimum required data points

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check array lengths
        if len(time) != len(flux):
            return False, "Time and flux arrays have different lengths"

        # Check minimum points
        valid_mask = np.isfinite(time) & np.isfinite(flux)
        n_valid = np.sum(valid_mask)

        if n_valid < min_points:
            return False, f"Insufficient valid data points ({n_valid} < {min_points})"

        # Check baseline
        baseline = time[valid_mask].max() - time[valid_mask].min()
        if baseline < 1.0:
            return False, f"Insufficient time baseline ({baseline:.2f} days < 1 day)"

        # Check flux range
        flux_valid = flux[valid_mask]
        flux_range = flux_valid.max() - flux_valid.min()
        if flux_range < 1e-6:
            return False, "Flux values have no variation"

        return True, ""


def run_bls_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period: float = 0.5,
    max_period: float = 50.0,
    min_snr: float = 7.0,
) -> Tuple[PeriodogramData, Optional[dict]]:
    """
    Convenience function to run BLS analysis.

    Args:
        time: Time array (days)
        flux: Normalized flux array
        flux_err: Optional flux errors
        min_period: Minimum period (days)
        max_period: Maximum period (days)
        min_snr: Minimum SNR threshold

    Returns:
        Tuple of (PeriodogramData, candidate_dict or None)
    """
    engine = BLSEngine(
        min_period=min_period,
        max_period=max_period,
        min_snr=min_snr,
    )

    periodogram, candidate = engine.compute(time, flux, flux_err)

    candidate_dict = candidate.to_dict() if candidate else None

    return periodogram, candidate_dict
