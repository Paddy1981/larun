"""
Phase Folding Module
====================
Utilities for phase folding light curves.

This module wraps the existing phase folding functions from
src/skills/periodogram.py and provides enhanced functionality.

Author: Agent ALPHA
"""

import numpy as np
from typing import Optional, Tuple
import logging

from .models import PhaseFoldedData

# Import existing phase folding functions
from src.skills.periodogram import phase_fold, bin_phase_curve

logger = logging.getLogger(__name__)


class PhaseFolding:
    """
    Phase folding utilities for transit analysis.

    Example:
        >>> folder = PhaseFolding(n_bins=100)
        >>> result = folder.fold(time, flux, period=3.5, t0=2458000.0)
        >>> print(f"Transit depth estimate: {result.estimate_depth():.1f} ppm")
    """

    def __init__(self, n_bins: int = 100):
        """
        Initialize phase folder.

        Args:
            n_bins: Number of bins for binned output
        """
        self.n_bins = n_bins
        logger.debug(f"PhaseFolding initialized with {n_bins} bins")

    def fold(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float = 0.0,
        flux_err: Optional[np.ndarray] = None,
    ) -> PhaseFoldedData:
        """
        Phase fold a light curve.

        Args:
            time: Time array (days, BJD/BTJD)
            flux: Flux array
            period: Orbital period (days)
            t0: Reference epoch / mid-transit time
            flux_err: Optional flux errors

        Returns:
            PhaseFoldedData with raw and binned phase-folded data
        """
        logger.info(f"Phase folding {len(time)} points at period={period:.4f}d")

        # Use existing phase_fold function
        phase, flux_sorted = phase_fold(time, flux, period, t0)

        # Handle flux errors
        if flux_err is not None:
            # Sort errors the same way as flux
            sort_idx = np.argsort(((time - t0) % period) / period)
            phase_for_sort = ((time - t0) % period) / period
            phase_for_sort[phase_for_sort > 0.5] -= 1.0
            sort_idx = np.argsort(phase_for_sort)
            flux_err_sorted = flux_err[sort_idx]
        else:
            flux_err_sorted = np.full_like(flux_sorted, np.std(flux_sorted))

        # Bin the phase-folded data
        bin_centers, bin_flux, bin_err = bin_phase_curve(
            phase, flux_sorted, self.n_bins
        )

        # Handle NaN in binned data
        valid_bins = ~np.isnan(bin_flux)
        if not np.all(valid_bins):
            # Interpolate over NaN bins
            bin_flux = np.interp(
                bin_centers,
                bin_centers[valid_bins],
                bin_flux[valid_bins]
            )
            bin_err = np.interp(
                bin_centers,
                bin_centers[valid_bins],
                bin_err[valid_bins]
            )

        return PhaseFoldedData(
            phase=phase.tolist(),
            flux=flux_sorted.tolist(),
            flux_err=flux_err_sorted.tolist(),
            binned_phase=bin_centers.tolist(),
            binned_flux=bin_flux.tolist(),
            binned_flux_err=bin_err.tolist(),
        )

    def estimate_depth(self, phase_data: PhaseFoldedData) -> float:
        """
        Estimate transit depth from phase-folded data.

        Args:
            phase_data: Phase-folded light curve data

        Returns:
            Transit depth in fractional units
        """
        binned_flux = np.array(phase_data.binned_flux)
        binned_phase = np.array(phase_data.binned_phase)

        # Get out-of-transit baseline
        out_of_transit = np.abs(binned_phase) > 0.15
        baseline = np.median(binned_flux[out_of_transit])

        # Get minimum flux (transit center)
        min_flux = np.min(binned_flux)

        depth = baseline - min_flux
        return float(depth)

    def estimate_duration(
        self,
        phase_data: PhaseFoldedData,
        period: float,
        depth: Optional[float] = None
    ) -> float:
        """
        Estimate transit duration from phase-folded data.

        Args:
            phase_data: Phase-folded light curve data
            period: Orbital period (days)
            depth: Transit depth (if known, otherwise estimated)

        Returns:
            Transit duration in hours
        """
        binned_flux = np.array(phase_data.binned_flux)
        binned_phase = np.array(phase_data.binned_phase)

        if depth is None:
            depth = self.estimate_depth(phase_data)

        # Get baseline
        out_of_transit = np.abs(binned_phase) > 0.15
        baseline = np.median(binned_flux[out_of_transit])

        # Find points below half-depth
        threshold = baseline - 0.5 * depth
        in_transit = binned_flux < threshold

        if np.sum(in_transit) < 2:
            return 0.0

        # Duration is range of phases in transit
        transit_phases = binned_phase[in_transit]
        duration_phase = np.max(transit_phases) - np.min(transit_phases)
        duration_days = duration_phase * period
        duration_hours = duration_days * 24

        return float(duration_hours)

    def find_transit_center(self, phase_data: PhaseFoldedData) -> float:
        """
        Find the phase of transit center.

        Args:
            phase_data: Phase-folded light curve data

        Returns:
            Phase of transit center (-0.5 to 0.5)
        """
        binned_flux = np.array(phase_data.binned_flux)
        binned_phase = np.array(phase_data.binned_phase)

        min_idx = np.argmin(binned_flux)
        return float(binned_phase[min_idx])


def phase_fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float = 0.0,
    flux_err: Optional[np.ndarray] = None,
    n_bins: int = 100,
) -> PhaseFoldedData:
    """
    Convenience function to phase fold a light curve.

    Args:
        time: Time array (days)
        flux: Flux array
        period: Orbital period (days)
        t0: Reference epoch
        flux_err: Optional flux errors
        n_bins: Number of bins

    Returns:
        PhaseFoldedData with folded light curve
    """
    folder = PhaseFolding(n_bins=n_bins)
    return folder.fold(time, flux, period, t0, flux_err)


def compute_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    buffer_factor: float = 1.5,
) -> np.ndarray:
    """
    Compute a boolean mask for in-transit points.

    Args:
        time: Time array (days)
        period: Orbital period (days)
        t0: Mid-transit epoch
        duration_hours: Transit duration (hours)
        buffer_factor: Factor to expand transit window

    Returns:
        Boolean array marking in-transit points
    """
    duration_days = duration_hours / 24.0
    half_duration = duration_days / 2.0 * buffer_factor

    # Phase fold
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # Convert duration to phase
    duration_phase = half_duration / period

    # Mark in-transit points
    in_transit = np.abs(phase) < duration_phase

    return in_transit
