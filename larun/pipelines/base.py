"""Base class for all LARUN data pipelines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base for data access pipelines.

    All pipelines return light curves as dicts with:
        times:     np.ndarray  — time stamps (MJD or BTJD or similar)
        flux:      np.ndarray  — flux / magnitude values (normalized or raw)
        flux_err:  np.ndarray  — uncertainties (same units as flux)
        meta:      dict        — additional metadata (TIC ID, sector, etc.)
    """

    SOURCE: str = ""

    @abstractmethod
    def fetch_light_curve(self, *args, **kwargs) -> dict | None:
        """Fetch a single target light curve."""

    def is_available(self) -> bool:
        """Check if the required dependencies are installed."""
        return True

    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        """Median-normalize flux to ~1.0."""
        med = np.nanmedian(flux)
        return flux / med if med != 0 else flux

    def _clean(self, times: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None):
        """Remove NaNs and sort by time."""
        mask = np.isfinite(times) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err)
        sort_idx = np.argsort(times[mask])
        t = times[mask][sort_idx]
        f = flux[mask][sort_idx]
        e = flux_err[mask][sort_idx] if flux_err is not None else None
        return t, f, e
