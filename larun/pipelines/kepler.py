"""
Kepler Pipeline — Access Kepler legacy light curves via MAST.

Data source: NASA/STScI MAST
Library: Lightkurve
Mission: Kepler (2009–2018), 17 quarters, ~200,000 targets
"""

from __future__ import annotations

import logging

import numpy as np

from larun.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class KeplerPipeline(BasePipeline):
    """
    Fetch and process Kepler light curves.

    Long cadence (LC): 30-minute cadence, best for periods >1 day
    Short cadence (SC): 1-minute cadence, best for asteroseismology
    """

    SOURCE = "kepler"

    def is_available(self) -> bool:
        try:
            import lightkurve  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch_light_curve(
        self,
        kic_id: int | None = None,
        ra: float | None = None,
        dec: float | None = None,
        quarter: int | None = None,
        cadence: str = "long",
    ) -> dict | None:
        """
        Fetch a Kepler light curve.

        Args:
            kic_id:   KIC (Kepler Input Catalog) ID
            ra/dec:   Sky coordinates (degrees)
            quarter:  Kepler quarter number 1–17 (None = download all)
            cadence:  'long' (30 min) or 'short' (1 min)

        Returns:
            dict with keys: times, flux, flux_err, meta
            or None if no data found
        """
        try:
            import lightkurve as lk
        except ImportError:
            logger.error("lightkurve is not installed. pip install lightkurve")
            return None

        if kic_id is not None:
            target = f"KIC {kic_id}"
        elif ra is not None and dec is not None:
            target = f"{ra} {dec}"
        else:
            raise ValueError("Provide either kic_id or ra+dec")

        try:
            search = lk.search_lightcurve(
                target,
                mission="Kepler",
                quarter=quarter,
                cadence=cadence,
                author="Kepler",
            )

            if len(search) == 0:
                logger.info(f"No Kepler data for {target}")
                return None

            logger.info(f"Found {len(search)} Kepler observations for {target}")

            # Stitch all quarters
            lc_collection = search.download_all(flux_column="pdcsap_flux")
            lc = lc_collection.stitch()
            lc = lc.remove_nans().remove_outliers(sigma=5).normalize()

            times = np.asarray(lc.time.value, dtype=float)
            flux = np.asarray(lc.flux.value, dtype=float)
            flux_err = (
                np.asarray(lc.flux_err.value, dtype=float)
                if lc.flux_err is not None
                else np.zeros_like(flux)
            )

            times, flux, flux_err = self._clean(times, flux, flux_err)

            return {
                "times": times,
                "flux": flux,
                "flux_err": flux_err,
                "meta": {
                    "source": "kepler",
                    "kic_id": kic_id,
                    "quarter": quarter,
                    "cadence": cadence,
                    "n_points": len(times),
                    "time_span_days": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
                    "target": target,
                },
            }

        except Exception as exc:
            logger.error(f"Kepler pipeline error for {target}: {exc}")
            return None

    def fetch_short_cadence(self, kic_id: int, quarter: int | None = None) -> dict | None:
        """Convenience wrapper for short-cadence (asteroseismology) data."""
        return self.fetch_light_curve(kic_id=kic_id, quarter=quarter, cadence="short")

    def bulk_analyze(
        self,
        targets: list[dict],
        federation=None,
    ) -> list[dict]:
        """Batch process multiple Kepler targets."""
        results = []
        for target in targets:
            lc = self.fetch_light_curve(**target)
            if lc is None:
                continue
            row = {"target": target, "light_curve": lc}
            if federation is not None:
                row["classifications"] = federation.run_all(lc)
            results.append(row)
        return results
