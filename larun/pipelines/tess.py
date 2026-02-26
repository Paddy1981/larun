"""
TESS Pipeline — Access TESS light curves via MAST API and Lightkurve.

Data source: NASA/STScI MAST (Mikulski Archive for Space Telescopes)
API: https://mast.stsci.edu/api/v0/
Library: Lightkurve (https://lightkurve.heliospython.org/)
"""

from __future__ import annotations

import logging

import numpy as np

from larun.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class TESSPipeline(BasePipeline):
    """
    Fetch and process TESS light curves.

    Supports:
        - Single target by TIC ID or RA/Dec
        - Sector-specific or all-sector downloads
        - PDCSAP_FLUX (systematics-corrected) or SAP_FLUX
        - Bulk analysis of target lists
    """

    SOURCE = "tess"

    def __init__(self, flux_type: str = "PDCSAP_FLUX"):
        """
        Args:
            flux_type: 'PDCSAP_FLUX' (default, systematics-corrected)
                       or 'SAP_FLUX' (raw aperture photometry)
        """
        self.flux_type = flux_type

    def is_available(self) -> bool:
        try:
            import lightkurve  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch_light_curve(
        self,
        tic_id: int | None = None,
        ra: float | None = None,
        dec: float | None = None,
        sector: int | None = None,
        author: str = "SPOC",
    ) -> dict | None:
        """
        Fetch a TESS light curve.

        Args:
            tic_id:  TIC (TESS Input Catalog) ID
            ra/dec:  Sky coordinates (degrees) — used if tic_id not provided
            sector:  TESS sector number (None = download all available sectors)
            author:  Pipeline author ('SPOC', 'QLP', 'TGLC', etc.)

        Returns:
            dict with keys: times, flux, flux_err, crowdsap, flfrcsap, meta
            or None if no data found
        """
        try:
            import lightkurve as lk
        except ImportError:
            logger.error("lightkurve is not installed. pip install lightkurve")
            return None

        # Build search target
        if tic_id is not None:
            target = f"TIC {tic_id}"
        elif ra is not None and dec is not None:
            target = f"{ra} {dec}"
        else:
            raise ValueError("Provide either tic_id or ra+dec")

        try:
            search = lk.search_lightcurve(
                target,
                mission="TESS",
                sector=sector,
                author=author,
            )

            if len(search) == 0:
                logger.info(f"No TESS data found for {target}")
                return None

            logger.info(f"Found {len(search)} TESS observations for {target}")

            if sector is None:
                # Download and stitch all sectors
                lc_collection = search.download_all(flux_column=self.flux_type)
                lc = lc_collection.stitch()
            else:
                lc = search.download(flux_column=self.flux_type)

            lc = lc.remove_nans().remove_outliers(sigma=5)
            lc = lc.normalize()

            # Extract crowding info from header if available
            crowdsap = None
            flfrcsap = None
            if hasattr(lc, "meta"):
                crowdsap = lc.meta.get("CROWDSAP")
                flfrcsap = lc.meta.get("FLFRCSAP")

            times = np.asarray(lc.time.value, dtype=float)
            flux = np.asarray(lc.flux.value, dtype=float)
            flux_err = np.asarray(lc.flux_err.value, dtype=float) if lc.flux_err is not None else np.zeros_like(flux)

            times, flux, flux_err = self._clean(times, flux, flux_err)

            return {
                "times": times,
                "flux": flux,
                "flux_err": flux_err,
                "crowdsap": crowdsap,
                "flfrcsap": flfrcsap,
                "meta": {
                    "source": "tess",
                    "tic_id": tic_id,
                    "sector": sector,
                    "author": author,
                    "n_points": len(times),
                    "time_span_days": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
                    "target": target,
                },
            }

        except Exception as exc:
            logger.error(f"TESS pipeline error for {target}: {exc}")
            return None

    def fetch_by_coordinates(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 21.0,  # TESS pixel scale ~21 arcsec
        sector: int | None = None,
    ) -> list[dict]:
        """
        Fetch all TESS targets within a search radius.
        Returns list of light curve dicts.
        """
        try:
            import lightkurve as lk
        except ImportError:
            return []

        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        results = lk.search_lightcurve(
            coord,
            mission="TESS",
            sector=sector,
            radius=radius_arcsec * u.arcsec,
        )

        light_curves = []
        for row in results:
            lc = self.fetch_light_curve(
                tic_id=int(str(row.target_name).replace("TIC ", "")) if "TIC" in str(row.target_name) else None,
                ra=ra,
                dec=dec,
                sector=row.exptime,
            )
            if lc is not None:
                light_curves.append(lc)

        return light_curves

    def bulk_analyze(
        self,
        targets: list[dict],
        federation=None,
    ) -> list[dict]:
        """
        Fetch + analyze multiple targets through the model federation.

        Args:
            targets: list of dicts, each with 'tic_id' or 'ra'+'dec'
            federation: ModelFederation instance (optional)

        Returns:
            list of result dicts with light_curve + classifications
        """
        results = []
        for target in targets:
            lc = self.fetch_light_curve(**target)
            if lc is None:
                continue

            row = {"target": target, "light_curve": lc}

            if federation is not None:
                classifications = federation.run_all(lc)
                row["classifications"] = classifications

            results.append(row)
            logger.info(f"Analyzed target {target}: {len(lc['times'])} points")

        return results
