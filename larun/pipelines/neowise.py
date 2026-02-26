"""
NEOWISE Pipeline — Access NEOWISE infrared photometry via IRSA API.

Data source: NASA/IPAC Infrared Science Archive (IRSA)
API: https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query

NEOWISE data:
    W1 band: 3.4 μm  — traces stellar photosphere, dust emission
    W2 band: 4.6 μm  — similar but more sensitive to warmer dust

Key use case: VARnet-inspired variable source detection in infrared.
NEOWISE has ~10 years of baseline (2013–2024), ~13 epochs/year.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urlencode

import numpy as np

from larun.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)

_IRSA_BASE = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
_NEOWISE_CATALOG = "neowiser_p1bs_psd"  # Single-exposure source table


class NEOWISEPipeline(BasePipeline):
    """
    Access NEOWISE single-exposure photometry for a sky position.

    No bulk data download needed — queries per-object via IRSA Gator API.
    Returns time-series light curves in W1 and W2 bands.
    """

    SOURCE = "neowise"
    REQUEST_TIMEOUT = 30  # seconds

    def fetch_light_curve(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 10.0,
        band: str = "w1",
        min_qual_frame: int = 0,
    ) -> dict | None:
        """
        Fetch NEOWISE light curve for a sky position.

        Args:
            ra/dec:         Sky position (degrees, J2000)
            radius_arcsec:  Search cone radius (default 10 arcsec)
            band:           'w1', 'w2', or 'both'
            min_qual_frame: Minimum quality frame flag (0 = accept all)

        Returns:
            dict with keys: times, flux/mags, flux_err/mags_err, meta
            (uses magnitudes, not flux units, for NEOWISE)
        """
        import requests

        params = {
            "catalog": _NEOWISE_CATALOG,
            "spatial": "Cone",
            "objstr": f"{ra} {dec}",
            "radius": str(radius_arcsec),
            "outfmt": "1",  # IPAC table format
            "selcols": "mjd,w1mpro,w1sigmpro,w2mpro,w2sigmpro,ra,dec,qual_frame,saa_sep,moon_masked",
            "constraints": f"qual_frame>{min_qual_frame}",
            "nrec": "5000",
        }

        url = f"{_IRSA_BASE}?{urlencode(params)}"
        logger.debug(f"NEOWISE query: ra={ra:.4f}, dec={dec:.4f}, r={radius_arcsec}\"")

        try:
            resp = requests.get(url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(f"NEOWISE API error: {exc}")
            return None

        rows = self._parse_ipac_table(resp.text)
        if not rows:
            logger.info(f"No NEOWISE data at ra={ra:.4f}, dec={dec:.4f}")
            return None

        # Build arrays
        try:
            mjd = np.array([float(r["mjd"]) for r in rows if r.get("mjd")])
            w1 = np.array([float(r["w1mpro"]) if r.get("w1mpro") and r["w1mpro"] != "null" else np.nan
                           for r in rows])
            w1_err = np.array([float(r["w1sigmpro"]) if r.get("w1sigmpro") and r["w1sigmpro"] != "null" else np.nan
                               for r in rows])
            w2 = np.array([float(r["w2mpro"]) if r.get("w2mpro") and r["w2mpro"] != "null" else np.nan
                           for r in rows])
            w2_err = np.array([float(r["w2sigmpro"]) if r.get("w2sigmpro") and r["w2sigmpro"] != "null" else np.nan
                               for r in rows])
        except Exception as exc:
            logger.error(f"NEOWISE data parsing error: {exc}")
            return None

        sort_idx = np.argsort(mjd)
        mjd, w1, w1_err, w2, w2_err = mjd[sort_idx], w1[sort_idx], w1_err[sort_idx], w2[sort_idx], w2_err[sort_idx]

        # Select primary band
        if band == "w1":
            primary_mag, primary_err = w1, w1_err
        elif band == "w2":
            primary_mag, primary_err = w2, w2_err
        else:
            primary_mag, primary_err = w1, w1_err  # default to W1

        # Filter NaNs
        mask = np.isfinite(mjd) & np.isfinite(primary_mag)
        mjd, primary_mag, primary_err = mjd[mask], primary_mag[mask], primary_err[mask]

        if len(mjd) < 3:
            logger.info(f"Insufficient NEOWISE data at ra={ra:.4f}, dec={dec:.4f} ({len(mjd)} points)")
            return None

        return {
            "times": mjd,
            "mags": primary_mag,
            "mags_err": primary_err,
            "flux": primary_mag,          # alias for pipeline compatibility
            "flux_err": primary_err,
            "w1_mag": w1[mask] if band != "w2" else w1[mask],
            "w1_err": w1_err[mask],
            "w2_mag": w2[mask],
            "w2_err": w2_err[mask],
            "meta": {
                "source": "neowise",
                "ra": ra,
                "dec": dec,
                "radius_arcsec": radius_arcsec,
                "band": band,
                "n_points": int(len(mjd)),
                "time_span_days": float(mjd[-1] - mjd[0]) if len(mjd) > 1 else 0.0,
                "mjd_start": float(mjd[0]),
                "mjd_end": float(mjd[-1]),
            },
        }

    def run_federation(self, ra: float, dec: float, federation=None) -> dict:
        """
        Fetch NEOWISE data and run all TinyML models.

        Args:
            ra/dec: sky position (degrees)
            federation: ModelFederation instance (optional)

        Returns:
            dict with 'light_curve' and 'classifications'
        """
        lc = self.fetch_light_curve(ra, dec)
        if lc is None:
            return {"error": f"No NEOWISE data at ra={ra}, dec={dec}"}

        result = {"light_curve": lc}
        if federation is not None:
            result["classifications"] = federation.run_all(lc)

        return result

    # -------------------------------------------------------------------------
    # IPAC Table Parser
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_ipac_table(text: str) -> list[dict]:
        """
        Parse IPAC ASCII table format.
        Lines starting with '|' are header lines; data follows.
        """
        lines = text.strip().split("\n")
        rows = []
        col_names = []
        data_started = False

        for line in lines:
            line = line.rstrip()
            if not line or line.startswith("\\"):
                continue  # comment or empty
            if line.startswith("|") and not data_started:
                # First | line is column names
                if not col_names:
                    col_names = [c.strip() for c in line.strip("|").split("|")]
                continue
            if not line.startswith("|") and col_names and not data_started:
                data_started = True
            if data_started and not line.startswith("|") and not line.startswith("\\"):
                # Data row — fixed-width, aligned with header
                parts = line.split()
                if len(parts) >= len(col_names):
                    row = {col_names[i]: parts[i] for i in range(len(col_names))}
                    rows.append(row)
                elif parts:
                    # Try positional split matching header
                    row = {}
                    for i, col in enumerate(col_names):
                        row[col] = parts[i] if i < len(parts) else "null"
                    rows.append(row)

        return rows

    def query_cone(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
        max_sources: int = 100,
    ) -> list[dict]:
        """
        Find all NEOWISE sources in a cone (for region-based discovery).

        Returns:
            list of {'ra', 'dec', 'n_detections'} dicts
        """
        import requests

        params = {
            "catalog": _NEOWISE_CATALOG,
            "spatial": "Cone",
            "objstr": f"{ra} {dec}",
            "radius": str(radius_arcmin * 60),  # convert to arcsec
            "outfmt": "1",
            "selcols": "ra,dec,cntr",
            "nrec": str(max_sources),
        }
        url = f"{_IRSA_BASE}?{urlencode(params)}"

        try:
            resp = requests.get(url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            rows = self._parse_ipac_table(resp.text)
            # Deduplicate by unique source (cntr)
            seen = set()
            unique = []
            for r in rows:
                cntr = r.get("cntr", "")
                if cntr not in seen:
                    seen.add(cntr)
                    unique.append({
                        "ra": float(r.get("ra", 0)),
                        "dec": float(r.get("dec", 0)),
                    })
            return unique
        except Exception as exc:
            logger.error(f"NEOWISE cone query error: {exc}")
            return []
