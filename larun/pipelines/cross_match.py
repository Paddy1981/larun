"""
Cross-Match Pipeline — Determine if a detected variable is known or new.

Cross-matches against 6+ major astronomical catalogs:
    VSX      — AAVSO Variable Star Index (~2.2M known variables)
    Gaia DR3 — Gaia Data Release 3 (~10M variable candidates)
    ASAS-SN  — All-Sky Automated Survey for Supernovae (~700K)
    ZTF      — Zwicky Transient Facility (~1M variables)
    SIMBAD   — Simbad Astronomical Database (general reference)
    VarWISE  — Paz (2025) catalog of 1.5M new variables (when released)

Critical for the Citizen Discovery Engine: determines if something is "new".
"""

from __future__ import annotations

import logging

import numpy as np

from larun.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)

CATALOGS = {
    "vsx": "AAVSO Variable Star Index (~2.2M known variables)",
    "gaia_dr3": "Gaia DR3 variability catalog (~10M candidates)",
    "asassn": "ASAS-SN Variable Stars Database (~700K)",
    "ztf": "ZTF Variable Star Catalog (~1M)",
    "simbad": "SIMBAD Astronomical Database",
    "varwise": "VarWISE Catalog (Paz 2025, 1.5M new variables)",
}

# VizieR catalog IDs
_VIZIER_CATALOGS = {
    "vsx": "B/vsx/vsx",
    "gaia_dr3": "I/358/vclasre",  # Gaia DR3 variability classification
    "asassn": "J/MNRAS/486/1907/catalog",
    "ztf": "J/ApJS/249/18/table2",
}


class CrossMatchPipeline(BasePipeline):
    """
    Cross-match coordinates against multiple astronomical catalogs.

    Used to determine:
        - Is this object already known?
        - What is the novelty score (0=well-known, 1=completely unknown)?
        - Has it been classified consistently?
    """

    SOURCE = "cross_match"

    def fetch_light_curve(self, *args, **kwargs) -> dict | None:
        """Not applicable for cross-matching."""
        return None

    def cross_match(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 3.0,
        catalogs: list[str] | str = "all",
    ) -> dict:
        """
        Check if an object at (ra, dec) is already known.

        Args:
            ra/dec:        Sky position (degrees, J2000)
            radius_arcsec: Match radius (default 3 arcsec)
            catalogs:      List of catalog keys or 'all'

        Returns:
            {
                'known': bool,
                'matches': list of {catalog, name, type, ...},
                'novelty_score': float (0=known, 1=unknown),
                'checked_catalogs': list of catalog names,
                'errors': list of catalog-specific errors,
            }
        """
        if catalogs == "all":
            target_catalogs = list(CATALOGS.keys())
        else:
            target_catalogs = [c for c in catalogs if c in CATALOGS]

        matches = []
        errors = []
        checked = []

        for cat_id in target_catalogs:
            try:
                cat_matches = self._query_catalog(ra, dec, radius_arcsec, cat_id)
                matches.extend(cat_matches)
                checked.append(cat_id)
            except Exception as exc:
                logger.warning(f"Cross-match failed for {cat_id}: {exc}")
                errors.append({"catalog": cat_id, "error": str(exc)})

        n_checked = max(len(checked), 1)
        n_matched = len(set(m["catalog"] for m in matches))
        novelty_score = float(1.0 - n_matched / n_checked)

        return {
            "known": len(matches) > 0,
            "matches": matches,
            "novelty_score": round(novelty_score, 3),
            "checked_catalogs": checked,
            "n_matches": len(matches),
            "errors": errors,
        }

    def novelty_score(self, ra: float, dec: float, radius_arcsec: float = 3.0) -> float:
        """Convenience method — return just the novelty score."""
        result = self.cross_match(ra, dec, radius_arcsec)
        return result["novelty_score"]

    # -------------------------------------------------------------------------
    # Per-catalog queries
    # -------------------------------------------------------------------------

    def _query_catalog(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        catalog: str,
    ) -> list[dict]:
        """Dispatch to per-catalog query method."""
        if catalog == "simbad":
            return self._query_simbad(ra, dec, radius_arcsec)
        elif catalog in _VIZIER_CATALOGS:
            return self._query_vizier(ra, dec, radius_arcsec, catalog)
        elif catalog == "varwise":
            return self._query_varwise(ra, dec, radius_arcsec)
        else:
            logger.debug(f"No query implementation for catalog: {catalog}")
            return []

    def _query_simbad(self, ra: float, dec: float, radius_arcsec: float) -> list[dict]:
        """Query SIMBAD for known objects."""
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        result = Simbad.query_region(coord, radius=radius_arcsec * u.arcsec)

        if result is None or len(result) == 0:
            return []

        return [
            {
                "catalog": "SIMBAD",
                "name": str(row["MAIN_ID"]),
                "type": str(row.get("OTYPES", "?")),
                "ra": float(row["RA_d"]) if "RA_d" in result.colnames else ra,
                "dec": float(row["DEC_d"]) if "DEC_d" in result.colnames else dec,
            }
            for row in result
        ]

    def _query_vizier(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        catalog: str,
    ) -> list[dict]:
        """Query a VizieR catalog."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        vizier_id = _VIZIER_CATALOGS[catalog]

        v = Vizier(row_limit=10)
        result = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=vizier_id)

        if not result or len(result) == 0:
            return []

        matches = []
        for table in result:
            for row in table:
                entry = {
                    "catalog": catalog.upper(),
                    "vizier_catalog": vizier_id,
                }
                # Extract name/type if available
                for col in ["Name", "ID", "Star", "ASASSN-V", "ObjID"]:
                    if col in table.colnames:
                        entry["name"] = str(row[col])
                        break
                for col in ["Type", "VarType", "Class", "type"]:
                    if col in table.colnames:
                        entry["type"] = str(row[col])
                        break
                matches.append(entry)

        return matches

    def _query_varwise(self, ra: float, dec: float, radius_arcsec: float) -> list[dict]:
        """
        Query local VarWISE catalog if loaded.
        Falls back to empty list if catalog not available yet.
        """
        from larun.catalogs.varwise import VarWISEBrowser
        try:
            browser = VarWISEBrowser.instance()
            if browser is None:
                return []
            results = browser.search_by_position(ra, dec, radius_arcmin=radius_arcsec / 60.0)
            return [{"catalog": "VarWISE", "name": r.get("id", ""), "type": r.get("type", "")} for r in results]
        except Exception:
            return []

    def batch_cross_match(
        self,
        positions: list[dict],
        radius_arcsec: float = 3.0,
    ) -> list[dict]:
        """
        Cross-match a list of positions.

        Args:
            positions: list of {'ra': float, 'dec': float} dicts
            radius_arcsec: match radius

        Returns:
            list of cross_match result dicts
        """
        results = []
        for pos in positions:
            result = self.cross_match(pos["ra"], pos["dec"], radius_arcsec)
            result["query_ra"] = pos["ra"]
            result["query_dec"] = pos["dec"]
            results.append(result)
        return results
