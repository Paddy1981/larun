"""
VarWISE Pipeline — Access the Paz (2025) catalog of 1.5M variable sources.

Reference:
    Paz, M. (2024). "A Sub-Millisecond Fourier and Wavelet Based Model to
    Extract Variable Candidates from the NEOWISE Single-Exposure Database."
    arXiv:2409.15499

The VarWISE catalog (~1.5M objects) is expected in FITS or CSV format.
When released, this pipeline provides:
- Spatial search by position
- Classification-based search
- Integration with NEOWISE pipeline for deeper analysis
- Comparison between VARnet (4-class) and LARUN (12-model) classifications

Unique value: VARnet used 4 classes; LARUN adds 8 more specialized models
              for deeper characterization of the same 1.5M objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CATALOG_PATH = Path(__file__).parent.parent.parent / "data" / "varwise_catalog.fits"

_VARWISE_CLASSES = {
    0: "NON_VARIABLE",
    1: "TRANSIENT",
    2: "PULSATOR",
    3: "ECLIPSING",
}


class VarWISEBrowser:
    """
    Interactive browser for the VarWISE catalog.

    Loads catalog into memory with spatial indexing (KD-tree).
    Provides search by position, classification, and properties.

    Singleton pattern — use VarWISEBrowser.instance() after loading.
    """

    _instance: ClassVar["VarWISEBrowser | None"] = None

    def __init__(self, catalog_path: str | Path | None = None):
        self._catalog_path = Path(catalog_path or _DEFAULT_CATALOG_PATH)
        self._data = None
        self._kdtree = None
        self._loaded = False

    @classmethod
    def instance(cls) -> "VarWISEBrowser | None":
        """Return the singleton instance if loaded, else None."""
        return cls._instance

    @classmethod
    def load_catalog(cls, catalog_path: str | Path | None = None) -> "VarWISEBrowser":
        """
        Load VarWISE catalog and register as singleton.

        Args:
            catalog_path: path to FITS or CSV catalog file

        Returns:
            VarWISEBrowser instance with catalog loaded
        """
        browser = cls(catalog_path)
        browser._load()
        cls._instance = browser
        return browser

    def _load(self) -> None:
        """Load catalog from FITS or CSV."""
        if not self._catalog_path.exists():
            logger.warning(
                f"VarWISE catalog not found at {self._catalog_path}. "
                "Download from https://irsa.ipac.caltech.edu when released."
            )
            self._loaded = False
            return

        try:
            suffix = self._catalog_path.suffix.lower()
            if suffix in (".fits", ".fit"):
                from astropy.table import Table
                table = Table.read(self._catalog_path)
                self._data = {
                    "ra": np.array(table["ra"]),
                    "dec": np.array(table["dec"]),
                    "class": np.array(table["class"]) if "class" in table.colnames else None,
                    "amplitude": np.array(table["amplitude"]) if "amplitude" in table.colnames else None,
                    "period": np.array(table["period"]) if "period" in table.colnames else None,
                    "id": np.array(table["id"]) if "id" in table.colnames else np.arange(len(table)),
                }
            elif suffix in (".csv", ".txt"):
                import pandas as pd
                df = pd.read_csv(self._catalog_path)
                self._data = {
                    "ra": df["ra"].values,
                    "dec": df["dec"].values,
                    "class": df["class"].values if "class" in df.columns else None,
                    "amplitude": df["amplitude"].values if "amplitude" in df.columns else None,
                    "period": df["period"].values if "period" in df.columns else None,
                    "id": df.index.values,
                }
            else:
                logger.error(f"Unsupported catalog format: {suffix}")
                return

            self._build_spatial_index()
            self._loaded = True
            n = len(self._data["ra"])
            logger.info(f"VarWISE catalog loaded: {n:,} objects from {self._catalog_path}")

        except Exception as exc:
            logger.error(f"Failed to load VarWISE catalog: {exc}")
            self._loaded = False

    def _build_spatial_index(self) -> None:
        """Build KD-tree for fast spatial queries."""
        from scipy.spatial import KDTree

        # Convert RA/Dec to 3D Cartesian for KD-tree
        ra_rad = np.deg2rad(self._data["ra"])
        dec_rad = np.deg2rad(self._data["dec"])
        xyz = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])
        self._kdtree = KDTree(xyz)
        logger.debug("VarWISE spatial index built.")

    # -------------------------------------------------------------------------
    # Search Methods
    # -------------------------------------------------------------------------

    def search_by_position(
        self,
        ra: float,
        dec: float,
        radius_arcmin: float = 5.0,
    ) -> list[dict]:
        """
        Find VarWISE objects near a position.

        Args:
            ra/dec:        Center position (degrees)
            radius_arcmin: Search radius (arcminutes)

        Returns:
            list of object dicts with id, ra, dec, class, amplitude, period
        """
        if not self._loaded or self._kdtree is None:
            return []

        # Convert search radius to chord distance for KD-tree
        r_rad = np.deg2rad(radius_arcmin / 60.0)
        chord = 2 * np.sin(r_rad / 2)

        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        xyz = np.array([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])

        idx = self._kdtree.query_ball_point(xyz, chord)
        return [self._row_to_dict(i) for i in idx]

    def search_by_class(
        self,
        var_class: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Find objects by VARnet classification.

        Args:
            var_class: 'TRANSIENT', 'PULSATOR', 'ECLIPSING', 'NON_VARIABLE'
            limit:     max results to return
            offset:    skip first N results (for pagination)
        """
        if not self._loaded or self._data["class"] is None:
            return []

        class_id = next((k for k, v in _VARWISE_CLASSES.items() if v == var_class.upper()), None)
        if class_id is None:
            logger.warning(f"Unknown VarWISE class: {var_class}")
            return []

        idx = np.where(self._data["class"] == class_id)[0]
        page = idx[offset : offset + limit]
        return [self._row_to_dict(i) for i in page]

    def search_by_properties(
        self,
        min_amplitude: float | None = None,
        max_period: float | None = None,
        min_period: float | None = None,
        var_class: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Advanced filter search."""
        if not self._loaded:
            return []

        n = len(self._data["ra"])
        mask = np.ones(n, dtype=bool)

        if min_amplitude is not None and self._data["amplitude"] is not None:
            mask &= self._data["amplitude"] >= min_amplitude
        if max_period is not None and self._data["period"] is not None:
            mask &= self._data["period"] <= max_period
        if min_period is not None and self._data["period"] is not None:
            mask &= self._data["period"] >= min_period
        if var_class is not None and self._data["class"] is not None:
            class_id = next((k for k, v in _VARWISE_CLASSES.items() if v == var_class.upper()), -1)
            mask &= self._data["class"] == class_id

        idx = np.where(mask)[0][:limit]
        return [self._row_to_dict(i) for i in idx]

    def get_light_curve(self, varwise_id: int) -> dict | None:
        """
        Fetch the original NEOWISE light curve for a VarWISE object.
        Uses NEOWISEPipeline to query IRSA.
        """
        if not self._loaded:
            return None

        obj = self._row_to_dict(varwise_id)
        from larun.pipelines.neowise import NEOWISEPipeline
        return NEOWISEPipeline().fetch_light_curve(obj["ra"], obj["dec"])

    def run_larun_models(self, varwise_id: int) -> dict:
        """
        Run LARUN model federation on a VarWISE object.

        Value: VARnet gave 4-class labels; LARUN adds deeper characterization
        with 8+ specialized models (asteroseismology, flare, microlensing, etc.)
        """
        lc = self.get_light_curve(varwise_id)
        if lc is None:
            return {"error": f"No NEOWISE data for VarWISE ID {varwise_id}"}

        from larun.models.federation import ModelFederation
        federation = ModelFederation()
        return federation.run_all(lc)

    def stats(self) -> dict:
        """Return catalog statistics."""
        if not self._loaded:
            return {"loaded": False}
        n = len(self._data["ra"])
        stats = {"loaded": True, "n_objects": n}
        if self._data["class"] is not None:
            from collections import Counter
            counts = Counter(self._data["class"])
            stats["class_counts"] = {_VARWISE_CLASSES.get(k, str(k)): int(v) for k, v in counts.items()}
        return stats

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _row_to_dict(self, idx: int) -> dict:
        row = {
            "id": int(self._data["id"][idx]),
            "ra": float(self._data["ra"][idx]),
            "dec": float(self._data["dec"][idx]),
        }
        if self._data["class"] is not None:
            row["varnet_class"] = _VARWISE_CLASSES.get(int(self._data["class"][idx]), "?")
        if self._data["amplitude"] is not None:
            row["amplitude"] = float(self._data["amplitude"][idx])
        if self._data["period"] is not None:
            row["period"] = float(self._data["period"][idx])
        return row

    def __len__(self) -> int:
        return len(self._data["ra"]) if self._loaded and self._data else 0

    def __repr__(self) -> str:
        status = f"{len(self):,} objects" if self._loaded else "not loaded"
        return f"VarWISEBrowser({status})"
