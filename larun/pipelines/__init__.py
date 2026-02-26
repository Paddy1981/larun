"""
LARUN Data Pipelines — Access to NASA Space Archives

Pipelines:
    TESS      — Transiting Exoplanet Survey Satellite (MAST API + Lightkurve)
    Kepler    — Kepler mission legacy data (MAST API + Lightkurve)
    NEOWISE   — Near-Earth Object Wide-field Infrared Survey Explorer (IRSA API)
    CrossMatch — Cross-match against 6+ catalogs (VSX, SIMBAD, Gaia DR3, etc.)
    VarWISE   — Paz (2024) catalog of 1.5M variable sources
"""

from larun.pipelines.tess import TESSPipeline
from larun.pipelines.kepler import KeplerPipeline
from larun.pipelines.neowise import NEOWISEPipeline
from larun.pipelines.cross_match import CrossMatchPipeline

__all__ = [
    "TESSPipeline",
    "KeplerPipeline",
    "NEOWISEPipeline",
    "CrossMatchPipeline",
]
