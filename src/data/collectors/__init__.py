"""
LARUN Data Collectors
=====================
Automated data collection from astronomical archives and surveys.

Data Sources:
- MAST/TESS: Exoplanet light curves
- Gaia: Stellar parameters and photometry
- ZTF/ATLAS: Transient events
- OGLE: Variable stars and microlensing
- Galaxy Zoo: Galaxy morphology labels
"""

from src.data.collectors.base import BaseDataCollector, DatasetInfo
from src.data.collectors.mast_collector import MASTCollector
from src.data.collectors.gaia_collector import GaiaCollector
from src.data.collectors.transient_collector import TransientCollector
from src.data.collectors.variable_star_collector import VariableStarCollector
from src.data.collectors.galaxy_collector import GalaxyCollector

__all__ = [
    'BaseDataCollector',
    'DatasetInfo',
    'MASTCollector',
    'GaiaCollector',
    'TransientCollector',
    'VariableStarCollector',
    'GalaxyCollector',
]
