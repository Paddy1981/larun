"""
LARUN Integrations
==================

Integration modules for external astronomical data sources.

Modules:
- gaia: Gaia DR3 stellar parameter queries
- mast: NASA MAST archive access
"""

from .gaia import GaiaClient, StellarParams

__all__ = ['GaiaClient', 'StellarParams']
