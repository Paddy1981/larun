"""
LARUN Integrations Package
==========================
Data integrations with external astronomical services.

Available integrations:
- Gaia DR3: Stellar parameters
"""

from .gaia import GaiaClient, StellarParams, GaiaResult

__all__ = ['GaiaClient', 'StellarParams', 'GaiaResult']
