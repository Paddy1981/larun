"""
Gaia DR3 Integration for LARUN
==============================

Provides stellar parameters from Gaia Data Release 3 for
target characterization in exoplanet analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import warnings


@dataclass
class StellarParams:
    """
    Stellar parameters from Gaia DR3.

    Contains astrophysical parameters derived from Gaia
    photometry, astrometry, and spectroscopy.
    """
    tic_id: Optional[str] = None
    gaia_id: Optional[str] = None
    teff: float = 5778.0  # Effective temperature (K)
    logg: float = 4.44    # Surface gravity (log cgs)
    radius: float = 1.0   # Stellar radius (R_sun)
    mass: float = 1.0     # Stellar mass (M_sun)
    luminosity: float = 1.0  # Luminosity (L_sun)
    metallicity: float = 0.0  # [Fe/H]
    age_gyr: float = 4.6  # Age in Gyr
    distance_pc: float = 10.0  # Distance in parsecs
    parallax_mas: float = 100.0  # Parallax in milliarcseconds
    pmra: float = 0.0  # Proper motion in RA (mas/yr)
    pmdec: float = 0.0  # Proper motion in Dec (mas/yr)
    radial_velocity: Optional[float] = None  # km/s
    bp_rp: float = 0.0  # BP-RP color
    g_mag: float = 10.0  # G-band magnitude
    is_valid: bool = True
    quality_flags: List[str] = field(default_factory=list)
    source: str = "gaia_dr3"

    def spectral_type(self) -> str:
        """Estimate spectral type from effective temperature."""
        if self.teff >= 30000:
            return "O"
        elif self.teff >= 10000:
            return "B"
        elif self.teff >= 7500:
            return "A"
        elif self.teff >= 6000:
            return "F"
        elif self.teff >= 5200:
            return "G"
        elif self.teff >= 3700:
            return "K"
        elif self.teff >= 2400:
            return "M"
        else:
            return "L"

    def luminosity_class(self) -> str:
        """Estimate luminosity class from surface gravity."""
        if self.logg >= 4.0:
            return "V"  # Dwarf (main sequence)
        elif self.logg >= 3.0:
            return "IV"  # Subgiant
        elif self.logg >= 1.5:
            return "III"  # Giant
        elif self.logg >= 0.5:
            return "II"  # Bright giant
        else:
            return "I"  # Supergiant

    def full_spectral_type(self) -> str:
        """Return full spectral classification."""
        return f"{self.spectral_type()}{self.luminosity_class()}"

    def is_dwarf(self) -> bool:
        """Check if star is a main-sequence dwarf."""
        return self.logg >= 4.0

    def is_giant(self) -> bool:
        """Check if star is a giant."""
        return self.logg < 3.5

    def habitable_zone(self) -> tuple:
        """
        Calculate conservative habitable zone boundaries.

        Returns:
            Tuple of (inner_au, outer_au) for habitable zone
        """
        # Simple luminosity-based calculation
        inner = 0.95 * (self.luminosity ** 0.5)
        outer = 1.67 * (self.luminosity ** 0.5)
        return (inner, outer)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'tic_id': self.tic_id,
            'gaia_id': self.gaia_id,
            'teff_k': self.teff,
            'logg': self.logg,
            'radius_rsun': self.radius,
            'mass_msun': self.mass,
            'luminosity_lsun': self.luminosity,
            'metallicity': self.metallicity,
            'age_gyr': self.age_gyr,
            'distance_pc': self.distance_pc,
            'parallax_mas': self.parallax_mas,
            'pmra_mas_yr': self.pmra,
            'pmdec_mas_yr': self.pmdec,
            'radial_velocity_km_s': self.radial_velocity,
            'bp_rp': self.bp_rp,
            'g_mag': self.g_mag,
            'spectral_type': self.full_spectral_type(),
            'is_dwarf': self.is_dwarf(),
            'is_valid': self.is_valid,
            'quality_flags': self.quality_flags,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StellarParams':
        """Create from dictionary."""
        return cls(
            tic_id=d.get('tic_id'),
            gaia_id=d.get('gaia_id'),
            teff=d.get('teff_k', d.get('teff', 5778.0)),
            logg=d.get('logg', 4.44),
            radius=d.get('radius_rsun', d.get('radius', 1.0)),
            mass=d.get('mass_msun', d.get('mass', 1.0)),
            luminosity=d.get('luminosity_lsun', d.get('luminosity', 1.0)),
            metallicity=d.get('metallicity', 0.0),
            age_gyr=d.get('age_gyr', 4.6),
            distance_pc=d.get('distance_pc', 10.0),
            parallax_mas=d.get('parallax_mas', 100.0),
            pmra=d.get('pmra_mas_yr', d.get('pmra', 0.0)),
            pmdec=d.get('pmdec_mas_yr', d.get('pmdec', 0.0)),
            radial_velocity=d.get('radial_velocity_km_s', d.get('radial_velocity')),
            bp_rp=d.get('bp_rp', 0.0),
            g_mag=d.get('g_mag', 10.0),
            is_valid=d.get('is_valid', True),
            quality_flags=d.get('quality_flags', []),
            source=d.get('source', 'unknown'),
        )


class GaiaClient:
    """
    Client for querying Gaia DR3 stellar parameters.

    Uses the Gaia TAP service to retrieve astrophysical
    parameters for stars identified by TIC ID or coordinates.
    """

    GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize Gaia client.

        Args:
            cache_enabled: Whether to cache query results
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, StellarParams] = {}

    def query_by_tic(self, tic_id: str) -> StellarParams:
        """
        Query stellar parameters by TIC ID.

        Args:
            tic_id: TESS Input Catalog identifier

        Returns:
            StellarParams for the star
        """
        cache_key = f"tic_{tic_id}"

        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            params = self._query_gaia_for_tic(tic_id)
        except Exception as e:
            warnings.warn(f"Gaia query failed for TIC {tic_id}: {e}")
            params = self._fallback_params(tic_id)

        if self.cache_enabled:
            self._cache[cache_key] = params

        return params

    def query_by_coords(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 2.0,
    ) -> StellarParams:
        """
        Query stellar parameters by coordinates.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius_arcsec: Search radius in arcseconds

        Returns:
            StellarParams for the closest match
        """
        cache_key = f"coords_{ra:.6f}_{dec:.6f}"

        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            params = self._query_gaia_for_coords(ra, dec, radius_arcsec)
        except Exception as e:
            warnings.warn(f"Gaia query failed for coords ({ra}, {dec}): {e}")
            params = self._fallback_params(f"coords_{ra}_{dec}")

        if self.cache_enabled:
            self._cache[cache_key] = params

        return params

    def _query_gaia_for_tic(self, tic_id: str) -> StellarParams:
        """
        Query Gaia using TIC crossmatch.

        This is a stub - in production, use astroquery.gaia
        """
        # In production, use:
        # from astroquery.gaia import Gaia
        # result = Gaia.launch_job_async(query).get_results()

        # For now, return fallback
        return self._fallback_params(tic_id)

    def _query_gaia_for_coords(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
    ) -> StellarParams:
        """
        Query Gaia by coordinates.

        This is a stub - in production, use astroquery.gaia
        """
        return self._fallback_params(f"coords_{ra}_{dec}")

    def _fallback_params(self, target_id: str) -> StellarParams:
        """
        Return solar-like fallback parameters.

        Used when Gaia query fails or returns no results.
        """
        return StellarParams(
            tic_id=target_id if target_id.startswith("TIC") else None,
            teff=5778.0,  # Solar values
            logg=4.44,
            radius=1.0,
            mass=1.0,
            luminosity=1.0,
            metallicity=0.0,
            is_valid=False,
            quality_flags=['fallback', 'no_gaia_match'],
            source='fallback',
        )

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'entries': len(self._cache),
            'valid': sum(1 for p in self._cache.values() if p.is_valid),
            'fallback': sum(1 for p in self._cache.values() if not p.is_valid),
        }
