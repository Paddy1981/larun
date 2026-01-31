"""
LARUN Integration: Gaia DR3
============================
Stellar parameter retrieval from ESA Gaia Data Release 3.

Features:
- Query by TIC ID (cross-match with TESS Input Catalog)
- Query by coordinates (RA, Dec)
- Retrieve: Teff, logg, radius, luminosity, distance, proper motion
- Batch cross-matching

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: https://gea.esac.esa.int/archive/
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StellarParams:
    """
    Stellar parameters from Gaia DR3.
    
    All values in standard units with uncertainties.
    """
    # Identification
    source_id: str = ""
    tic_id: str = ""
    
    # Astrometry
    ra: float = 0.0           # Right ascension (degrees)
    dec: float = 0.0          # Declination (degrees)
    parallax: float = 0.0     # Parallax (mas)
    parallax_err: float = 0.0
    distance_pc: float = 0.0  # Distance (parsecs)
    distance_err: float = 0.0
    pmra: float = 0.0         # Proper motion RA (mas/yr)
    pmdec: float = 0.0        # Proper motion Dec (mas/yr)
    
    # Photometry
    g_mag: float = 0.0        # Gaia G magnitude
    bp_mag: float = 0.0       # Gaia BP magnitude
    rp_mag: float = 0.0       # Gaia RP magnitude
    bp_rp: float = 0.0        # BP-RP color
    
    # Stellar parameters (from GSPPHOT or GSPSPEC)
    teff: float = 5778.0      # Effective temperature (K)
    teff_err: float = 0.0
    logg: float = 4.44        # Surface gravity (dex)
    logg_err: float = 0.0
    metallicity: float = 0.0  # [M/H] or [Fe/H]
    metallicity_err: float = 0.0
    
    # Derived parameters
    radius: float = 1.0       # Stellar radius (R_sun)
    radius_err: float = 0.0
    luminosity: float = 1.0   # Stellar luminosity (L_sun)
    luminosity_err: float = 0.0
    mass: float = 1.0         # Stellar mass (M_sun) - estimated
    
    # Quality flags
    is_valid: bool = True
    quality_flags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_id': self.source_id,
            'tic_id': self.tic_id,
            'coordinates': {'ra': self.ra, 'dec': self.dec},
            'parallax_mas': round(self.parallax, 4),
            'distance_pc': round(self.distance_pc, 2),
            'teff_k': round(self.teff, 0),
            'logg': round(self.logg, 3),
            'metallivity': round(self.metallicity, 3),
            'radius_rsun': round(self.radius, 3),
            'luminosity_lsun': round(self.luminosity, 3),
            'mass_msun': round(self.mass, 3),
            'g_mag': round(self.g_mag, 3),
            'bp_rp': round(self.bp_rp, 3),
            'is_valid': self.is_valid
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"  STELLAR PARAMETERS (Gaia DR3)",
            f"═══════════════════════════════════════════════════════════",
            f"  Source: {self.source_id}" if self.source_id else "",
            f"  TIC: {self.tic_id}" if self.tic_id else "",
            f"───────────────────────────────────────────────────────────",
            f"  Teff:        {self.teff:.0f} ± {self.teff_err:.0f} K",
            f"  log(g):      {self.logg:.3f} ± {self.logg_err:.3f}",
            f"  [M/H]:       {self.metallicity:+.3f} ± {self.metallicity_err:.3f}",
            f"  Radius:      {self.radius:.3f} ± {self.radius_err:.3f} R☉",
            f"  Luminosity:  {self.luminosity:.3f} ± {self.luminosity_err:.3f} L☉",
            f"  Mass:        {self.mass:.3f} M☉ (estimated)",
            f"───────────────────────────────────────────────────────────",
            f"  Distance:    {self.distance_pc:.1f} ± {self.distance_err:.1f} pc",
            f"  G mag:       {self.g_mag:.3f}",
            f"  BP-RP:       {self.bp_rp:.3f}",
            f"═══════════════════════════════════════════════════════════",
        ]
        return "\n".join([l for l in lines if l])
    
    def spectral_type(self) -> str:
        """Estimate spectral type from Teff."""
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
        else:
            return "M"
    
    def luminosity_class(self) -> str:
        """Estimate luminosity class from logg."""
        if self.logg < 2.0:
            return "I"   # Supergiant
        elif self.logg < 3.0:
            return "III" # Giant
        elif self.logg < 3.5:
            return "IV"  # Subgiant
        else:
            return "V"   # Dwarf


@dataclass
class GaiaResult:
    """Result of Gaia query."""
    success: bool
    params: Optional[StellarParams] = None
    error: Optional[str] = None
    query_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'params': self.params.to_dict() if self.params else None,
            'error': self.error,
            'query_type': self.query_type
        }


# ============================================================================
# Gaia Client
# ============================================================================

class GaiaClient:
    """
    Client for Gaia DR3 stellar parameter retrieval.
    
    Uses astroquery.gaia for TAP queries to ESA Gaia Archive.
    Falls back to estimated parameters if Gaia not available.
    
    Example:
        >>> client = GaiaClient()
        >>> result = client.query_by_tic("261136679")
        >>> if result.success:
        ...     print(result.params.summary())
    """
    
    # Gaia DR3 table names
    GAIA_SOURCE = "gaiadr3.gaia_source"
    GAIA_ASTROPHYSICAL = "gaiadr3.astrophysical_parameters"
    TIC_CROSSMATCH = "gaiadr3.tmass_psc_xsc_best_neighbour"  # Approximate
    
    def __init__(self, use_cache: bool = True, cache_dir: str = "data/gaia_cache"):
        """
        Initialize Gaia client.
        
        Args:
            use_cache: Whether to cache query results
            cache_dir: Directory for caching results
        """
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self._cache: Dict[str, StellarParams] = {}
        self._gaia_available = self._check_gaia()
    
    def _check_gaia(self) -> bool:
        """Check if astroquery.gaia is available."""
        try:
            from astroquery.gaia import Gaia
            return True
        except ImportError:
            logger.warning("astroquery.gaia not available. Using fallback methods.")
            return False
    
    def query_by_tic(self, tic_id: str) -> GaiaResult:
        """
        Query Gaia parameters by TESS Input Catalog ID.
        
        Args:
            tic_id: TIC ID (just the number, e.g., "261136679")
            
        Returns:
            GaiaResult with stellar parameters
        """
        # Clean TIC ID
        tic_id = str(tic_id).replace("TIC", "").strip()
        
        # Check cache
        cache_key = f"tic_{tic_id}"
        if cache_key in self._cache:
            return GaiaResult(
                success=True,
                params=self._cache[cache_key],
                query_type="tic_cached"
            )
        
        try:
            # First, get coordinates from TIC using lightkurve
            params = self._query_tic_catalog(tic_id)
            
            if params:
                self._cache[cache_key] = params
                return GaiaResult(
                    success=True,
                    params=params,
                    query_type="tic"
                )
            else:
                return GaiaResult(
                    success=False,
                    error=f"TIC {tic_id} not found in catalog",
                    query_type="tic"
                )
                
        except Exception as e:
            logger.error(f"TIC query failed: {e}")
            return GaiaResult(
                success=False,
                error=str(e),
                query_type="tic"
            )
    
    def _query_tic_catalog(self, tic_id: str) -> Optional[StellarParams]:
        """Query TIC catalog for basic stellar parameters."""
        try:
            from astroquery.mast import Catalogs
            
            # Query TIC
            tic_data = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
            
            if len(tic_data) == 0:
                return None
            
            row = tic_data[0]
            
            # Extract parameters
            params = StellarParams(
                tic_id=tic_id,
                ra=float(row.get('ra', 0) or 0),
                dec=float(row.get('dec', 0) or 0),
                teff=float(row.get('Teff', 5778) or 5778),
                teff_err=float(row.get('e_Teff', 100) or 100),
                logg=float(row.get('logg', 4.44) or 4.44),
                logg_err=float(row.get('e_logg', 0.1) or 0.1),
                radius=float(row.get('rad', 1.0) or 1.0),
                radius_err=float(row.get('e_rad', 0.1) or 0.1),
                mass=float(row.get('mass', 1.0) or 1.0),
                distance_pc=float(row.get('d', 100) or 100),
                g_mag=float(row.get('Gmag', 10) or 10),
                is_valid=True
            )
            
            # Calculate luminosity from radius and Teff
            teff_solar = 5778.0
            params.luminosity = (params.radius ** 2) * ((params.teff / teff_solar) ** 4)
            
            return params
            
        except ImportError:
            logger.warning("astroquery.mast not available")
            return self._fallback_params(tic_id)
        except Exception as e:
            logger.error(f"TIC catalog query failed: {e}")
            return self._fallback_params(tic_id)
    
    def query_by_coords(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 3.0
    ) -> GaiaResult:
        """
        Query Gaia by celestial coordinates.
        
        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcsec: Search radius in arcseconds
            
        Returns:
            GaiaResult with parameters of nearest source
        """
        cache_key = f"coords_{ra:.5f}_{dec:.5f}"
        if cache_key in self._cache:
            return GaiaResult(
                success=True,
                params=self._cache[cache_key],
                query_type="coords_cached"
            )
        
        if not self._gaia_available:
            # Return estimated parameters based on typical star
            params = self._fallback_params(f"coords_{ra}_{dec}")
            params.ra = ra
            params.dec = dec
            return GaiaResult(
                success=True,
                params=params,
                query_type="coords_fallback"
            )
        
        try:
            from astroquery.gaia import Gaia
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            
            # Query Gaia
            query = f"""
            SELECT TOP 1
                source_id, ra, dec, parallax, parallax_error,
                pmra, pmdec, phot_g_mean_mag, bp_rp,
                teff_gspphot, teff_gspphot_upper, teff_gspphot_lower,
                logg_gspphot, logg_gspphot_upper, logg_gspphot_lower,
                mh_gspphot, mh_gspphot_upper, mh_gspphot_lower,
                radius_gspphot, radius_gspphot_upper, radius_gspphot_lower,
                lum_gspphot, lum_gspphot_upper, lum_gspphot_lower
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})
            )
            ORDER BY angular_distance ASC
            """
            
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            if len(results) == 0:
                return GaiaResult(
                    success=False,
                    error="No Gaia source found at coordinates",
                    query_type="coords"
                )
            
            row = results[0]
            params = self._parse_gaia_row(row)
            self._cache[cache_key] = params
            
            return GaiaResult(
                success=True,
                params=params,
                query_type="coords"
            )
            
        except Exception as e:
            logger.error(f"Gaia coordinate query failed: {e}")
            return GaiaResult(
                success=False,
                error=str(e),
                query_type="coords"
            )
    
    def _parse_gaia_row(self, row) -> StellarParams:
        """Parse Gaia query result row into StellarParams."""
        # Helper for safe float conversion
        def safe_float(val, default=0.0):
            try:
                if val is None or (hasattr(val, 'mask') and val.mask):
                    return default
                return float(val)
            except:
                return default
        
        parallax = safe_float(row.get('parallax', 0))
        distance = 1000.0 / parallax if parallax > 0 else 100.0
        
        teff = safe_float(row.get('teff_gspphot', 5778), 5778)
        logg = safe_float(row.get('logg_gspphot', 4.44), 4.44)
        radius = safe_float(row.get('radius_gspphot', 1.0), 1.0)
        lum = safe_float(row.get('lum_gspphot', 1.0), 1.0)
        
        # Calculate errors from upper/lower bounds
        teff_up = safe_float(row.get('teff_gspphot_upper', teff), teff)
        teff_low = safe_float(row.get('teff_gspphot_lower', teff), teff)
        teff_err = (teff_up - teff_low) / 2 if teff_up != teff_low else 100
        
        return StellarParams(
            source_id=str(row.get('source_id', '')),
            ra=safe_float(row.get('ra', 0)),
            dec=safe_float(row.get('dec', 0)),
            parallax=parallax,
            parallax_err=safe_float(row.get('parallax_error', 0)),
            distance_pc=distance,
            pmra=safe_float(row.get('pmra', 0)),
            pmdec=safe_float(row.get('pmdec', 0)),
            g_mag=safe_float(row.get('phot_g_mean_mag', 10)),
            bp_rp=safe_float(row.get('bp_rp', 0.8)),
            teff=teff,
            teff_err=teff_err,
            logg=logg,
            metallicity=safe_float(row.get('mh_gspphot', 0)),
            radius=radius,
            luminosity=lum,
            mass=self._estimate_mass(teff, logg, radius),
            is_valid=True
        )
    
    def _estimate_mass(self, teff: float, logg: float, radius: float) -> float:
        """Estimate stellar mass from Teff, logg, and radius."""
        # Use logg and radius to estimate mass
        # logg = log(G * M / R^2), so M = 10^logg * R^2 / G
        # In solar units: M_sun = 10^(logg - 4.44) * R^2
        try:
            mass = 10 ** (logg - 4.44) * (radius ** 2)
            return max(0.1, min(mass, 100.0))  # Reasonable limits
        except:
            return 1.0
    
    def _fallback_params(self, identifier: str) -> StellarParams:
        """Return default Sun-like parameters as fallback."""
        return StellarParams(
            tic_id=identifier if "TIC" in str(identifier).upper() else "",
            teff=5778.0,
            teff_err=100.0,
            logg=4.44,
            logg_err=0.1,
            radius=1.0,
            radius_err=0.1,
            luminosity=1.0,
            luminosity_err=0.1,
            mass=1.0,
            distance_pc=100.0,
            is_valid=False,
            quality_flags={'fallback': True}
        )
    
    def cross_match_tic(
        self,
        tic_ids: List[str],
        batch_size: int = 100
    ) -> Dict[str, StellarParams]:
        """
        Batch cross-match TIC IDs with Gaia.
        
        Args:
            tic_ids: List of TIC IDs
            batch_size: Number of IDs per batch query
            
        Returns:
            Dict mapping TIC ID to StellarParams
        """
        results = {}
        
        for i in range(0, len(tic_ids), batch_size):
            batch = tic_ids[i:i + batch_size]
            for tic_id in batch:
                result = self.query_by_tic(tic_id)
                if result.success and result.params:
                    results[tic_id] = result.params
            
            if i + batch_size < len(tic_ids):
                logger.info(f"Cross-matched {i + batch_size}/{len(tic_ids)} TIC IDs")
        
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def get_stellar_params(target: str) -> GaiaResult:
    """
    Convenience function to get stellar parameters.
    
    Automatically detects if target is TIC ID or coordinates.
    
    Args:
        target: TIC ID (e.g., "TIC 261136679") or "ra,dec" coordinates
        
    Returns:
        GaiaResult with stellar parameters
    """
    client = GaiaClient()
    
    # Check if target is TIC ID
    target_clean = str(target).upper().replace("TIC", "").strip()
    if target_clean.isdigit():
        return client.query_by_tic(target_clean)
    
    # Check if coordinates
    if "," in target:
        try:
            ra, dec = map(float, target.split(","))
            return client.query_by_coords(ra, dec)
        except ValueError:
            pass
    
    # Try as target name using astropy
    try:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord.from_name(target)
        return client.query_by_coords(coord.ra.deg, coord.dec.deg)
    except:
        pass
    
    return GaiaResult(
        success=False,
        error=f"Could not resolve target: {target}",
        query_type="unknown"
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Gaia Integration...")
    print("=" * 60)
    
    client = GaiaClient()
    
    # Test with a known TIC ID
    print("\nQuerying TIC 261136679 (TOI-700)...")
    result = client.query_by_tic("261136679")
    
    if result.success:
        print(result.params.summary())
        print(f"\nSpectral type: {result.params.spectral_type()}{result.params.luminosity_class()}")
    else:
        print(f"Query failed: {result.error}")
        # Use fallback
        fallback = client._fallback_params("261136679")
        print("\nUsing fallback parameters:")
        print(fallback.summary())
    
    print("\nTest complete!")
