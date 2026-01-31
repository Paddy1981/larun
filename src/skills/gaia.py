"""
LARUN Skill: Gaia DR3 Integration
=================================
Query Gaia DR3 archive for stellar parameters and cross-match with TIC.

Skill ID: MISSION-003
Command: larun mission gaia

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)
Reference: docs/integrations/GAIA_INTEGRATION.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GaiaSource:
    """Gaia DR3 source data."""
    source_id: int
    ra: float                      # Right ascension (deg)
    dec: float                     # Declination (deg)
    parallax: Optional[float]      # Parallax (mas)
    parallax_error: Optional[float]
    pmra: Optional[float]          # Proper motion in RA (mas/yr)
    pmdec: Optional[float]         # Proper motion in Dec (mas/yr)
    phot_g_mean_mag: Optional[float]    # G magnitude
    phot_bp_mean_mag: Optional[float]   # BP magnitude
    phot_rp_mean_mag: Optional[float]   # RP magnitude
    bp_rp: Optional[float]              # BP-RP color
    teff_gspphot: Optional[float]       # Effective temperature (K)
    teff_gspphot_lower: Optional[float]
    teff_gspphot_upper: Optional[float]
    logg_gspphot: Optional[float]       # Surface gravity
    logg_gspphot_lower: Optional[float]
    logg_gspphot_upper: Optional[float]
    mh_gspphot: Optional[float]         # Metallicity [M/H]
    mh_gspphot_lower: Optional[float]
    mh_gspphot_upper: Optional[float]
    radius_gspphot: Optional[float]     # Stellar radius (R_sun)
    radius_gspphot_lower: Optional[float]
    radius_gspphot_upper: Optional[float]
    distance_gspphot: Optional[float]   # Distance (pc)
    ruwe: Optional[float]               # Renormalized Unit Weight Error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'ra_deg': round(self.ra, 6),
            'dec_deg': round(self.dec, 6),
            'parallax_mas': round(self.parallax, 4) if self.parallax else None,
            'parallax_error_mas': round(self.parallax_error, 4) if self.parallax_error else None,
            'distance_pc': round(1000 / self.parallax, 1) if self.parallax and self.parallax > 0 else None,
            'pmra_mas_yr': round(self.pmra, 3) if self.pmra else None,
            'pmdec_mas_yr': round(self.pmdec, 3) if self.pmdec else None,
            'g_mag': round(self.phot_g_mean_mag, 3) if self.phot_g_mean_mag else None,
            'bp_mag': round(self.phot_bp_mean_mag, 3) if self.phot_bp_mean_mag else None,
            'rp_mag': round(self.phot_rp_mean_mag, 3) if self.phot_rp_mean_mag else None,
            'bp_rp': round(self.bp_rp, 3) if self.bp_rp else None,
            'teff_K': round(self.teff_gspphot, 0) if self.teff_gspphot else None,
            'logg': round(self.logg_gspphot, 2) if self.logg_gspphot else None,
            'metallicity_feh': round(self.mh_gspphot, 2) if self.mh_gspphot else None,
            'radius_rsun': round(self.radius_gspphot, 3) if self.radius_gspphot else None,
            'ruwe': round(self.ruwe, 3) if self.ruwe else None,
        }

    @property
    def distance_pc(self) -> Optional[float]:
        """Calculate distance from parallax."""
        if self.parallax and self.parallax > 0:
            return 1000 / self.parallax
        return self.distance_gspphot

    def __str__(self) -> str:
        dist = f", d={self.distance_pc:.1f}pc" if self.distance_pc else ""
        teff = f", Teff={self.teff_gspphot:.0f}K" if self.teff_gspphot else ""
        return f"Gaia DR3 {self.source_id}: G={self.phot_g_mean_mag:.2f}{dist}{teff}"


@dataclass
class TICCrossmatch:
    """TIC cross-match result."""
    tic_id: int
    gaia_id: int
    angular_distance: float  # arcsec
    tmag: Optional[float]    # TESS magnitude
    teff: Optional[float]
    logg: Optional[float]
    radius: Optional[float]
    mass: Optional[float]
    luminosity: Optional[float]
    disposition: Optional[str]  # e.g., "CONFIRMED PLANET HOST"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tic_id': self.tic_id,
            'gaia_id': self.gaia_id,
            'angular_distance_arcsec': round(self.angular_distance, 2),
            'tmag': round(self.tmag, 3) if self.tmag else None,
            'teff_K': round(self.teff, 0) if self.teff else None,
            'logg': round(self.logg, 2) if self.logg else None,
            'radius_rsun': round(self.radius, 3) if self.radius else None,
            'mass_msun': round(self.mass, 3) if self.mass else None,
            'luminosity_lsun': round(self.luminosity, 4) if self.luminosity else None,
            'disposition': self.disposition
        }


# ============================================================================
# Gaia Query Client
# ============================================================================

class GaiaClient:
    """
    Client for querying Gaia DR3 archive.

    Uses astroquery.gaia for TAP queries to the ESA Gaia archive.

    Example:
        >>> client = GaiaClient()
        >>> source = client.query_by_name("Proxima Centauri")
        >>> print(f"Teff: {source.teff_gspphot} K")

        >>> sources = client.cone_search(ra=83.633, dec=-5.391, radius=0.1)
        >>> for s in sources:
        ...     print(s)
    """

    # Gaia DR3 table names
    GAIA_TABLE = "gaiadr3.gaia_source"
    ASTROPHYS_TABLE = "gaiadr3.astrophysical_parameters"

    # Default columns to query
    DEFAULT_COLUMNS = [
        "source_id", "ra", "dec", "parallax", "parallax_error",
        "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
        "bp_rp", "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
        "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper",
        "mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper",
        "radius_gspphot", "radius_gspphot_lower", "radius_gspphot_upper",
        "distance_gspphot", "ruwe"
    ]

    def __init__(self):
        self._gaia = None
        self._simbad = None
        logger.info("GaiaClient initialized")

    @property
    def gaia(self):
        """Lazy load astroquery Gaia module."""
        if self._gaia is None:
            try:
                from astroquery.gaia import Gaia
                Gaia.MAIN_GAIA_TABLE = self.GAIA_TABLE
                Gaia.ROW_LIMIT = -1  # No limit by default
                self._gaia = Gaia
                logger.info("Connected to Gaia TAP service")
            except ImportError:
                raise ImportError("astroquery is required: pip install astroquery")
        return self._gaia

    @property
    def simbad(self):
        """Lazy load astroquery SIMBAD module."""
        if self._simbad is None:
            try:
                from astroquery.simbad import Simbad
                self._simbad = Simbad
            except ImportError:
                raise ImportError("astroquery is required: pip install astroquery")
        return self._simbad

    def query_by_source_id(self, source_id: Union[int, str]) -> Optional[GaiaSource]:
        """
        Query Gaia DR3 by source ID.

        Args:
            source_id: Gaia DR3 source ID

        Returns:
            GaiaSource object or None if not found
        """
        logger.info(f"Querying Gaia DR3 source_id={source_id}")

        columns = ", ".join(self.DEFAULT_COLUMNS)
        query = f"""
        SELECT {columns}
        FROM {self.GAIA_TABLE}
        WHERE source_id = {source_id}
        """

        try:
            job = self.gaia.launch_job(query)
            result = job.get_results()

            if len(result) == 0:
                logger.warning(f"No Gaia source found for ID {source_id}")
                return None

            return self._row_to_source(result[0])

        except Exception as e:
            logger.error(f"Gaia query failed: {e}")
            raise

    def query_by_name(self, name: str) -> Optional[GaiaSource]:
        """
        Query Gaia DR3 by object name (resolved via SIMBAD).

        Args:
            name: Object name (e.g., "Proxima Centauri", "HD 10700")

        Returns:
            GaiaSource object or None if not found
        """
        logger.info(f"Resolving '{name}' via SIMBAD...")

        try:
            # Resolve name to coordinates using SkyCoord directly
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            try:
                # Try direct coordinate resolution (works for most objects)
                coord = SkyCoord.from_name(name)
            except Exception:
                # Fall back to SIMBAD query
                result = self.simbad.query_object(name)
                if result is None:
                    logger.warning(f"Object '{name}' not found in SIMBAD")
                    return None

                # Handle different column name formats in astroquery versions
                if 'ra' in result.colnames:
                    ra = result['ra'][0]
                    dec = result['dec'][0]
                    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                elif 'RA' in result.colnames:
                    ra = result['RA'][0]
                    dec = result['DEC'][0]
                    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                else:
                    # Try to find coordinate columns
                    logger.error(f"Unknown SIMBAD column format: {result.colnames}")
                    return None
            ra_deg = coord.ra.deg
            dec_deg = coord.dec.deg

            logger.info(f"Resolved to RA={ra_deg:.4f}, Dec={dec_deg:.4f}")

            # Query nearest Gaia source
            sources = self.cone_search(ra_deg, dec_deg, radius=0.01)  # 36 arcsec
            if sources:
                return sources[0]

            # Try larger radius
            sources = self.cone_search(ra_deg, dec_deg, radius=0.05)
            return sources[0] if sources else None

        except Exception as e:
            logger.error(f"Name resolution failed: {e}")
            raise

    def cone_search(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        limit: int = 100
    ) -> List[GaiaSource]:
        """
        Cone search around coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius: Search radius (degrees)
            limit: Maximum number of results

        Returns:
            List of GaiaSource objects, sorted by distance
        """
        logger.info(f"Cone search: RA={ra:.4f}, Dec={dec:.4f}, r={radius}deg")

        columns = ", ".join(self.DEFAULT_COLUMNS)
        query = f"""
        SELECT TOP {limit} {columns},
               DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS dist
        FROM {self.GAIA_TABLE}
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        ORDER BY dist ASC
        """

        try:
            job = self.gaia.launch_job(query)
            result = job.get_results()

            sources = []
            for row in result:
                source = self._row_to_source(row)
                sources.append(source)

            logger.info(f"Found {len(sources)} Gaia sources")
            return sources

        except Exception as e:
            logger.error(f"Cone search failed: {e}")
            raise

    def query_by_tic(self, tic_id: int) -> Optional[GaiaSource]:
        """
        Query Gaia DR3 by TIC ID using cross-match tables.

        Args:
            tic_id: TESS Input Catalog ID

        Returns:
            GaiaSource object or None
        """
        logger.info(f"Querying Gaia for TIC {tic_id}")

        # First, get coordinates from TIC via MAST
        try:
            from astroquery.mast import Catalogs

            tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
            if tic_data is None or len(tic_data) == 0:
                logger.warning(f"TIC {tic_id} not found")
                return None

            ra = float(tic_data['ra'][0])
            dec = float(tic_data['dec'][0])

            # Check if TIC has Gaia ID
            if 'GAIA' in tic_data.colnames:
                gaia_id = tic_data['GAIA'][0]
                if gaia_id and not np.ma.is_masked(gaia_id):
                    return self.query_by_source_id(int(gaia_id))

            # Fall back to cone search
            return self.cone_search(ra, dec, radius=0.005)[0] if self.cone_search(ra, dec, radius=0.005) else None

        except ImportError:
            logger.warning("astroquery.mast not available, using coordinate lookup")
            return None
        except Exception as e:
            logger.error(f"TIC query failed: {e}")
            return None

    def get_stellar_params(self, source: GaiaSource) -> Dict[str, Any]:
        """
        Extract stellar parameters from Gaia source.

        Returns dict with Teff, logg, [M/H], radius, and derived quantities.
        """
        params = {
            'source_id': source.source_id,
            'teff': source.teff_gspphot,
            'teff_lower': source.teff_gspphot_lower,
            'teff_upper': source.teff_gspphot_upper,
            'logg': source.logg_gspphot,
            'logg_lower': source.logg_gspphot_lower,
            'logg_upper': source.logg_gspphot_upper,
            'metallicity': source.mh_gspphot,
            'metallicity_lower': source.mh_gspphot_lower,
            'metallicity_upper': source.mh_gspphot_upper,
            'radius': source.radius_gspphot,
            'radius_lower': source.radius_gspphot_lower,
            'radius_upper': source.radius_gspphot_upper,
            'distance_pc': source.distance_pc,
            'parallax': source.parallax,
            'g_mag': source.phot_g_mean_mag,
            'bp_rp': source.bp_rp,
        }

        # Calculate absolute magnitude if distance available
        if source.distance_pc and source.phot_g_mean_mag:
            params['abs_g_mag'] = source.phot_g_mean_mag - 5 * np.log10(source.distance_pc / 10)

        # Estimate mass from logg and radius
        if source.logg_gspphot and source.radius_gspphot:
            # log(g) = log(G*M/R^2) in cgs
            # M/M_sun = (R/R_sun)^2 * 10^(logg - logg_sun)
            logg_sun = 4.44
            params['mass'] = source.radius_gspphot**2 * 10**(source.logg_gspphot - logg_sun)

        # Calculate luminosity from Teff and radius
        if source.teff_gspphot and source.radius_gspphot:
            T_sun = 5778
            params['luminosity'] = source.radius_gspphot**2 * (source.teff_gspphot / T_sun)**4

        return params

    def _row_to_source(self, row) -> GaiaSource:
        """Convert astropy table row to GaiaSource."""
        def safe_float(val):
            if val is None or (hasattr(val, 'mask') and val.mask):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_int(val):
            if val is None or (hasattr(val, 'mask') and val.mask):
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        return GaiaSource(
            source_id=safe_int(row['source_id']),
            ra=safe_float(row['ra']),
            dec=safe_float(row['dec']),
            parallax=safe_float(row['parallax']),
            parallax_error=safe_float(row['parallax_error']),
            pmra=safe_float(row['pmra']),
            pmdec=safe_float(row['pmdec']),
            phot_g_mean_mag=safe_float(row['phot_g_mean_mag']),
            phot_bp_mean_mag=safe_float(row['phot_bp_mean_mag']),
            phot_rp_mean_mag=safe_float(row['phot_rp_mean_mag']),
            bp_rp=safe_float(row['bp_rp']),
            teff_gspphot=safe_float(row['teff_gspphot']),
            teff_gspphot_lower=safe_float(row['teff_gspphot_lower']),
            teff_gspphot_upper=safe_float(row['teff_gspphot_upper']),
            logg_gspphot=safe_float(row['logg_gspphot']),
            logg_gspphot_lower=safe_float(row['logg_gspphot_lower']),
            logg_gspphot_upper=safe_float(row['logg_gspphot_upper']),
            mh_gspphot=safe_float(row['mh_gspphot']),
            mh_gspphot_lower=safe_float(row['mh_gspphot_lower']),
            mh_gspphot_upper=safe_float(row['mh_gspphot_upper']),
            radius_gspphot=safe_float(row['radius_gspphot']),
            radius_gspphot_lower=safe_float(row['radius_gspphot_lower']),
            radius_gspphot_upper=safe_float(row['radius_gspphot_upper']),
            distance_gspphot=safe_float(row['distance_gspphot']),
            ruwe=safe_float(row['ruwe'])
        )


# ============================================================================
# TIC Cross-matcher
# ============================================================================

class TICCrossmatcher:
    """
    Cross-match Gaia sources with TESS Input Catalog.
    """

    def __init__(self):
        self._catalogs = None
        logger.info("TICCrossmatcher initialized")

    @property
    def catalogs(self):
        """Lazy load MAST Catalogs."""
        if self._catalogs is None:
            try:
                from astroquery.mast import Catalogs
                self._catalogs = Catalogs
            except ImportError:
                raise ImportError("astroquery is required: pip install astroquery")
        return self._catalogs

    def crossmatch(self, gaia_source: GaiaSource) -> Optional[TICCrossmatch]:
        """
        Find TIC entry for a Gaia source.

        Args:
            gaia_source: GaiaSource object

        Returns:
            TICCrossmatch or None
        """
        logger.info(f"Cross-matching Gaia {gaia_source.source_id} with TIC")

        try:
            # Query TIC by coordinates
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coord = SkyCoord(ra=gaia_source.ra*u.deg, dec=gaia_source.dec*u.deg)

            result = self.catalogs.query_region(coord, radius=0.01*u.deg, catalog="TIC")

            if result is None or len(result) == 0:
                logger.warning("No TIC match found")
                return None

            # Find closest match
            closest = result[0]

            # Check if Gaia ID matches
            if 'GAIA' in result.colnames:
                for row in result:
                    if row['GAIA'] == gaia_source.source_id:
                        closest = row
                        break

            def safe_float(val):
                if val is None or (hasattr(val, 'mask') and val.mask):
                    return None
                return float(val)

            return TICCrossmatch(
                tic_id=int(closest['ID']),
                gaia_id=gaia_source.source_id,
                angular_distance=float(closest['dstArcSec']) if 'dstArcSec' in closest.colnames else 0.0,
                tmag=safe_float(closest.get('Tmag')),
                teff=safe_float(closest.get('Teff')),
                logg=safe_float(closest.get('logg')),
                radius=safe_float(closest.get('rad')),
                mass=safe_float(closest.get('mass')),
                luminosity=safe_float(closest.get('lum')),
                disposition=closest.get('disposition') if 'disposition' in closest.colnames else None
            )

        except Exception as e:
            logger.error(f"TIC crossmatch failed: {e}")
            return None

    def get_tic_params(self, tic_id: int) -> Optional[Dict[str, Any]]:
        """
        Get TIC parameters by TIC ID.

        Args:
            tic_id: TIC ID

        Returns:
            Dictionary of stellar parameters
        """
        logger.info(f"Querying TIC {tic_id}")

        try:
            result = self.catalogs.query_object(f"TIC {tic_id}", catalog="TIC")

            if result is None or len(result) == 0:
                return None

            row = result[0]

            def safe_float(key):
                if key not in row.colnames:
                    return None
                val = row[key]
                if val is None or (hasattr(val, 'mask') and val.mask):
                    return None
                return float(val)

            return {
                'tic_id': int(row['ID']),
                'ra': safe_float('ra'),
                'dec': safe_float('dec'),
                'tmag': safe_float('Tmag'),
                'teff': safe_float('Teff'),
                'logg': safe_float('logg'),
                'radius': safe_float('rad'),
                'mass': safe_float('mass'),
                'luminosity': safe_float('lum'),
                'distance': safe_float('d'),
                'gaia_id': int(row['GAIA']) if 'GAIA' in row.colnames and row['GAIA'] else None,
            }

        except Exception as e:
            logger.error(f"TIC query failed: {e}")
            return None


# ============================================================================
# Skill Registration
# ============================================================================

SKILL_INFO = {
    'MISSION-003': {
        'id': 'MISSION-003',
        'name': 'Gaia Integration',
        'command': 'mission gaia',
        'class': GaiaClient,
        'description': 'Query Gaia DR3 for stellar parameters'
    }
}


# ============================================================================
# CLI Functions
# ============================================================================

def query_gaia(target: str) -> Optional[GaiaSource]:
    """Query Gaia by name or ID (convenience function)."""
    client = GaiaClient()

    # Check if it's a numeric ID
    if target.isdigit():
        return client.query_by_source_id(int(target))

    # Check if it's a TIC ID
    if target.upper().startswith('TIC'):
        tic_id = int(target.upper().replace('TIC', '').strip())
        return client.query_by_tic(tic_id)

    # Try name resolution
    return client.query_by_name(target)


def get_stellar_params(target: str) -> Optional[Dict[str, Any]]:
    """Get stellar parameters for a target (convenience function)."""
    source = query_gaia(target)
    if source:
        client = GaiaClient()
        return client.get_stellar_params(source)
    return None


def crossmatch_tic(gaia_source: GaiaSource) -> Optional[TICCrossmatch]:
    """Cross-match Gaia source with TIC (convenience function)."""
    matcher = TICCrossmatcher()
    return matcher.crossmatch(gaia_source)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Gaia DR3 Integration...")
    print("=" * 50)

    # Note: This requires internet connection and astroquery
    try:
        client = GaiaClient()

        # Test with a well-known star
        print("\nQuerying Proxima Centauri...")
        source = client.query_by_name("Proxima Centauri")

        if source:
            print(f"  {source}")
            print(f"  Parallax: {source.parallax:.2f} mas")
            print(f"  Distance: {source.distance_pc:.2f} pc")
            print(f"  Teff: {source.teff_gspphot:.0f} K" if source.teff_gspphot else "  Teff: N/A")
            print(f"  Radius: {source.radius_gspphot:.3f} R_sun" if source.radius_gspphot else "  Radius: N/A")

            # Get full stellar parameters
            print("\n  Full stellar parameters:")
            params = client.get_stellar_params(source)
            for key, val in params.items():
                if val is not None:
                    print(f"    {key}: {val}")

            # Cross-match with TIC
            print("\n  Cross-matching with TIC...")
            matcher = TICCrossmatcher()
            tic = matcher.crossmatch(source)
            if tic:
                print(f"    TIC ID: {tic.tic_id}")
                print(f"    TESS mag: {tic.tmag}")
        else:
            print("  Source not found")

        # Test cone search
        print("\nCone search around Orion Nebula (M42)...")
        sources = client.cone_search(ra=83.8221, dec=-5.3911, radius=0.1, limit=5)
        for s in sources:
            print(f"  {s}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Install required packages: pip install astroquery")
    except Exception as e:
        print(f"Error: {e}")
        print("(This test requires internet connection)")
