# NASA Data Sources - Research Documentation

## Overview

This document provides comprehensive information about NASA astronomical data sources that LARUN can access for analysis.

---

## 1. MAST (Mikulski Archive for Space Telescopes)

### About
MAST is NASA's primary archive for optical, ultraviolet, and near-infrared astronomy data. It hosts data from major missions including Hubble, JWST, TESS, and Kepler.

### API Endpoint
```
Base URL: https://mast.stsci.edu/api/v0/
Documentation: https://mast.stsci.edu/api/v0/
```

### Python Access (lightkurve)
```python
import lightkurve as lk

# Search for TESS data
search_result = lk.search_lightcurve('TIC 307210830', mission='TESS')

# Search for Kepler data
search_result = lk.search_lightcurve('Kepler-186', mission='Kepler')

# Download light curve
lc = search_result[0].download()

# Basic processing
lc = lc.remove_nans().normalize().flatten()
```

### Available Data Products

| Mission | Data Type | Cadence | Coverage |
|---------|-----------|---------|----------|
| TESS | Light curves | 2-min, 10-min, 30-min | Near full sky |
| Kepler | Light curves | 1-min, 30-min | Cygnus-Lyra field |
| K2 | Light curves | 1-min, 30-min | Ecliptic plane |
| HST | Images, Spectra | Varies | Pointed |
| JWST | Images, Spectra | Varies | Pointed |

### Query Examples

```python
from astroquery.mast import Observations, Catalogs

# Query by coordinates
observations = Observations.query_region("19:51:52.99 +08:56:02.6", radius=0.02)

# Query by target name
observations = Observations.query_object("TRAPPIST-1")

# Query TIC catalog
tic_data = Catalogs.query_object("TIC 307210830", catalog="TIC")

# Get data products
products = Observations.get_product_list(observations)
```

### Data Formats
- **FITS**: Primary format for light curves and images
- **CSV**: Tabular data exports
- **HLSP**: High Level Science Products (processed data)

---

## 2. NASA Exoplanet Archive

### About
The NASA Exoplanet Archive is the authoritative source for confirmed exoplanet data, containing parameters for 5,000+ confirmed planets.

### API Endpoint
```
Base URL: https://exoplanetarchive.ipac.caltech.edu/
TAP Service: https://exoplanetarchive.ipac.caltech.edu/TAP/
```

### Python Access
```python
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Query confirmed planets
planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name,hostname,pl_orbper,pl_rade,pl_bmasse,disc_facility",
    where="disc_facility LIKE '%TESS%'",
    order="pl_name"
)

# Convert to pandas
df = planets.to_pandas()
```

### Key Tables

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `pscomppars` | Planetary Systems Composite Parameters | pl_name, hostname, pl_orbper, pl_rade |
| `ps` | Planetary Systems | All published parameters |
| `stellarhosts` | Stellar Host Parameters | st_teff, st_rad, st_mass |
| `toi` | TESS Objects of Interest | TOI name, disposition, parameters |
| `koi` | Kepler Objects of Interest | KOI name, disposition, parameters |

### Important Columns

**Planetary Parameters:**
- `pl_name`: Planet name
- `pl_orbper`: Orbital period (days)
- `pl_rade`: Planet radius (Earth radii)
- `pl_bmasse`: Planet mass (Earth masses)
- `pl_eqt`: Equilibrium temperature (K)
- `pl_insol`: Insolation flux (Earth flux)

**Stellar Parameters:**
- `hostname`: Host star name
- `st_teff`: Effective temperature (K)
- `st_rad`: Stellar radius (Solar radii)
- `st_mass`: Stellar mass (Solar masses)
- `st_met`: Metallicity [Fe/H]
- `st_logg`: Surface gravity (log g)

**Discovery:**
- `disc_year`: Discovery year
- `disc_facility`: Discovery facility
- `disc_telescope`: Discovery telescope
- `discoverymethod`: Detection method

### Query Examples

```python
# Get all TESS discoveries
tess_planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    where="disc_facility LIKE '%TESS%'"
)

# Get habitable zone planets
hz_planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    where="pl_insol BETWEEN 0.25 AND 1.75"
)

# Get planets with known masses and radii
characterized = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    where="pl_rade IS NOT NULL AND pl_bmasse IS NOT NULL"
)
```

---

## 3. Gaia Archive

### About
ESA's Gaia mission provides precise astrometry, photometry, and spectroscopy for nearly 2 billion objects.

### API Endpoint
```
Base URL: https://gea.esac.esa.int/archive/
TAP Service: https://gea.esac.esa.int/tap-server/tap
```

### Python Access
```python
from astroquery.gaia import Gaia

# Query by coordinates
job = Gaia.launch_job("""
    SELECT TOP 100 
        source_id, ra, dec, parallax, phot_g_mean_mag,
        bp_rp, teff_gspphot, logg_gspphot
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', 280.0, -60.0, 0.1)
    )
""")
result = job.get_results()

# Cross-match with other catalogs
job = Gaia.launch_job("""
    SELECT g.source_id, g.ra, g.dec, g.parallax,
           t.teff, t.logg, t.fe_h
    FROM gaiadr3.gaia_source AS g
    JOIN gaiadr3.astrophysical_parameters AS t
        ON g.source_id = t.source_id
    WHERE g.parallax > 10
""")
```

### Key Gaia DR3 Tables

| Table | Description |
|-------|-------------|
| `gaia_source` | Main source catalog (1.8B sources) |
| `astrophysical_parameters` | Stellar parameters (470M sources) |
| `vari_summary` | Variable star summary |
| `nss_two_body_orbit` | Binary star orbits |
| `rvs_mean_spectrum` | Mean RVS spectra |

### Important Columns

**Astrometry:**
- `ra`, `dec`: Position (deg)
- `parallax`: Parallax (mas)
- `pmra`, `pmdec`: Proper motion (mas/yr)
- `radial_velocity`: Radial velocity (km/s)

**Photometry:**
- `phot_g_mean_mag`: G-band magnitude
- `phot_bp_mean_mag`: BP magnitude
- `phot_rp_mean_mag`: RP magnitude
- `bp_rp`: BP-RP color

**Astrophysical Parameters:**
- `teff_gspphot`: Effective temperature (K)
- `logg_gspphot`: Surface gravity
- `mh_gspphot`: Metallicity [M/H]
- `distance_gspphot`: Distance (pc)
- `ag_gspphot`: Extinction A_G

---

## 4. IRSA (Infrared Science Archive)

### About
NASA's archive for infrared and submillimeter astronomy data.

### Key Catalogs

| Survey | Wavelength | Objects |
|--------|------------|---------|
| 2MASS | J, H, K (1-2.5 μm) | 470M sources |
| WISE/NEOWISE | 3.4-22 μm | 747M sources |
| Spitzer | 3.6-160 μm | Various |

### Python Access
```python
from astroquery.irsa import Irsa

# Query 2MASS
result = Irsa.query_region(
    "19:51:52.99 +08:56:02.6",
    catalog="fp_psc",  # 2MASS Point Source Catalog
    radius="0d0m30s"
)

# Query WISE
result = Irsa.query_region(
    "19:51:52.99 +08:56:02.6",
    catalog="allwise_p3as_psd",
    radius="0d0m30s"
)
```

---

## 5. Data Access Patterns for LARUN

### Pattern 1: Light Curve Analysis Pipeline
```python
async def fetch_light_curve(target: str, mission: str = "TESS") -> LightCurve:
    """Fetch and preprocess light curve for analysis."""
    import lightkurve as lk
    
    # Search
    search = lk.search_lightcurve(target, mission=mission)
    if len(search) == 0:
        raise ValueError(f"No data found for {target}")
    
    # Download
    lc = search[0].download()
    
    # Preprocess
    lc = (lc
        .remove_nans()
        .remove_outliers(sigma=5)
        .normalize()
        .flatten(window_length=401))
    
    return lc
```

### Pattern 2: Cross-Match Pipeline
```python
def cross_match_catalogs(ra: float, dec: float, radius: float = 0.001):
    """Cross-match position across multiple catalogs."""
    from astroquery.mast import Catalogs
    from astroquery.gaia import Gaia
    from astroquery.irsa import Irsa
    
    results = {}
    
    # TIC (TESS Input Catalog)
    results['tic'] = Catalogs.query_region(
        f"{ra} {dec}", catalog="TIC", radius=radius
    )
    
    # Gaia
    results['gaia'] = Gaia.query_object(f"{ra} {dec}", radius=radius)
    
    # 2MASS
    results['2mass'] = Irsa.query_region(
        f"{ra} {dec}", catalog="fp_psc", radius=f"0d0m{radius*3600}s"
    )
    
    return results
```

### Pattern 3: Bulk Data Download
```python
def download_training_data(n_planets: int = 100):
    """Download light curves for confirmed planets."""
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
    import lightkurve as lk
    
    # Get confirmed planets
    planets = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="hostname,pl_name,disc_facility",
        where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'",
        order="pl_name"
    ).to_pandas()
    
    light_curves = []
    for _, row in planets.head(n_planets).iterrows():
        try:
            search = lk.search_lightcurve(row['hostname'])
            if len(search) > 0:
                lc = search[0].download()
                light_curves.append({
                    'hostname': row['hostname'],
                    'planet': row['pl_name'],
                    'flux': lc.flux.value,
                    'time': lc.time.value
                })
        except Exception:
            continue
    
    return light_curves
```

---

## 6. Rate Limits and Best Practices

### MAST
- No strict rate limits for authenticated users
- Use `astroquery.mast.Observations.enable_cloud_dataset()` for faster access
- Cache downloaded data locally

### Exoplanet Archive
- No strict rate limits
- Use batch queries where possible
- Cache results for repeated queries

### Gaia
- Asynchronous queries for large results
- Use `WHERE` clauses to limit results
- Maximum 3 million rows per query

### Best Practices
```python
# 1. Cache data locally
from pathlib import Path
import pickle

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cached_query(key: str, query_func, *args, **kwargs):
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    result = query_func(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    return result

# 2. Handle errors gracefully
def safe_download(target: str):
    try:
        return lk.search_lightcurve(target)[0].download()
    except (IndexError, Exception) as e:
        logger.warning(f"Failed to download {target}: {e}")
        return None

# 3. Use async for multiple downloads
import asyncio

async def download_multiple(targets: list):
    tasks = [fetch_light_curve(t) for t in targets]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 7. Data Quality Flags

### TESS Quality Flags
| Bit | Meaning |
|-----|---------|
| 1 | Attitude tweak |
| 2 | Safe mode |
| 4 | Coarse point |
| 8 | Earth point |
| 16 | Argabrightening |
| 32 | Reaction wheel desaturation |
| 64 | Manual exclude |
| 128 | Discontinuity corrected |

### Kepler Quality Flags
| Bit | Meaning |
|-----|---------|
| 1 | Attitude tweak |
| 2 | Safe mode |
| 4 | Coarse point |
| 8 | Earth point |
| 16 | Zero crossing |
| 32 | Desaturation event |

### Usage
```python
# Filter out bad quality data
quality_mask = lc.quality == 0
clean_lc = lc[quality_mask]

# Or use lightkurve's built-in
clean_lc = lc.remove_outliers().flatten()
```

---

## References

1. MAST API Documentation: https://mast.stsci.edu/api/v0/
2. NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/docs/
3. Gaia Archive: https://gea.esac.esa.int/archive/documentation/
4. IRSA: https://irsa.ipac.caltech.edu/docs/
5. Lightkurve: https://docs.lightkurve.org/

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
