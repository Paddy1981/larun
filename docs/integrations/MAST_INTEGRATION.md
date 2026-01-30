# MAST Integration Guide

## Overview

Integration guide for MAST (Mikulski Archive for Space Telescopes) data access in LARUN.

---

## Quick Start

```python
import lightkurve as lk
from astroquery.mast import Observations, Catalogs

# Search by target name
results = lk.search_lightcurve('TIC 307210830', mission='TESS')

# Download light curve
lc = results[0].download()

# Process
lc = lc.remove_nans().normalize().flatten()
```

---

## Supported Missions

| Mission | Command | Data Type |
|---------|---------|-----------|
| TESS | `mission='TESS'` | Light curves, FFIs |
| Kepler | `mission='Kepler'` | Light curves |
| K2 | `mission='K2'` | Light curves |
| HST | Direct API | Images, spectra |
| JWST | Direct API | Images, spectra |

---

## TIC (TESS Input Catalog) Query

```python
from astroquery.mast import Catalogs

# Query by TIC ID
tic_data = Catalogs.query_object('TIC 307210830', catalog='TIC')

# Query by coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u

coord = SkyCoord(ra=285.0, dec=45.0, unit='deg')
tic_data = Catalogs.query_region(coord, radius=0.1*u.deg, catalog='TIC')

# Key columns
# - ID: TIC identifier
# - Tmag: TESS magnitude
# - Teff: Effective temperature
# - rad: Stellar radius
# - mass: Stellar mass
# - d: Distance
```

---

## Light Curve Products

### SPOC Pipeline (2-minute cadence)

```python
# Search for SPOC products
results = lk.search_lightcurve('TIC 307210830', 
                                mission='TESS',
                                author='SPOC')

# PDCSAP flux (detrended)
lc = results[0].download()
flux = lc.pdcsap_flux

# SAP flux (raw)
flux_raw = lc.sap_flux
```

### QLP (Quick Look Pipeline)

```python
# FFI-based light curves
results = lk.search_lightcurve('TIC 307210830',
                                mission='TESS',
                                author='QLP')
```

### TESS-SPOC FFI

```python
# 10-minute/30-minute cadence from FFIs
results = lk.search_lightcurve('TIC 307210830',
                                mission='TESS',
                                exptime=600)  # 10-minute
```

---

## Bulk Downloads

```python
async def download_multiple_lightcurves(tic_ids, max_concurrent=5):
    """
    Download multiple light curves efficiently.
    """
    import asyncio
    
    async def download_one(tic_id):
        try:
            results = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')
            if len(results) > 0:
                return results[0].download()
        except:
            return None
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_download(tic_id):
        async with semaphore:
            return await download_one(tic_id)
    
    tasks = [bounded_download(tic_id) for tic_id in tic_ids]
    return await asyncio.gather(*tasks)
```

---

## Error Handling

```python
def safe_download(target, mission='TESS'):
    """
    Safe download with error handling.
    """
    try:
        results = lk.search_lightcurve(target, mission=mission)
        
        if len(results) == 0:
            logger.warning(f"No data found for {target}")
            return None
        
        lc = results[0].download()
        
        # Quality checks
        if len(lc.flux) < 100:
            logger.warning(f"Insufficient data points for {target}")
            return None
        
        return lc
    
    except Exception as e:
        logger.error(f"Failed to download {target}: {e}")
        return None
```

---

## Caching

```python
import pickle
from pathlib import Path

CACHE_DIR = Path('data/cache/mast')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cached_download(target, mission='TESS'):
    """
    Download with local caching.
    """
    cache_key = f"{target}_{mission}".replace(' ', '_')
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    lc = safe_download(target, mission)
    
    if lc is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump(lc, f)
    
    return lc
```

---

*LARUN - Larun. × Astrodata*
