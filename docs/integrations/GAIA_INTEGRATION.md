# Gaia DR3 Integration

```
╔══════════════════════════════════════════════════════════════════════════╗
║  LARUN TinyML - Gaia DR3 Integration Guide                               ║
║  Skill ID: MISSION-003                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Overview

The Gaia DR3 integration provides access to the ESA Gaia mission's third data release, enabling:
- Precise stellar parallaxes and distances
- Stellar parameters (Teff, log g, metallicity, radius)
- Cross-matching with TESS Input Catalog (TIC)
- Support for exoplanet host star characterization

## Quick Start

### CLI Commands

```bash
# Query by object name
larun> /gaia "Proxima Centauri"

# Query by TIC ID
larun> /gaia tic 307210830

# Cone search
larun> /gaia cone 83.633 -5.391 0.1
```

### Python API

```python
from src.skills.gaia import GaiaClient, query_gaia, get_stellar_params

# Simple query
source = query_gaia("Proxima Centauri")
print(f"Distance: {source.distance_pc:.2f} pc")
print(f"Teff: {source.teff_gspphot:.0f} K")

# Full stellar parameters
params = get_stellar_params("HD 10700")
print(f"Mass: {params['mass']:.2f} M_sun")

# Advanced usage
client = GaiaClient()
sources = client.cone_search(ra=83.633, dec=-5.391, radius=0.1)
for s in sources:
    print(s)
```

## Data Available

### Gaia DR3 Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `source_id` | Unique Gaia identifier | - |
| `ra`, `dec` | Coordinates (ICRS) | degrees |
| `parallax` | Trigonometric parallax | mas |
| `pmra`, `pmdec` | Proper motion | mas/yr |
| `phot_g_mean_mag` | G-band magnitude | mag |
| `phot_bp_mean_mag` | BP-band magnitude | mag |
| `phot_rp_mean_mag` | RP-band magnitude | mag |
| `bp_rp` | BP-RP color index | mag |
| `teff_gspphot` | Effective temperature | K |
| `logg_gspphot` | Surface gravity | log(cm/s²) |
| `mh_gspphot` | Metallicity [M/H] | dex |
| `radius_gspphot` | Stellar radius | R_sun |
| `ruwe` | Astrometric quality flag | - |

### TIC Cross-Match

When cross-matching with TIC, additional parameters are available:

| Parameter | Description | Unit |
|-----------|-------------|------|
| `tic_id` | TESS Input Catalog ID | - |
| `tmag` | TESS magnitude | mag |
| `mass` | Stellar mass | M_sun |
| `luminosity` | Stellar luminosity | L_sun |

## Use Cases

### 1. Exoplanet Host Characterization

```python
from src.skills.gaia import query_gaia, GaiaClient
from src.skills.stellar import StellarClassifier

# Get stellar parameters for a known exoplanet host
source = query_gaia("TOI-700")
client = GaiaClient()
params = client.get_stellar_params(source)

# Calculate planet radius from transit depth
transit_depth = 0.00064  # 640 ppm
stellar_radius = params['radius']  # R_sun
planet_radius = stellar_radius * np.sqrt(transit_depth) * 109.2  # R_earth

print(f"Planet radius: {planet_radius:.2f} R_earth")
```

### 2. Target Selection

```python
from src.skills.gaia import GaiaClient

client = GaiaClient()

# Find nearby bright stars for TESS follow-up
sources = client.cone_search(ra=180.0, dec=30.0, radius=1.0, limit=100)

# Filter for good targets
good_targets = [
    s for s in sources
    if s.phot_g_mean_mag and s.phot_g_mean_mag < 12
    and s.teff_gspphot and 4000 < s.teff_gspphot < 6500
    and s.ruwe and s.ruwe < 1.4  # Good astrometric fit
]
```

### 3. Distance-Limited Sample

```python
# Find all stars within 50 pc in a field
sources = client.cone_search(ra=83.0, dec=-5.0, radius=0.5)
nearby = [s for s in sources if s.parallax and s.parallax > 20]  # >20 mas = <50 pc
```

## Integration with Other Skills

### Stellar Classification

```python
from src.skills.gaia import query_gaia
from src.skills.stellar import StellarClassifier

source = query_gaia("HD 10700")  # Tau Ceti
classifier = StellarClassifier()
result = classifier.classify_from_teff(
    source.teff_gspphot,
    logg=source.logg_gspphot
)
print(f"Spectral type: {result.spectral_type}")  # G8V
```

### BLS Transit Search

```python
from src.skills.gaia import query_gaia
from src.skills.periodogram import BLSPeriodogram

# Get stellar radius for transit depth -> planet radius
source = query_gaia("TIC 307210830")
stellar_radius = source.radius_gspphot

# Run BLS and convert depth to planet radius
bls = BLSPeriodogram()
result = bls.compute(time, flux)
if result.candidates:
    depth = result.candidates[0].depth
    planet_radius = stellar_radius * np.sqrt(depth) * 109.2  # R_earth
```

## API Reference

### GaiaClient

```python
class GaiaClient:
    def query_by_source_id(self, source_id: int) -> Optional[GaiaSource]
    def query_by_name(self, name: str) -> Optional[GaiaSource]
    def query_by_tic(self, tic_id: int) -> Optional[GaiaSource]
    def cone_search(self, ra: float, dec: float, radius: float, limit: int = 100) -> List[GaiaSource]
    def get_stellar_params(self, source: GaiaSource) -> Dict[str, Any]
```

### TICCrossmatcher

```python
class TICCrossmatcher:
    def crossmatch(self, gaia_source: GaiaSource) -> Optional[TICCrossmatch]
    def get_tic_params(self, tic_id: int) -> Optional[Dict[str, Any]]
```

### GaiaSource

```python
@dataclass
class GaiaSource:
    source_id: int
    ra: float
    dec: float
    parallax: Optional[float]
    teff_gspphot: Optional[float]
    logg_gspphot: Optional[float]
    radius_gspphot: Optional[float]
    # ... and more

    @property
    def distance_pc(self) -> Optional[float]:
        """Calculate distance from parallax."""
```

## Requirements

```
astroquery>=0.4.6
astropy>=5.3
```

## Data Source

- **Archive**: ESA Gaia Archive (https://gea.esac.esa.int/archive/)
- **Table**: `gaiadr3.gaia_source`
- **Access**: TAP query via astroquery.gaia

## References

1. Gaia Collaboration et al. (2023). "Gaia Data Release 3"
2. Stassun et al. (2019). "The TESS Input Catalog"
3. ESA Gaia Archive Documentation

---

*Created by: Padmanaban Veeraragavalu (Larun Engineering)*
*With AI assistance from: Claude (Anthropic)*
*Last Updated: January 2026*
