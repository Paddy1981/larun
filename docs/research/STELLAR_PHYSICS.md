# Stellar Physics - Research Documentation

## Overview

This document covers stellar classification, characterization, and analysis methods for LARUN's stellar science capabilities.

---

## 1. Stellar Classification Systems

### Harvard Spectral Classification (OBAFGKM)

```
O → B → A → F → G → K → M
Hot ←――――――――――――――――――→ Cool
Blue ←――――――――――――――――――→ Red

Mnemonic: "Oh Be A Fine Girl/Guy Kiss Me"
```

### Spectral Class Properties

| Class | Temperature (K) | Color | Mass (M☉) | Luminosity (L☉) | Examples |
|-------|-----------------|-------|-----------|-----------------|----------|
| O | 30,000-50,000 | Blue | 16-150 | 30,000-1,000,000 | Alnitak, ζ Ophiuchi |
| B | 10,000-30,000 | Blue-white | 2.1-16 | 25-30,000 | Rigel, Spica |
| A | 7,500-10,000 | White | 1.4-2.1 | 5-25 | Sirius, Vega |
| F | 6,000-7,500 | Yellow-white | 1.04-1.4 | 1.5-5 | Canopus, Procyon |
| G | 5,200-6,000 | Yellow | 0.8-1.04 | 0.6-1.5 | Sun, Alpha Centauri A |
| K | 3,700-5,200 | Orange | 0.45-0.8 | 0.08-0.6 | Arcturus, Aldebaran |
| M | 2,400-3,700 | Red | 0.08-0.45 | <0.08 | Betelgeuse, Proxima Centauri |

### Luminosity Classes (Yerkes/MKK)

| Class | Type | Description |
|-------|------|-------------|
| Ia | Luminous Supergiant | Most luminous |
| Ib | Less Luminous Supergiant | |
| II | Bright Giant | |
| III | Giant | |
| IV | Subgiant | |
| V | Main Sequence (Dwarf) | Like the Sun |
| VI | Subdwarf | |
| VII | White Dwarf | |

### Full Classification Example
```
Sun: G2V
- G = Spectral class (yellow)
- 2 = Subclass (0-9, where 0 is hottest in class)
- V = Luminosity class (main sequence)

Betelgeuse: M1Ia
- M = Spectral class (red)
- 1 = Subclass
- Ia = Luminosity class (luminous supergiant)
```

---

## 2. Color-Temperature Relations

### B-V Color Index

The B-V color index is the difference between blue (B) and visual (V) magnitudes:

```python
def temperature_from_bv(bv):
    """
    Estimate effective temperature from B-V color.
    Based on Flower (1996) calibration.
    
    Args:
        bv: B-V color index
    
    Returns:
        T_eff: Effective temperature (K)
    """
    # Polynomial fit coefficients
    log_teff = (3.979145 
                - 0.654499 * bv 
                + 1.740690 * bv**2 
                - 4.608815 * bv**3
                + 6.792600 * bv**4
                - 5.396909 * bv**5
                + 2.192970 * bv**6
                - 0.359496 * bv**7)
    
    return 10**log_teff


# B-V values for reference
BV_REFERENCE = {
    'O5': -0.33,
    'B0': -0.30,
    'B5': -0.17,
    'A0': -0.02,
    'A5': 0.15,
    'F0': 0.30,
    'F5': 0.44,
    'G0': 0.58,
    'G2': 0.65,  # Sun
    'G5': 0.68,
    'K0': 0.81,
    'K5': 1.15,
    'M0': 1.40,
    'M5': 1.64,
}
```

### Gaia Color-Temperature Relations

```python
def temperature_from_gaia_bp_rp(bp_rp, logg=4.5, feh=0.0):
    """
    Estimate temperature from Gaia BP-RP color.
    Based on Andrae et al. (2018).
    
    Args:
        bp_rp: Gaia BP-RP color
        logg: Surface gravity (optional refinement)
        feh: Metallicity [Fe/H] (optional refinement)
    
    Returns:
        T_eff: Effective temperature (K)
    """
    # Simple relation (good for main sequence)
    if bp_rp < 0.5:
        T_eff = 10000 - 8000 * bp_rp
    elif bp_rp < 1.5:
        T_eff = 7500 - 2500 * (bp_rp - 0.5)
    else:
        T_eff = 5000 - 1500 * (bp_rp - 1.5)
    
    return max(T_eff, 2500)  # Floor at 2500K
```

---

## 3. Stellar Parameters from Photometry

### Luminosity Calculation

```python
import numpy as np

def calculate_luminosity(magnitude, distance_pc, bc=0):
    """
    Calculate stellar luminosity from apparent magnitude and distance.
    
    Args:
        magnitude: Apparent visual magnitude
        distance_pc: Distance in parsecs
        bc: Bolometric correction
    
    Returns:
        L: Luminosity in solar luminosities
    """
    # Absolute magnitude
    M_V = magnitude - 5 * np.log10(distance_pc) + 5
    
    # Bolometric magnitude
    M_bol = M_V + bc
    
    # Solar bolometric magnitude
    M_bol_sun = 4.74
    
    # Luminosity ratio
    L = 10**((M_bol_sun - M_bol) / 2.5)
    
    return L


def bolometric_correction(T_eff):
    """
    Approximate bolometric correction from temperature.
    Based on Flower (1996).
    """
    log_T = np.log10(T_eff)
    
    bc = (-0.190537291496456e5
          + 0.155144866764412e5 * log_T
          - 0.421278819301717e4 * log_T**2
          + 0.381476328422343e3 * log_T**3)
    
    return bc
```

### Stellar Radius

```python
def calculate_radius(L_solar, T_eff):
    """
    Calculate stellar radius from luminosity and temperature.
    Stefan-Boltzmann law.
    
    Args:
        L_solar: Luminosity in solar luminosities
        T_eff: Effective temperature (K)
    
    Returns:
        R: Radius in solar radii
    """
    T_sun = 5778  # K
    
    # L = 4πR²σT⁴
    # L/L☉ = (R/R☉)² × (T/T☉)⁴
    # R/R☉ = √(L/L☉) × (T☉/T)²
    
    R_solar = np.sqrt(L_solar) * (T_sun / T_eff)**2
    
    return R_solar
```

### Mass-Luminosity Relation

```python
def estimate_mass_from_luminosity(L_solar):
    """
    Estimate stellar mass from luminosity (main sequence only).
    
    L ∝ M^α where α ≈ 3.5 for solar-type stars
    """
    if L_solar < 0.03:
        alpha = 2.3  # Low-mass stars
    elif L_solar < 16:
        alpha = 4.0  # Solar-type
    elif L_solar < 54000:
        alpha = 3.5  # Intermediate mass
    else:
        alpha = 1.0  # Very massive
    
    M_solar = L_solar**(1/alpha)
    
    return M_solar
```

---

## 4. Stellar Evolution Stages

### Hertzsprung-Russell (HR) Diagram Regions

```
           Luminosity
                ↑
    10⁶ L☉  |  ★ Supergiants
            |    ↗
    10⁴ L☉  |      Giants
            |        ↗
    10² L☉  |   ★-----★  
            |   Main Sequence
      1 L☉  |   ★ (Sun)
            |         ↘
   10⁻² L☉ |            ★ White Dwarfs
            |
            └─────────────────────→
                 40,000   10,000   5,000   3,000
                          Temperature (K)
```

### Evolution Tracks

```python
def stellar_evolution_stage(mass_solar, luminosity_solar, temperature):
    """
    Estimate stellar evolution stage from parameters.
    """
    # Main sequence properties (approximate)
    ms_luminosity = mass_solar**3.5
    ms_radius = mass_solar**0.8
    ms_temp = 5778 * mass_solar**0.5
    
    # Luminosity ratio to MS
    l_ratio = luminosity_solar / ms_luminosity
    
    # Temperature ratio
    t_ratio = temperature / ms_temp
    
    if l_ratio < 2 and 0.7 < t_ratio < 1.3:
        return "main_sequence"
    elif l_ratio > 100 and temperature < 5000:
        if luminosity_solar > 10000:
            return "supergiant"
        else:
            return "giant"
    elif l_ratio > 2 and l_ratio < 10:
        return "subgiant"
    elif luminosity_solar < 0.01 and temperature > 10000:
        return "white_dwarf"
    else:
        return "unknown"
```

### Main Sequence Lifetime

```python
def main_sequence_lifetime(mass_solar):
    """
    Estimate main sequence lifetime.
    
    t_MS ∝ M/L ∝ M/M^3.5 = M^(-2.5)
    
    Args:
        mass_solar: Stellar mass in solar masses
    
    Returns:
        t_MS: Main sequence lifetime in Gyr
    """
    t_sun = 10  # Gyr (Sun's MS lifetime)
    
    return t_sun * mass_solar**(-2.5)
```

---

## 5. Stellar Activity and Variability

### Rotation-Activity Relations

```python
def estimate_rotation_period(mass_solar, age_gyr):
    """
    Estimate rotation period from gyrochronology.
    Based on Barnes (2010).
    
    Args:
        mass_solar: Stellar mass
        age_gyr: Age in Gyr
    
    Returns:
        P_rot: Rotation period in days
    """
    # Skumanich relation: v_rot ∝ t^(-1/2)
    # P_rot ∝ t^(1/2)
    
    # Approximate for solar-type stars
    if 0.5 < mass_solar < 1.2:
        P_rot = 1.0 * (age_gyr / 0.1)**0.5 * (1.0 / mass_solar)**0.6
        return min(P_rot, 50)  # Cap at slow rotators
    else:
        return None  # Gyrochronology less reliable
```

### Flare Detection

```python
def detect_flares(time, flux, sigma_threshold=5):
    """
    Detect stellar flares in light curve.
    
    Args:
        time: Time array
        flux: Flux array (normalized)
        sigma_threshold: Detection threshold in sigma
    
    Returns:
        flare_times: Times of detected flares
        flare_amplitudes: Amplitudes of flares
    """
    # Baseline and noise
    baseline = np.median(flux)
    noise = np.std(flux[flux < baseline + 0.01])
    
    # Find positive outliers (flares brighten)
    threshold = baseline + sigma_threshold * noise
    flare_mask = flux > threshold
    
    # Group consecutive points
    flares = []
    in_flare = False
    flare_start = 0
    
    for i, is_flare in enumerate(flare_mask):
        if is_flare and not in_flare:
            flare_start = i
            in_flare = True
        elif not is_flare and in_flare:
            # Flare ended
            flare_idx = slice(flare_start, i)
            peak_time = time[flare_start + np.argmax(flux[flare_idx])]
            amplitude = np.max(flux[flare_idx]) - baseline
            flares.append({
                'time': peak_time,
                'amplitude': amplitude,
                'duration': time[i-1] - time[flare_start]
            })
            in_flare = False
    
    return flares


def flare_energy(amplitude, duration_hours, L_star_solar):
    """
    Estimate flare energy.
    
    Args:
        amplitude: Flare amplitude (fractional flux increase)
        duration_hours: Flare duration in hours
        L_star_solar: Stellar luminosity in solar luminosities
    
    Returns:
        E: Flare energy in ergs
    """
    L_sun_erg = 3.828e33  # erg/s
    
    # Energy ≈ L × ΔF × Δt
    E = L_star_solar * L_sun_erg * amplitude * (duration_hours * 3600)
    
    return E
```

---

## 6. Binary Star Detection

### Radial Velocity Variations

```python
def detect_rv_binary(rv_measurements, times, min_amplitude=1.0):
    """
    Detect binary companion from radial velocity variations.
    
    Args:
        rv_measurements: Radial velocities (km/s)
        times: Observation times (days)
        min_amplitude: Minimum RV amplitude to detect (km/s)
    
    Returns:
        is_binary: Boolean
        period: Orbital period if binary
        k: RV semi-amplitude
    """
    # Check for significant variation
    rv_std = np.std(rv_measurements)
    rv_range = np.max(rv_measurements) - np.min(rv_measurements)
    
    if rv_range < min_amplitude:
        return False, None, None
    
    # Simple period search (Lomb-Scargle)
    from scipy.signal import lombscargle
    
    # Period grid
    periods = np.linspace(1, 1000, 10000)
    frequencies = 2 * np.pi / periods
    
    # Compute periodogram
    power = lombscargle(times, rv_measurements - np.mean(rv_measurements), frequencies)
    
    # Find best period
    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    
    # Semi-amplitude
    K = rv_range / 2
    
    return True, best_period, K
```

### Eclipsing Binary Classification

```python
def classify_eclipsing_binary(period, depth_primary, depth_secondary):
    """
    Classify eclipsing binary type.
    
    Args:
        period: Orbital period (days)
        depth_primary: Primary eclipse depth
        depth_secondary: Secondary eclipse depth
    
    Returns:
        eb_type: 'EA' (Algol), 'EB' (Beta Lyrae), or 'EW' (W UMa)
    """
    depth_ratio = depth_secondary / depth_primary if depth_primary > 0 else 0
    
    if period > 1 and depth_ratio < 0.5:
        return "EA"  # Algol: detached, unequal depths
    elif period > 0.5 and depth_ratio > 0.3:
        return "EB"  # Beta Lyrae: semi-detached, continuous variation
    elif period < 1 and depth_ratio > 0.5:
        return "EW"  # W UMa: contact binary, nearly equal depths
    else:
        return "unknown"
```

---

## 7. Metallicity and Abundances

### Metallicity Notation

```
[Fe/H] = log₁₀(N_Fe/N_H)_star - log₁₀(N_Fe/N_H)_sun

Where:
- [Fe/H] = 0: Solar metallicity
- [Fe/H] = -1: 10× less iron than Sun
- [Fe/H] = +0.3: 2× more iron than Sun
```

### Photometric Metallicity Estimates

```python
def estimate_metallicity_from_photometry(bp_rp, g_mag, parallax):
    """
    Rough metallicity estimate from Gaia photometry.
    Based on position relative to solar metallicity isochrone.
    
    Note: This is approximate. Spectroscopy needed for accuracy.
    """
    # Absolute magnitude
    M_G = g_mag + 5 * np.log10(parallax / 1000) + 5
    
    # Expected M_G for solar metallicity MS at this color
    # (simplified polynomial)
    if 0.5 < bp_rp < 2.5:
        M_G_solar = 2.0 + 3.0 * bp_rp + 0.5 * bp_rp**2
        
        # Offset from solar metallicity sequence
        delta_M = M_G - M_G_solar
        
        # Rough calibration: brighter = lower metallicity
        feh_estimate = -0.5 * delta_M
        
        return np.clip(feh_estimate, -2.5, 0.5)
    else:
        return None
```

---

## 8. Distance Measurements

### Parallax

```python
def distance_from_parallax(parallax_mas, parallax_error_mas=None):
    """
    Calculate distance from parallax.
    
    Args:
        parallax_mas: Parallax in milliarcseconds
        parallax_error_mas: Parallax uncertainty
    
    Returns:
        distance_pc: Distance in parsecs
        distance_error: Distance uncertainty (if error provided)
    """
    if parallax_mas <= 0:
        return None, None
    
    distance_pc = 1000 / parallax_mas
    
    if parallax_error_mas is not None:
        # Approximate error propagation
        distance_error = distance_pc * (parallax_error_mas / parallax_mas)
        return distance_pc, distance_error
    
    return distance_pc, None
```

### Spectroscopic Parallax

```python
def spectroscopic_distance(spectral_type, luminosity_class, apparent_mag):
    """
    Estimate distance using spectroscopic parallax.
    
    Args:
        spectral_type: e.g., 'G2'
        luminosity_class: e.g., 'V'
        apparent_mag: Apparent visual magnitude
    
    Returns:
        distance_pc: Estimated distance
    """
    # Absolute magnitude lookup table (partial)
    MV_TABLE = {
        ('O5', 'V'): -5.7,
        ('B0', 'V'): -4.0,
        ('A0', 'V'): +0.6,
        ('F0', 'V'): +2.7,
        ('G0', 'V'): +4.4,
        ('G2', 'V'): +4.8,  # Sun
        ('K0', 'V'): +5.9,
        ('M0', 'V'): +8.8,
        ('K0', 'III'): +0.7,  # Giant
        ('M0', 'III'): -0.4,
    }
    
    key = (spectral_type, luminosity_class)
    if key not in MV_TABLE:
        return None
    
    M_V = MV_TABLE[key]
    
    # Distance modulus: m - M = 5 log(d) - 5
    distance_pc = 10**((apparent_mag - M_V + 5) / 5)
    
    return distance_pc
```

---

## 9. TinyML Stellar Classification

### Lightweight Classifier

```python
import tensorflow as tf

def create_stellar_classifier_tiny(n_features=10, n_classes=7):
    """
    Lightweight stellar classifier for edge deployment.
    
    Features: Temperature indicators (colors), magnitudes, etc.
    Classes: O, B, A, F, G, K, M
    
    Target: <50KB model size
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    return model


def extract_stellar_features(gaia_data):
    """
    Extract features for stellar classification.
    
    Args:
        gaia_data: Dict with Gaia photometry
    
    Returns:
        features: numpy array of features
    """
    features = [
        gaia_data.get('bp_rp', 0),           # Color
        gaia_data.get('phot_g_mean_mag', 0),  # G magnitude
        gaia_data.get('parallax', 0),         # Parallax
        gaia_data.get('bp_rp', 0)**2,        # Color squared
        np.log10(max(gaia_data.get('parallax', 0.1), 0.1)),  # Log parallax
    ]
    
    return np.array(features)
```

---

## 10. Stellar Catalogs

### Key Catalogs for LARUN

| Catalog | Stars | Data | Access |
|---------|-------|------|--------|
| Gaia DR3 | 1.8B | Astrometry, photometry | `astroquery.gaia` |
| TIC | 1.7B | TESS targets | `astroquery.mast` |
| Hipparcos | 118K | Precise parallax | `astroquery.vizier` |
| APOGEE | 700K | Spectra, abundances | SDSS |
| LAMOST | 10M | Spectra | LAMOST archive |

### Query Examples

```python
from astroquery.gaia import Gaia

def query_gaia_stellar_params(ra, dec, radius_deg=0.01):
    """
    Query Gaia for stellar parameters.
    """
    query = f"""
    SELECT 
        source_id, ra, dec, parallax, parallax_error,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        bp_rp, teff_gspphot, logg_gspphot, mh_gspphot,
        radial_velocity, radial_velocity_error
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    AND parallax > 0
    AND teff_gspphot IS NOT NULL
    """
    
    job = Gaia.launch_job(query)
    return job.get_results()
```

---

## References

1. Gray, R. O., & Corbally, C. J. (2009). "Stellar Spectral Classification." Princeton University Press.
2. Flower, P. J. (1996). "Transformations from Theoretical Hertzsprung-Russell Diagrams to Color-Magnitude Diagrams." ApJ, 469, 355.
3. Barnes, S. A. (2010). "A Simple Nonlinear Model for the Rotation of Main-sequence Cool Stars." ApJ, 722, 222.
4. Andrae, R., et al. (2018). "Gaia Data Release 2: First stellar parameters from Apsis." A&A, 616, A8.
5. Pecaut, M. J., & Mamajek, E. E. (2013). "Intrinsic Colors, Temperatures, and Bolometric Corrections of Pre-main-sequence Stars." ApJS, 208, 9.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
