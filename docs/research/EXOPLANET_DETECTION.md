# Exoplanet Detection Methods - Research Documentation

## Overview

This document covers the scientific methods for detecting exoplanets from photometric data, with focus on transit detection algorithms suitable for LARUN implementation.

---

## 1. Transit Photometry Fundamentals

### What is a Transit?
A planetary transit occurs when a planet passes in front of its host star as seen from Earth, causing a temporary dimming of the star's light.

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Transit Depth | δ | Fractional flux decrease | 0.01% - 3% |
| Transit Duration | T₁₄ | Total duration | 1 - 24 hours |
| Orbital Period | P | Time between transits | 0.5 - 1000 days |
| Impact Parameter | b | Projected distance from center | 0 - 1 |
| Ingress/Egress Time | τ | Time for partial transit | Minutes |

### Transit Depth Formula
```
δ = (Rp / Rs)²

Where:
- δ = transit depth (fractional)
- Rp = planet radius
- Rs = star radius
```

### Example Transit Depths

| Planet Type | Radius (R⊕) | Transit Depth (Sun-like) |
|-------------|-------------|--------------------------|
| Earth | 1.0 | 0.0084% (84 ppm) |
| Super-Earth | 1.5-2.0 | 0.019-0.034% |
| Neptune | 4.0 | 0.13% |
| Jupiter | 11.2 | 1.0% |

---

## 2. Box Least Squares (BLS) Algorithm

### Theory
BLS is the gold standard for periodic transit detection. It models the transit as a box-shaped dip in the light curve.

### Algorithm Steps
1. Define period grid P_i
2. For each period, phase-fold the data
3. For each phase, fit a box model
4. Calculate BLS statistic (signal residue)
5. Find peak in periodogram

### Mathematical Formulation
```
SR = max[(s² / (r(1-r)))]^0.5

Where:
- SR = Signal Residue (BLS statistic)
- s = sum of residuals in transit
- r = fractional transit duration (q/P)
```

### Python Implementation
```python
import numpy as np
from scipy import optimize

def bls_periodogram(time, flux, periods, durations):
    """
    Compute BLS periodogram.
    
    Args:
        time: Time array (days)
        flux: Normalized flux array
        periods: Period grid to search (days)
        durations: Transit duration grid (fraction of period)
    
    Returns:
        power: BLS power at each period
        best_params: Dict with best period, epoch, duration, depth
    """
    power = np.zeros(len(periods))
    
    for i, period in enumerate(periods):
        # Phase fold
        phase = (time % period) / period
        
        best_sr = 0
        for q in durations:
            # Slide box across phase
            for phi in np.linspace(0, 1-q, 100):
                in_transit = (phase >= phi) & (phase < phi + q)
                out_transit = ~in_transit
                
                if np.sum(in_transit) < 3:
                    continue
                
                # Mean flux in and out of transit
                f_in = np.mean(flux[in_transit])
                f_out = np.mean(flux[out_transit])
                
                # BLS statistic
                r = np.sum(in_transit) / len(flux)
                s = np.sum(flux[in_transit] - f_out)
                sr = np.abs(s) / np.sqrt(r * (1 - r) * len(flux))
                
                if sr > best_sr:
                    best_sr = sr
        
        power[i] = best_sr
    
    return power
```

### Optimized BLS (astropy)
```python
from astropy.timeseries import BoxLeastSquares

def compute_bls(time, flux, flux_err=None):
    """Compute BLS periodogram using astropy."""
    # Create BLS model
    bls = BoxLeastSquares(time, flux, dy=flux_err)
    
    # Define period grid
    periods = np.linspace(0.5, 50, 10000)
    
    # Compute periodogram
    periodogram = bls.power(periods, duration=np.linspace(0.01, 0.1, 10))
    
    # Find best period
    best_idx = np.argmax(periodogram.power)
    best_period = periodogram.period[best_idx]
    best_power = periodogram.power[best_idx]
    
    # Get transit parameters
    stats = bls.compute_stats(best_period, 
                               periodogram.duration[best_idx],
                               periodogram.transit_time[best_idx])
    
    return {
        'period': best_period,
        'power': best_power,
        'depth': stats['depth'],
        'duration': stats['duration'],
        'snr': stats['snr'],
        't0': periodogram.transit_time[best_idx]
    }
```

---

## 3. Phase Folding

### Theory
Phase folding collapses multiple transits onto a single orbital phase, improving signal-to-noise.

### Formula
```
φ = ((t - t₀) mod P) / P

Where:
- φ = orbital phase (0-1)
- t = observation time
- t₀ = reference epoch (mid-transit)
- P = orbital period
```

### Implementation
```python
def phase_fold(time, flux, period, t0=0):
    """
    Phase fold light curve.
    
    Args:
        time: Time array
        flux: Flux array
        period: Orbital period
        t0: Reference epoch (mid-transit time)
    
    Returns:
        phase: Phase array (-0.5 to 0.5, transit at 0)
        folded_flux: Flux sorted by phase
    """
    # Calculate phase
    phase = ((time - t0) % period) / period
    
    # Center transit at phase 0
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    
    return phase[sort_idx], flux[sort_idx]


def bin_phase_curve(phase, flux, n_bins=100):
    """Bin phase-folded data."""
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    binned_flux = np.zeros(n_bins)
    binned_err = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.mean(flux[mask])
            binned_err[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
    
    return bin_centers, binned_flux, binned_err
```

---

## 4. Transit Model Fitting

### Mandel-Agol Model
The standard analytic transit model accounting for limb darkening.

### Parameters
| Parameter | Description | Typical Prior |
|-----------|-------------|---------------|
| Rp/Rs | Planet-to-star radius ratio | U(0, 0.3) |
| a/Rs | Semi-major axis / stellar radius | U(1, 100) |
| i | Orbital inclination (degrees) | U(70, 90) |
| e | Eccentricity | Fixed at 0 or U(0, 0.5) |
| ω | Argument of periastron | U(0, 360) |
| u1, u2 | Limb darkening coefficients | From tables |
| t0 | Mid-transit time | Gaussian |
| P | Period | Gaussian |

### Implementation with batman
```python
import batman
import numpy as np
from scipy.optimize import minimize

def create_transit_model(params, time):
    """
    Create transit model using batman.
    
    Args:
        params: Dict with t0, per, rp, a, inc, ecc, w, u1, u2
        time: Time array
    
    Returns:
        model_flux: Normalized flux
    """
    # Initialize batman parameters
    bm_params = batman.TransitParams()
    bm_params.t0 = params['t0']        # Mid-transit time
    bm_params.per = params['per']      # Orbital period
    bm_params.rp = params['rp']        # Rp/Rs
    bm_params.a = params['a']          # a/Rs
    bm_params.inc = params['inc']      # Inclination
    bm_params.ecc = params['ecc']      # Eccentricity
    bm_params.w = params['w']          # Argument of periastron
    bm_params.u = [params['u1'], params['u2']]  # Limb darkening
    bm_params.limb_dark = "quadratic"
    
    # Generate model
    m = batman.TransitModel(bm_params, time)
    return m.light_curve(bm_params)


def fit_transit(time, flux, flux_err, initial_params):
    """
    Fit transit model to data.
    """
    def neg_log_likelihood(theta):
        params = {
            't0': theta[0],
            'per': initial_params['per'],  # Fixed
            'rp': theta[1],
            'a': theta[2],
            'inc': theta[3],
            'ecc': 0,
            'w': 90,
            'u1': 0.3,
            'u2': 0.2
        }
        model = create_transit_model(params, time)
        residuals = (flux - model) / flux_err
        return 0.5 * np.sum(residuals**2)
    
    # Initial guess
    x0 = [initial_params['t0'], initial_params['rp'], 
          initial_params['a'], initial_params['inc']]
    
    # Optimize
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead')
    
    return result
```

---

## 5. Signal-to-Noise Ratio (SNR)

### Transit SNR Formula
```
SNR = δ × √(N_transit × N_points)  / σ

Where:
- δ = transit depth
- N_transit = number of observed transits
- N_points = number of data points per transit
- σ = photometric precision
```

### Implementation
```python
def calculate_transit_snr(depth, duration, period, total_time, cadence, noise):
    """
    Calculate expected transit SNR.
    
    Args:
        depth: Transit depth (fractional)
        duration: Transit duration (hours)
        period: Orbital period (days)
        total_time: Total observation time (days)
        cadence: Observation cadence (minutes)
        noise: Photometric noise (ppm)
    
    Returns:
        snr: Signal-to-noise ratio
    """
    # Number of transits observed
    n_transits = total_time / period
    
    # Number of in-transit points per transit
    n_points = (duration * 60) / cadence
    
    # Total in-transit points
    n_total = n_transits * n_points
    
    # SNR
    snr = (depth * 1e6) * np.sqrt(n_total) / noise
    
    return snr
```

---

## 6. False Positive Identification

### Common False Positives

| Type | Description | Identification |
|------|-------------|----------------|
| Eclipsing Binary | Background EB | Depth difference odd/even |
| Grazing EB | Partial eclipse | V-shaped transit |
| Blended EB | EB in aperture | Centroid shift |
| Stellar Variability | Spots, pulsations | Periodogram analysis |
| Systematic | Instrumental | Correlation with centroid |

### Odd-Even Test
```python
def odd_even_test(time, flux, period, t0):
    """
    Check for depth difference between odd and even transits.
    """
    # Calculate transit number
    transit_num = np.floor((time - t0) / period)
    
    odd_transits = transit_num % 2 == 1
    even_transits = transit_num % 2 == 0
    
    # Phase fold separately
    phase_odd, flux_odd = phase_fold(time[odd_transits], flux[odd_transits], period, t0)
    phase_even, flux_even = phase_fold(time[even_transits], flux[even_transits], period, t0)
    
    # Calculate depths
    in_transit = np.abs(phase_odd) < 0.05
    depth_odd = 1 - np.mean(flux_odd[in_transit])
    
    in_transit = np.abs(phase_even) < 0.05
    depth_even = 1 - np.mean(flux_even[in_transit])
    
    # Significance of difference
    depth_diff = np.abs(depth_odd - depth_even)
    combined_err = np.sqrt(np.var(flux_odd[in_transit]) + np.var(flux_even[in_transit]))
    significance = depth_diff / combined_err
    
    return {
        'depth_odd': depth_odd,
        'depth_even': depth_even,
        'difference': depth_diff,
        'significance': significance,
        'is_eb': significance > 3  # If > 3σ, likely EB
    }
```

### Centroid Analysis
```python
def centroid_shift(flux, centroid_x, centroid_y, in_transit_mask):
    """
    Check for centroid shift during transit.
    A shift indicates the transit is not on the target star.
    """
    # Centroid during transit
    x_in = np.mean(centroid_x[in_transit_mask])
    y_in = np.mean(centroid_y[in_transit_mask])
    
    # Centroid out of transit
    x_out = np.mean(centroid_x[~in_transit_mask])
    y_out = np.mean(centroid_y[~in_transit_mask])
    
    # Shift
    shift = np.sqrt((x_in - x_out)**2 + (y_in - y_out)**2)
    
    # Significance
    x_std = np.std(centroid_x[~in_transit_mask])
    y_std = np.std(centroid_y[~in_transit_mask])
    significance = shift / np.sqrt(x_std**2 + y_std**2)
    
    return {
        'shift_pixels': shift,
        'significance': significance,
        'is_blend': significance > 3
    }
```

---

## 7. Multi-Planet Detection

### TTV (Transit Timing Variations)
Gravitational interactions between planets cause deviations from strict periodicity.

```python
def detect_ttv(transit_times, period):
    """
    Detect transit timing variations.
    """
    # Expected transit times
    n_transits = len(transit_times)
    expected = transit_times[0] + np.arange(n_transits) * period
    
    # O-C (Observed minus Calculated)
    oc = transit_times - expected
    
    # Fit linear ephemeris
    coeffs = np.polyfit(np.arange(n_transits), transit_times, 1)
    refined_period = coeffs[0]
    refined_t0 = coeffs[1]
    
    # Residuals
    residuals = transit_times - (refined_t0 + np.arange(n_transits) * refined_period)
    
    # TTV amplitude
    ttv_amplitude = np.std(residuals) * 24 * 60  # in minutes
    
    return {
        'oc_minutes': oc * 24 * 60,
        'residuals_minutes': residuals * 24 * 60,
        'ttv_amplitude': ttv_amplitude,
        'has_ttv': ttv_amplitude > 1  # > 1 minute
    }
```

### Iterative Multi-Planet Search
```python
def search_additional_planets(time, flux, known_periods):
    """
    Search for additional planets after removing known transits.
    """
    residual_flux = flux.copy()
    
    # Remove known transits
    for period in known_periods:
        # Fit and remove transit model
        # ... (simplified)
        pass
    
    # BLS search on residuals
    bls_result = compute_bls(time, residual_flux)
    
    if bls_result['snr'] > 7:
        return {
            'new_planet': True,
            'period': bls_result['period'],
            'snr': bls_result['snr']
        }
    
    return {'new_planet': False}
```

---

## 8. Planet Characterization

### Radius Calculation
```python
def calculate_planet_radius(depth, stellar_radius_solar):
    """
    Calculate planet radius from transit depth.
    
    Args:
        depth: Transit depth (fractional)
        stellar_radius_solar: Stellar radius in solar radii
    
    Returns:
        planet_radius_earth: Planet radius in Earth radii
    """
    SOLAR_RADIUS_EARTH = 109.2  # Solar radii in Earth radii
    
    # Rp/Rs = sqrt(depth)
    rp_rs = np.sqrt(depth)
    
    # Rp in Earth radii
    planet_radius = rp_rs * stellar_radius_solar * SOLAR_RADIUS_EARTH
    
    return planet_radius
```

### Equilibrium Temperature
```python
def calculate_equilibrium_temp(stellar_teff, stellar_radius, semi_major_axis, albedo=0.3):
    """
    Calculate planet equilibrium temperature.
    
    Args:
        stellar_teff: Stellar effective temperature (K)
        stellar_radius: Stellar radius (solar radii)
        semi_major_axis: Orbital distance (AU)
        albedo: Bond albedo (default 0.3)
    
    Returns:
        T_eq: Equilibrium temperature (K)
    """
    # Convert units
    Rs_AU = stellar_radius * 0.00465  # Solar radius in AU
    
    # Equilibrium temperature formula
    T_eq = stellar_teff * np.sqrt(Rs_AU / (2 * semi_major_axis)) * (1 - albedo)**0.25
    
    return T_eq
```

### Habitable Zone
```python
def habitable_zone(stellar_teff, stellar_luminosity):
    """
    Calculate habitable zone boundaries.
    
    Based on Kopparapu et al. (2013)
    
    Args:
        stellar_teff: Stellar temperature (K)
        stellar_luminosity: Stellar luminosity (solar luminosities)
    
    Returns:
        dict: Inner and outer HZ boundaries (AU)
    """
    # Coefficients for moist greenhouse (inner) and maximum greenhouse (outer)
    S_eff_sun = {
        'inner': 1.0140,
        'outer': 0.3438
    }
    
    # Temperature dependence
    T_star = stellar_teff - 5780
    
    S_eff = {}
    for key in ['inner', 'outer']:
        # Polynomial correction (simplified)
        S_eff[key] = S_eff_sun[key] + 8.1774e-5 * T_star + 1.7063e-9 * T_star**2
    
    # Distance = sqrt(L / S_eff)
    hz = {
        'inner_au': np.sqrt(stellar_luminosity / S_eff['inner']),
        'outer_au': np.sqrt(stellar_luminosity / S_eff['outer'])
    }
    
    return hz
```

---

## 9. TinyML Considerations

### Model Architecture for Transit Detection
```python
import tensorflow as tf

def create_transit_detector_tinyml():
    """
    Create TinyML-compatible transit detector.
    Target: <100KB model size
    """
    model = tf.keras.Sequential([
        # Input: 1024 flux points
        tf.keras.layers.Conv1D(8, 15, activation='relu', input_shape=(1024, 1)),
        tf.keras.layers.MaxPooling1D(4),
        
        tf.keras.layers.Conv1D(16, 7, activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Transit probability
    ])
    
    return model
```

### Feature Engineering for Small Models
```python
def extract_transit_features(flux, time):
    """
    Extract features for lightweight classification.
    """
    features = []
    
    # Statistical features
    features.append(np.mean(flux))
    features.append(np.std(flux))
    features.append(np.min(flux))
    features.append(np.max(flux))
    features.append(np.percentile(flux, 5))
    features.append(np.percentile(flux, 95))
    
    # Shape features
    features.append(np.sum(flux < np.mean(flux) - 2*np.std(flux)))  # Dip count
    
    # Autocorrelation
    acf = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
    features.append(acf[len(acf)//2 + 1] / acf[len(acf)//2])
    
    return np.array(features)
```

---

## References

1. Kovács, G., Zucker, S., & Mazeh, T. (2002). "A box-fitting algorithm in the search for periodic transits." A&A, 391, 369.
2. Mandel, K., & Agol, E. (2002). "Analytic Light Curves for Planetary Transit Searches." ApJL, 580, L171.
3. Holman, M. J., & Murray, N. W. (2005). "The Use of Transit Timing to Detect Terrestrial-Mass Extrasolar Planets." Science, 307, 1288.
4. Kopparapu, R. K., et al. (2013). "Habitable Zones around Main-sequence Stars." ApJ, 765, 131.
5. Christiansen, J. L., et al. (2012). "The Derivation, Properties, and Value of Kepler's Combined Differential Photometric Precision." PASP, 124, 1279.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
