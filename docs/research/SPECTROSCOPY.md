# Spectroscopy - Research Documentation

## Overview

This document covers spectroscopic analysis methods for LARUN, including spectral classification, feature extraction, and analysis techniques.

---

## 1. Spectroscopy Fundamentals

### Electromagnetic Spectrum Regions

| Region | Wavelength | Astronomy Applications |
|--------|------------|------------------------|
| Gamma Ray | <0.01 nm | High-energy phenomena |
| X-Ray | 0.01-10 nm | Compact objects, hot gas |
| UV | 10-400 nm | Hot stars, emission lines |
| Optical | 400-700 nm | Stars, galaxies (visible) |
| NIR | 700-2500 nm | Cool stars, dust |
| MIR | 2.5-25 μm | Dust, molecules |
| FIR | 25-350 μm | Cold dust |
| Radio | >1 mm | Gas, synchrotron |

### Spectral Resolution

```
R = λ / Δλ

Where:
- R = spectral resolution
- λ = wavelength
- Δλ = smallest resolvable wavelength difference

Categories:
- Low: R < 1000 (broadband)
- Medium: R = 1000-10000
- High: R = 10000-100000
- Very High: R > 100000 (exoplanet atmospheres)
```

---

## 2. Light Curve as "Spectroscopy in Time"

### Spectral Information from Photometry

LARUN primarily works with photometric time-series, which contain spectral information:

```python
def extract_spectral_features_from_lightcurve(time, flux, flux_err=None):
    """
    Extract quasi-spectral features from light curve.
    
    Transit depth varies with wavelength due to:
    - Limb darkening
    - Atmospheric absorption
    - Rayleigh scattering
    
    Returns features useful for classification.
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(flux)
    features['std'] = np.std(flux)
    features['median'] = np.median(flux)
    
    # Variability metrics
    features['mad'] = np.median(np.abs(flux - np.median(flux)))
    features['iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)
    
    # Shape metrics
    features['skewness'] = scipy.stats.skew(flux)
    features['kurtosis'] = scipy.stats.kurtosis(flux)
    
    # Percentiles
    for p in [5, 10, 25, 75, 90, 95]:
        features[f'percentile_{p}'] = np.percentile(flux, p)
    
    # Time-domain features
    diff = np.diff(flux)
    features['diff_std'] = np.std(diff)
    features['diff_mean'] = np.mean(np.abs(diff))
    
    # Autocorrelation
    acf = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    features['acf_1'] = acf[1] if len(acf) > 1 else 0
    features['acf_10'] = acf[10] if len(acf) > 10 else 0
    
    return features
```

### Multi-Band Photometry

```python
def analyze_multiband(data_dict):
    """
    Analyze multi-band photometric data.
    
    Args:
        data_dict: Dict with band names as keys, (time, flux) as values
    
    Returns:
        color_info: Color indices and variations
    """
    results = {}
    
    bands = list(data_dict.keys())
    
    for i, band1 in enumerate(bands):
        for band2 in bands[i+1:]:
            time1, flux1 = data_dict[band1]
            time2, flux2 = data_dict[band2]
            
            # Interpolate to common time grid
            common_time = np.union1d(time1, time2)
            flux1_interp = np.interp(common_time, time1, flux1)
            flux2_interp = np.interp(common_time, time2, flux2)
            
            # Color (magnitude difference)
            mag1 = -2.5 * np.log10(flux1_interp)
            mag2 = -2.5 * np.log10(flux2_interp)
            color = mag1 - mag2
            
            results[f'{band1}-{band2}'] = {
                'mean_color': np.mean(color),
                'color_variation': np.std(color),
                'color_timeseries': (common_time, color)
            }
    
    return results
```

---

## 3. Spectral Classification

### Template Matching

```python
def spectral_template_matching(wavelength, flux, templates):
    """
    Match observed spectrum to template library.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array (normalized)
        templates: Dict of {spectral_type: (wave, flux)}
    
    Returns:
        best_match: Best matching spectral type
        scores: Dict of chi-squared scores
    """
    scores = {}
    
    for spec_type, (temp_wave, temp_flux) in templates.items():
        # Interpolate template to observed wavelength grid
        temp_interp = np.interp(wavelength, temp_wave, temp_flux)
        
        # Normalize
        temp_interp = temp_interp / np.median(temp_interp)
        flux_norm = flux / np.median(flux)
        
        # Chi-squared
        chi2 = np.sum((flux_norm - temp_interp)**2)
        scores[spec_type] = chi2
    
    best_match = min(scores, key=scores.get)
    
    return best_match, scores
```

### CNN for Spectral Classification

```python
def create_spectral_classifier(input_length=4000, num_classes=7):
    """
    CNN for stellar spectral classification.
    
    Input: Normalized spectrum (4000 wavelength points)
    Output: Spectral class (O, B, A, F, G, K, M)
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_length, 1)),
        
        # Conv blocks
        tf.keras.layers.Conv1D(16, 15, activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv1D(32, 7, activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Classification head
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

---

## 4. Spectral Line Analysis

### Line Detection

```python
def detect_spectral_lines(wavelength, flux, threshold=3):
    """
    Detect absorption/emission lines in spectrum.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        threshold: Detection threshold in sigma
    
    Returns:
        lines: List of detected lines with properties
    """
    from scipy import ndimage
    from scipy.signal import find_peaks
    
    # Normalize flux
    continuum = ndimage.median_filter(flux, size=50)
    normalized = flux / continuum
    
    # Find absorption lines (dips)
    inverted = 1 - normalized
    peaks_abs, properties = find_peaks(inverted, height=threshold * np.std(normalized))
    
    # Find emission lines (peaks)
    peaks_em, properties = find_peaks(normalized - 1, height=threshold * np.std(normalized))
    
    lines = []
    
    for idx in peaks_abs:
        lines.append({
            'wavelength': wavelength[idx],
            'type': 'absorption',
            'depth': 1 - normalized[idx],
            'ew': estimate_equivalent_width(wavelength, normalized, idx)
        })
    
    for idx in peaks_em:
        lines.append({
            'wavelength': wavelength[idx],
            'type': 'emission',
            'height': normalized[idx] - 1,
            'ew': estimate_equivalent_width(wavelength, normalized, idx)
        })
    
    return lines


def estimate_equivalent_width(wavelength, normalized_flux, line_idx, window=20):
    """
    Estimate equivalent width of spectral line.
    
    EW = ∫ (1 - F/F_c) dλ
    """
    start = max(0, line_idx - window)
    end = min(len(wavelength), line_idx + window)
    
    wave_segment = wavelength[start:end]
    flux_segment = normalized_flux[start:end]
    
    # Numerical integration
    ew = np.trapz(1 - flux_segment, wave_segment)
    
    return ew
```

### Common Spectral Lines

```python
# Important astronomical spectral lines (Angstroms)

SPECTRAL_LINES = {
    # Hydrogen Balmer series
    'H_alpha': 6562.8,
    'H_beta': 4861.3,
    'H_gamma': 4340.5,
    'H_delta': 4101.7,
    
    # Calcium
    'Ca_K': 3933.7,
    'Ca_H': 3968.5,
    'Ca_triplet_1': 8498.0,
    'Ca_triplet_2': 8542.1,
    'Ca_triplet_3': 8662.1,
    
    # Sodium
    'Na_D1': 5895.9,
    'Na_D2': 5889.9,
    
    # Magnesium
    'Mg_b': 5183.6,
    
    # Iron (many lines)
    'Fe_5270': 5270.0,
    'Fe_5335': 5335.0,
    
    # Oxygen
    'O_triplet': 7774.0,  # Actually 7771.9, 7774.2, 7775.4
    '[O_III]_5007': 5006.8,
    '[O_III]_4959': 4958.9,
    
    # Nitrogen
    '[N_II]_6583': 6583.4,
    '[N_II]_6548': 6548.0,
    
    # Lithium (important for young stars)
    'Li_6707': 6707.8,
    
    # Helium
    'He_I_5876': 5875.6,
    'He_II_4686': 4685.7,
}


def identify_lines(detected_lines, tolerance=5):
    """
    Identify detected lines by matching to known lines.
    
    Args:
        detected_lines: List of detected line dicts
        tolerance: Wavelength tolerance in Angstroms
    
    Returns:
        identified: List of lines with identifications
    """
    identified = []
    
    for line in detected_lines:
        wave = line['wavelength']
        
        # Find closest known line
        best_match = None
        best_diff = tolerance
        
        for name, lab_wave in SPECTRAL_LINES.items():
            diff = abs(wave - lab_wave)
            if diff < best_diff:
                best_diff = diff
                best_match = name
        
        line['identification'] = best_match
        line['velocity'] = None
        
        if best_match is not None:
            # Calculate radial velocity
            lab_wave = SPECTRAL_LINES[best_match]
            v_over_c = (wave - lab_wave) / lab_wave
            line['velocity'] = v_over_c * 299792.458  # km/s
        
        identified.append(line)
    
    return identified
```

---

## 5. Radial Velocity

### Cross-Correlation Method

```python
def measure_radial_velocity(observed_wave, observed_flux, 
                            template_wave, template_flux,
                            velocity_range=(-500, 500)):
    """
    Measure radial velocity via cross-correlation.
    
    Args:
        observed_wave: Observed wavelength array
        observed_flux: Observed flux array
        template_wave: Template wavelength array
        template_flux: Template flux array
        velocity_range: Velocity search range (km/s)
    
    Returns:
        rv: Radial velocity (km/s)
        rv_err: Velocity uncertainty
        ccf: Cross-correlation function
    """
    c = 299792.458  # km/s
    
    # Velocity grid
    velocities = np.linspace(velocity_range[0], velocity_range[1], 1000)
    
    ccf = np.zeros_like(velocities)
    
    for i, v in enumerate(velocities):
        # Doppler shift template
        doppler_factor = 1 + v / c
        shifted_wave = template_wave * doppler_factor
        
        # Interpolate to observed wavelength grid
        shifted_flux = np.interp(observed_wave, shifted_wave, template_flux)
        
        # Cross-correlation
        ccf[i] = np.corrcoef(observed_flux, shifted_flux)[0, 1]
    
    # Find peak
    peak_idx = np.argmax(ccf)
    rv = velocities[peak_idx]
    
    # Fit parabola for sub-pixel precision
    if 1 < peak_idx < len(velocities) - 2:
        x = velocities[peak_idx-1:peak_idx+2]
        y = ccf[peak_idx-1:peak_idx+2]
        coeffs = np.polyfit(x, y, 2)
        rv = -coeffs[1] / (2 * coeffs[0])
    
    # Estimate uncertainty from CCF width
    half_max = (ccf.max() + ccf.min()) / 2
    above_half = velocities[ccf > half_max]
    if len(above_half) > 1:
        rv_err = (above_half[-1] - above_half[0]) / 2.35 / np.sqrt(ccf.max() / (1 - ccf.max()))
    else:
        rv_err = 10  # Default uncertainty
    
    return rv, rv_err, (velocities, ccf)
```

### Bisector Analysis

```python
def line_bisector(wavelength, flux, line_center, window=5):
    """
    Calculate line bisector for asymmetry analysis.
    
    Asymmetric bisectors can indicate:
    - Stellar activity (spots)
    - Convection
    - Blended binaries
    """
    # Extract line region
    mask = np.abs(wavelength - line_center) < window
    wave = wavelength[mask]
    flux_line = flux[mask]
    
    # Normalize
    continuum = np.max(flux_line)
    flux_norm = flux_line / continuum
    
    # Find line core
    core_idx = np.argmin(flux_norm)
    
    # Calculate bisector at different flux levels
    flux_levels = np.linspace(flux_norm[core_idx], 0.95, 20)
    bisector_velocities = []
    
    c = 299792.458  # km/s
    
    for level in flux_levels:
        # Find intersections with flux level
        above = flux_norm > level
        crossings = np.where(np.diff(above.astype(int)))[0]
        
        if len(crossings) >= 2:
            # Blue and red wings
            blue_wave = wave[crossings[0]]
            red_wave = wave[crossings[-1]]
            
            # Bisector wavelength
            bisector_wave = (blue_wave + red_wave) / 2
            
            # Convert to velocity
            v = (bisector_wave - line_center) / line_center * c
            bisector_velocities.append((level, v))
    
    return bisector_velocities
```

---

## 6. Atmospheric Parameters from Spectra

### Effective Temperature from Line Ratios

```python
def estimate_teff_from_lines(spectrum, wavelength):
    """
    Estimate temperature from spectral line ratios.
    
    Uses temperature-sensitive line ratios.
    """
    # Example: H-alpha to H-beta ratio
    ha_ew = measure_ew_at_line(wavelength, spectrum, 6562.8)
    hb_ew = measure_ew_at_line(wavelength, spectrum, 4861.3)
    
    if ha_ew > 0 and hb_ew > 0:
        ratio = ha_ew / hb_ew
        # Empirical calibration (simplified)
        teff = 8000 - 1500 * np.log10(ratio)
        return teff
    
    return None


def estimate_logg_from_pressure_lines(spectrum, wavelength):
    """
    Estimate surface gravity from pressure-broadened lines.
    
    Uses Ca II and Mg I lines which are sensitive to log g.
    """
    # Calcium K line width
    ca_k_ew = measure_ew_at_line(wavelength, spectrum, 3933.7)
    
    # Simplified relation
    if ca_k_ew > 0:
        logg = 4.5 - 0.5 * np.log10(ca_k_ew)
        return np.clip(logg, 0, 5.5)
    
    return None
```

---

## 7. Feature Extraction for TinyML

### Spectral Feature Vector

```python
def extract_spectral_features_tinyml(wavelength, flux, n_features=32):
    """
    Extract compact feature vector for TinyML classification.
    
    Args:
        wavelength: Wavelength array
        flux: Flux array
        n_features: Number of features to extract
    
    Returns:
        features: Numpy array of features
    """
    # Normalize
    flux_norm = flux / np.median(flux)
    
    features = []
    
    # 1. Binned flux (8 features)
    n_bins = 8
    bins = np.array_split(flux_norm, n_bins)
    for b in bins:
        features.append(np.median(b))
    
    # 2. Key spectral indices (8 features)
    indices = [
        (4000, 4100),  # Ca II H&K break
        (4300, 4400),  # G-band
        (4800, 4900),  # H-beta region
        (5150, 5250),  # Mg I
        (5850, 5950),  # Na D
        (6500, 6600),  # H-alpha region
        (7500, 7600),  # O2 band
        (8500, 8700),  # Ca triplet
    ]
    
    for low, high in indices:
        mask = (wavelength >= low) & (wavelength <= high)
        if np.sum(mask) > 0:
            features.append(np.mean(flux_norm[mask]))
        else:
            features.append(1.0)
    
    # 3. Statistical features (8 features)
    features.append(np.std(flux_norm))
    features.append(scipy.stats.skew(flux_norm))
    features.append(scipy.stats.kurtosis(flux_norm))
    features.append(np.percentile(flux_norm, 10))
    features.append(np.percentile(flux_norm, 90))
    
    # Gradient features
    gradient = np.gradient(flux_norm)
    features.append(np.std(gradient))
    features.append(np.max(np.abs(gradient)))
    features.append(np.mean(gradient**2))
    
    # 4. Color-like indices (8 features)
    # Split into pseudo-bands
    n_pseudo = 4
    pseudo_bands = np.array_split(flux_norm, n_pseudo)
    for i in range(n_pseudo - 1):
        color = np.median(pseudo_bands[i]) / np.median(pseudo_bands[i+1])
        features.append(color)
    
    # Pad or truncate to n_features
    features = np.array(features[:n_features])
    if len(features) < n_features:
        features = np.pad(features, (0, n_features - len(features)))
    
    return features.astype(np.float32)
```

### Lightweight Spectral Classifier

```python
def create_tinyml_spectral_classifier(n_features=32, n_classes=7):
    """
    Tiny classifier for spectral classification.
    
    Target: <50KB model size
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    return model
```

---

## 8. Spectral Data Sources

### SDSS Spectra

```python
from astroquery.sdss import SDSS

def get_sdss_spectrum(ra, dec, radius=2):
    """
    Query SDSS for spectrum at position.
    
    Args:
        ra, dec: Position in degrees
        radius: Search radius in arcsec
    
    Returns:
        wavelength, flux arrays if found
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    coord = SkyCoord(ra, dec, unit='deg')
    
    # Search for spectroscopic observation
    result = SDSS.query_region(coord, spectro=True, radius=radius*u.arcsec)
    
    if result is not None and len(result) > 0:
        # Download spectrum
        spectra = SDSS.get_spectra(matches=result)
        
        if spectra:
            spec = spectra[0]
            wavelength = 10**spec[1].data['loglam']
            flux = spec[1].data['flux']
            return wavelength, flux
    
    return None, None
```

### LAMOST Spectra

```python
def query_lamost(ra, dec, radius=3):
    """
    Query LAMOST DR for spectra.
    """
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    coord = SkyCoord(ra, dec, unit='deg')
    
    # Query LAMOST catalog
    result = Vizier.query_region(coord, radius=radius*u.arcsec, 
                                  catalog='V/164')  # LAMOST DR8
    
    return result
```

---

## References

1. Gray, R. O., & Corbally, C. J. (2009). "Stellar Spectral Classification." Princeton University Press.
2. Tonry, J., & Davis, M. (1979). "A survey of galaxy redshifts. I. Data reduction techniques." AJ, 84, 1511.
3. Prugniel, P., & Soubiran, C. (2001). "A database of high and medium-resolution stellar spectra." A&A, 369, 1048.
4. Allende Prieto, C. (2016). "Stellar spectroscopy." arXiv:1606.08242.
5. Bochanski, J. J., et al. (2007). "The Luminosity and Mass Functions of Low-Mass Stars in the Galactic Disk." AJ, 133, 531.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
