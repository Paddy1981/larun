# Image Processing - Research Documentation

## Overview

This document covers astronomical image processing techniques for LARUN's image-based analysis capabilities.

---

## 1. Astronomical Image Fundamentals

### CCD/CMOS Basics

```
Raw Image = (Science + Dark + Bias) × Flat + Noise

Where:
- Science: Photons from celestial sources
- Dark: Thermal electrons
- Bias: Electronic offset
- Flat: Pixel sensitivity variations
- Noise: Random fluctuations
```

### Image Processing Pipeline

```
Raw Frame
    ↓
Bias Subtraction → Remove electronic offset
    ↓
Dark Subtraction → Remove thermal signal
    ↓
Flat Fielding → Correct sensitivity variations
    ↓
Cosmic Ray Removal → Remove particle hits
    ↓
Calibrated Science Frame
```

---

## 2. Basic Preprocessing

### Bias Subtraction

```python
import numpy as np
from astropy.io import fits

def create_master_bias(bias_frames):
    """
    Create master bias from multiple bias frames.
    
    Args:
        bias_frames: List of bias frame file paths
    
    Returns:
        master_bias: Median combined bias
    """
    biases = []
    for frame in bias_frames:
        data = fits.getdata(frame)
        biases.append(data)
    
    # Median combine to reject outliers
    master_bias = np.median(biases, axis=0)
    
    return master_bias


def subtract_bias(science_frame, master_bias):
    """Subtract bias from science frame."""
    return science_frame - master_bias
```

### Dark Subtraction

```python
def create_master_dark(dark_frames, master_bias, exposure_time):
    """
    Create master dark frame.
    
    Args:
        dark_frames: List of dark frame paths
        master_bias: Master bias frame
        exposure_time: Exposure time of darks
    
    Returns:
        master_dark: Bias-subtracted, normalized dark
    """
    darks = []
    for frame in dark_frames:
        data = fits.getdata(frame)
        # Subtract bias
        data = data - master_bias
        darks.append(data)
    
    # Median combine
    master_dark = np.median(darks, axis=0)
    
    # Normalize to 1 second
    master_dark = master_dark / exposure_time
    
    return master_dark


def subtract_dark(science_frame, master_dark, exposure_time):
    """Subtract scaled dark from science frame."""
    return science_frame - (master_dark * exposure_time)
```

### Flat Fielding

```python
def create_master_flat(flat_frames, master_bias, master_dark):
    """
    Create master flat field.
    
    Args:
        flat_frames: List of flat field paths
        master_bias: Master bias frame
        master_dark: Master dark frame (per second)
    
    Returns:
        master_flat: Normalized flat field
    """
    flats = []
    for frame in flat_frames:
        data = fits.getdata(frame)
        header = fits.getheader(frame)
        exp_time = header.get('EXPTIME', 1)
        
        # Calibrate
        data = data - master_bias - (master_dark * exp_time)
        
        # Normalize to median
        data = data / np.median(data)
        
        flats.append(data)
    
    # Median combine
    master_flat = np.median(flats, axis=0)
    
    return master_flat


def apply_flat(science_frame, master_flat):
    """Apply flat field correction."""
    return science_frame / master_flat
```

### Complete Calibration

```python
def calibrate_image(science_path, master_bias, master_dark, master_flat):
    """
    Full calibration pipeline.
    """
    # Load science frame
    science = fits.getdata(science_path)
    header = fits.getheader(science_path)
    exp_time = header.get('EXPTIME', 1)
    
    # Calibrate
    calibrated = science - master_bias
    calibrated = calibrated - (master_dark * exp_time)
    calibrated = calibrated / master_flat
    
    return calibrated
```

---

## 3. Cosmic Ray Removal

### LA Cosmic Algorithm

```python
def lacosmic(image, gain=1.0, readnoise=5.0, sigclip=4.5, sigfrac=0.3, objlim=2.0, niter=4):
    """
    Laplacian Cosmic Ray Identification (van Dokkum 2001).
    
    Args:
        image: Input image array
        gain: CCD gain (e-/ADU)
        readnoise: Read noise (e-)
        sigclip: Detection threshold
        sigfrac: Fractional threshold for neighbors
        objlim: Minimum contrast for cosmic rays
        niter: Number of iterations
    
    Returns:
        cleaned: Cleaned image
        mask: Cosmic ray mask (True = cosmic ray)
    """
    from scipy import ndimage
    
    cleaned = image.copy() * gain  # Convert to electrons
    mask = np.zeros_like(image, dtype=bool)
    
    for _ in range(niter):
        # Median filter to estimate background
        med5 = ndimage.median_filter(cleaned, size=5)
        
        # Laplacian for edge detection
        laplacian_kernel = np.array([[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]])
        laplacian = ndimage.convolve(cleaned, laplacian_kernel)
        
        # Normalize by noise
        noise = np.sqrt(np.abs(med5) + readnoise**2)
        
        # Signal-to-noise of Laplacian
        sn = laplacian / noise / 2  # Factor of 2 for Laplacian
        
        # Fine structure (to distinguish stars from cosmics)
        med3 = ndimage.median_filter(cleaned, size=3)
        fine = med3 - ndimage.median_filter(med3, size=7)
        
        # Cosmic ray candidates
        cr_mask = (sn > sigclip) & (fine / noise < objlim)
        
        # Grow mask slightly
        cr_mask = ndimage.binary_dilation(cr_mask)
        
        # Replace cosmic rays with median
        cleaned[cr_mask] = med5[cr_mask]
        mask |= cr_mask
    
    return cleaned / gain, mask
```

### Using astroscrappy

```python
def remove_cosmic_rays(image, gain=1.0, readnoise=5.0):
    """
    Remove cosmic rays using astroscrappy.
    """
    try:
        import astroscrappy
        
        mask, cleaned = astroscrappy.detect_cosmics(
            image,
            gain=gain,
            readnoise=readnoise,
            sigclip=4.5,
            sigfrac=0.3,
            objlim=5.0,
            cleantype='medmask',
            niter=4
        )
        
        return cleaned, mask
    
    except ImportError:
        # Fallback to simple method
        return simple_cosmic_removal(image)


def simple_cosmic_removal(image, sigma=5):
    """
    Simple sigma-clipping cosmic ray removal.
    """
    from scipy import ndimage
    
    # Median filter
    median = ndimage.median_filter(image, size=5)
    
    # Deviation from median
    diff = image - median
    
    # Sigma clip
    std = np.std(diff)
    mask = np.abs(diff) > sigma * std
    
    # Replace with median
    cleaned = image.copy()
    cleaned[mask] = median[mask]
    
    return cleaned, mask
```

---

## 4. Source Extraction

### Simple Peak Detection

```python
def detect_sources_simple(image, threshold=5, fwhm=3):
    """
    Simple source detection using local maxima.
    
    Args:
        image: Calibrated image
        threshold: Detection threshold in sigma
        fwhm: Expected FWHM in pixels
    
    Returns:
        sources: List of (x, y, flux) tuples
    """
    from scipy import ndimage
    
    # Estimate background
    background = np.median(image)
    noise = np.std(image)
    
    # Smooth image
    sigma = fwhm / 2.35
    smoothed = ndimage.gaussian_filter(image, sigma)
    
    # Find local maxima
    data_max = ndimage.maximum_filter(smoothed, size=int(fwhm * 2))
    peaks = (smoothed == data_max) & (smoothed > background + threshold * noise)
    
    # Get coordinates
    y_coords, x_coords = np.where(peaks)
    
    sources = []
    for y, x in zip(y_coords, x_coords):
        flux = image[y, x] - background
        sources.append((x, y, flux))
    
    return sources
```

### Using photutils

```python
def detect_sources_photutils(image, threshold=5, fwhm=3):
    """
    Source detection using photutils.
    """
    from photutils.detection import DAOStarFinder
    from photutils.background import Background2D, MedianBackground
    
    # Background estimation
    bkg = Background2D(image, (64, 64), filter_size=(3, 3),
                       bkg_estimator=MedianBackground())
    
    image_sub = image - bkg.background
    
    # Find sources
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg.background_rms_median)
    sources = daofind(image_sub)
    
    return sources


def detect_sources_sep(image, threshold=5):
    """
    Source detection using SEP (Source Extractor Python).
    """
    import sep
    
    # Ensure C-contiguous array
    image = np.ascontiguousarray(image.astype(np.float64))
    
    # Background estimation
    bkg = sep.Background(image)
    
    # Subtract background
    image_sub = image - bkg
    
    # Extract sources
    sources = sep.extract(image_sub, threshold, err=bkg.globalrms)
    
    return sources
```

---

## 5. Aperture Photometry

### Simple Aperture Photometry

```python
def aperture_photometry(image, x, y, r_aperture=5, r_inner=10, r_outer=15):
    """
    Perform aperture photometry at given position.
    
    Args:
        image: Calibrated image
        x, y: Source position
        r_aperture: Aperture radius (pixels)
        r_inner: Inner annulus radius for sky
        r_outer: Outer annulus radius for sky
    
    Returns:
        flux: Background-subtracted flux
        flux_err: Flux uncertainty
    """
    # Create coordinate grids
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((xx - x)**2 + (yy - y)**2)
    
    # Aperture mask
    aperture_mask = r <= r_aperture
    
    # Sky annulus mask
    annulus_mask = (r >= r_inner) & (r <= r_outer)
    
    # Sky background
    sky_values = image[annulus_mask]
    sky_median = np.median(sky_values)
    sky_std = np.std(sky_values)
    
    # Aperture sum
    aperture_sum = np.sum(image[aperture_mask])
    n_pixels = np.sum(aperture_mask)
    
    # Background-subtracted flux
    flux = aperture_sum - n_pixels * sky_median
    
    # Uncertainty (photon noise + sky noise)
    flux_err = np.sqrt(flux + n_pixels * sky_std**2)
    
    return flux, flux_err


def photometry_all_sources(image, sources, r_aperture=5):
    """
    Perform photometry on all detected sources.
    """
    results = []
    
    for source in sources:
        x, y = source['xcentroid'], source['ycentroid']
        flux, flux_err = aperture_photometry(image, x, y, r_aperture)
        
        # Magnitude (instrumental)
        if flux > 0:
            mag = -2.5 * np.log10(flux)
            mag_err = 2.5 / np.log(10) * flux_err / flux
        else:
            mag = np.nan
            mag_err = np.nan
        
        results.append({
            'x': x,
            'y': y,
            'flux': flux,
            'flux_err': flux_err,
            'mag': mag,
            'mag_err': mag_err
        })
    
    return results
```

### PSF Photometry

```python
def psf_photometry(image, sources, psf_model=None, fwhm=3):
    """
    PSF fitting photometry for crowded fields.
    """
    from photutils.psf import DAOPhotPSFPhotometry, IntegratedGaussianPRF
    from photutils.background import MMMBackground
    
    # Create PSF model if not provided
    if psf_model is None:
        psf_model = IntegratedGaussianPRF(sigma=fwhm/2.35)
    
    # Background estimator
    bkgrms = MMMBackground()
    
    # PSF photometry
    photometry = DAOPhotPSFPhotometry(
        crit_separation=fwhm * 2,
        threshold=5,
        fwhm=fwhm,
        psf_model=psf_model,
        fitshape=(11, 11),
        bkgrms=bkgrms
    )
    
    result = photometry(image)
    
    return result
```

---

## 6. Astrometric Calibration

### WCS (World Coordinate System)

```python
def solve_astrometry_local(image, sources, catalog='gaia'):
    """
    Solve astrometry by matching to reference catalog.
    
    Args:
        image: Image array
        sources: Detected source positions
        catalog: Reference catalog ('gaia', '2mass', etc.)
    
    Returns:
        wcs: Astropy WCS object
    """
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Query reference catalog (need approximate center)
    # This requires an initial WCS guess or manual center
    
    # For now, assume we have matching sources
    # pixel_coords: (x, y) from image
    # sky_coords: (ra, dec) from catalog
    
    # Fit WCS transformation
    from astropy.wcs.utils import fit_wcs_from_points
    
    wcs = fit_wcs_from_points(
        (pixel_coords[:, 0], pixel_coords[:, 1]),
        SkyCoord(sky_coords[:, 0], sky_coords[:, 1], unit=u.deg)
    )
    
    return wcs


def solve_astrometry_net(image_path):
    """
    Solve astrometry using astrometry.net.
    
    Requires local installation or API access.
    """
    import subprocess
    
    result = subprocess.run([
        'solve-field',
        '--no-plots',
        '--overwrite',
        image_path
    ], capture_output=True)
    
    # Load WCS from solution
    wcs_path = image_path.replace('.fits', '.wcs')
    if os.path.exists(wcs_path):
        from astropy.wcs import WCS
        return WCS(wcs_path)
    
    return None
```

---

## 7. Image Stacking

### Alignment and Combination

```python
def align_images(images, reference_idx=0):
    """
    Align images to a reference using cross-correlation.
    
    Args:
        images: List of image arrays
        reference_idx: Index of reference image
    
    Returns:
        aligned: List of aligned images
        shifts: List of (dx, dy) shifts
    """
    from scipy import ndimage
    from scipy.signal import correlate2d
    
    reference = images[reference_idx]
    aligned = []
    shifts = []
    
    for img in images:
        # Cross-correlation
        correlation = correlate2d(reference, img, mode='same')
        
        # Find peak
        y_max, x_max = np.unravel_index(correlation.argmax(), correlation.shape)
        
        # Calculate shift from center
        dy = y_max - reference.shape[0] // 2
        dx = x_max - reference.shape[1] // 2
        
        # Shift image
        shifted = ndimage.shift(img, (dy, dx))
        
        aligned.append(shifted)
        shifts.append((dx, dy))
    
    return aligned, shifts


def stack_images(images, method='median'):
    """
    Stack aligned images.
    
    Args:
        images: List of aligned image arrays
        method: 'mean', 'median', or 'sigma_clip'
    
    Returns:
        stacked: Combined image
    """
    stack = np.array(images)
    
    if method == 'mean':
        stacked = np.mean(stack, axis=0)
    
    elif method == 'median':
        stacked = np.median(stack, axis=0)
    
    elif method == 'sigma_clip':
        # Iterative sigma clipping
        stacked = sigma_clip_stack(stack, sigma=3, iterations=3)
    
    return stacked


def sigma_clip_stack(stack, sigma=3, iterations=3):
    """
    Sigma-clipped mean combination.
    """
    result = np.mean(stack, axis=0)
    
    for _ in range(iterations):
        # Calculate deviation from mean
        deviation = stack - result
        std = np.std(deviation, axis=0)
        
        # Mask outliers
        mask = np.abs(deviation) > sigma * std
        masked_stack = np.ma.array(stack, mask=mask)
        
        # Recalculate mean
        result = np.ma.mean(masked_stack, axis=0).data
    
    return result
```

---

## 8. Image Subtraction

### HOTPANTS-style Subtraction

```python
def image_subtraction(science, reference, kernel_size=11):
    """
    Simple image subtraction with PSF matching.
    
    Args:
        science: Science image
        reference: Reference image (template)
        kernel_size: Convolution kernel size
    
    Returns:
        difference: Subtracted image
    """
    from scipy import ndimage
    from scipy.optimize import least_squares
    
    # Estimate PSF difference
    # Convolve reference to match science PSF
    
    def residual(kernel_flat):
        kernel = kernel_flat.reshape((kernel_size, kernel_size))
        convolved = ndimage.convolve(reference, kernel)
        return (science - convolved).ravel()
    
    # Initial kernel (delta function)
    kernel0 = np.zeros((kernel_size, kernel_size))
    kernel0[kernel_size//2, kernel_size//2] = 1
    
    # Solve for optimal kernel
    result = least_squares(residual, kernel0.ravel())
    best_kernel = result.x.reshape((kernel_size, kernel_size))
    
    # Apply kernel and subtract
    convolved_ref = ndimage.convolve(reference, best_kernel)
    difference = science - convolved_ref
    
    return difference


def detect_transients(difference, threshold=5):
    """
    Detect transients in difference image.
    """
    # Statistics
    std = np.std(difference)
    
    # Find significant positive residuals (new sources)
    positive = difference > threshold * std
    
    # Find significant negative residuals (disappeared sources)
    negative = difference < -threshold * std
    
    # Get positions
    from scipy import ndimage
    
    pos_labels, n_pos = ndimage.label(positive)
    neg_labels, n_neg = ndimage.label(negative)
    
    transients = []
    
    for i in range(1, n_pos + 1):
        mask = pos_labels == i
        y, x = ndimage.center_of_mass(difference * mask)
        flux = np.sum(difference[mask])
        transients.append({
            'x': x, 'y': y, 'flux': flux, 'type': 'positive'
        })
    
    return transients
```

---

## 9. Image Quality Metrics

### FWHM Measurement

```python
def measure_fwhm(image, sources, method='gaussian'):
    """
    Measure FWHM from detected sources.
    """
    fwhms = []
    
    for source in sources:
        x, y = int(source['x']), int(source['y'])
        
        # Extract cutout
        size = 15
        cutout = image[y-size:y+size+1, x-size:x+size+1]
        
        if cutout.shape != (2*size+1, 2*size+1):
            continue
        
        if method == 'gaussian':
            fwhm = fit_gaussian_fwhm(cutout)
        else:
            fwhm = measure_fwhm_curve_of_growth(cutout)
        
        if fwhm is not None:
            fwhms.append(fwhm)
    
    return np.median(fwhms)


def fit_gaussian_fwhm(cutout):
    """
    Fit 2D Gaussian to measure FWHM.
    """
    from scipy.optimize import curve_fit
    
    def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
        x, y = coords
        return (offset + amplitude * 
                np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))))
    
    y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
    
    try:
        # Initial guess
        p0 = [cutout.max(), cutout.shape[1]/2, cutout.shape[0]/2, 2, 2, cutout.min()]
        
        popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), 
                           cutout.ravel(), p0=p0)
        
        sigma = np.mean([popt[3], popt[4]])
        fwhm = 2.35 * sigma
        
        return fwhm
    except:
        return None
```

### Signal-to-Noise Ratio

```python
def calculate_snr(image, sources):
    """
    Calculate SNR for detected sources.
    """
    results = []
    
    for source in sources:
        x, y = source['x'], source['y']
        
        # Source region
        r_source = 5
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((xx - x)**2 + (yy - y)**2)
        
        source_mask = r <= r_source
        sky_mask = (r > 10) & (r <= 20)
        
        # Flux and noise
        source_flux = np.sum(image[source_mask])
        sky_flux = np.median(image[sky_mask])
        sky_std = np.std(image[sky_mask])
        
        n_pix = np.sum(source_mask)
        signal = source_flux - n_pix * sky_flux
        noise = np.sqrt(signal + n_pix * sky_std**2)
        
        snr = signal / noise if noise > 0 else 0
        
        results.append(snr)
    
    return results
```

---

## 10. TinyML Image Processing

### Efficient Preprocessing

```python
def preprocess_for_tinyml(image, target_size=(64, 64)):
    """
    Preprocess image for TinyML classification.
    
    - Resize to target size
    - Normalize
    - Convert to INT8
    """
    from scipy import ndimage
    
    # Resize
    zoom_factors = (target_size[0] / image.shape[0],
                   target_size[1] / image.shape[1])
    resized = ndimage.zoom(image, zoom_factors, order=1)
    
    # Normalize to [0, 1]
    normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    
    # Convert to INT8 for TinyML
    int8_image = (normalized * 255).astype(np.uint8)
    
    return int8_image
```

---

## References

1. Howell, S. B. (2006). "Handbook of CCD Astronomy." Cambridge University Press.
2. van Dokkum, P. G. (2001). "Cosmic-Ray Rejection by Laplacian Edge Detection." PASP, 113, 1420.
3. Bertin, E., & Arnouts, S. (1996). "SExtractor: Software for source extraction." A&AS, 117, 393.
4. Stetson, P. B. (1987). "DAOPHOT: A Computer Program for Crowded-Field Stellar Photometry." PASP, 99, 191.
5. Alard, C., & Lupton, R. H. (1998). "A Method for Optimal Image Subtraction." ApJ, 503, 325.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
