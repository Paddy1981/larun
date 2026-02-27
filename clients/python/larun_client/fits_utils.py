"""
FITS File Utilities for LARUN Client

Extracts light curves from FITS files (TESS, Kepler, K2, CHEOPS, etc.)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

try:
    from astropy.io import fits
except ImportError:
    raise ImportError(
        "astropy is required for FITS file support. "
        "Install with: pip install astropy"
    )


def extract_lightcurve_from_fits(
    fits_path: Union[str, Path],
    hdu: int = 1,
    time_col: str = 'TIME',
    flux_col: str = 'FLUX',
    flux_err_col: Optional[str] = 'FLUX_ERR',
    quality_col: Optional[str] = 'QUALITY',
    remove_nans: bool = True,
    remove_outliers: bool = False,
    sigma_clip: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract light curve from FITS file

    Args:
        fits_path: Path to FITS file
        hdu: HDU index (default: 1 for TESS/Kepler)
        time_col: Time column name
        flux_col: Flux column name
        flux_err_col: Flux error column name (optional)
        quality_col: Quality flag column name (optional)
        remove_nans: Remove NaN values
        remove_outliers: Remove outliers using sigma clipping
        sigma_clip: Sigma threshold for outlier removal

    Returns:
        (time, flux) tuple as numpy arrays

    Raises:
        FileNotFoundError: If FITS file not found
        KeyError: If column not found
    """
    fits_path = Path(fits_path)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    # Open FITS file
    with fits.open(fits_path) as hdul:
        # Get data from specified HDU
        data = hdul[hdu].data

        # Extract columns
        try:
            time = data[time_col]
            flux = data[flux_col]
        except KeyError as e:
            available_cols = data.columns.names
            raise KeyError(
                f"Column {e} not found. Available columns: {available_cols}"
            )

        # Extract quality flags if available
        quality = None
        if quality_col and quality_col in data.columns.names:
            quality = data[quality_col]

        # Convert to numpy arrays
        time = np.asarray(time, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float32)

    # Filter by quality flags (TESS/Kepler convention: 0 = good)
    if quality is not None:
        good = quality == 0
        time = time[good]
        flux = flux[good]

    # Remove NaNs
    if remove_nans:
        finite = np.isfinite(time) & np.isfinite(flux)
        time = time[finite]
        flux = flux[finite]

    # Remove outliers using sigma clipping
    if remove_outliers and len(flux) > 10:
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        sigma = mad * 1.4826  # Convert MAD to std

        if sigma > 0:
            outliers = np.abs(flux - median) < sigma_clip * sigma
            time = time[outliers]
            flux = flux[outliers]

    return time, flux


def extract_lightcurve_from_tess(
    fits_path: Union[str, Path],
    flux_type: str = 'PDCSAP',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract light curve from TESS FITS file

    Args:
        fits_path: Path to TESS FITS file
        flux_type: Flux type ('SAP' or 'PDCSAP')
        **kwargs: Additional arguments for extract_lightcurve_from_fits

    Returns:
        (time, flux) tuple
    """
    if flux_type.upper() == 'PDCSAP':
        flux_col = 'PDCSAP_FLUX'
    elif flux_type.upper() == 'SAP':
        flux_col = 'SAP_FLUX'
    else:
        raise ValueError(f"Unknown flux type: {flux_type}")

    return extract_lightcurve_from_fits(
        fits_path,
        hdu=1,
        time_col='TIME',
        flux_col=flux_col,
        quality_col='QUALITY',
        **kwargs
    )


def extract_lightcurve_from_kepler(
    fits_path: Union[str, Path],
    flux_type: str = 'PDCSAP',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract light curve from Kepler/K2 FITS file

    Args:
        fits_path: Path to Kepler FITS file
        flux_type: Flux type ('SAP' or 'PDCSAP')
        **kwargs: Additional arguments for extract_lightcurve_from_fits

    Returns:
        (time, flux) tuple
    """
    if flux_type.upper() == 'PDCSAP':
        flux_col = 'PDCSAP_FLUX'
    elif flux_type.upper() == 'SAP':
        flux_col = 'SAP_FLUX'
    else:
        raise ValueError(f"Unknown flux type: {flux_type}")

    return extract_lightcurve_from_fits(
        fits_path,
        hdu=1,
        time_col='TIME',
        flux_col=flux_col,
        quality_col='SAP_QUALITY',
        **kwargs
    )


def normalize_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = 'median'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize light curve

    Args:
        time: Time array
        flux: Flux array
        method: Normalization method ('median', 'mean', or 'minmax')

    Returns:
        (time, normalized_flux) tuple
    """
    if method == 'median':
        flux = flux / np.median(flux)
    elif method == 'mean':
        flux = flux / np.mean(flux)
    elif method == 'minmax':
        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return time, flux


def detrend_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrend light curve

    Args:
        time: Time array
        flux: Flux array
        method: Detrending method ('linear' or 'polynomial')

    Returns:
        (time, detrended_flux) tuple
    """
    if method == 'linear':
        # Linear detrending
        coeffs = np.polyfit(time, flux, 1)
        trend = np.polyval(coeffs, time)
        flux = flux - trend + np.median(flux)
    elif method == 'polynomial':
        # Polynomial detrending (order 2)
        coeffs = np.polyfit(time, flux, 2)
        trend = np.polyval(coeffs, time)
        flux = flux - trend + np.median(flux)
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    return time, flux


def bin_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    bin_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin light curve by time

    Args:
        time: Time array
        flux: Flux array
        bin_size: Bin size in time units

    Returns:
        (binned_time, binned_flux) tuple
    """
    # Calculate bin edges
    time_min = np.min(time)
    time_max = np.max(time)
    bins = np.arange(time_min, time_max + bin_size, bin_size)

    # Bin data
    binned_flux = []
    binned_time = []

    for i in range(len(bins) - 1):
        mask = (time >= bins[i]) & (time < bins[i + 1])
        if np.any(mask):
            binned_time.append(np.mean(time[mask]))
            binned_flux.append(np.mean(flux[mask]))

    return np.array(binned_time), np.array(binned_flux)


def get_fits_info(fits_path: Union[str, Path]) -> dict:
    """
    Get information about FITS file

    Args:
        fits_path: Path to FITS file

    Returns:
        Dictionary with FITS file information
    """
    fits_path = Path(fits_path)

    with fits.open(fits_path) as hdul:
        info = {
            'filename': fits_path.name,
            'num_hdus': len(hdul),
            'hdus': []
        }

        for i, hdu in enumerate(hdul):
            hdu_info = {
                'index': i,
                'name': hdu.name,
                'type': type(hdu).__name__
            }

            if hasattr(hdu, 'columns'):
                hdu_info['columns'] = hdu.columns.names
                hdu_info['num_rows'] = len(hdu.data) if hdu.data is not None else 0

            if hasattr(hdu, 'header'):
                # Extract key header values
                header_keys = ['TELESCOP', 'INSTRUME', 'OBJECT', 'RA_OBJ', 'DEC_OBJ']
                hdu_info['header'] = {}
                for key in header_keys:
                    if key in hdu.header:
                        hdu_info['header'][key] = hdu.header[key]

            info['hdus'].append(hdu_info)

    return info
