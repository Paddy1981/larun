"""
LARUN Client Library
Python client for communicating with LARUN TinyML devices

Usage:
    from larun_client import LarunDevice

    # Connect via USB serial
    device = LarunDevice('/dev/ttyUSB0')

    # Select model
    device.select_model('EXOPLANET-001')

    # Analyze light curve
    result = device.analyze_fits('lightcurve.fits')
    print(f"Classification: {result.classification}")
    print(f"Confidence: {result.confidence:.2%}")
"""

from .device import LarunDevice, DeviceInfo, InferenceResult
from .serial_client import SerialClient
from .fits_utils import extract_lightcurve_from_fits

__version__ = '2.0.0'
__all__ = [
    'LarunDevice',
    'DeviceInfo',
    'InferenceResult',
    'SerialClient',
    'extract_lightcurve_from_fits',
]
