"""
LARUN Device Client
Main interface for communicating with LARUN TinyML devices
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .serial_client import SerialClient
from .fits_utils import extract_lightcurve_from_fits


@dataclass
class DeviceInfo:
    """Device information"""
    platform: str
    version: str
    free_heap: int
    total_heap: int
    tensor_arena_size: int
    total_inferences: int
    avg_inference_time_ms: float
    current_model: int


@dataclass
class InferenceResult:
    """Inference result from device"""
    node_id: str
    classification: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    memory_used_kb: int
    success: bool
    metadata: Dict[str, Any]

    def __str__(self):
        return (f"InferenceResult(classification={self.classification}, "
                f"confidence={self.confidence:.2%}, "
                f"time={self.inference_time_ms:.1f}ms)")


class LarunDevice:
    """
    LARUN TinyML Device Client

    Provides high-level interface for:
    - Connecting to device (serial, WiFi, BLE)
    - Selecting models
    - Uploading data
    - Running inference
    - Analyzing FITS files

    Example:
        device = LarunDevice('/dev/ttyUSB0')
        device.select_model('EXOPLANET-001')
        result = device.analyze_fits('transit.fits')
        print(result.classification, result.confidence)
    """

    # Model name to ID mapping
    MODEL_IDS = {
        'EXOPLANET-001': 0x01,
        'VSTAR-001': 0x02,
        'FLARE-001': 0x03,
        'ASTERO-001': 0x04,
        'SUPERNOVA-001': 0x05,
        'GALAXY-001': 0x06,
        'SPECTYPE-001': 0x07,
        'MICROLENS-001': 0x08,
    }

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 5.0):
        """
        Initialize device client

        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Serial baud rate (default: 115200)
            timeout: Communication timeout in seconds
        """
        self.client = SerialClient(port, baudrate, timeout)
        self._current_model = None
        self._device_info = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def connect(self) -> bool:
        """
        Connect to device

        Returns:
            True on success

        Raises:
            ConnectionError: If connection fails
        """
        if not self.client.connect():
            raise ConnectionError(f"Failed to connect to device")

        # Ping device to verify connection
        if not self.ping():
            raise ConnectionError("Device not responding to ping")

        # Get device info
        self._device_info = self.get_info()
        return True

    def close(self):
        """Close connection to device"""
        self.client.close()

    def ping(self, timeout: float = 1.0) -> bool:
        """
        Ping device to check if it's responding

        Args:
            timeout: Ping timeout in seconds

        Returns:
            True if device responds
        """
        return self.client.ping(timeout)

    def get_info(self) -> DeviceInfo:
        """
        Get device information

        Returns:
            DeviceInfo object
        """
        response = self.client.get_info()
        return DeviceInfo(
            platform=response['platform'],
            version=response['version'],
            free_heap=response['free_heap'],
            total_heap=response['total_heap'],
            tensor_arena_size=response['tensor_arena_size'],
            total_inferences=response['total_inferences'],
            avg_inference_time_ms=response['avg_inference_time_ms'],
            current_model=response['current_model'],
        )

    def select_model(self, model_name: str) -> bool:
        """
        Select active model

        Args:
            model_name: Model name (e.g., 'EXOPLANET-001')

        Returns:
            True on success

        Raises:
            ValueError: If model name is invalid
        """
        if model_name not in self.MODEL_IDS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.MODEL_IDS.keys())}")

        model_id = self.MODEL_IDS[model_name]
        success = self.client.select_model(model_id)

        if success:
            self._current_model = model_name

        return success

    def upload_lightcurve(self, data: np.ndarray) -> bool:
        """
        Upload light curve data to device

        Args:
            data: Light curve data (1D numpy array)

        Returns:
            True on success
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if data.ndim != 1:
            data = data.flatten()

        return self.client.upload_lightcurve(data.astype(np.float32))

    def infer(self) -> InferenceResult:
        """
        Run inference on uploaded data

        Returns:
            InferenceResult object

        Raises:
            RuntimeError: If inference fails
        """
        result = self.client.infer()

        if not result.get('success', False):
            raise RuntimeError(f"Inference failed: {result.get('error', 'Unknown error')}")

        return InferenceResult(
            node_id=result['node_id'],
            classification=result['classification'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            inference_time_ms=result['inference_time_ms'],
            memory_used_kb=result['memory_used_kb'],
            success=result['success'],
            metadata=result.get('metadata', {}),
        )

    def analyze(self, time_array: Optional[np.ndarray] = None,
                flux_array: np.ndarray = None) -> InferenceResult:
        """
        Upload and analyze light curve data

        Args:
            time_array: Time values (optional, not used by model)
            flux_array: Flux values

        Returns:
            InferenceResult object
        """
        if flux_array is None:
            raise ValueError("flux_array is required")

        # Upload data
        if not self.upload_lightcurve(flux_array):
            raise RuntimeError("Failed to upload data")

        # Run inference
        return self.infer()

    def analyze_fits(self, fits_path: Union[str, Path],
                     hdu: int = 1,
                     time_col: str = 'TIME',
                     flux_col: str = 'FLUX') -> InferenceResult:
        """
        Analyze light curve from FITS file

        Args:
            fits_path: Path to FITS file
            hdu: HDU index (default: 1)
            time_col: Time column name
            flux_col: Flux column name

        Returns:
            InferenceResult object
        """
        # Extract light curve from FITS
        time, flux = extract_lightcurve_from_fits(
            fits_path, hdu=hdu, time_col=time_col, flux_col=flux_col
        )

        # Analyze
        return self.analyze(time, flux)

    def analyze_csv(self, csv_path: Union[str, Path],
                    time_col: int = 0,
                    flux_col: int = 1,
                    skiprows: int = 0) -> InferenceResult:
        """
        Analyze light curve from CSV file

        Args:
            csv_path: Path to CSV file
            time_col: Time column index
            flux_col: Flux column index
            skiprows: Number of rows to skip (header)

        Returns:
            InferenceResult object
        """
        data = np.loadtxt(csv_path, delimiter=',', skiprows=skiprows)

        if data.ndim == 1:
            flux = data
        else:
            flux = data[:, flux_col]

        return self.analyze(flux_array=flux)

    def reset(self):
        """Reset device"""
        self.client.reset()
        time.sleep(2.0)  # Wait for device to reboot

    @property
    def current_model(self) -> Optional[str]:
        """Get currently selected model name"""
        return self._current_model

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Get cached device info"""
        return self._device_info

    def __repr__(self):
        if self._device_info:
            return (f"<LarunDevice port={self.client.port} "
                   f"platform={self._device_info.platform} "
                   f"model={self._current_model}>")
        else:
            return f"<LarunDevice port={self.client.port} (not connected)>"
