"""
LARUN Detection Module
======================
Core astronomical analysis pipeline for transit detection.

This module provides:
- DetectionService: Main service for analyzing targets
- TransitDetector: Low-level transit detection
- BLSEngine: Box Least Squares periodogram wrapper
- PhaseFolding: Phase folding utilities

Author: Agent ALPHA
Version: 1.0.0 (MVP)
"""

from .models import (
    DetectionResult,
    VettingResult,
    PhaseFoldedData,
    PeriodogramData,
    LightCurveData,
    TestResult,
    TestFlag,
    Disposition,
)
from .service import DetectionService
from .detector import TransitDetector
from .bls_engine import BLSEngine
from .phase_folder import PhaseFolding

__all__ = [
    # Main service
    "DetectionService",
    # Detection components
    "TransitDetector",
    "BLSEngine",
    "PhaseFolding",
    # Data models
    "DetectionResult",
    "VettingResult",
    "PhaseFoldedData",
    "PeriodogramData",
    "LightCurveData",
    "TestResult",
    "TestFlag",
    "Disposition",
]

__version__ = "1.0.0"
