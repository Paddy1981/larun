"""
LARUN Model Federation — 12 TinyML Models for Space Science

Layer 1 (browser-side, ONNX, <100ms):
    EXOPLANET-001, VSTAR-001, FLARE-001, MICROLENS-001,
    SUPERNOVA-001, SPECTYPE-001, ASTERO-001, GALAXY-001

Layer 2 (server-side, Python, <1s):
    VARDET-001, ANOMALY-001, DEBLEND-001, PERIODOGRAM-001

Layer 3 (orchestration):
    Claude API for NL routing, report generation, anomaly explanation
"""

from larun.models.federation import ModelFederation
from larun.models.vardet import VARDET001
from larun.models.anomaly import ANOMALY001
from larun.models.deblend import DEBLEND001
from larun.models.periodogram import PERIODOGRAM001

__all__ = [
    "ModelFederation",
    "VARDET001",
    "ANOMALY001",
    "DEBLEND001",
    "PERIODOGRAM001",
]
