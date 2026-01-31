"""
LARUN Federated Model System
============================
Central infrastructure for managing multiple specialized TinyML models.

This package provides:
- ModelRegistry: Central registry for model management
- ModelOrchestrator: Ensemble prediction orchestration
- Protocol: Federated inference protocol definitions

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

from .registry import ModelRegistry, ModelMetadata
from .orchestrator import ModelOrchestrator, EnsemblePrediction
from .protocol import InferenceRequest, InferenceResponse

__all__ = [
    'ModelRegistry',
    'ModelMetadata',
    'ModelOrchestrator',
    'EnsemblePrediction',
    'InferenceRequest',
    'InferenceResponse',
]

__version__ = '1.0.0'
