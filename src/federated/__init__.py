"""
LARUN Federated Multi-Model System
===================================

Provides infrastructure for running multiple TinyML models
in a federated architecture for astronomical analysis.

Components:
- registry: Model metadata and version management
- protocol: Inter-model communication protocol
- orchestrator: Model execution coordination
"""

from .registry import ModelRegistry, ModelMetadata
from .protocol import InferenceRequest, InferenceResponse
from .orchestrator import ModelOrchestrator, EnsemblePrediction, ModelPrediction

__all__ = [
    'ModelRegistry',
    'ModelMetadata',
    'InferenceRequest',
    'InferenceResponse',
    'ModelOrchestrator',
    'EnsemblePrediction',
    'ModelPrediction',
]
