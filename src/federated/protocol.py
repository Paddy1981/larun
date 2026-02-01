"""
Federated Inference Protocol for LARUN
======================================

Defines the communication protocol for coordinating inference
across multiple TinyML models in the federated system.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import base64
import numpy as np


@dataclass
class InferenceRequest:
    """
    Request for model inference in the federated system.

    Can be used to request inference from a single model or
    ensemble of models for a specific task.
    """
    task: str
    data: Optional[np.ndarray] = None
    models: List[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    timeout_ms: int = 1000
    ensemble_method: str = 'average'  # average, voting, weighted
    weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize request to dictionary."""
        d = {
            'request_id': self.request_id,
            'task': self.task,
            'models': self.models,
            'timestamp': self.timestamp,
            'timeout_ms': self.timeout_ms,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'metadata': self.metadata,
        }

        # Encode numpy array as base64
        if self.data is not None:
            d['data'] = {
                'dtype': str(self.data.dtype),
                'shape': list(self.data.shape),
                'data_b64': base64.b64encode(self.data.tobytes()).decode('ascii'),
            }

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InferenceRequest':
        """Deserialize request from dictionary."""
        data = None
        if 'data' in d and d['data'] is not None:
            data_info = d['data']
            data_bytes = base64.b64decode(data_info['data_b64'])
            data = np.frombuffer(data_bytes, dtype=data_info['dtype'])
            data = data.reshape(data_info['shape'])

        return cls(
            request_id=d['request_id'],
            task=d['task'],
            models=d.get('models', []),
            data=data,
            timestamp=d.get('timestamp', ''),
            timeout_ms=d.get('timeout_ms', 1000),
            ensemble_method=d.get('ensemble_method', 'average'),
            weights=d.get('weights'),
            metadata=d.get('metadata', {}),
        )


@dataclass
class InferenceResponse:
    """
    Response from model inference.

    Contains prediction results from one or more models.
    """
    request_id: str
    success: bool
    predicted_class: Optional[int] = None
    confidence: float = 0.0
    probabilities: Optional[np.ndarray] = None
    model_id: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        d = {
            'request_id': self.request_id,
            'success': self.success,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'model_id': self.model_id,
            'latency_ms': self.latency_ms,
            'error': self.error,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }

        if self.probabilities is not None:
            d['probabilities'] = self.probabilities.tolist()

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InferenceResponse':
        """Deserialize response from dictionary."""
        probabilities = None
        if 'probabilities' in d and d['probabilities'] is not None:
            probabilities = np.array(d['probabilities'])

        return cls(
            request_id=d['request_id'],
            success=d['success'],
            predicted_class=d.get('predicted_class'),
            confidence=d.get('confidence', 0.0),
            probabilities=probabilities,
            model_id=d.get('model_id'),
            latency_ms=d.get('latency_ms', 0.0),
            error=d.get('error'),
            timestamp=d.get('timestamp', ''),
            metadata=d.get('metadata', {}),
        )

    @classmethod
    def error_response(
        cls,
        request_id: str,
        error: str,
    ) -> 'InferenceResponse':
        """Create an error response."""
        return cls(
            request_id=request_id,
            success=False,
            error=error,
        )

    @classmethod
    def success_response(
        cls,
        request_id: str,
        predicted_class: int,
        confidence: float,
        probabilities: Optional[np.ndarray] = None,
        model_id: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> 'InferenceResponse':
        """Create a successful response."""
        return cls(
            request_id=request_id,
            success=True,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            model_id=model_id,
            latency_ms=latency_ms,
        )


@dataclass
class HealthCheck:
    """Health check message for federated nodes."""
    node_id: str
    status: str  # healthy, degraded, unhealthy
    models_available: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    inference_count: int = 0
    avg_latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSyncRequest:
    """Request to sync model from registry."""
    model_id: str
    version: str
    source_url: str
    checksum: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSyncResponse:
    """Response for model sync operation."""
    request_id: str
    success: bool
    model_id: str
    version: str
    error: Optional[str] = None
    sync_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
