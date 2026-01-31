"""
LARUN Federated Inference Protocol
===================================
Protocol definitions for distributed model inference.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np


# ============================================================================
# Request/Response Protocol
# ============================================================================

@dataclass
class InferenceRequest:
    """
    Request for federated inference.
    
    Can be used for local or distributed inference.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = "transit_detection"
    data: Optional[np.ndarray] = None
    data_base64: Optional[str] = None  # For serialization
    models: Optional[List[str]] = None  # Specific models to use
    ensemble_method: str = "weighted_voting"
    return_all_predictions: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'request_id': self.request_id,
            'task': self.task,
            'models': self.models,
            'ensemble_method': self.ensemble_method,
            'return_all_predictions': self.return_all_predictions,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        
        if self.data_base64:
            result['data_base64'] = self.data_base64
        elif self.data is not None:
            import base64
            result['data_base64'] = base64.b64encode(
                self.data.astype(np.float32).tobytes()
            ).decode('ascii')
            result['data_shape'] = list(self.data.shape)
            result['data_dtype'] = str(self.data.dtype)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceRequest':
        """Create from dictionary."""
        request = cls(
            request_id=data.get('request_id', str(uuid.uuid4())),
            task=data.get('task', 'transit_detection'),
            data_base64=data.get('data_base64'),
            models=data.get('models'),
            ensemble_method=data.get('ensemble_method', 'weighted_voting'),
            return_all_predictions=data.get('return_all_predictions', True),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        )
        
        # Decode data if provided
        if request.data_base64 and 'data_shape' in data:
            import base64
            raw_bytes = base64.b64decode(request.data_base64)
            dtype = data.get('data_dtype', 'float32')
            request.data = np.frombuffer(raw_bytes, dtype=dtype).reshape(data['data_shape'])
        
        return request


@dataclass
class InferenceResponse:
    """
    Response from federated inference.
    """
    request_id: str
    success: bool
    predicted_class: int = 0
    confidence: float = 0.0
    probabilities: Optional[np.ndarray] = None
    model_agreement: float = 0.0
    predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'request_id': self.request_id,
            'success': self.success,
            'predicted_class': self.predicted_class,
            'confidence': round(self.confidence, 4),
            'probabilities': self.probabilities.tolist() if self.probabilities is not None else None,
            'model_agreement': round(self.model_agreement, 4),
            'predictions': self.predictions,
            'model_versions': self.model_versions,
            'timing': self.timing,
            'error': self.error,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResponse':
        """Create from dictionary."""
        probs = data.get('probabilities')
        if probs is not None:
            probs = np.array(probs)
        
        return cls(
            request_id=data['request_id'],
            success=data.get('success', True),
            predicted_class=data.get('predicted_class', 0),
            confidence=data.get('confidence', 0.0),
            probabilities=probs,
            model_agreement=data.get('model_agreement', 0.0),
            predictions=data.get('predictions', {}),
            model_versions=data.get('model_versions', {}),
            timing=data.get('timing', {}),
            error=data.get('error'),
            timestamp=data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        )
    
    @classmethod
    def error_response(cls, request_id: str, error: str) -> 'InferenceResponse':
        """Create an error response."""
        return cls(
            request_id=request_id,
            success=False,
            error=error
        )


# ============================================================================
# Worker Protocol
# ============================================================================

@dataclass
class WorkerStatus:
    """Status of a federated inference worker."""
    worker_id: str
    hostname: str
    models_loaded: List[str]
    is_available: bool
    current_load: float  # 0-1
    total_requests: int
    avg_latency_ms: float
    last_heartbeat: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'hostname': self.hostname,
            'models_loaded': self.models_loaded,
            'is_available': self.is_available,
            'current_load': round(self.current_load, 2),
            'total_requests': self.total_requests,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'last_heartbeat': self.last_heartbeat
        }


@dataclass
class ModelSyncRequest:
    """Request to sync a model to a worker."""
    model_id: str
    version: str
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'source_url': self.source_url,
            'source_path': self.source_path,
            'checksum': self.checksum
        }


# ============================================================================
# Message Types
# ============================================================================

class MessageType:
    """Protocol message types for federated communication."""
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    HEARTBEAT = "heartbeat"
    MODEL_SYNC = "model_sync"
    WORKER_REGISTER = "worker_register"
    WORKER_DEREGISTER = "worker_deregister"
    WORKER_STATUS = "worker_status"


@dataclass
class ProtocolMessage:
    """
    Base protocol message for federated communication.
    """
    message_type: str
    payload: Dict[str, Any]
    sender_id: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_type': self.message_type,
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'payload': self.payload
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolMessage':
        return cls(
            message_type=data['message_type'],
            payload=data['payload'],
            sender_id=data.get('sender_id', ''),
            message_id=data.get('message_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        )


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Federated Protocol...")
    print("=" * 60)
    
    # Test request/response
    test_data = np.random.randn(1024, 1).astype(np.float32)
    
    request = InferenceRequest(
        task="transit_detection",
        data=test_data,
        models=["transit_v1", "transit_v2"]
    )
    
    print(f"Request ID: {request.request_id}")
    print(f"Task: {request.task}")
    
    # Serialize and deserialize
    req_dict = request.to_dict()
    req_restored = InferenceRequest.from_dict(req_dict)
    
    print(f"Serialization OK: {req_restored.request_id == request.request_id}")
    print(f"Data shape preserved: {req_restored.data.shape == test_data.shape}")
    
    # Test response
    response = InferenceResponse(
        request_id=request.request_id,
        success=True,
        predicted_class=2,
        confidence=0.85,
        probabilities=np.array([0.05, 0.05, 0.85, 0.03, 0.02]),
        model_agreement=0.9
    )
    
    print(f"\nResponse success: {response.success}")
    print(f"Predicted: class {response.predicted_class} ({response.confidence:.0%})")
    
    print("\nTest complete!")
