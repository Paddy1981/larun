"""
LARUN Federated Model Orchestrator
===================================
Ensemble prediction orchestration across multiple specialized models.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .registry import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_id: str
    probabilities: np.ndarray
    predicted_class: int
    confidence: float
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'predicted_class': int(self.predicted_class),
            'confidence': round(float(self.confidence), 4),
            'latency_ms': round(self.latency_ms, 2),
            'probabilities': self.probabilities.tolist()
        }


@dataclass
class EnsemblePrediction:
    """
    Combined prediction from multiple models.
    
    Uses weighted voting to combine predictions.
    """
    predicted_class: int
    confidence: float
    probabilities: np.ndarray
    individual_predictions: List[ModelPrediction]
    model_agreement: float  # 0-1, how much models agree
    total_latency_ms: float
    ensemble_method: str = "weighted_voting"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_class': int(self.predicted_class),
            'confidence': round(float(self.confidence), 4),
            'probabilities': self.probabilities.tolist(),
            'model_agreement': round(self.model_agreement, 4),
            'total_latency_ms': round(self.total_latency_ms, 2),
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.individual_predictions),
            'individual_predictions': [p.to_dict() for p in self.individual_predictions]
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "  FEDERATED PREDICTION RESULT",
            "═══════════════════════════════════════════════════════════════",
            f"  Predicted Class: {self.predicted_class}",
            f"  Confidence: {self.confidence:.1%}",
            f"  Model Agreement: {self.model_agreement:.1%}",
            f"  Total Latency: {self.total_latency_ms:.1f} ms",
            "───────────────────────────────────────────────────────────────",
            "  Individual Models:"
        ]
        
        for pred in self.individual_predictions:
            lines.append(
                f"    • {pred.model_id}: class {pred.predicted_class} "
                f"({pred.confidence:.1%}, {pred.latency_ms:.1f}ms)"
            )
        
        lines.append("═══════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ============================================================================
# Model Orchestrator
# ============================================================================

class ModelOrchestrator:
    """
    Federated inference orchestrator.
    
    Combines predictions from multiple specialized models using
    weighted ensemble voting.
    
    Supported ensemble methods:
    - weighted_voting: Weight by model accuracy
    - simple_voting: Equal weight voting
    - stacking: Meta-learner combination (future)
    
    Example:
        >>> registry = ModelRegistry()
        >>> orchestrator = ModelOrchestrator(registry)
        >>> result = orchestrator.predict(data, task="transit_detection")
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            registry: Model registry for loading models
            default_weights: Optional custom weights per model
        """
        self.registry = registry
        self.weights = default_weights or {}
        self._class_labels: Dict[str, List[str]] = {}
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for ensemble voting.
        
        Args:
            weights: Dict of model_id -> weight (0-1)
        """
        self.weights = weights.copy()
    
    def set_class_labels(self, task: str, labels: List[str]) -> None:
        """
        Set class labels for a task.
        
        Args:
            task: Task type
            labels: List of class label strings
        """
        self._class_labels[task] = labels
    
    def predict(
        self,
        data: np.ndarray,
        task: str = "transit_detection",
        models: Optional[List[str]] = None,
        method: str = "weighted_voting"
    ) -> EnsemblePrediction:
        """
        Run federated prediction across multiple models.
        
        Args:
            data: Input data matching model input shape
            task: Task type for model selection
            models: Specific model IDs to use (None = use all for task)
            method: Ensemble method ("weighted_voting", "simple_voting")
            
        Returns:
            EnsemblePrediction with combined result
        """
        import time
        
        # Get models for this task
        if models:
            model_metas = [self.registry.get(m) for m in models if self.registry.get(m)]
        else:
            model_metas = self.registry.get_by_task(task)
        
        if not model_metas:
            logger.warning(f"No models found for task: {task}")
            # Return empty prediction
            return EnsemblePrediction(
                predicted_class=0,
                confidence=0.0,
                probabilities=np.array([1.0]),
                individual_predictions=[],
                model_agreement=0.0,
                total_latency_ms=0.0,
                ensemble_method=method
            )
        
        # Run inference on each model
        predictions = []
        total_start = time.time()
        
        for meta in model_metas:
            model = self.registry.load_model(meta.model_id)
            if model is None:
                continue
            
            start = time.time()
            try:
                probs = self._run_inference(model, data, meta)
                latency = (time.time() - start) * 1000
                
                predictions.append(ModelPrediction(
                    model_id=meta.model_id,
                    probabilities=probs,
                    predicted_class=int(np.argmax(probs)),
                    confidence=float(np.max(probs)),
                    latency_ms=latency
                ))
            except Exception as e:
                logger.error(f"Inference failed for {meta.model_id}: {e}")
        
        total_latency = (time.time() - total_start) * 1000
        
        if not predictions:
            return EnsemblePrediction(
                predicted_class=0,
                confidence=0.0,
                probabilities=np.array([1.0]),
                individual_predictions=[],
                model_agreement=0.0,
                total_latency_ms=total_latency,
                ensemble_method=method
            )
        
        # Combine predictions
        if method == "weighted_voting":
            result = self._weighted_voting(predictions, model_metas)
        else:
            result = self._simple_voting(predictions)
        
        # Calculate model agreement
        pred_classes = [p.predicted_class for p in predictions]
        if pred_classes:
            most_common = max(set(pred_classes), key=pred_classes.count)
            agreement = pred_classes.count(most_common) / len(pred_classes)
        else:
            agreement = 0.0
        
        return EnsemblePrediction(
            predicted_class=result['class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            individual_predictions=predictions,
            model_agreement=agreement,
            total_latency_ms=total_latency,
            ensemble_method=method
        )
    
    def _run_inference(
        self,
        model: Any,
        data: np.ndarray,
        metadata: ModelMetadata
    ) -> np.ndarray:
        """
        Run inference on a single model.
        
        Handles both TFLite interpreters and Keras models.
        """
        import tensorflow as tf
        
        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1, 1)
        elif data.ndim == 2:
            data = data.reshape(1, *data.shape)
        
        # Handle TFLite interpreter
        if isinstance(model, tf.lite.Interpreter):
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Resize input if needed
            expected_shape = input_details[0]['shape']
            if data.shape != tuple(expected_shape):
                # Try to interpolate to expected size
                target_len = expected_shape[1]
                if len(data.shape) >= 2:
                    from scipy.ndimage import zoom
                    current_len = data.shape[1]
                    if current_len != target_len:
                        ratio = target_len / current_len
                        data = zoom(data, (1, ratio, 1), order=1)
            
            data = data.astype(input_details[0]['dtype'])
            model.set_tensor(input_details[0]['index'], data)
            model.invoke()
            output = model.get_tensor(output_details[0]['index'])
            return output[0]
        
        # Handle Keras model
        else:
            output = model.predict(data, verbose=0)
            return output[0] if output.ndim > 1 else output
    
    def _weighted_voting(
        self,
        predictions: List[ModelPrediction],
        metadata: List[ModelMetadata]
    ) -> Dict[str, Any]:
        """
        Combine predictions using accuracy-weighted voting.
        """
        # Get weights from accuracy or custom weights
        weights = []
        for i, pred in enumerate(predictions):
            if pred.model_id in self.weights:
                weights.append(self.weights[pred.model_id])
            elif i < len(metadata):
                weights.append(metadata[i].accuracy)
            else:
                weights.append(1.0)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        # Weighted average of probabilities
        n_classes = max(len(p.probabilities) for p in predictions)
        combined = np.zeros(n_classes)
        
        for pred, weight in zip(predictions, weights):
            probs = pred.probabilities
            # Pad if necessary
            if len(probs) < n_classes:
                probs = np.pad(probs, (0, n_classes - len(probs)))
            combined += probs * weight
        
        return {
            'class': int(np.argmax(combined)),
            'confidence': float(np.max(combined)),
            'probabilities': combined
        }
    
    def _simple_voting(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """
        Combine predictions using simple majority voting.
        """
        n_classes = max(len(p.probabilities) for p in predictions)
        combined = np.zeros(n_classes)
        
        for pred in predictions:
            probs = pred.probabilities
            if len(probs) < n_classes:
                probs = np.pad(probs, (0, n_classes - len(probs)))
            combined += probs
        
        combined /= len(predictions)
        
        return {
            'class': int(np.argmax(combined)),
            'confidence': float(np.max(combined)),
            'probabilities': combined
        }
    
    def benchmark(
        self,
        data: np.ndarray,
        task: str,
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark ensemble inference speed.
        
        Args:
            data: Test input data
            task: Task type
            n_iterations: Number of iterations
            
        Returns:
            Dict with timing statistics
        """
        import time
        
        latencies = []
        for _ in range(n_iterations):
            start = time.time()
            self.predict(data, task)
            latencies.append((time.time() - start) * 1000)
        
        return {
            'task': task,
            'n_iterations': n_iterations,
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95))
        }


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Model Orchestrator...")
    print("=" * 60)
    
    # Create registry and orchestrator
    from .registry import ModelRegistry
    
    registry = ModelRegistry()
    orchestrator = ModelOrchestrator(registry)
    
    # Test with synthetic data
    test_data = np.random.randn(1024, 1).astype(np.float32)
    
    print(f"Available tasks: {registry.list_tasks()}")
    
    if registry.list_tasks():
        task = registry.list_tasks()[0]
        result = orchestrator.predict(test_data, task=task)
        print(result.summary())
    else:
        print("No models registered. Register models first.")
    
    print("\nTest complete!")
