"""
Model Orchestrator for LARUN Federated System
==============================================

Coordinates inference across multiple TinyML models,
implementing ensemble methods for improved accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import numpy as np
import time

from .registry import ModelRegistry, ModelMetadata
from .protocol import InferenceRequest, InferenceResponse


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
            'probabilities': self.probabilities.tolist(),
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
        }


@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple models."""
    predicted_class: int
    confidence: float
    probabilities: np.ndarray
    individual_predictions: List[ModelPrediction]
    model_agreement: float
    total_latency_ms: float
    ensemble_method: str = 'average'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'probabilities': self.probabilities.tolist(),
            'num_models': len(self.individual_predictions),
            'model_agreement': self.model_agreement,
            'total_latency_ms': self.total_latency_ms,
            'ensemble_method': self.ensemble_method,
            'individual_predictions': [p.to_dict() for p in self.individual_predictions],
        }


class ModelOrchestrator:
    """
    Orchestrates inference across multiple TinyML models.

    Supports:
    - Single model inference
    - Ensemble inference (averaging, voting, weighted)
    - Model selection based on task
    - Latency tracking
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_loader: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            registry: Model registry for model discovery
            model_loader: Function to load models by path
        """
        self.registry = registry
        self.model_loader = model_loader
        self._loaded_models: Dict[str, Any] = {}

    def _load_model(self, model_id: str) -> Optional[Any]:
        """Load a model if not already cached."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        metadata = self.registry.get(model_id)
        if metadata is None:
            return None

        if self.model_loader is None:
            return None

        try:
            model = self.model_loader(metadata.file_path)
            self._loaded_models[model_id] = model
            return model
        except Exception:
            return None

    def _run_inference(
        self,
        model_id: str,
        data: np.ndarray,
    ) -> Optional[ModelPrediction]:
        """Run inference on a single model."""
        model = self._load_model(model_id)
        if model is None:
            return None

        start_time = time.time()

        try:
            # Assume model has predict method
            output = model.predict(data)

            # Handle different output formats
            if isinstance(output, np.ndarray):
                if output.ndim == 1:
                    probabilities = output
                else:
                    probabilities = output[0]
            else:
                probabilities = np.array(output)

            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            latency_ms = (time.time() - start_time) * 1000

            return ModelPrediction(
                model_id=model_id,
                probabilities=probabilities,
                predicted_class=predicted_class,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        except Exception:
            return None

    def predict(
        self,
        data: np.ndarray,
        task: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        ensemble_method: str = 'average',
        weights: Optional[Dict[str, float]] = None,
    ) -> EnsemblePrediction:
        """
        Run inference across one or more models.

        Args:
            data: Input data
            task: Task name (to auto-select models)
            model_ids: Specific model IDs to use
            ensemble_method: How to combine predictions
            weights: Optional weights for each model

        Returns:
            EnsemblePrediction with combined result
        """
        # Determine which models to use
        if model_ids:
            models_to_use = model_ids
        elif task:
            task_models = self.registry.get_by_task(task)
            models_to_use = [m.model_id for m in task_models]
        else:
            models_to_use = []

        if not models_to_use:
            # Return empty prediction
            return EnsemblePrediction(
                predicted_class=0,
                confidence=0.0,
                probabilities=np.array([]),
                individual_predictions=[],
                model_agreement=0.0,
                total_latency_ms=0.0,
                ensemble_method=ensemble_method,
            )

        # Run inference on each model
        predictions = []
        total_latency = 0.0

        for model_id in models_to_use:
            pred = self._run_inference(model_id, data)
            if pred is not None:
                predictions.append(pred)
                total_latency += pred.latency_ms

        if not predictions:
            return EnsemblePrediction(
                predicted_class=0,
                confidence=0.0,
                probabilities=np.array([]),
                individual_predictions=[],
                model_agreement=0.0,
                total_latency_ms=total_latency,
                ensemble_method=ensemble_method,
            )

        # Combine predictions
        if ensemble_method == 'voting':
            combined = self._voting_ensemble(predictions)
        elif ensemble_method == 'weighted':
            combined = self._weighted_ensemble(predictions, weights or {})
        else:  # average
            combined = self._average_ensemble(predictions)

        # Calculate model agreement
        classes = [p.predicted_class for p in predictions]
        agreement = classes.count(combined[0]) / len(classes)

        return EnsemblePrediction(
            predicted_class=combined[0],
            confidence=combined[1],
            probabilities=combined[2],
            individual_predictions=predictions,
            model_agreement=agreement,
            total_latency_ms=total_latency,
            ensemble_method=ensemble_method,
        )

    def _average_ensemble(
        self,
        predictions: List[ModelPrediction],
    ) -> tuple:
        """Average probabilities across models."""
        # Stack probabilities
        all_probs = np.array([p.probabilities for p in predictions])

        # Average
        avg_probs = np.mean(all_probs, axis=0)

        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        return predicted_class, confidence, avg_probs

    def _voting_ensemble(
        self,
        predictions: List[ModelPrediction],
    ) -> tuple:
        """Majority voting across models."""
        classes = [p.predicted_class for p in predictions]
        num_classes = len(predictions[0].probabilities)

        # Count votes
        votes = np.zeros(num_classes)
        for c in classes:
            votes[c] += 1

        predicted_class = int(np.argmax(votes))
        confidence = float(votes[predicted_class] / len(predictions))

        # Normalize votes as probabilities
        probs = votes / votes.sum()

        return predicted_class, confidence, probs

    def _weighted_ensemble(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> tuple:
        """Weighted average of probabilities."""
        all_probs = []
        all_weights = []

        for p in predictions:
            w = weights.get(p.model_id, 1.0)
            all_probs.append(p.probabilities * w)
            all_weights.append(w)

        # Weighted average
        weighted_sum = np.sum(all_probs, axis=0)
        total_weight = sum(all_weights)
        avg_probs = weighted_sum / total_weight

        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        return predicted_class, confidence, avg_probs

    def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request.

        Args:
            request: InferenceRequest object

        Returns:
            InferenceResponse with results
        """
        if request.data is None:
            return InferenceResponse.error_response(
                request.request_id,
                "No data provided",
            )

        start_time = time.time()

        try:
            result = self.predict(
                data=request.data,
                task=request.task,
                model_ids=request.models if request.models else None,
                ensemble_method=request.ensemble_method,
                weights=request.weights,
            )

            latency_ms = (time.time() - start_time) * 1000

            return InferenceResponse.success_response(
                request_id=request.request_id,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                probabilities=result.probabilities,
                latency_ms=latency_ms,
            )

        except Exception as e:
            return InferenceResponse.error_response(
                request.request_id,
                str(e),
            )
