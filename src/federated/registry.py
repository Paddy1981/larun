"""
Model Registry for LARUN Federated System
==========================================

Manages model metadata, versions, and provides model discovery
for the federated multi-model TinyML architecture.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    version: str
    task: str
    accuracy: float
    input_shape: Tuple[int, ...]
    created_at: str
    file_path: str
    description: str = ""
    author: str = "LARUN"
    license: str = "MIT"
    quantization: str = "int8"
    model_size_kb: int = 0
    inference_time_ms: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    data_version: str = ""
    compatible_larun: str = ">=2.0.0"
    checksum_sha256: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d['input_shape'] = list(self.input_shape)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        # Convert list back to tuple
        if 'input_shape' in d:
            d['input_shape'] = tuple(d['input_shape'])
        return cls(**d)


class ModelRegistry:
    """
    Central registry for LARUN models.

    Provides:
    - Model registration and unregistration
    - Querying by task, accuracy, or other criteria
    - Persistence to JSON file
    - Version management
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to JSON file for persistence.
                          If None, operates in memory only.
        """
        self.registry_path = registry_path
        self.models: Dict[str, ModelMetadata] = {}

        if registry_path and registry_path.exists():
            self._load()

    def _load(self) -> None:
        """Load registry from file."""
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                for model_id, meta in data.get('models', {}).items():
                    self.models[model_id] = ModelMetadata.from_dict(meta)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load registry: {e}")
            self.models = {}

    def _save(self) -> None:
        """Save registry to file."""
        if self.registry_path:
            data = {
                'version': '1.0',
                'updated_at': datetime.utcnow().isoformat() + 'Z',
                'models': {
                    model_id: meta.to_dict()
                    for model_id, meta in self.models.items()
                }
            }
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

    def register(
        self,
        metadata: ModelMetadata,
        save: bool = True,
        overwrite: bool = False,
    ) -> bool:
        """
        Register a model in the registry.

        Args:
            metadata: Model metadata
            save: Whether to persist to file
            overwrite: Whether to overwrite existing model

        Returns:
            True if registered successfully
        """
        if metadata.model_id in self.models and not overwrite:
            return False

        self.models[metadata.model_id] = metadata

        if save:
            self._save()

        return True

    def unregister(self, model_id: str, save: bool = True) -> bool:
        """
        Remove a model from the registry.

        Args:
            model_id: Model identifier
            save: Whether to persist to file

        Returns:
            True if model was found and removed
        """
        if model_id not in self.models:
            return False

        del self.models[model_id]

        if save:
            self._save()

        return True

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def get_by_task(self, task: str) -> List[ModelMetadata]:
        """Get all models for a specific task."""
        return [m for m in self.models.values() if m.task == task]

    def get_best_for_task(
        self,
        task: str,
        metric: str = 'accuracy',
    ) -> Optional[ModelMetadata]:
        """
        Get the best-performing model for a task.

        Args:
            task: Task name
            metric: Metric to optimize (accuracy, f1_score, etc.)

        Returns:
            Best model or None if no models exist
        """
        task_models = self.get_by_task(task)
        if not task_models:
            return None

        return max(task_models, key=lambda m: getattr(m, metric, 0))

    def get_by_tag(self, tag: str) -> List[ModelMetadata]:
        """Get all models with a specific tag."""
        return [m for m in self.models.values() if tag in m.tags]

    def list_tasks(self) -> List[str]:
        """Get list of all unique tasks."""
        return list(set(m.task for m in self.models.values()))

    def list_models(self) -> List[str]:
        """Get list of all model IDs."""
        return list(self.models.keys())

    def search(
        self,
        task: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        max_size_kb: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """
        Search models by multiple criteria.

        Args:
            task: Filter by task
            min_accuracy: Minimum accuracy threshold
            max_size_kb: Maximum model size
            tags: Required tags (all must match)

        Returns:
            List of matching models
        """
        results = list(self.models.values())

        if task is not None:
            results = [m for m in results if m.task == task]

        if min_accuracy is not None:
            results = [m for m in results if m.accuracy >= min_accuracy]

        if max_size_kb is not None:
            results = [m for m in results if m.model_size_kb <= max_size_kb]

        if tags is not None:
            results = [m for m in results if all(t in m.tags for t in tags)]

        return results

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the registry."""
        if not self.models:
            return {'count': 0}

        accuracies = [m.accuracy for m in self.models.values()]
        sizes = [m.model_size_kb for m in self.models.values() if m.model_size_kb > 0]

        return {
            'count': len(self.models),
            'tasks': self.list_tasks(),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'avg_size_kb': sum(sizes) / len(sizes) if sizes else 0,
            'total_size_kb': sum(sizes),
        }
