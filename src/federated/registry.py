"""
LARUN Federated Model Registry
==============================
Central registry for managing multiple specialized TinyML models.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    
    Attributes:
        model_id: Unique identifier for the model
        version: Semantic version string (e.g., "1.0.0")
        task: Task type ("transit_detection", "stellar_classification", etc.)
        accuracy: Validation accuracy (0-1)
        input_shape: Expected input shape as tuple
        created_at: ISO format timestamp
        file_path: Path to the model file (.tflite or .h5)
        is_quantized: Whether the model is INT8 quantized
        description: Human-readable description
        metrics: Additional metrics dict (precision, recall, etc.)
    """
    model_id: str
    version: str
    task: str
    accuracy: float
    input_shape: Tuple[int, ...]
    created_at: str
    file_path: str
    is_quantized: bool = False
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'task': self.task,
            'accuracy': round(self.accuracy, 4),
            'input_shape': list(self.input_shape),
            'created_at': self.created_at,
            'file_path': self.file_path,
            'is_quantized': self.is_quantized,
            'description': self.description,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data['input_shape'] = tuple(data.get('input_shape', (1024, 1)))
        return cls(**data)


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Central registry for federated TinyML models.
    
    Manages model registration, versioning, and lazy loading.
    Supports multiple models for each task type for ensemble predictions.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register(ModelMetadata(
        ...     model_id="transit_v1",
        ...     version="1.0.0",
        ...     task="transit_detection",
        ...     accuracy=0.85,
        ...     input_shape=(1024, 1),
        ...     created_at="2026-01-31T00:00:00Z",
        ...     file_path="models/real/transit_v1.tflite"
        ... ))
        >>> model = registry.load_model("transit_v1")
    """
    
    # Supported task types
    TASK_TYPES = [
        'transit_detection',
        'stellar_classification',
        'binary_discrimination',
        'habitability_assessment',
        'anomaly_detection',
        'spectral_classification',
    ]
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to registry JSON file (optional)
                          If not provided, uses default location
        """
        self.registry_path = registry_path or Path("models/registry.json")
        self.models: Dict[str, ModelMetadata] = {}
        self._loaded_models: Dict[str, Any] = {}  # Lazy loading cache
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk if it exists."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                for model_data in data.get('models', []):
                    metadata = ModelMetadata.from_dict(model_data)
                    self.models[metadata.model_id] = metadata
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'version': '1.0',
            'updated_at': datetime.utcnow().isoformat() + 'Z',
            'models': [m.to_dict() for m in self.models.values()]
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved registry with {len(self.models)} models")
    
    def register(self, metadata: ModelMetadata, save: bool = True) -> None:
        """
        Register a model in the registry.
        
        Args:
            metadata: Model metadata
            save: Whether to persist to disk immediately
        
        Raises:
            ValueError: If model_id already exists with higher version
        """
        if metadata.model_id in self.models:
            existing = self.models[metadata.model_id]
            if existing.version >= metadata.version:
                logger.warning(
                    f"Model {metadata.model_id} v{existing.version} already exists. "
                    f"Skipping v{metadata.version}"
                )
                return
        
        self.models[metadata.model_id] = metadata
        logger.info(f"Registered model: {metadata.model_id} v{metadata.version}")
        
        if save:
            self._save_registry()
    
    def unregister(self, model_id: str, save: bool = True) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: Model identifier to remove
            save: Whether to persist change to disk
            
        Returns:
            True if model was removed, False if not found
        """
        if model_id in self.models:
            del self.models[model_id]
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
            if save:
                self._save_registry()
            logger.info(f"Unregistered model: {model_id}")
            return True
        return False
    
    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelMetadata if found, None otherwise
        """
        return self.models.get(model_id)
    
    def get_by_task(self, task: str) -> List[ModelMetadata]:
        """
        Get all models for a specific task.
        
        Args:
            task: Task type (e.g., "transit_detection")
            
        Returns:
            List of ModelMetadata for that task, sorted by version (newest first)
        """
        models = [m for m in self.models.values() if m.task == task]
        return sorted(models, key=lambda m: m.version, reverse=True)
    
    def get_best_for_task(self, task: str) -> Optional[ModelMetadata]:
        """
        Get the best performing model for a task.
        
        Args:
            task: Task type
            
        Returns:
            ModelMetadata with highest accuracy, or None if no models
        """
        models = self.get_by_task(task)
        if not models:
            return None
        return max(models, key=lambda m: m.accuracy)
    
    def list_tasks(self) -> List[str]:
        """
        Get list of tasks with registered models.
        
        Returns:
            Unique task types with at least one registered model
        """
        return list(set(m.task for m in self.models.values()))
    
    def list_all(self) -> List[ModelMetadata]:
        """
        Get all registered models.
        
        Returns:
            List of all ModelMetadata
        """
        return list(self.models.values())
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load a model for inference (lazy loading with caching).
        
        Args:
            model_id: Model identifier
            
        Returns:
            TFLite interpreter or Keras model, None if not found
        """
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        metadata = self.get(model_id)
        if metadata is None:
            logger.error(f"Model not found: {model_id}")
            return None
        
        model_path = Path(metadata.file_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            if model_path.suffix == '.tflite':
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                self._loaded_models[model_id] = interpreter
                logger.info(f"Loaded TFLite model: {model_id}")
            else:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(model_path))
                self._loaded_models[model_id] = model
                logger.info(f"Loaded Keras model: {model_id}")
            
            return self._loaded_models[model_id]
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if unloaded, False if not loaded
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            return True
        return False
    
    def summary(self) -> str:
        """
        Get human-readable registry summary.
        
        Returns:
            Formatted summary string
        """
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "  LARUN Model Registry",
            "═══════════════════════════════════════════════════════════════",
            f"  Total Models: {len(self.models)}",
            f"  Active Tasks: {len(self.list_tasks())}",
            "───────────────────────────────────────────────────────────────",
        ]
        
        by_task = {}
        for m in self.models.values():
            by_task.setdefault(m.task, []).append(m)
        
        for task, models in sorted(by_task.items()):
            lines.append(f"  {task}:")
            for m in sorted(models, key=lambda x: -x.accuracy):
                q_mark = "Q" if m.is_quantized else " "
                lines.append(
                    f"    [{q_mark}] {m.model_id} v{m.version} "
                    f"({m.accuracy:.0%})"
                )
        
        lines.append("═══════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_registry(path: Optional[str] = None) -> ModelRegistry:
    """
    Create a new model registry.
    
    Args:
        path: Optional path to registry file
        
    Returns:
        ModelRegistry instance
    """
    registry_path = Path(path) if path else None
    return ModelRegistry(registry_path)


def auto_register_models(
    registry: ModelRegistry,
    models_dir: str = "models/real"
) -> int:
    """
    Auto-register all models found in a directory.
    
    Scans for .tflite and .h5 files and registers them with
    inferred metadata.
    
    Args:
        registry: ModelRegistry instance
        models_dir: Directory to scan
        
    Returns:
        Number of models registered
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return 0
    
    count = 0
    for model_file in models_path.glob("*.tflite"):
        model_id = model_file.stem
        if registry.get(model_id):
            continue  # Already registered
        
        # Infer task from filename
        task = "transit_detection"  # Default
        if "stellar" in model_id.lower():
            task = "stellar_classification"
        elif "binary" in model_id.lower():
            task = "binary_discrimination"
        elif "habit" in model_id.lower():
            task = "habitability_assessment"
        
        metadata = ModelMetadata(
            model_id=model_id,
            version="1.0.0",
            task=task,
            accuracy=0.8,  # Placeholder
            input_shape=(1024, 1),
            created_at=datetime.utcnow().isoformat() + 'Z',
            file_path=str(model_file),
            is_quantized='int8' in model_id.lower() or 'quant' in model_id.lower()
        )
        registry.register(metadata, save=False)
        count += 1
    
    if count > 0:
        registry._save_registry()
    
    return count


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Model Registry...")
    print("=" * 60)
    
    # Create a test registry
    registry = ModelRegistry(Path("test_registry.json"))
    
    # Register some test models
    registry.register(ModelMetadata(
        model_id="transit_detector_v1",
        version="1.0.0",
        task="transit_detection",
        accuracy=0.82,
        input_shape=(1024, 1),
        created_at="2026-01-31T00:00:00Z",
        file_path="models/real/astro_tinyml_real.tflite",
        is_quantized=False,
        description="Primary transit detection model"
    ))
    
    registry.register(ModelMetadata(
        model_id="binary_discriminator_v1",
        version="1.0.0",
        task="binary_discrimination",
        accuracy=0.78,
        input_shape=(1024, 1),
        created_at="2026-01-31T00:00:00Z",
        file_path="models/real/binary_v1.tflite",
        is_quantized=True,
        description="Eclipsing binary detector"
    ))
    
    # Print summary
    print(registry.summary())
    
    # Test queries
    print(f"\nModels for transit_detection: {len(registry.get_by_task('transit_detection'))}")
    print(f"Active tasks: {registry.list_tasks()}")
    
    # Cleanup test file
    Path("test_registry.json").unlink(missing_ok=True)
    print("\nTest complete!")
