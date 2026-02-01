"""
Model Manager for ASTRA - Hot-swap and Parallel Deployment
===========================================================

Manages model versions with support for:
- Hot-swapping models without downtime
- Parallel deployment (A/B testing)
- Automatic updates on significant improvements
- Rollback capabilities
- Version tracking and metrics
"""

import os
import json
import shutil
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class ModelStatus(Enum):
    """Status of a deployed model."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    STANDBY = "standby"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    REPLACE = "replace"          # Replace current model
    PARALLEL = "parallel"        # Run alongside current
    BLUE_GREEN = "blue_green"    # Switch between two slots
    CANARY = "canary"            # Gradual rollout
    SHADOW = "shadow"            # Run in background, compare


@dataclass
class ModelVersion:
    """A specific version of a model."""
    model_id: str
    version: str
    path: str
    checksum: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ModelStatus = ModelStatus.PENDING
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deployment_slot: Optional[str] = None  # "blue", "green", or None

    @property
    def version_id(self) -> str:
        return f"{self.model_id}@{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'path': self.path,
            'checksum': self.checksum,
            'created_at': self.created_at,
            'status': self.status.value,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'deployment_slot': self.deployment_slot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        data = data.copy()
        data['status'] = ModelStatus(data.get('status', 'pending'))
        return cls(**data)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    auto_update: bool = True
    min_improvement_threshold: float = 0.05  # 5% improvement required
    canary_percentage: float = 0.1  # 10% traffic for canary
    health_check_interval_seconds: int = 60
    rollback_on_error: bool = True
    max_parallel_versions: int = 2


@dataclass
class DeploymentEvent:
    """Record of a deployment event."""
    event_type: str  # deploy, rollback, promote, retire
    model_version: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


# =============================================================================
# Model Loader Interface
# =============================================================================

class ModelLoader(ABC):
    """Abstract interface for loading models."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load a model from path."""
        pass

    @abstractmethod
    def predict(self, model: Any, input_data: Any) -> Any:
        """Run inference with a model."""
        pass

    @abstractmethod
    def validate(self, model: Any) -> bool:
        """Validate a loaded model."""
        pass


class TFLiteModelLoader(ModelLoader):
    """Loader for TFLite models."""

    def load(self, path: str) -> Any:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return None

    def predict(self, model: Any, input_data: Any) -> Any:
        import numpy as np

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()

        return model.get_tensor(output_details[0]['index'])

    def validate(self, model: Any) -> bool:
        try:
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            return len(input_details) > 0 and len(output_details) > 0
        except Exception:
            return False


class OllamaModelLoader(ModelLoader):
    """Loader for Ollama LLM models."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def load(self, path: str) -> Any:
        # For Ollama, path is the model name
        return {'model_name': path, 'base_url': self.base_url}

    def predict(self, model: Any, input_data: Any) -> Any:
        import requests

        response = requests.post(
            f"{model['base_url']}/api/generate",
            json={
                'model': model['model_name'],
                'prompt': input_data,
                'stream': False,
            },
            timeout=120,
        )
        return response.json().get('response', '')

    def validate(self, model: Any) -> bool:
        import requests
        try:
            response = requests.get(
                f"{model['base_url']}/api/tags",
                timeout=5,
            )
            models = response.json().get('models', [])
            return any(m['name'] == model['model_name'] for m in models)
        except Exception:
            return False


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """
    Manages model versions with hot-swap and parallel deployment.

    Features:
    - Blue-green deployment for zero-downtime updates
    - Automatic model updates when improvements detected
    - Rollback capabilities
    - A/B testing support
    - Metrics tracking
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        config: Optional[DeploymentConfig] = None,
        loader: Optional[ModelLoader] = None,
    ):
        if models_dir is None:
            models_dir = Path.home() / '.larun' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = models_dir
        self.config = config or DeploymentConfig()
        self.loader = loader or TFLiteModelLoader()

        self._versions: Dict[str, ModelVersion] = {}
        self._active_models: Dict[str, Any] = {}  # model_id -> loaded model
        self._slots = {'blue': None, 'green': None}  # Blue-green slots
        self._current_slot = 'blue'

        self._lock = threading.RLock()
        self._events: List[DeploymentEvent] = []

        self._registry_file = models_dir / 'registry.json'
        self._load_registry()

        # Background health checker
        self._health_thread = None
        self._stop_health_check = threading.Event()

    # -------------------------------------------------------------------------
    # Registry Management
    # -------------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file) as f:
                    data = json.load(f)
                    for vid, vdata in data.get('versions', {}).items():
                        self._versions[vid] = ModelVersion.from_dict(vdata)
                    self._current_slot = data.get('current_slot', 'blue')
                    self._slots = data.get('slots', {'blue': None, 'green': None})
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        with self._lock:
            data = {
                'versions': {vid: v.to_dict() for vid, v in self._versions.items()},
                'current_slot': self._current_slot,
                'slots': self._slots,
                'last_updated': datetime.utcnow().isoformat(),
            }
            with open(self._registry_file, 'w') as f:
                json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # Model Registration
    # -------------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        version: str,
        source_path: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_id: Unique model identifier
            version: Semantic version string
            source_path: Path to model file
            metrics: Performance metrics
            metadata: Additional metadata

        Returns:
            Registered ModelVersion
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")

        # Compute checksum
        with open(source, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Copy to models directory
        dest_dir = self.models_dir / model_id / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / source.name
        shutil.copy2(source, dest_path)

        # Create version record
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            path=str(dest_path),
            checksum=checksum,
            metrics=metrics or {},
            metadata=metadata or {},
        )

        with self._lock:
            self._versions[model_version.version_id] = model_version
            self._save_registry()

        logger.info(f"Registered model: {model_version.version_id}")
        return model_version

    # -------------------------------------------------------------------------
    # Deployment
    # -------------------------------------------------------------------------

    def deploy(
        self,
        model_id: str,
        version: str,
        strategy: Optional[DeploymentStrategy] = None,
    ) -> bool:
        """
        Deploy a model version.

        Args:
            model_id: Model identifier
            version: Version to deploy
            strategy: Deployment strategy (uses config default if None)

        Returns:
            True if deployment successful
        """
        strategy = strategy or self.config.strategy
        version_id = f"{model_id}@{version}"

        with self._lock:
            if version_id not in self._versions:
                logger.error(f"Version not found: {version_id}")
                return False

            model_version = self._versions[version_id]
            model_version.status = ModelStatus.DEPLOYING

        try:
            # Load the model
            loaded_model = self.loader.load(model_version.path)
            if not loaded_model or not self.loader.validate(loaded_model):
                raise ValueError("Model validation failed")

            with self._lock:
                if strategy == DeploymentStrategy.BLUE_GREEN:
                    self._deploy_blue_green(model_id, model_version, loaded_model)
                elif strategy == DeploymentStrategy.PARALLEL:
                    self._deploy_parallel(model_id, model_version, loaded_model)
                else:
                    self._deploy_replace(model_id, model_version, loaded_model)

                model_version.status = ModelStatus.ACTIVE
                self._save_registry()

            self._log_event('deploy', version_id, {'strategy': strategy.value})
            logger.info(f"Deployed {version_id} with strategy {strategy.value}")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            with self._lock:
                model_version.status = ModelStatus.FAILED
                self._save_registry()

            self._log_event('deploy', version_id, {'error': str(e)}, success=False)
            return False

    def _deploy_blue_green(
        self,
        model_id: str,
        version: ModelVersion,
        loaded_model: Any,
    ) -> None:
        """Blue-green deployment: deploy to inactive slot, then switch."""
        # Determine target slot (opposite of current)
        target_slot = 'green' if self._current_slot == 'blue' else 'blue'

        # Deploy to target slot
        version.deployment_slot = target_slot
        self._slots[target_slot] = version.version_id
        self._active_models[f"{model_id}_{target_slot}"] = loaded_model

        # Switch active slot
        self._current_slot = target_slot

    def _deploy_parallel(
        self,
        model_id: str,
        version: ModelVersion,
        loaded_model: Any,
    ) -> None:
        """Parallel deployment: run alongside existing."""
        version.deployment_slot = 'parallel'
        key = f"{model_id}_parallel_{version.version}"
        self._active_models[key] = loaded_model

    def _deploy_replace(
        self,
        model_id: str,
        version: ModelVersion,
        loaded_model: Any,
    ) -> None:
        """Simple replacement deployment."""
        # Retire old version
        if model_id in self._active_models:
            old_version = self._get_active_version(model_id)
            if old_version:
                old_version.status = ModelStatus.DEPRECATED

        self._active_models[model_id] = loaded_model

    # -------------------------------------------------------------------------
    # Model Access
    # -------------------------------------------------------------------------

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get the currently active model."""
        with self._lock:
            # Try current slot first (blue-green)
            slot_key = f"{model_id}_{self._current_slot}"
            if slot_key in self._active_models:
                return self._active_models[slot_key]

            # Fall back to direct key
            return self._active_models.get(model_id)

    def get_all_active(self, model_id: str) -> Dict[str, Any]:
        """Get all active versions of a model (for A/B testing)."""
        with self._lock:
            result = {}
            for key, model in self._active_models.items():
                if key.startswith(model_id):
                    result[key] = model
            return result

    def _get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the active version record for a model."""
        for v in self._versions.values():
            if v.model_id == model_id and v.status == ModelStatus.ACTIVE:
                return v
        return None

    # -------------------------------------------------------------------------
    # Auto-Update
    # -------------------------------------------------------------------------

    def check_for_updates(
        self,
        model_id: str,
        new_metrics: Dict[str, float],
        new_path: str,
        new_version: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new model version should be deployed.

        Args:
            model_id: Model identifier
            new_metrics: Metrics of new version
            new_path: Path to new model
            new_version: Version string

        Returns:
            Tuple of (should_deploy, reason)
        """
        current = self._get_active_version(model_id)

        if not current:
            return True, "No active version exists"

        if not self.config.auto_update:
            return False, "Auto-update disabled"

        # Compare key metrics
        improvement = self._calculate_improvement(current.metrics, new_metrics)

        if improvement >= self.config.min_improvement_threshold:
            return True, f"Improvement of {improvement:.1%} exceeds threshold"

        return False, f"Improvement of {improvement:.1%} below threshold"

    def _calculate_improvement(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
    ) -> float:
        """Calculate overall improvement percentage."""
        key_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        improvements = []

        for metric in key_metrics:
            if metric in old_metrics and metric in new_metrics:
                if old_metrics[metric] > 0:
                    imp = (new_metrics[metric] - old_metrics[metric]) / old_metrics[metric]
                    improvements.append(imp)

        return sum(improvements) / len(improvements) if improvements else 0

    def auto_deploy_if_improved(
        self,
        model_id: str,
        new_path: str,
        new_version: str,
        new_metrics: Dict[str, float],
    ) -> bool:
        """
        Automatically deploy if the new version is significantly better.

        Returns True if deployed, False otherwise.
        """
        should_deploy, reason = self.check_for_updates(
            model_id, new_metrics, new_path, new_version
        )

        logger.info(f"Update check for {model_id}: {reason}")

        if should_deploy:
            # Register and deploy
            self.register_model(model_id, new_version, new_path, new_metrics)
            return self.deploy(model_id, new_version)

        return False

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    def rollback(self, model_id: str) -> bool:
        """
        Rollback to the previous version.

        For blue-green: switch back to previous slot
        For others: redeploy previous version
        """
        with self._lock:
            if self.config.strategy == DeploymentStrategy.BLUE_GREEN:
                # Simply switch slots
                self._current_slot = 'green' if self._current_slot == 'blue' else 'blue'
                self._save_registry()
                self._log_event('rollback', model_id, {'new_slot': self._current_slot})
                logger.info(f"Rolled back {model_id} to slot {self._current_slot}")
                return True

            # Find previous active version
            versions = [
                v for v in self._versions.values()
                if v.model_id == model_id and v.status in [ModelStatus.DEPRECATED, ModelStatus.STANDBY]
            ]

            if not versions:
                logger.error("No previous version to rollback to")
                return False

            # Sort by creation date, get most recent
            versions.sort(key=lambda x: x.created_at, reverse=True)
            previous = versions[0]

            return self.deploy(model_id, previous.version)

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------

    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_thread and self._health_thread.is_alive():
            return

        self._stop_health_check.clear()
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self._stop_health_check.set()
        if self._health_thread:
            self._health_thread.join(timeout=5)

    def _health_loop(self) -> None:
        """Background health check loop."""
        while not self._stop_health_check.is_set():
            try:
                self._check_model_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._stop_health_check.wait(self.config.health_check_interval_seconds)

    def _check_model_health(self) -> None:
        """Check health of all active models."""
        with self._lock:
            for key, model in list(self._active_models.items()):
                try:
                    if not self.loader.validate(model):
                        logger.warning(f"Model {key} failed health check")
                        # Could trigger rollback here
                except Exception as e:
                    logger.error(f"Health check failed for {key}: {e}")

    # -------------------------------------------------------------------------
    # Event Logging
    # -------------------------------------------------------------------------

    def _log_event(
        self,
        event_type: str,
        version_id: str,
        details: Dict[str, Any] = None,
        success: bool = True,
    ) -> None:
        """Log a deployment event."""
        event = DeploymentEvent(
            event_type=event_type,
            model_version=version_id,
            details=details or {},
            success=success,
        )
        self._events.append(event)

        # Keep last 100 events
        if len(self._events) > 100:
            self._events = self._events[-100:]

    def get_deployment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent deployment events."""
        return [
            {
                'event_type': e.event_type,
                'model_version': e.model_version,
                'timestamp': e.timestamp,
                'details': e.details,
                'success': e.success,
            }
            for e in self._events[-limit:]
        ]

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        with self._lock:
            return {
                'current_slot': self._current_slot,
                'slots': self._slots,
                'active_models': list(self._active_models.keys()),
                'versions': {
                    vid: {
                        'status': v.status.value,
                        'slot': v.deployment_slot,
                        'metrics': v.metrics,
                    }
                    for vid, v in self._versions.items()
                },
                'strategy': self.config.strategy.value,
                'auto_update': self.config.auto_update,
            }
