"""
Base model class for all LARUN TinyML models.
All Layer 2 (server-side) models inherit from this.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class BaseModel(ABC):
    """
    Abstract base class for all LARUN TinyML models.

    Subclasses must implement:
    - extract_features(times, magnitudes, errors=None) -> np.ndarray
    - predict(times, magnitudes, errors=None) -> tuple[str, np.ndarray]
    - CLASSES: dict[int, str]
    - MODEL_ID: str  (e.g. "VARDET-001")
    """

    MODEL_ID: str = ""
    CLASSES: dict[int, str] = {}

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self._model_path = model_path
        self._loaded = False

    def load(self, path: str | Path | None = None) -> "BaseModel":
        """Load model weights from .npz file."""
        target = Path(path or self._model_path or MODELS_DIR / f"{self.MODEL_ID.lower().replace('-', '_')}.npz")
        if not target.exists():
            logger.warning(f"Model file not found: {target}. Model will train on first predict().")
            return self
        data = np.load(target, allow_pickle=True)
        self._load_weights(data)
        self._loaded = True
        logger.info(f"Loaded {self.MODEL_ID} from {target}")
        return self

    def save(self, path: str | Path | None = None) -> Path:
        """Save model weights to .npz file."""
        target = Path(path or MODELS_DIR / f"{self.MODEL_ID.lower().replace('-', '_')}.npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        weights = self._get_weights()
        np.savez_compressed(target, **weights)
        logger.info(f"Saved {self.MODEL_ID} to {target}")
        return target

    @abstractmethod
    def extract_features(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract feature vector from a light curve."""

    @abstractmethod
    def predict(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Classify a light curve.

        Returns:
            (label, probability_vector)
        """

    def _load_weights(self, data: Any) -> None:
        """Override to load model-specific weights from npz data."""

    def _get_weights(self) -> dict[str, np.ndarray]:
        """Override to return model-specific weights for saving."""
        return {}

    def result_dict(
        self,
        label: str,
        proba: np.ndarray,
        extra: dict | None = None,
    ) -> dict:
        """Standard result format."""
        return {
            "model_id": self.MODEL_ID,
            "label": label,
            "confidence": float(np.max(proba)),
            "probabilities": {self.CLASSES[i]: float(p) for i, p in enumerate(proba)},
            **(extra or {}),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.MODEL_ID}, loaded={self._loaded})"
