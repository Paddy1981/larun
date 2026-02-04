"""
Base Data Collector
===================
Abstract base class for all astronomical data collectors.
"""

import os
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a collected dataset."""
    name: str
    source: str
    n_samples: int
    n_classes: int
    class_distribution: Dict[str, int]
    collection_date: str
    input_shape: Tuple[int, ...]
    labels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "class_distribution": self.class_distribution,
            "collection_date": self.collection_date,
            "input_shape": list(self.input_shape),
            "labels": self.labels,
            "metadata": self.metadata,
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        data["input_shape"] = tuple(data["input_shape"])
        return cls(**data)


class BaseDataCollector(ABC):
    """
    Abstract base class for astronomical data collectors.

    Each collector is responsible for:
    1. Connecting to a data source (API, archive, etc.)
    2. Querying and downloading raw data
    3. Preprocessing data into model-ready format
    4. Caching data locally
    5. Providing labeled training datasets
    """

    def __init__(self, cache_dir: str = "data/cache",
                 output_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._rate_limit_delay = 1.0  # seconds between API calls
        self._last_request_time = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the data collector."""
        pass

    @property
    @abstractmethod
    def source_url(self) -> str:
        """URL of the data source."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of model IDs this collector provides data for."""
        pass

    @abstractmethod
    def collect(self, n_samples: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
        """
        Collect and preprocess data.

        Args:
            n_samples: Number of samples to collect
            **kwargs: Additional collection parameters

        Returns:
            Tuple of (X, y, dataset_info)
        """
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get list of class labels."""
        pass

    def _rate_limit(self):
        """Enforce rate limiting for API calls."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.name}_{hash_key}.npz"

    def _load_from_cache(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        return None

    def _save_to_cache(self, key: str, data: Dict[str, np.ndarray]):
        """Save data to cache."""
        cache_path = self._get_cache_path(key)
        np.savez_compressed(cache_path, **data)
        logger.info(f"Saved to cache: {cache_path}")

    def save_dataset(self, X: np.ndarray, y: np.ndarray,
                    info: DatasetInfo, filename: str = None) -> Path:
        """
        Save collected dataset to disk.

        Args:
            X: Feature array
            y: Label array
            info: Dataset information
            filename: Optional custom filename

        Returns:
            Path to saved dataset
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}"

        data_path = self.output_dir / f"{filename}.npz"
        info_path = self.output_dir / f"{filename}_info.json"

        # Compute checksum
        checksum = hashlib.md5(X.tobytes() + y.tobytes()).hexdigest()
        info.checksum = checksum

        # Save data
        np.savez_compressed(data_path, X=X, y=y)

        # Save info
        with open(info_path, 'w') as f:
            json.dump(info.to_dict(), f, indent=2)

        logger.info(f"Dataset saved: {data_path}")
        return data_path

    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
        """
        Load a saved dataset.

        Args:
            filename: Dataset filename (without extension)

        Returns:
            Tuple of (X, y, dataset_info)
        """
        data_path = self.output_dir / f"{filename}.npz"
        info_path = self.output_dir / f"{filename}_info.json"

        data = np.load(data_path)
        X, y = data['X'], data['y']

        with open(info_path, 'r') as f:
            info = DatasetInfo.from_dict(json.load(f))

        return X, y, info

    def list_cached_datasets(self) -> List[str]:
        """List all cached datasets for this collector."""
        datasets = []
        for path in self.output_dir.glob(f"{self.name}_*_info.json"):
            datasets.append(path.stem.replace("_info", ""))
        return datasets

    def clear_cache(self):
        """Clear all cached data for this collector."""
        for path in self.cache_dir.glob(f"{self.name}_*"):
            path.unlink()
        logger.info(f"Cleared cache for {self.name}")

    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate collected data.

        Args:
            X: Feature array
            y: Label array

        Returns:
            True if data is valid
        """
        if X.shape[0] != y.shape[0]:
            logger.error(f"Shape mismatch: X={X.shape}, y={y.shape}")
            return False

        if np.isnan(X).any():
            logger.warning("Data contains NaN values")
            return False

        if np.isinf(X).any():
            logger.warning("Data contains infinite values")
            return False

        return True

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Standard preprocessing for collected data.
        Override in subclasses for custom preprocessing.

        Args:
            X: Raw feature array

        Returns:
            Preprocessed feature array
        """
        # Remove NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to zero mean, unit variance per sample
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        X = (X - mean) / std

        return X.astype(np.float32)
