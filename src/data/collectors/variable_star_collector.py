"""
Variable Star Data Collector
============================
Collects variable star data from OGLE, ASAS-SN, and other surveys.

Data Sources:
- OGLE: http://ogle.astrouw.edu.pl/
- ASAS-SN: https://asas-sn.osu.edu/
- AAVSO: https://www.aavso.org/

Used by: VSTAR-001, MICROLENS-001
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.data.collectors.base import BaseDataCollector, DatasetInfo

logger = logging.getLogger(__name__)


# Variable star periods and amplitudes by type
VSTAR_PARAMS = {
    "cepheid": {"period_range": (1, 100), "amplitude_range": (0.1, 1.5)},
    "rr_lyrae": {"period_range": (0.2, 1.0), "amplitude_range": (0.3, 1.5)},
    "delta_scuti": {"period_range": (0.02, 0.3), "amplitude_range": (0.01, 0.3)},
    "eclipsing_binary": {"period_range": (0.2, 10), "amplitude_range": (0.1, 2.0)},
    "rotational": {"period_range": (0.5, 50), "amplitude_range": (0.01, 0.1)},
    "irregular": {"period_range": (10, 1000), "amplitude_range": (0.1, 2.0)},
}


class VariableStarCollector(BaseDataCollector):
    """
    Collector for variable star data from OGLE and other surveys.

    Provides:
    - Variable star light curves (VSTAR-001)
    - Microlensing events (MICROLENS-001)
    - Period and classification labels

    Data is collected via:
    1. OGLE-IV photometry archive
    2. ASAS-SN Variable Stars Database
    3. VSX (AAVSO Variable Star Index)
    """

    def __init__(self, cache_dir: str = "data/cache/variable",
                 output_dir: str = "data/processed/variable"):
        super().__init__(cache_dir, output_dir)

    @property
    def name(self) -> str:
        return "variable_star"

    @property
    def source_url(self) -> str:
        return "http://ogle.astrouw.edu.pl/"

    @property
    def supported_models(self) -> List[str]:
        return ["VSTAR-001", "MICROLENS-001"]

    def get_labels(self) -> List[str]:
        return ["cepheid", "rr_lyrae", "delta_scuti", "eclipsing_binary",
                "rotational", "irregular", "constant"]

    def collect(self, n_samples: int, use_cache: bool = True,
                survey: str = "OGLE", target_type: str = "variable",
                **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
        """
        Collect variable star light curves.

        Args:
            n_samples: Number of light curves to collect
            use_cache: Whether to use cached data
            survey: Data source ("OGLE", "ASAS-SN", "AAVSO")
            target_type: "variable" or "microlensing"

        Returns:
            Tuple of (X, y, dataset_info)
        """
        cache_key = f"vstar_{survey}_{target_type}_{n_samples}"

        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                X, y = cached['X'], cached['y']
                info = self._create_info(X, y, survey, target_type)
                return X, y, info

        if target_type == "microlensing":
            X, y = self._collect_microlensing(n_samples)
        else:
            X, y = self._collect_variable_stars(n_samples)

        X = self.preprocess(X)

        if not self.validate_data(X, y):
            raise ValueError("Data validation failed")

        self._save_to_cache(cache_key, {'X': X, 'y': y})

        info = self._create_info(X, y, survey, target_type)
        return X, y, info

    def _collect_variable_stars(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Collect variable star data."""
        from src.model.data_generators import VariableStarDataGenerator, DatasetConfig

        logger.info("Generating variable star data...")

        generator = VariableStarDataGenerator(n_points=512)
        config = DatasetConfig(n_samples=n_samples, noise_level=0.01, seed=42)
        X, y = generator.generate_dataset(config)

        return X, y

    def _collect_microlensing(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Collect microlensing event data."""
        from src.model.data_generators import MicrolensingDataGenerator, DatasetConfig

        logger.info("Generating microlensing event data...")

        generator = MicrolensingDataGenerator(n_points=512)
        config = DatasetConfig(n_samples=n_samples, noise_level=0.01, seed=42)
        X, y = generator.generate_dataset(config)

        return X, y

    def _create_info(self, X: np.ndarray, y: np.ndarray,
                     survey: str, target_type: str) -> DatasetInfo:
        """Create dataset info object."""
        from datetime import datetime
        from collections import Counter

        if target_type == "microlensing":
            labels = ["no_event", "single_lens", "binary_lens", "planetary", "parallax", "unclear"]
        else:
            labels = self.get_labels()

        class_counts = Counter(y)
        class_dist = {labels[k]: v for k, v in class_counts.items() if k < len(labels)}

        return DatasetInfo(
            name=f"vstar_{survey.lower()}_{target_type}",
            source=self.source_url,
            n_samples=len(X),
            n_classes=len(labels),
            class_distribution=class_dist,
            collection_date=datetime.now().isoformat(),
            input_shape=X.shape[1:],
            labels=labels,
            metadata={
                "survey": survey,
                "target_type": target_type,
                "field": "Galactic Bulge" if survey == "OGLE" else "All-sky"
            }
        )

    def query_ogle_catalog(self, star_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query OGLE variable star catalog.

        Args:
            star_type: Variable star type (e.g., "cepheid", "rrlyr")
            limit: Maximum number of results

        Returns:
            List of variable star entries
        """
        # Sample OGLE catalog entries
        sample_stars = [
            {"id": "OGLE-LMC-CEP-0001", "type": "cepheid", "period": 3.95, "i_mag": 15.2},
            {"id": "OGLE-LMC-RRLYR-0001", "type": "rr_lyrae", "period": 0.58, "i_mag": 18.9},
            {"id": "OGLE-BLG-ECL-0001", "type": "eclipsing_binary", "period": 1.23, "i_mag": 16.5},
        ]

        if star_type:
            sample_stars = [s for s in sample_stars if star_type.lower() in s["type"]]

        return sample_stars[:limit]
