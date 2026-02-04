"""
Gaia Data Collector
===================
Collects stellar parameters from ESA Gaia DR3 archive.

Data Source: https://gea.esac.esa.int/archive/
Used by: SPECTYPE-001, provides stellar context for other models
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from src.data.collectors.base import BaseDataCollector, DatasetInfo

logger = logging.getLogger(__name__)


# Typical stellar parameters by spectral type (for synthetic data generation)
# [Teff, log(g), [Fe/H], BP-RP, G-RP, BP-G, parallax_error_ratio, phot_g_mean_mag]
SPECTRAL_TYPE_PARAMS = {
    "O": [35000, 4.0, 0.0, -0.3, -0.1, -0.2, 0.02, 6.0],
    "B": [18000, 4.0, 0.0, -0.15, -0.05, -0.1, 0.02, 8.0],
    "A": [8500, 4.2, 0.0, 0.1, 0.05, 0.05, 0.03, 10.0],
    "F": [6500, 4.3, 0.0, 0.4, 0.2, 0.2, 0.03, 11.0],
    "G": [5500, 4.4, 0.0, 0.8, 0.4, 0.4, 0.04, 12.0],
    "K": [4500, 4.5, 0.0, 1.2, 0.6, 0.6, 0.05, 13.0],
    "M": [3200, 4.8, 0.0, 2.5, 1.3, 1.2, 0.08, 15.0],
    "L": [2000, 5.0, 0.0, 4.0, 2.0, 2.0, 0.15, 18.0],
}


class GaiaCollector(BaseDataCollector):
    """
    Collector for ESA Gaia DR3 archive data.

    Provides:
    - Stellar photometry for spectral type classification (SPECTYPE-001)
    - Stellar parameters (Teff, log(g), metallicity)
    - Astrometry (parallax, proper motion)
    - Nearby source information for contamination assessment

    Data is collected via:
    1. astroquery.gaia for TAP queries
    2. Direct HTTP API calls
    3. Bulk downloads for large queries
    """

    def __init__(self, cache_dir: str = "data/cache/gaia",
                 output_dir: str = "data/processed/gaia"):
        super().__init__(cache_dir, output_dir)
        self._astroquery_available = self._check_astroquery()

    def _check_astroquery(self) -> bool:
        """Check if astroquery is available."""
        try:
            from astroquery.gaia import Gaia
            return True
        except ImportError:
            logger.warning("astroquery not installed. Install with: pip install astroquery")
            return False

    @property
    def name(self) -> str:
        return "gaia"

    @property
    def source_url(self) -> str:
        return "https://gea.esac.esa.int/archive/"

    @property
    def supported_models(self) -> List[str]:
        return ["SPECTYPE-001"]

    def get_labels(self) -> List[str]:
        return ["O", "B", "A", "F", "G", "K", "M", "L"]

    def collect(self, n_samples: int, use_cache: bool = True,
                magnitude_range: Tuple[float, float] = (6.0, 16.0),
                **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
        """
        Collect stellar parameters from Gaia DR3.

        Args:
            n_samples: Number of stars to collect
            use_cache: Whether to use cached data
            magnitude_range: G magnitude range to query

        Returns:
            Tuple of (X, y, dataset_info)
        """
        cache_key = f"gaia_dr3_{n_samples}_{magnitude_range}"

        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                X, y = cached['X'], cached['y']
                info = self._create_info(X, y)
                return X, y, info

        if self._astroquery_available:
            X, y = self._collect_with_astroquery(n_samples, magnitude_range)
        else:
            X, y = self._collect_synthetic(n_samples)
            logger.warning("Using synthetic data - install astroquery for real Gaia data")

        # Validate
        if not self.validate_data(X, y):
            raise ValueError("Data validation failed")

        # Cache
        self._save_to_cache(cache_key, {'X': X, 'y': y})

        info = self._create_info(X, y)
        return X, y, info

    def _collect_with_astroquery(self, n_samples: int,
                                  magnitude_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data using astroquery."""
        from astroquery.gaia import Gaia

        logger.info(f"Collecting {n_samples} stars from Gaia DR3...")

        X_list = []
        y_list = []

        # Collect samples for each spectral type
        samples_per_type = n_samples // len(self.get_labels())

        for type_idx, spec_type in enumerate(self.get_labels()):
            logger.info(f"Querying {spec_type}-type stars...")

            # Temperature range for this spectral type
            teff_ranges = {
                "O": (30000, 50000),
                "B": (10000, 30000),
                "A": (7500, 10000),
                "F": (6000, 7500),
                "G": (5200, 6000),
                "K": (3700, 5200),
                "M": (2400, 3700),
                "L": (1300, 2400),
            }

            teff_min, teff_max = teff_ranges[spec_type]

            query = f"""
            SELECT TOP {samples_per_type}
                source_id,
                teff_gspphot,
                logg_gspphot,
                mh_gspphot,
                bp_rp,
                g_rp,
                bp_g,
                parallax,
                phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE teff_gspphot BETWEEN {teff_min} AND {teff_max}
                AND phot_g_mean_mag BETWEEN {magnitude_range[0]} AND {magnitude_range[1]}
                AND parallax > 0
                AND bp_rp IS NOT NULL
            ORDER BY random_index
            """

            try:
                self._rate_limit()
                job = Gaia.launch_job_async(query)
                results = job.get_results()

                for row in results:
                    features = [
                        float(row['teff_gspphot']) / 10000,  # Normalize Teff
                        float(row['logg_gspphot']),
                        float(row['mh_gspphot']) if row['mh_gspphot'] else 0.0,
                        float(row['bp_rp']),
                        float(row['g_rp']),
                        float(row['bp_g']),
                        float(row['parallax']) / 100,  # Normalize parallax
                        float(row['phot_g_mean_mag']) / 20,  # Normalize magnitude
                    ]
                    X_list.append(features)
                    y_list.append(type_idx)

            except Exception as e:
                logger.warning(f"Error querying {spec_type} stars: {e}")
                # Fall back to synthetic for this type
                for _ in range(samples_per_type):
                    features = self._generate_synthetic_star(spec_type)
                    X_list.append(features)
                    y_list.append(type_idx)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        return X, y

    def _collect_synthetic(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic stellar parameters."""
        from src.model.data_generators import SpectralTypeDataGenerator, DatasetConfig

        logger.info("Generating synthetic Gaia-like data...")

        generator = SpectralTypeDataGenerator(n_features=8)
        config = DatasetConfig(n_samples=n_samples, noise_level=0.05, seed=42)
        X, y = generator.generate_dataset(config)

        return X, y

    def _generate_synthetic_star(self, spec_type: str) -> List[float]:
        """Generate synthetic parameters for a single star."""
        base_params = SPECTRAL_TYPE_PARAMS[spec_type]
        noise = np.random.normal(0, 0.1, len(base_params))

        features = [
            (base_params[0] + noise[0] * 1000) / 10000,  # Teff
            base_params[1] + noise[1] * 0.2,  # log(g)
            base_params[2] + noise[2] * 0.2,  # [Fe/H]
            base_params[3] + noise[3] * 0.1,  # BP-RP
            base_params[4] + noise[4] * 0.1,  # G-RP
            base_params[5] + noise[5] * 0.1,  # BP-G
            base_params[6] + noise[6] * 0.01,  # parallax
            base_params[7] + noise[7] * 1.0,  # G mag
        ]

        return features

    def _create_info(self, X: np.ndarray, y: np.ndarray) -> DatasetInfo:
        """Create dataset info object."""
        from datetime import datetime
        from collections import Counter

        labels = self.get_labels()
        class_counts = Counter(y)
        class_dist = {labels[k]: v for k, v in class_counts.items()}

        return DatasetInfo(
            name="gaia_dr3_spectral_types",
            source=self.source_url,
            n_samples=len(X),
            n_classes=len(labels),
            class_distribution=class_dist,
            collection_date=datetime.now().isoformat(),
            input_shape=X.shape[1:],
            labels=labels,
            metadata={
                "catalog": "Gaia DR3",
                "features": ["teff", "logg", "mh", "bp_rp", "g_rp", "bp_g", "parallax", "phot_g_mean_mag"]
            }
        )

    def query_by_coordinates(self, ra: float, dec: float, radius: float = 0.01) -> List[Dict[str, Any]]:
        """
        Query Gaia sources near coordinates.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees

        Returns:
            List of Gaia source entries
        """
        if not self._astroquery_available:
            logger.error("astroquery required for coordinate queries")
            return []

        from astroquery.gaia import Gaia
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        try:
            self._rate_limit()
            j = Gaia.cone_search_async(coord, radius=radius * u.degree)
            results = j.get_results()

            sources = []
            for row in results:
                sources.append({
                    "source_id": row['source_id'],
                    "ra": row['ra'],
                    "dec": row['dec'],
                    "parallax": row['parallax'],
                    "phot_g_mean_mag": row['phot_g_mean_mag'],
                    "bp_rp": row['bp_rp'],
                    "teff_gspphot": row.get('teff_gspphot'),
                })

            return sources

        except Exception as e:
            logger.error(f"Error querying coordinates: {e}")
            return []

    def get_stellar_params(self, source_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed stellar parameters for a Gaia source.

        Args:
            source_id: Gaia DR3 source ID

        Returns:
            Dictionary of stellar parameters
        """
        if not self._astroquery_available:
            logger.error("astroquery required for parameter queries")
            return None

        from astroquery.gaia import Gaia

        query = f"""
        SELECT
            source_id,
            ra, dec,
            parallax, parallax_error,
            pmra, pmdec,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            bp_rp, bp_g, g_rp,
            teff_gspphot, logg_gspphot, mh_gspphot,
            radius_gspphot, distance_gspphot,
            ruwe
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
        """

        try:
            self._rate_limit()
            job = Gaia.launch_job(query)
            results = job.get_results()

            if len(results) == 0:
                return None

            row = results[0]
            return {k: float(row[k]) if row[k] is not None else None for k in results.colnames}

        except Exception as e:
            logger.error(f"Error getting params for {source_id}: {e}")
            return None

    def get_nfpp_sources(self, ra: float, dec: float, radius_arcsec: float = 60) -> List[Dict[str, Any]]:
        """
        Get nearby sources for NFPP (Nearby False Positive Probability) calculation.

        Args:
            ra: Target RA in degrees
            dec: Target Dec in degrees
            radius_arcsec: Search radius in arcseconds

        Returns:
            List of nearby sources with magnitudes
        """
        radius_deg = radius_arcsec / 3600.0
        return self.query_by_coordinates(ra, dec, radius_deg)
