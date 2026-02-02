"""
Detection Service
=================
Main service class for transit detection.

This is the primary interface for Agent BETA to use. It provides
async methods for analyzing TESS targets and light curves.

Author: Agent ALPHA
Reference: .coordination/MVP_INTERFACES.md
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
import time as time_module

from .models import (
    DetectionResult,
    LightCurveData,
    TargetNotFoundError,
    DataUnavailableError,
    AnalysisError,
)
from .detector import TransitDetector

logger = logging.getLogger(__name__)


class IDetectionService(ABC):
    """
    Interface for detection service.

    This abstract base class defines the contract that DetectionService
    must implement. Agent BETA should code against this interface.
    """

    @abstractmethod
    async def analyze(self, tic_id: str) -> DetectionResult:
        """
        Analyze a target by TIC ID.

        Args:
            tic_id: TESS Input Catalog ID (e.g., "TIC 12345678" or "12345678")

        Returns:
            DetectionResult with all analysis data

        Raises:
            TargetNotFoundError: If TIC ID not found in MAST
            DataUnavailableError: If no light curve data available
            AnalysisError: If analysis fails
        """
        pass

    @abstractmethod
    async def analyze_lightcurve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        tic_id: Optional[str] = None,
    ) -> DetectionResult:
        """
        Analyze provided light curve data directly.

        Args:
            time: Time array (BJD or BTJD)
            flux: Normalized flux array
            flux_err: Optional flux error array
            tic_id: Optional TIC ID for metadata

        Returns:
            DetectionResult with all analysis data
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get service health status."""
        pass


class DetectionService(IDetectionService):
    """
    Main detection service for transit analysis.

    This service wraps the TransitDetector and provides async methods
    for analyzing targets. It handles data fetching from MAST and
    coordinates the full detection pipeline.

    Example:
        >>> service = DetectionService()
        >>> result = await service.analyze("TIC 470710327")
        >>> if result.detection:
        ...     print(f"Planet candidate: P={result.period_days:.4f}d")

    For direct light curve analysis:
        >>> result = await service.analyze_lightcurve(time, flux)
    """

    def __init__(
        self,
        min_period: float = 0.5,
        max_period: float = 50.0,
        min_snr: float = 7.0,
        cache_dir: str = "data/cache",
    ):
        """
        Initialize the detection service.

        Args:
            min_period: Minimum search period (days)
            max_period: Maximum search period (days)
            min_snr: Minimum SNR threshold
            cache_dir: Directory for caching data
        """
        self.min_period = min_period
        self.max_period = max_period
        self.min_snr = min_snr
        self.cache_dir = cache_dir

        # Initialize detector
        self.detector = TransitDetector(
            min_period=min_period,
            max_period=max_period,
            min_snr=min_snr,
        )

        # Track service stats
        self._analyses_count = 0
        self._start_time = time_module.time()

        logger.info(
            f"DetectionService initialized: period=[{min_period}, {max_period}]d"
        )

    async def analyze(self, tic_id: str) -> DetectionResult:
        """
        Analyze a target by TIC ID.

        Fetches light curve data from MAST and runs the full detection
        pipeline including BLS periodogram and vetting tests.

        Args:
            tic_id: TESS Input Catalog ID (e.g., "TIC 12345678" or "12345678")

        Returns:
            DetectionResult with all analysis data

        Raises:
            TargetNotFoundError: If TIC ID not found in MAST
            DataUnavailableError: If no light curve data available
            AnalysisError: If analysis fails
        """
        start_time = time_module.time()
        logger.info(f"Starting analysis for {tic_id}")

        # Normalize TIC ID
        tic_id = self._normalize_tic_id(tic_id)

        try:
            # Fetch light curve data
            time, flux, flux_err, metadata = await self._fetch_lightcurve(tic_id)

            # Run detection
            result = await self.analyze_lightcurve(
                time=time,
                flux=flux,
                flux_err=flux_err,
                tic_id=tic_id,
            )

            # Update with metadata
            result.ra = metadata.get("ra")
            result.dec = metadata.get("dec")
            result.sectors_used = metadata.get("sectors", [])

            self._analyses_count += 1

            logger.info(
                f"Analysis complete for {tic_id}: "
                f"detection={result.detection}, "
                f"time={time_module.time() - start_time:.2f}s"
            )

            return result

        except TargetNotFoundError:
            raise
        except DataUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Analysis failed for {tic_id}: {e}")
            raise AnalysisError(f"Analysis failed: {e}") from e

    async def analyze_lightcurve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        tic_id: Optional[str] = None,
    ) -> DetectionResult:
        """
        Analyze provided light curve data directly.

        Use this method when you already have light curve data and
        don't need to fetch from MAST.

        Args:
            time: Time array (BJD or BTJD)
            flux: Normalized flux array
            flux_err: Optional flux error array
            tic_id: Optional TIC ID for metadata

        Returns:
            DetectionResult with all analysis data
        """
        # Ensure arrays are numpy
        time = np.asarray(time, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if flux_err is not None:
            flux_err = np.asarray(flux_err, dtype=np.float64)

        # Run detection in executor to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.detector.detect,
            time,
            flux,
            flux_err,
            tic_id or "direct_input",
        )

        self._analyses_count += 1

        return result

    async def analyze_batch(
        self,
        tic_ids: List[str],
        max_concurrent: int = 5,
    ) -> List[DetectionResult]:
        """
        Analyze multiple targets in parallel.

        Args:
            tic_ids: List of TIC IDs to analyze
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of DetectionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(tic_id: str) -> DetectionResult:
            async with semaphore:
                try:
                    return await self.analyze(tic_id)
                except Exception as e:
                    logger.error(f"Batch analysis failed for {tic_id}: {e}")
                    return DetectionResult(
                        tic_id=tic_id,
                        detection=False,
                        error=str(e),
                    )

        tasks = [analyze_with_limit(tic_id) for tic_id in tic_ids]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def get_status(self) -> Dict[str, Any]:
        """
        Get service health status.

        Returns:
            Dictionary with service status information
        """
        uptime = time_module.time() - self._start_time

        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "analyses_completed": self._analyses_count,
            "config": {
                "min_period": self.min_period,
                "max_period": self.max_period,
                "min_snr": self.min_snr,
            },
        }

    async def _fetch_lightcurve(
        self, tic_id: str
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Fetch light curve data from MAST.

        Args:
            tic_id: Normalized TIC ID

        Returns:
            Tuple of (time, flux, flux_err, metadata)

        Raises:
            TargetNotFoundError: If target not found
            DataUnavailableError: If no data available
        """
        logger.info(f"Fetching light curve for {tic_id}")

        try:
            # Try to use lightkurve
            from lightkurve import search_lightcurve

            # Search for TESS data
            search_result = search_lightcurve(
                tic_id,
                mission="TESS",
                author="SPOC",  # Prefer SPOC pipeline
            )

            if len(search_result) == 0:
                # Try without SPOC filter
                search_result = search_lightcurve(tic_id, mission="TESS")

            if len(search_result) == 0:
                raise DataUnavailableError(f"No TESS data found for {tic_id}")

            # Download all available sectors
            logger.info(f"Found {len(search_result)} sectors for {tic_id}")

            # Download and stitch light curves
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                raise DataUnavailableError(f"Failed to download data for {tic_id}")

            # Stitch together
            lc = lc_collection.stitch()

            # Use PDCSAP flux if available (de-trended)
            if hasattr(lc, 'flux'):
                flux = lc.flux.value
                time = lc.time.value
            else:
                raise DataUnavailableError("No flux data in light curve")

            # Get flux errors
            flux_err = None
            if hasattr(lc, 'flux_err') and lc.flux_err is not None:
                flux_err = lc.flux_err.value

            # Get metadata
            sectors = []
            for item in search_result:
                if hasattr(item, 'sequence_number'):
                    sectors.append(int(item.sequence_number))
                elif hasattr(item, 'sector'):
                    sectors.append(int(item.sector))

            # Get coordinates if available
            ra = None
            dec = None
            if hasattr(lc, 'meta'):
                ra = lc.meta.get('RA')
                dec = lc.meta.get('DEC')

            metadata = {
                "sectors": list(set(sectors)),
                "ra": ra,
                "dec": dec,
                "n_points": len(time),
                "baseline_days": float(time.max() - time.min()) if len(time) > 0 else 0,
            }

            logger.info(
                f"Downloaded {len(time)} points from {len(metadata['sectors'])} sectors"
            )

            return time, flux, flux_err, metadata

        except ImportError:
            logger.warning("lightkurve not available, using fallback")
            return await self._fetch_lightcurve_fallback(tic_id)

        except Exception as e:
            if "not found" in str(e).lower():
                raise TargetNotFoundError(f"Target {tic_id} not found in MAST")
            if "no data" in str(e).lower():
                raise DataUnavailableError(f"No data available for {tic_id}")
            raise

    async def _fetch_lightcurve_fallback(
        self, tic_id: str
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Fallback light curve fetch using astroquery.

        Used when lightkurve is not available.
        """
        try:
            from src.pipeline.nasa_pipeline import NASADataPipeline

            pipeline = NASADataPipeline({}, cache_dir=self.cache_dir)
            results = await pipeline.fetch_tess_lightcurve(tic_id)

            if not results:
                raise DataUnavailableError(f"No data found for {tic_id}")

            # Combine all results
            all_time = []
            all_flux = []
            all_flux_err = []

            for data in results:
                if data.time is not None:
                    all_time.extend(data.time)
                    all_flux.extend(data.flux)
                    if data.flux_error is not None:
                        all_flux_err.extend(data.flux_error)

            time = np.array(all_time)
            flux = np.array(all_flux)
            flux_err = np.array(all_flux_err) if all_flux_err else None

            metadata = {
                "sectors": [],
                "ra": None,
                "dec": None,
                "n_points": len(time),
            }

            return time, flux, flux_err, metadata

        except Exception as e:
            raise DataUnavailableError(f"Failed to fetch data: {e}")

    def _normalize_tic_id(self, tic_id: str) -> str:
        """
        Normalize TIC ID format.

        Accepts formats: "TIC 12345678", "TIC12345678", "12345678"
        Returns: "TIC 12345678"
        """
        # Remove common prefixes and whitespace
        tic_id = tic_id.strip().upper()

        if tic_id.startswith("TIC"):
            tic_id = tic_id[3:].strip()

        # Ensure it's numeric
        if not tic_id.isdigit():
            raise ValueError(f"Invalid TIC ID format: {tic_id}")

        return f"TIC {tic_id}"


# Factory function for creating service with config
def create_detection_service(
    config: Optional[Dict[str, Any]] = None
) -> DetectionService:
    """
    Create a DetectionService with configuration.

    Args:
        config: Optional configuration dictionary with keys:
            - min_period: Minimum search period (default 0.5)
            - max_period: Maximum search period (default 50.0)
            - min_snr: Minimum SNR threshold (default 7.0)
            - cache_dir: Cache directory (default "data/cache")

    Returns:
        Configured DetectionService instance
    """
    if config is None:
        config = {}

    return DetectionService(
        min_period=config.get("min_period", 0.5),
        max_period=config.get("max_period", 50.0),
        min_snr=config.get("min_snr", 7.0),
        cache_dir=config.get("cache_dir", "data/cache"),
    )
