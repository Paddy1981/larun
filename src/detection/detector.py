"""
Transit Detector
================
Core transit detection logic combining BLS and vetting.

This module orchestrates the detection pipeline, using BLS for
period finding and vetting tests for validation.

Author: Agent ALPHA
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
import time as time_module

from .models import (
    DetectionResult,
    VettingResult,
    TestResult,
    TestFlag,
    Disposition,
    PeriodogramData,
    PhaseFoldedData,
    LightCurveData,
)
from .bls_engine import BLSEngine
from .phase_folder import PhaseFolding

# Import existing vetting from skills
from src.skills.vetting import TransitVetter, VettingResult as SkillVettingResult

logger = logging.getLogger(__name__)


class TransitDetector:
    """
    Transit detection and vetting pipeline.

    Combines BLS periodogram analysis with vetting tests to detect
    and validate transit signals in light curve data.

    Example:
        >>> detector = TransitDetector()
        >>> result = detector.detect(time, flux, tic_id="TIC 470710327")
        >>> if result.detection:
        ...     print(f"Found planet candidate with period {result.period_days:.4f} days")
    """

    def __init__(
        self,
        min_period: float = 0.5,
        max_period: float = 50.0,
        min_snr: float = 7.0,
        n_bins: int = 100,
        run_vetting: bool = True,
    ):
        """
        Initialize the transit detector.

        Args:
            min_period: Minimum period to search (days)
            max_period: Maximum period to search (days)
            min_snr: Minimum SNR threshold for detection
            n_bins: Number of bins for phase-folded data
            run_vetting: Whether to run vetting tests
        """
        self.min_period = min_period
        self.max_period = max_period
        self.min_snr = min_snr
        self.n_bins = n_bins
        self.run_vetting = run_vetting

        # Initialize components
        self.bls_engine = BLSEngine(
            min_period=min_period,
            max_period=max_period,
            min_snr=min_snr,
        )
        self.phase_folder = PhaseFolding(n_bins=n_bins)
        self.vetter = TransitVetter()

        logger.info(
            f"TransitDetector initialized: period=[{min_period}, {max_period}]d, "
            f"min_snr={min_snr}"
        )

    def detect(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        tic_id: str = "unknown",
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        sectors: Optional[list] = None,
    ) -> DetectionResult:
        """
        Run full transit detection pipeline.

        Args:
            time: Time array (days, BJD/BTJD)
            flux: Normalized flux array
            flux_err: Optional flux errors
            tic_id: TESS Input Catalog ID
            ra: Right ascension (optional)
            dec: Declination (optional)
            sectors: List of TESS sectors used (optional)

        Returns:
            DetectionResult with full analysis results
        """
        start_time = time_module.time()
        logger.info(f"Starting detection for {tic_id}")

        # Validate input
        is_valid, error_msg = self.bls_engine.validate_light_curve(time, flux)
        if not is_valid:
            logger.warning(f"Invalid light curve: {error_msg}")
            return DetectionResult(
                tic_id=tic_id,
                ra=ra,
                dec=dec,
                detection=False,
                error=error_msg,
                processing_time_seconds=time_module.time() - start_time,
            )

        # Clean data
        time_clean, flux_clean, flux_err_clean = self._clean_data(
            time, flux, flux_err
        )

        # Create raw light curve data
        raw_lightcurve = LightCurveData.from_arrays(
            time_clean, flux_clean, flux_err_clean
        )

        try:
            # Run BLS
            periodogram, candidate = self.bls_engine.compute(
                time_clean, flux_clean, flux_err_clean
            )

            # Check if we have a detection
            if candidate is None:
                logger.info(f"No significant transit signal found for {tic_id}")
                return DetectionResult(
                    tic_id=tic_id,
                    ra=ra,
                    dec=dec,
                    detection=False,
                    confidence=0.0,
                    periodogram=periodogram,
                    raw_lightcurve=raw_lightcurve,
                    sectors_used=sectors or [],
                    processing_time_seconds=time_module.time() - start_time,
                )

            # Extract transit parameters
            period = candidate.period
            t0 = candidate.t0
            depth = candidate.depth
            snr = candidate.snr

            logger.info(
                f"Candidate found: P={period:.4f}d, depth={depth*1e6:.1f}ppm, "
                f"SNR={snr:.1f}"
            )

            # Phase fold
            phase_folded = self.phase_folder.fold(
                time_clean, flux_clean, period, t0, flux_err_clean
            )

            # Estimate duration
            duration_hours = self.phase_folder.estimate_duration(
                phase_folded, period, depth
            )

            # Run vetting tests
            vetting_result = None
            if self.run_vetting:
                vetting_result = self._run_vetting(
                    time_clean, flux_clean, period, t0
                )

            # Compute overall confidence
            confidence = self._compute_confidence(snr, vetting_result)

            # Determine if this is a detection
            detection = snr >= self.min_snr

            result = DetectionResult(
                tic_id=tic_id,
                ra=ra,
                dec=dec,
                detection=detection,
                confidence=confidence,
                period_days=float(period),
                depth_ppm=float(depth * 1e6),
                duration_hours=float(duration_hours) if duration_hours > 0 else None,
                epoch_btjd=float(t0),
                snr=float(snr),
                vetting=vetting_result,
                periodogram=periodogram,
                phase_folded=phase_folded,
                raw_lightcurve=raw_lightcurve,
                sectors_used=sectors or [],
                processing_time_seconds=time_module.time() - start_time,
            )

            logger.info(
                f"Detection complete for {tic_id}: "
                f"detection={detection}, confidence={confidence:.2%}"
            )

            return result

        except Exception as e:
            logger.error(f"Detection failed for {tic_id}: {e}")
            return DetectionResult(
                tic_id=tic_id,
                ra=ra,
                dec=dec,
                detection=False,
                error=str(e),
                raw_lightcurve=raw_lightcurve,
                processing_time_seconds=time_module.time() - start_time,
            )

    def _clean_data(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Clean and normalize input data."""
        # Remove NaN/Inf values
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err)

        time_clean = np.asarray(time[mask], dtype=np.float64)
        flux_clean = np.asarray(flux[mask], dtype=np.float64)

        if flux_err is not None:
            flux_err_clean = np.asarray(flux_err[mask], dtype=np.float64)
        else:
            flux_err_clean = None

        # Normalize flux to median = 1
        median_flux = np.median(flux_clean)
        if median_flux > 0 and not np.isclose(median_flux, 1.0, rtol=0.01):
            flux_clean = flux_clean / median_flux
            if flux_err_clean is not None:
                flux_err_clean = flux_err_clean / median_flux

        logger.debug(
            f"Data cleaned: {len(time_clean)} points, "
            f"baseline={time_clean.max()-time_clean.min():.1f} days"
        )

        return time_clean, flux_clean, flux_err_clean

    def _run_vetting(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
    ) -> VettingResult:
        """
        Run vetting tests and convert to our VettingResult format.

        Args:
            time: Cleaned time array
            flux: Cleaned flux array
            period: Best-fit period
            t0: Mid-transit epoch

        Returns:
            VettingResult with all test results
        """
        logger.debug("Running vetting tests...")

        # Use existing vetter
        skill_result = self.vetter.run_all(time, flux, period, t0)

        # Convert individual test results
        test_results = {}
        for test in skill_result.tests:
            test_name = test.test_name.lower().replace(" ", "_").replace("-", "_")

            # Determine flag
            if test.passed:
                flag = TestFlag.PASS
            elif test.confidence < 0.5:
                flag = TestFlag.FAIL
            else:
                flag = TestFlag.WARNING

            # Extract value and threshold from details if available
            value = test.details.get("difference_sigma", 0.0)
            threshold = 3.0  # Default significance threshold

            test_results[test_name] = TestResult(
                test_name=test.test_name,
                flag=flag,
                confidence=test.confidence,
                value=value,
                threshold=threshold,
                message=test.message,
                details=test.details,
            )

        # Ensure we have all required tests (create defaults if missing)
        if "odd_even_depth" not in test_results:
            test_results["odd_even_depth"] = self._create_default_test("Odd-Even Depth")
        if "v_shape_grazing" not in test_results and "v_shape_(grazing)" not in test_results:
            test_results["v_shape"] = self._create_default_test("V-Shape (Grazing)")
        if "secondary_eclipse" not in test_results:
            test_results["secondary_eclipse"] = self._create_default_test("Secondary Eclipse")

        # Get the test results (handle varying key names)
        odd_even = test_results.get("odd_even_depth", test_results.get("odd_even", self._create_default_test("Odd-Even Depth")))
        v_shape = test_results.get("v_shape_(grazing)", test_results.get("v_shape", self._create_default_test("V-Shape (Grazing)")))
        secondary = test_results.get("secondary_eclipse", self._create_default_test("Secondary Eclipse"))

        # Create VettingResult
        return VettingResult.from_test_results(odd_even, v_shape, secondary)

    def _create_default_test(self, name: str) -> TestResult:
        """Create a default inconclusive test result."""
        return TestResult(
            test_name=name,
            flag=TestFlag.WARNING,
            confidence=0.5,
            value=0.0,
            threshold=0.0,
            message="Test not performed - insufficient data",
            details={},
        )

    def _compute_confidence(
        self,
        snr: float,
        vetting: Optional[VettingResult],
    ) -> float:
        """
        Compute overall detection confidence.

        Args:
            snr: Signal-to-noise ratio from BLS
            vetting: Vetting result (if available)

        Returns:
            Confidence score (0.0-1.0)
        """
        # SNR contribution (caps at 1.0 for SNR >= 20)
        snr_confidence = min(1.0, snr / 20.0)

        if vetting is None:
            return snr_confidence * 0.7  # Reduce if no vetting

        # Combine with vetting confidence
        vetting_weight = 0.4
        snr_weight = 0.6

        confidence = (
            snr_weight * snr_confidence +
            vetting_weight * vetting.confidence
        )

        # Apply disposition penalty
        if vetting.disposition == Disposition.LIKELY_FALSE_POSITIVE:
            confidence *= 0.3
        elif vetting.disposition == Disposition.INCONCLUSIVE:
            confidence *= 0.7

        return min(1.0, max(0.0, confidence))


def detect_transits(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    tic_id: str = "unknown",
    **kwargs,
) -> DetectionResult:
    """
    Convenience function for transit detection.

    Args:
        time: Time array (days)
        flux: Flux array
        flux_err: Optional flux errors
        tic_id: Target identifier
        **kwargs: Additional arguments for TransitDetector

    Returns:
        DetectionResult
    """
    detector = TransitDetector(**kwargs)
    return detector.detect(time, flux, flux_err, tic_id)
