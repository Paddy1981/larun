"""
Detection Data Models
=====================
Dataclasses for the detection pipeline results.

These models define the interface contract between Agent ALPHA (detection)
and Agent BETA (API service).

Reference: .coordination/MVP_INTERFACES.md
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class Disposition(str, Enum):
    """Classification disposition for a transit candidate."""
    PLANET_CANDIDATE = "PLANET_CANDIDATE"
    LIKELY_FALSE_POSITIVE = "LIKELY_FALSE_POSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


class TestFlag(str, Enum):
    """Result flag for individual vetting tests."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class TestResult:
    """
    Result of a single vetting test.

    Attributes:
        test_name: Name of the test (e.g., "odd_even", "v_shape")
        flag: Test result flag (PASS/WARNING/FAIL)
        confidence: Confidence in the result (0.0-1.0)
        value: The measured value from the test
        threshold: The threshold used for pass/fail decision
        message: Human-readable description of result
        details: Additional test-specific details
    """
    test_name: str
    flag: TestFlag
    confidence: float  # 0.0 - 1.0
    value: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "test_name": self.test_name,
            "flag": self.flag.value,
            "confidence": round(self.confidence, 4),
            "value": round(self.value, 6) if self.value is not None else None,
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class VettingResult:
    """
    Combined vetting results from all tests.

    Attributes:
        disposition: Overall classification (PLANET_CANDIDATE, etc.)
        confidence: Combined confidence score (0.0-1.0)
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        tests_warning: Number of tests with warnings
        odd_even: Result of odd-even depth test
        v_shape: Result of V-shape test
        secondary_eclipse: Result of secondary eclipse search
        recommendation: Human-readable recommendation
    """
    disposition: Disposition
    confidence: float  # 0.0 - 1.0
    tests_passed: int
    tests_failed: int
    tests_warning: int
    odd_even: TestResult
    v_shape: TestResult
    secondary_eclipse: TestResult
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "disposition": self.disposition.value,
            "confidence": round(self.confidence, 4),
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_warning": self.tests_warning,
            "odd_even": self.odd_even.to_dict(),
            "v_shape": self.v_shape.to_dict(),
            "secondary_eclipse": self.secondary_eclipse.to_dict(),
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_test_results(
        cls,
        odd_even: TestResult,
        v_shape: TestResult,
        secondary_eclipse: TestResult
    ) -> "VettingResult":
        """
        Create VettingResult from individual test results.

        Args:
            odd_even: Odd-even depth test result
            v_shape: V-shape test result
            secondary_eclipse: Secondary eclipse test result

        Returns:
            VettingResult with computed disposition and recommendation
        """
        tests = [odd_even, v_shape, secondary_eclipse]

        tests_passed = sum(1 for t in tests if t.flag == TestFlag.PASS)
        tests_failed = sum(1 for t in tests if t.flag == TestFlag.FAIL)
        tests_warning = sum(1 for t in tests if t.flag == TestFlag.WARNING)

        # Compute overall confidence as weighted average
        weights = [0.35, 0.35, 0.30]  # Weights for each test
        confidence = sum(t.confidence * w for t, w in zip(tests, weights))

        # Determine disposition
        if tests_failed >= 2:
            disposition = Disposition.LIKELY_FALSE_POSITIVE
            recommendation = "Multiple vetting tests failed. Likely a false positive (eclipsing binary or systematic)."
        elif tests_failed == 1 and tests_warning >= 1:
            disposition = Disposition.INCONCLUSIVE
            recommendation = "Some vetting tests raised concerns. Manual review recommended."
        elif tests_failed == 1:
            disposition = Disposition.INCONCLUSIVE
            recommendation = f"One test failed ({[t.test_name for t in tests if t.flag == TestFlag.FAIL][0]}). Further investigation needed."
        elif tests_warning >= 2:
            disposition = Disposition.INCONCLUSIVE
            recommendation = "Multiple warnings raised. Consider additional observations."
        else:
            disposition = Disposition.PLANET_CANDIDATE
            recommendation = "All vetting tests passed. Strong planet candidate for follow-up."

        return cls(
            disposition=disposition,
            confidence=confidence,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_warning=tests_warning,
            odd_even=odd_even,
            v_shape=v_shape,
            secondary_eclipse=secondary_eclipse,
            recommendation=recommendation,
        )


@dataclass
class LightCurveData:
    """
    Light curve data for visualization.

    Attributes:
        time: Time array (BJD or BTJD)
        flux: Normalized flux values
        flux_err: Flux error values
        quality: Quality flags for each point
    """
    time: List[float]  # BJD or BTJD
    flux: List[float]  # Normalized flux
    flux_err: List[float]  # Flux errors
    quality: List[int]  # Quality flags

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "time": self.time,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "quality": self.quality,
        }

    @classmethod
    def from_arrays(
        cls,
        time,
        flux,
        flux_err=None,
        quality=None
    ) -> "LightCurveData":
        """Create from numpy arrays."""
        import numpy as np

        time_list = time.tolist() if hasattr(time, 'tolist') else list(time)
        flux_list = flux.tolist() if hasattr(flux, 'tolist') else list(flux)

        if flux_err is not None:
            flux_err_list = flux_err.tolist() if hasattr(flux_err, 'tolist') else list(flux_err)
        else:
            flux_err_list = [0.0] * len(flux_list)

        if quality is not None:
            quality_list = quality.tolist() if hasattr(quality, 'tolist') else list(quality)
        else:
            quality_list = [0] * len(flux_list)

        return cls(
            time=time_list,
            flux=flux_list,
            flux_err=flux_err_list,
            quality=quality_list,
        )


@dataclass
class PhaseFoldedData:
    """
    Phase-folded light curve data.

    Attributes:
        phase: Phase values (-0.5 to 0.5, transit at 0)
        flux: Flux values sorted by phase
        flux_err: Flux error values
        binned_phase: Binned phase centers
        binned_flux: Binned flux values
        binned_flux_err: Binned flux errors
    """
    phase: List[float]  # -0.5 to 0.5
    flux: List[float]
    flux_err: List[float]
    binned_phase: List[float]  # Binned for visualization
    binned_flux: List[float]
    binned_flux_err: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "phase": self.phase,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "binned_phase": self.binned_phase,
            "binned_flux": self.binned_flux,
            "binned_flux_err": self.binned_flux_err,
        }


@dataclass
class PeriodogramData:
    """
    BLS periodogram results.

    Attributes:
        periods: Array of periods searched (days)
        powers: BLS power for each period
        best_period: Best-fit period (days)
        best_power: Power at best period
        top_periods: Top N period candidates (days)
        top_powers: Powers of top candidates
    """
    periods: List[float]  # Days
    powers: List[float]  # BLS power
    best_period: float
    best_power: float
    top_periods: List[float]  # Top 3 candidates
    top_powers: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "periods": self.periods,
            "powers": self.powers,
            "best_period": round(self.best_period, 6),
            "best_power": round(self.best_power, 6),
            "top_periods": [round(p, 6) for p in self.top_periods],
            "top_powers": [round(p, 6) for p in self.top_powers],
        }


@dataclass
class DetectionResult:
    """
    Complete detection analysis result.

    This is the main output from DetectionService.analyze().
    Contains all information about the detection attempt including
    transit parameters, vetting results, and visualization data.

    Attributes:
        tic_id: TESS Input Catalog ID
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        detection: Whether a transit was detected
        confidence: Overall detection confidence (0.0-1.0)
        period_days: Orbital period (days)
        depth_ppm: Transit depth in parts per million
        duration_hours: Transit duration (hours)
        epoch_btjd: Mid-transit time (BTJD)
        snr: Signal-to-noise ratio
        vetting: Vetting test results
        periodogram: BLS periodogram data
        phase_folded: Phase-folded light curve
        raw_lightcurve: Original light curve data
        sectors_used: List of TESS sectors used
        processing_time_seconds: Analysis duration
        error: Error message if analysis failed
    """
    # Target identification
    tic_id: str
    ra: Optional[float] = None
    dec: Optional[float] = None

    # Detection result
    detection: bool = False
    confidence: float = 0.0  # 0.0 - 1.0

    # Transit parameters (if detected)
    period_days: Optional[float] = None
    depth_ppm: Optional[float] = None
    duration_hours: Optional[float] = None
    epoch_btjd: Optional[float] = None
    snr: Optional[float] = None

    # Detailed results
    vetting: Optional[VettingResult] = None
    periodogram: Optional[PeriodogramData] = None
    phase_folded: Optional[PhaseFoldedData] = None
    raw_lightcurve: Optional[LightCurveData] = None

    # Metadata
    sectors_used: List[int] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "tic_id": self.tic_id,
            "ra": self.ra,
            "dec": self.dec,
            "detection": self.detection,
            "confidence": round(self.confidence, 4),
            "period_days": round(self.period_days, 6) if self.period_days else None,
            "depth_ppm": round(self.depth_ppm, 2) if self.depth_ppm else None,
            "duration_hours": round(self.duration_hours, 4) if self.duration_hours else None,
            "epoch_btjd": round(self.epoch_btjd, 6) if self.epoch_btjd else None,
            "snr": round(self.snr, 2) if self.snr else None,
            "vetting": self.vetting.to_dict() if self.vetting else None,
            "periodogram": self.periodogram.to_dict() if self.periodogram else None,
            "phase_folded": self.phase_folded.to_dict() if self.phase_folded else None,
            "raw_lightcurve": self.raw_lightcurve.to_dict() if self.raw_lightcurve else None,
            "sectors_used": self.sectors_used,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Create from dictionary."""
        # Handle nested objects
        vetting = None
        if data.get("vetting"):
            # Would need to reconstruct VettingResult
            pass

        return cls(
            tic_id=data["tic_id"],
            ra=data.get("ra"),
            dec=data.get("dec"),
            detection=data.get("detection", False),
            confidence=data.get("confidence", 0.0),
            period_days=data.get("period_days"),
            depth_ppm=data.get("depth_ppm"),
            duration_hours=data.get("duration_hours"),
            epoch_btjd=data.get("epoch_btjd"),
            snr=data.get("snr"),
            sectors_used=data.get("sectors_used", []),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            error=data.get("error"),
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Detection Result for {self.tic_id}",
            "=" * 40,
        ]

        if self.error:
            lines.append(f"ERROR: {self.error}")
            return "\n".join(lines)

        if self.detection:
            lines.append(f"TRANSIT DETECTED (confidence: {self.confidence:.1%})")
            lines.append(f"  Period: {self.period_days:.4f} days")
            lines.append(f"  Depth: {self.depth_ppm:.1f} ppm")
            if self.duration_hours:
                lines.append(f"  Duration: {self.duration_hours:.2f} hours")
            if self.snr:
                lines.append(f"  SNR: {self.snr:.1f}")
            if self.vetting:
                lines.append(f"  Disposition: {self.vetting.disposition.value}")
                lines.append(f"  Recommendation: {self.vetting.recommendation}")
        else:
            lines.append("NO TRANSIT DETECTED")

        lines.append(f"Processing time: {self.processing_time_seconds:.2f}s")

        return "\n".join(lines)


# Custom exceptions for the detection module
class DetectionError(Exception):
    """Base exception for detection errors."""
    pass


class TargetNotFoundError(DetectionError):
    """Raised when target TIC ID is not found in MAST."""
    pass


class DataUnavailableError(DetectionError):
    """Raised when no light curve data is available."""
    pass


class AnalysisError(DetectionError):
    """Raised when analysis fails."""
    pass
