"""
Tests for detection models.
"""

import pytest
import json
from src.detection.models import (
    DetectionResult,
    VettingResult,
    TestResult,
    TestFlag,
    Disposition,
    PhaseFoldedData,
    PeriodogramData,
    LightCurveData,
    DetectionError,
    TargetNotFoundError,
    DataUnavailableError,
    AnalysisError,
)


class TestTestFlag:
    """Tests for TestFlag enum."""

    def test_values(self):
        assert TestFlag.PASS.value == "PASS"
        assert TestFlag.WARNING.value == "WARNING"
        assert TestFlag.FAIL.value == "FAIL"

    def test_string_conversion(self):
        assert str(TestFlag.PASS) == "TestFlag.PASS"


class TestDisposition:
    """Tests for Disposition enum."""

    def test_values(self):
        assert Disposition.PLANET_CANDIDATE.value == "PLANET_CANDIDATE"
        assert Disposition.LIKELY_FALSE_POSITIVE.value == "LIKELY_FALSE_POSITIVE"
        assert Disposition.INCONCLUSIVE.value == "INCONCLUSIVE"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_creation(self):
        result = TestResult(
            test_name="odd_even",
            flag=TestFlag.PASS,
            confidence=0.95,
            value=1.2,
            threshold=3.0,
            message="Depths consistent",
            details={"depth_odd": 100, "depth_even": 102},
        )
        assert result.test_name == "odd_even"
        assert result.flag == TestFlag.PASS
        assert result.confidence == 0.95

    def test_to_dict(self):
        result = TestResult(
            test_name="v_shape",
            flag=TestFlag.WARNING,
            confidence=0.6,
            value=0.35,
            threshold=0.3,
            message="Slight V-shape detected",
        )
        d = result.to_dict()
        assert d["test_name"] == "v_shape"
        assert d["flag"] == "WARNING"
        assert d["confidence"] == 0.6


class TestVettingResult:
    """Tests for VettingResult dataclass."""

    @pytest.fixture
    def test_results(self):
        """Create test results for testing."""
        odd_even = TestResult(
            test_name="Odd-Even",
            flag=TestFlag.PASS,
            confidence=0.9,
            value=1.0,
            threshold=3.0,
            message="Pass",
        )
        v_shape = TestResult(
            test_name="V-Shape",
            flag=TestFlag.PASS,
            confidence=0.85,
            value=0.1,
            threshold=0.3,
            message="Pass",
        )
        secondary = TestResult(
            test_name="Secondary Eclipse",
            flag=TestFlag.PASS,
            confidence=0.8,
            value=0.5,
            threshold=3.0,
            message="Pass",
        )
        return odd_even, v_shape, secondary

    def test_from_test_results_all_pass(self, test_results):
        odd_even, v_shape, secondary = test_results
        result = VettingResult.from_test_results(odd_even, v_shape, secondary)

        assert result.disposition == Disposition.PLANET_CANDIDATE
        assert result.tests_passed == 3
        assert result.tests_failed == 0
        assert "Strong planet candidate" in result.recommendation

    def test_from_test_results_some_fail(self, test_results):
        odd_even, v_shape, secondary = test_results
        odd_even.flag = TestFlag.FAIL
        odd_even.confidence = 0.3
        v_shape.flag = TestFlag.FAIL
        v_shape.confidence = 0.2

        result = VettingResult.from_test_results(odd_even, v_shape, secondary)

        assert result.disposition == Disposition.LIKELY_FALSE_POSITIVE
        assert result.tests_failed == 2

    def test_to_dict(self, test_results):
        odd_even, v_shape, secondary = test_results
        result = VettingResult.from_test_results(odd_even, v_shape, secondary)
        d = result.to_dict()

        assert "disposition" in d
        assert "confidence" in d
        assert "odd_even" in d
        assert d["disposition"] == "PLANET_CANDIDATE"


class TestLightCurveData:
    """Tests for LightCurveData dataclass."""

    def test_creation(self):
        lc = LightCurveData(
            time=[0.0, 1.0, 2.0],
            flux=[1.0, 0.99, 1.0],
            flux_err=[0.001, 0.001, 0.001],
            quality=[0, 0, 0],
        )
        assert len(lc.time) == 3
        assert len(lc.flux) == 3

    def test_from_arrays(self):
        import numpy as np

        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.99, 1.0])

        lc = LightCurveData.from_arrays(time, flux)
        assert lc.time == [0.0, 1.0, 2.0]
        assert lc.flux == [1.0, 0.99, 1.0]

    def test_to_dict(self):
        lc = LightCurveData(
            time=[0.0, 1.0],
            flux=[1.0, 0.99],
            flux_err=[0.001, 0.001],
            quality=[0, 0],
        )
        d = lc.to_dict()
        assert "time" in d
        assert "flux" in d


class TestPhaseFoldedData:
    """Tests for PhaseFoldedData dataclass."""

    def test_creation(self):
        data = PhaseFoldedData(
            phase=[-0.5, 0.0, 0.5],
            flux=[1.0, 0.99, 1.0],
            flux_err=[0.001, 0.001, 0.001],
            binned_phase=[-0.25, 0.0, 0.25],
            binned_flux=[1.0, 0.99, 1.0],
            binned_flux_err=[0.0005, 0.0005, 0.0005],
        )
        assert len(data.phase) == 3
        assert len(data.binned_phase) == 3


class TestPeriodogramData:
    """Tests for PeriodogramData dataclass."""

    def test_creation(self):
        data = PeriodogramData(
            periods=[1.0, 2.0, 3.0],
            powers=[0.1, 0.5, 0.2],
            best_period=2.0,
            best_power=0.5,
            top_periods=[2.0, 3.0, 1.0],
            top_powers=[0.5, 0.2, 0.1],
        )
        assert data.best_period == 2.0
        assert len(data.top_periods) == 3

    def test_to_dict(self):
        data = PeriodogramData(
            periods=[1.0, 2.0],
            powers=[0.1, 0.5],
            best_period=2.0,
            best_power=0.5,
            top_periods=[2.0],
            top_powers=[0.5],
        )
        d = data.to_dict()
        assert d["best_period"] == 2.0


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_no_detection(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=False,
        )
        assert result.tic_id == "TIC 12345678"
        assert result.detection is False
        assert result.period_days is None

    def test_with_detection(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=True,
            confidence=0.85,
            period_days=3.5,
            depth_ppm=1000,
            duration_hours=2.5,
            snr=15.0,
        )
        assert result.detection is True
        assert result.period_days == 3.5
        assert result.snr == 15.0

    def test_to_dict(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=True,
            confidence=0.85,
            period_days=3.5,
        )
        d = result.to_dict()
        assert d["tic_id"] == "TIC 12345678"
        assert d["detection"] is True
        assert d["period_days"] == 3.5

    def test_to_json(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=True,
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["tic_id"] == "TIC 12345678"

    def test_summary(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=True,
            confidence=0.85,
            period_days=3.5,
            depth_ppm=1000,
            snr=15.0,
        )
        summary = result.summary()
        assert "TIC 12345678" in summary
        assert "DETECTED" in summary
        assert "3.5" in summary

    def test_summary_no_detection(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=False,
        )
        summary = result.summary()
        assert "NO TRANSIT" in summary

    def test_error_summary(self):
        result = DetectionResult(
            tic_id="TIC 12345678",
            detection=False,
            error="Insufficient data",
        )
        summary = result.summary()
        assert "ERROR" in summary
        assert "Insufficient data" in summary


class TestExceptions:
    """Tests for custom exceptions."""

    def test_detection_error(self):
        with pytest.raises(DetectionError):
            raise DetectionError("Test error")

    def test_target_not_found_error(self):
        with pytest.raises(TargetNotFoundError):
            raise TargetNotFoundError("TIC not found")

    def test_data_unavailable_error(self):
        with pytest.raises(DataUnavailableError):
            raise DataUnavailableError("No data")

    def test_analysis_error(self):
        with pytest.raises(AnalysisError):
            raise AnalysisError("Analysis failed")

    def test_inheritance(self):
        assert issubclass(TargetNotFoundError, DetectionError)
        assert issubclass(DataUnavailableError, DetectionError)
        assert issubclass(AnalysisError, DetectionError)
