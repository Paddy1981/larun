"""Tests for ModelFederation — the 12-model orchestrator."""

import numpy as np
import pytest


def make_test_lc(n: int = 150, period: float = 5.0) -> dict:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, n))
    flux = 1.0 + 0.1 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.005, n)
    return {
        "times": t,
        "flux": flux,
        "flux_err": np.ones(n) * 0.005,
    }


@pytest.fixture
def federation():
    from larun.models.federation import ModelFederation
    return ModelFederation()


class TestModelFederation:
    def test_run_layer2_all(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2(lc, models="all")
        # All 4 layer2 models should have results
        assert "VARDET-001" in results
        assert "ANOMALY-001" in results
        assert "DEBLEND-001" in results
        assert "PERIODOGRAM-001" in results

    def test_run_layer2_subset(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2(lc, models=["VARDET-001"])
        assert "VARDET-001" in results
        assert "ANOMALY-001" not in results

    def test_run_layer2_parallel(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2_parallel(lc)
        assert len(results) == 4
        for model_id, result in results.items():
            if "error" not in result:
                assert "model_id" in result

    def test_run_all_includes_layer1(self, federation):
        """run_all() should merge pre-computed layer1 results."""
        lc = make_test_lc()
        lc["layer1_results"] = {
            "EXOPLANET-001": {"model_id": "EXOPLANET-001", "label": "noise", "confidence": 0.9}
        }
        results = federation.run_all(lc)
        assert "EXOPLANET-001" in results
        assert "VARDET-001" in results

    def test_consensus_structure(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2(lc)
        consensus = federation.consensus(results)
        assert "consensus_label" in consensus
        assert "consensus_confidence" in consensus
        assert "is_variable" in consensus
        assert "anomaly_detected" in consensus
        assert "blend_detected" in consensus
        assert 0.0 <= consensus["consensus_confidence"] <= 1.0

    def test_inference_time_reported(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2(lc, models=["VARDET-001"])
        vardet_result = results.get("VARDET-001", {})
        if "error" not in vardet_result:
            assert "inference_ms" in vardet_result
            assert vardet_result["inference_ms"] >= 0

    def test_summary_text(self, federation):
        lc = make_test_lc()
        results = federation.run_layer2(lc)
        summary = federation.summary(results)
        assert "LARUN Model Federation Results" in summary
        assert "Consensus" in summary
