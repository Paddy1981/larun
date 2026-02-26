"""
ModelFederation — Orchestrator for LARUN's 12 TinyML Models

Manages the full model suite across Layer 1 (browser) and Layer 2 (server)
and provides a unified run_all() interface for the Citizen Discovery Engine.

Layer 1 models (existing, browser ONNX):
    EXOPLANET-001, VSTAR-001, FLARE-001, MICROLENS-001,
    SUPERNOVA-001, SPECTYPE-001, ASTERO-001, GALAXY-001

Layer 2 models (new, server Python):
    VARDET-001, ANOMALY-001, DEBLEND-001, PERIODOGRAM-001
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from larun.models.vardet import VARDET001
from larun.models.anomaly import ANOMALY001
from larun.models.deblend import DEBLEND001
from larun.models.periodogram import PERIODOGRAM001

logger = logging.getLogger(__name__)

# Layer 1 model stubs — these delegate to existing node implementations
_LAYER1_MODELS = [
    "EXOPLANET-001",
    "VSTAR-001",
    "FLARE-001",
    "MICROLENS-001",
    "SUPERNOVA-001",
    "SPECTYPE-001",
    "ASTERO-001",
    "GALAXY-001",
]

_LAYER2_MODEL_CLASSES = {
    "VARDET-001": VARDET001,
    "ANOMALY-001": ANOMALY001,
    "DEBLEND-001": DEBLEND001,
    "PERIODOGRAM-001": PERIODOGRAM001,
}


class ModelFederation:
    """
    Unified interface to all 12 LARUN TinyML models.

    Usage:
        federation = ModelFederation()
        results = federation.run_all({
            'times': np.array([...]),
            'flux': np.array([...]),
            'flux_err': np.array([...]),
        })
        # results is a dict: model_id → classification result
    """

    def __init__(self, auto_load: bool = True):
        """
        Initialize the model federation.

        Args:
            auto_load: if True, load/train Layer 2 models on first use
        """
        self._layer2_models: dict[str, Any] = {}
        self._auto_load = auto_load
        self._layer1_runner = None  # Optional: inject existing node runner

    def _get_layer2(self, model_id: str):
        """Lazy-initialize a Layer 2 model."""
        if model_id not in self._layer2_models:
            cls = _LAYER2_MODEL_CLASSES[model_id]
            instance = cls()
            if self._auto_load:
                instance.load()
            self._layer2_models[model_id] = instance
        return self._layer2_models[model_id]

    def run_layer2(
        self,
        light_curve: dict,
        models: list[str] | str = "all",
    ) -> dict[str, dict]:
        """
        Run Layer 2 (server-side) models on a light curve.

        Args:
            light_curve: dict with at minimum 'times' and 'flux' or 'mags' keys.
                         Optional: 'flux_err'/'errors', 'crowdsap', 'flfrcsap'
            models: 'all' or list of model IDs e.g. ['VARDET-001', 'ANOMALY-001']

        Returns:
            dict: model_id → result dict
        """
        if models == "all":
            target_ids = list(_LAYER2_MODEL_CLASSES.keys())
        else:
            target_ids = [m for m in models if m in _LAYER2_MODEL_CLASSES]

        times = np.asarray(light_curve.get("times", []), dtype=float)
        mags = np.asarray(
            light_curve.get("flux", light_curve.get("mags", [])), dtype=float
        )
        errors = light_curve.get("flux_err", light_curve.get("errors"))
        if errors is not None:
            errors = np.asarray(errors, dtype=float)

        results = {}

        for model_id in target_ids:
            t0 = time.perf_counter()
            try:
                model = self._get_layer2(model_id)

                if model_id == "PERIODOGRAM-001":
                    result = model.find_period(times, mags, errors)
                elif model_id == "DEBLEND-001":
                    label, proba = model.predict(
                        times, mags, errors,
                        light_curve.get("crowdsap"),
                        light_curve.get("flfrcsap"),
                    )
                    result = model.result_dict(label, proba)
                else:
                    label, proba = model.predict(times, mags, errors)
                    result = model.result_dict(label, proba)

                result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                results[model_id] = result

            except Exception as exc:
                logger.error(f"{model_id} inference failed: {exc}")
                results[model_id] = {
                    "model_id": model_id,
                    "error": str(exc),
                    "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
                }

        return results

    def run_layer2_parallel(
        self,
        light_curve: dict,
        models: list[str] | str = "all",
        max_workers: int = 4,
    ) -> dict[str, dict]:
        """Run Layer 2 models in parallel using ThreadPoolExecutor."""
        if models == "all":
            target_ids = list(_LAYER2_MODEL_CLASSES.keys())
        else:
            target_ids = [m for m in models if m in _LAYER2_MODEL_CLASSES]

        times = np.asarray(light_curve.get("times", []), dtype=float)
        mags = np.asarray(
            light_curve.get("flux", light_curve.get("mags", [])), dtype=float
        )
        errors = light_curve.get("flux_err", light_curve.get("errors"))

        results = {}

        def _run_one(model_id: str) -> tuple[str, dict]:
            t0 = time.perf_counter()
            model = self._get_layer2(model_id)
            if model_id == "PERIODOGRAM-001":
                result = model.find_period(times, mags, errors)
            elif model_id == "DEBLEND-001":
                label, proba = model.predict(times, mags, errors,
                                              light_curve.get("crowdsap"),
                                              light_curve.get("flfrcsap"))
                result = model.result_dict(label, proba)
            else:
                label, proba = model.predict(times, mags, errors)
                result = model.result_dict(label, proba)
            result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return model_id, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, mid): mid for mid in target_ids}
            for future in as_completed(futures):
                try:
                    model_id, result = future.result()
                    results[model_id] = result
                except Exception as exc:
                    model_id = futures[future]
                    results[model_id] = {"model_id": model_id, "error": str(exc)}

        return results

    def run_all(
        self,
        light_curve: dict,
        include_layer1: bool = True,
        parallel: bool = True,
    ) -> dict[str, dict]:
        """
        Run all available models on a light curve.

        Layer 1 results (browser models) may be pre-computed on the client
        and passed in via light_curve['layer1_results']. If present, they
        are merged into the output. Otherwise, Layer 1 is skipped server-side.

        Args:
            light_curve: dict with 'times', 'flux'/'mags', and optionally
                         'flux_err', 'crowdsap', 'flfrcsap', 'layer1_results'
            include_layer1: merge pre-computed layer1 results if provided
            parallel: use ThreadPoolExecutor for Layer 2 models

        Returns:
            dict: model_id → result dict (all 12 models if layer1 included)
        """
        results = {}

        # Merge Layer 1 results if pre-computed
        if include_layer1 and "layer1_results" in light_curve:
            results.update(light_curve["layer1_results"])

        # Run Layer 2 server-side models
        if parallel:
            layer2_results = self.run_layer2_parallel(light_curve)
        else:
            layer2_results = self.run_layer2(light_curve)

        results.update(layer2_results)

        return results

    def consensus(self, results: dict[str, dict]) -> dict:
        """
        Compute a consensus classification across all models.

        Returns:
            {
                'consensus_label': str,
                'consensus_confidence': float,
                'agreement_count': int,
                'is_variable': bool,
                'anomaly_detected': bool,
                'blend_detected': bool,
            }
        """
        is_variable = False
        anomaly_detected = False
        blend_detected = False
        labels = []

        for model_id, result in results.items():
            if "error" in result:
                continue

            label = result.get("label", "")
            conf = result.get("confidence", 0.0)

            if model_id == "VARDET-001" and label != "NON_VARIABLE" and conf > 0.6:
                is_variable = True
                labels.append(label)

            if model_id == "ANOMALY-001" and label in ("MILD_ANOMALY", "STRONG_ANOMALY") and conf > 0.5:
                anomaly_detected = True

            if model_id == "DEBLEND-001" and label in ("MILD_BLEND", "SEVERE_BLEND") and conf > 0.6:
                blend_detected = True

            if label and model_id != "PERIODOGRAM-001":
                labels.append(label)

        # Most common label
        if labels:
            from collections import Counter
            label_counts = Counter(labels)
            consensus_label, agreement_count = label_counts.most_common(1)[0]
            consensus_confidence = agreement_count / len(labels)
        else:
            consensus_label = "UNKNOWN"
            agreement_count = 0
            consensus_confidence = 0.0

        return {
            "consensus_label": consensus_label,
            "consensus_confidence": round(consensus_confidence, 3),
            "agreement_count": agreement_count,
            "total_models_run": len([r for r in results.values() if "error" not in r]),
            "is_variable": is_variable,
            "anomaly_detected": anomaly_detected,
            "blend_detected": blend_detected,
        }

    def summary(self, results: dict[str, dict]) -> str:
        """Human-readable summary of federation results."""
        lines = ["=== LARUN Model Federation Results ==="]
        for model_id, result in sorted(results.items()):
            if "error" in result:
                lines.append(f"  {model_id}: ERROR — {result['error']}")
            elif model_id == "PERIODOGRAM-001":
                lines.append(
                    f"  {model_id}: period={result.get('best_period', 0):.3f}d "
                    f"type={result.get('period_type', '?')} "
                    f"conf={result.get('confidence', 0):.2f}"
                )
            else:
                lines.append(
                    f"  {model_id}: {result.get('label', '?')} "
                    f"({result.get('confidence', 0):.1%}) "
                    f"[{result.get('inference_ms', 0):.0f}ms]"
                )
        consensus = self.consensus(results)
        lines.append(f"\nConsensus: {consensus['consensus_label']} "
                     f"({consensus['consensus_confidence']:.1%} agreement, "
                     f"{consensus['agreement_count']}/{consensus['total_models_run']} models)")
        if consensus["anomaly_detected"]:
            lines.append("⚠️  ANOMALY DETECTED — priority review recommended")
        if consensus["blend_detected"]:
            lines.append("⚠️  BLEND DETECTED — results may be unreliable")
        return "\n".join(lines)
