"""
Asteroseismology Analyzer Node (ASTERO-001)

Analyzes stellar oscillations from power spectra to classify
oscillation types and estimate global seismic parameters.

Model size: 60KB (INT8 quantized)
Input: 512-point power spectrum
Output: Oscillation classification + nu_max, delta_nu estimates
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class AsteroseismoNode(BaseNode):
    """Asteroseismology analysis from power spectra."""

    CLASSES = [
        'no_oscillation',
        'solar_like',
        'red_giant',
        'delta_scuti',
        'gamma_dor',
        'hybrid',
    ]

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 512
        self.tflite_path = self.model_path / 'analyzer.tflite'
        self._use_tflite = self.tflite_path.exists()

    def load_model(self) -> None:
        if self._use_tflite:
            try:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(
                    model_path=str(self.tflite_path)
                )
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
            except:
                self._model = AsteroStatisticalClassifier()
        else:
            self._model = AsteroStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """Preprocess power spectrum."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            data = np.zeros(self.input_length)

        # Resample
        if len(data) != self.input_length:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, self.input_length)
            data = np.interp(x_new, x_old, data)

        # Log transform (power spectra often span orders of magnitude)
        data = np.log10(np.maximum(data, 1e-10))

        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-10)

        return data.reshape(self.input_length, 1).astype(np.float32)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        if self._interpreter is not None:
            input_data = np.expand_dims(preprocessed_data, axis=0).astype(np.float32)
            self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
            self._interpreter.invoke()
            return self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        elif self._model is not None:
            return self._model.predict(preprocessed_data)[0]
        raise RuntimeError("No model loaded")

    def postprocess(self, model_output: np.ndarray,
                    raw_input: np.ndarray) -> NodeResult:
        probabilities = self._softmax(model_output)
        class_idx = np.argmax(probabilities)
        classification = self.CLASSES[class_idx]
        confidence = float(probabilities[class_idx])

        prob_dict = {cls: float(probabilities[i]) for i, cls in enumerate(self.CLASSES)}

        detections = []
        metadata = {}

        # Estimate seismic parameters
        seismic_params = self._estimate_seismic_params(raw_input, classification)
        metadata['seismic_params'] = seismic_params

        if classification != 'no_oscillation' and confidence > 0.5:
            detections.append({
                'type': 'stellar_oscillation',
                'classification': classification,
                **seismic_params
            })

        return NodeResult(
            node_id=self.node_id,
            classification=classification,
            confidence=confidence,
            probabilities=prob_dict,
            detections=detections,
            metadata=metadata,
            inference_time_ms=0.0,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _estimate_seismic_params(self, raw_input: np.ndarray,
                                  classification: str) -> Dict[str, Any]:
        """Estimate nu_max and delta_nu from power spectrum."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0 or classification == 'no_oscillation':
            return {'nu_max': None, 'delta_nu': None}

        # Simple peak finding for nu_max (frequency of maximum power)
        # This is a rough approximation
        peak_idx = np.argmax(data)
        nu_max_approx = peak_idx / len(data)  # Normalized frequency

        # For delta_nu, look for periodic spacing in the envelope
        # Using autocorrelation of the smoothed power spectrum
        smoothed = np.convolve(data, np.ones(5)/5, mode='same')
        autocorr = np.correlate(smoothed, smoothed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find first significant peak after zero lag
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1 * autocorr[0]:
                    peaks.append(i)
                    break

        delta_nu_approx = peaks[0] / len(data) if peaks else None

        return {
            'nu_max_normalized': float(nu_max_approx),
            'delta_nu_normalized': float(delta_nu_approx) if delta_nu_approx else None,
            'oscillation_type': classification,
        }


class AsteroStatisticalClassifier:
    """Fallback statistical classifier."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data).flatten()
        probs = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Check for significant peaks
        if len(x) > 10:
            peak_ratio = np.max(x) / np.median(x) if np.median(x) > 0 else 1
            if peak_ratio > 5:
                probs = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.1])

        return probs / np.sum(probs)
