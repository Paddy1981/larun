"""
Supernova/Transient Detector Node (SUPERNOVA-001)

Detects and classifies supernovae and transient events from light curves.

Model size: 80KB (INT8 quantized)
Input: 128-point light curve
Output: Transient classification (SN Ia, II, Ibc, kilonova, TDE)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class SupernovaNode(BaseNode):
    """Supernova and transient detection."""

    CLASSES = [
        'no_transient',
        'sn_ia',
        'sn_ii',
        'sn_ibc',
        'kilonova',
        'tde',
        'other_transient',
    ]

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 128
        self.tflite_path = self.model_path / 'detector.tflite'
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
                self._model = TransientStatisticalClassifier()
        else:
            self._model = TransientStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)] if np.any(np.isnan(data)) else data

        if len(data) == 0:
            data = np.zeros(self.input_length)

        if len(data) != self.input_length:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, self.input_length)
            data = np.interp(x_new, x_old, data)

        # Normalize
        data_min, data_max = np.min(data), np.max(data)
        if data_max - data_min > 1e-10:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data) + 0.5

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

        # Analyze transient properties
        transient_props = self._analyze_transient(raw_input)
        metadata['transient_properties'] = transient_props

        if classification != 'no_transient' and confidence > 0.5:
            detections.append({
                'type': 'transient',
                'classification': classification,
                **transient_props
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

    def _analyze_transient(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """Extract transient light curve properties."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {}

        peak_idx = np.argmax(data)
        peak_mag = data[peak_idx]
        baseline = np.median(data[:max(1, peak_idx//4)])

        # Rise time (to peak)
        rise_time = peak_idx / len(data)

        # Decline rate (mag/time after peak)
        post_peak = data[peak_idx:]
        if len(post_peak) > 10:
            decline_rate = (post_peak[0] - post_peak[-1]) / len(post_peak)
        else:
            decline_rate = 0

        return {
            'peak_idx': int(peak_idx),
            'peak_brightness': float(peak_mag),
            'amplitude': float(peak_mag - baseline),
            'rise_fraction': float(rise_time),
            'decline_rate': float(decline_rate),
        }


class TransientStatisticalClassifier:
    """Fallback statistical classifier."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data).flatten()
        x = x[~np.isnan(x)] if np.any(np.isnan(x)) else x

        probs = np.zeros(7)

        if len(x) == 0:
            probs[0] = 1.0
            return probs

        # Check for significant brightening
        baseline = np.median(x[:len(x)//4]) if len(x) > 4 else np.median(x)
        peak = np.max(x)
        amplitude = peak - baseline

        if amplitude < 0.1:
            probs[0] = 0.8  # no_transient
        elif amplitude < 0.3:
            probs[6] = 0.5  # other_transient
        else:
            # Classify based on shape
            peak_idx = np.argmax(x)
            rise_fraction = peak_idx / len(x)

            if rise_fraction < 0.2:
                probs[4] = 0.4; probs[5] = 0.3  # kilonova, TDE (fast rise)
            else:
                probs[1] = 0.3; probs[2] = 0.3; probs[3] = 0.2  # SN types

        return probs / np.sum(probs)
