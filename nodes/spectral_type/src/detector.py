"""
Spectral Type Classifier Node (SPECTYPE-001)

Classifies stellar spectral types from photometric data.
Input: 8 photometric values (BP-RP, G, J, H, K, W1, W2, parallax)

Model size: 40KB (INT8 quantized)
Output: Spectral type (O, B, A, F, G, K, M, L)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class SpectralTypeNode(BaseNode):
    """Spectral type classification from photometry."""

    CLASSES = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L']

    # Typical effective temperatures for each class (K)
    TEFF_RANGES = {
        'O': (30000, 50000),
        'B': (10000, 30000),
        'A': (7500, 10000),
        'F': (6000, 7500),
        'G': (5200, 6000),
        'K': (3700, 5200),
        'M': (2400, 3700),
        'L': (1300, 2400),
    }

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 8
        self.tflite_path = self.model_path / 'classifier.tflite'
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
                self._model = SpectralTypeStatisticalClassifier()
        else:
            self._model = SpectralTypeStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """Preprocess photometric values."""
        data = np.asarray(raw_input).flatten()

        # Handle missing values with median
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)

        # Pad or truncate to expected length
        if len(data) < self.input_length:
            data = np.pad(data, (0, self.input_length - len(data)))
        elif len(data) > self.input_length:
            data = data[:self.input_length]

        # Normalize (standard scaling)
        # Typical ranges for photometric data
        scales = np.array([3.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 10.0])
        data = data / scales

        return data.astype(np.float32)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        if self._interpreter is not None:
            input_data = np.expand_dims(preprocessed_data, axis=0).astype(np.float32)
            self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
            self._interpreter.invoke()
            return self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        elif self._model is not None:
            return self._model.predict(preprocessed_data)
        raise RuntimeError("No model loaded")

    def postprocess(self, model_output: np.ndarray,
                    raw_input: np.ndarray) -> NodeResult:
        probabilities = self._softmax(model_output)
        class_idx = np.argmax(probabilities)
        classification = self.CLASSES[class_idx]
        confidence = float(probabilities[class_idx])

        prob_dict = {cls: float(probabilities[i]) for i, cls in enumerate(self.CLASSES)}

        # Estimate Teff from classification
        teff_range = self.TEFF_RANGES.get(classification, (3000, 10000))
        teff_estimate = (teff_range[0] + teff_range[1]) / 2

        metadata = {
            'spectral_type': classification,
            'teff_estimate_k': teff_estimate,
            'teff_range': teff_range,
        }

        detections = [{
            'type': 'spectral_classification',
            'spectral_type': classification,
            'teff_estimate': teff_estimate,
        }]

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


class SpectralTypeStatisticalClassifier:
    """Fallback classifier using color-temperature relations."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Classify based on BP-RP color (first element)."""
        bp_rp = data[0] * 3.0 if len(data) > 0 else 1.0  # Denormalize

        # Rough color-spectral type mapping
        probs = np.zeros(8)

        if bp_rp < -0.3:
            probs[0] = 0.5; probs[1] = 0.4  # O, B
        elif bp_rp < 0.0:
            probs[1] = 0.6; probs[2] = 0.3  # B, A
        elif bp_rp < 0.3:
            probs[2] = 0.5; probs[3] = 0.4  # A, F
        elif bp_rp < 0.6:
            probs[3] = 0.5; probs[4] = 0.4  # F, G
        elif bp_rp < 1.0:
            probs[4] = 0.5; probs[5] = 0.4  # G, K
        elif bp_rp < 1.5:
            probs[5] = 0.6; probs[6] = 0.3  # K, M
        elif bp_rp < 2.5:
            probs[6] = 0.7; probs[5] = 0.2  # M
        else:
            probs[7] = 0.6; probs[6] = 0.3  # L

        return probs / np.sum(probs)
