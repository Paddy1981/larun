"""
Microlensing Event Detector Node (MICROLENS-001)

Detects and classifies gravitational microlensing events.

Model size: 72KB (INT8 quantized)
Input: 512-point light curve (long baseline)
Output: Microlensing classification + event parameters
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class MicrolensingNode(BaseNode):
    """Gravitational microlensing event detection."""

    CLASSES = [
        'no_event',
        'single_lens',
        'binary_lens',
        'planetary',
        'parallax',
        'unclear',
    ]

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 512
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
                self._model = MicrolensingStatisticalClassifier()
        else:
            self._model = MicrolensingStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """Preprocess long-baseline light curve."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)] if np.any(np.isnan(data)) else data

        if len(data) == 0:
            data = np.ones(self.input_length)  # Baseline = 1

        if len(data) != self.input_length:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, self.input_length)
            data = np.interp(x_new, x_old, data)

        # Normalize to baseline = 1 (magnification representation)
        baseline = np.median(data)
        if baseline > 1e-10:
            data = data / baseline
        else:
            data = np.ones_like(data)

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

        # Analyze event parameters
        event_params = self._analyze_event(raw_input)
        metadata['event_params'] = event_params

        if classification != 'no_event' and confidence > 0.5:
            detections.append({
                'type': 'microlensing',
                'classification': classification,
                **event_params
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

    def _analyze_event(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """Extract microlensing event parameters."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {}

        # Baseline flux
        baseline = np.median(data)

        # Peak magnification
        peak_idx = np.argmax(data)
        peak_mag = data[peak_idx] / baseline if baseline > 0 else 1

        # Einstein crossing time (FWHM of magnification)
        # Find where magnification drops to half of peak excess
        half_mag = (peak_mag + 1) / 2  # Midpoint between baseline and peak
        above_half = data / baseline > half_mag if baseline > 0 else np.zeros(len(data), dtype=bool)

        # Find FWHM
        left = peak_idx
        while left > 0 and above_half[left]:
            left -= 1
        right = peak_idx
        while right < len(data) - 1 and above_half[right]:
            right += 1

        fwhm = (right - left) / len(data)

        # Check for asymmetry (could indicate parallax or binary)
        left_side = data[left:peak_idx]
        right_side = data[peak_idx:right]
        if len(left_side) > 0 and len(right_side) > 0:
            asymmetry = np.abs(np.mean(left_side) - np.mean(right_side)) / peak_mag
        else:
            asymmetry = 0

        # Check for secondary peaks (binary lens signature)
        smoothed = np.convolve(data, np.ones(5)/5, mode='same')
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > baseline * 1.1:
                    peaks.append(i)

        return {
            'peak_magnification': float(peak_mag),
            'peak_idx': int(peak_idx),
            'fwhm_fraction': float(fwhm),
            'asymmetry': float(asymmetry),
            'n_peaks': len(peaks),
            'baseline': float(baseline),
        }


class MicrolensingStatisticalClassifier:
    """Fallback statistical classifier for microlensing."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Classify based on light curve shape."""
        x = np.asarray(data).flatten()

        probs = np.zeros(6)

        if len(x) == 0:
            probs[0] = 1.0
            return probs

        baseline = np.median(x)
        peak_mag = np.max(x) / baseline if baseline > 0 else 1

        # No significant magnification
        if peak_mag < 1.1:
            probs[0] = 0.8  # no_event
            return probs / np.sum(probs)

        # Check shape symmetry
        peak_idx = np.argmax(x)
        n = len(x)

        # Simple event
        if peak_mag < 2:
            probs[1] = 0.5  # single_lens
            probs[5] = 0.3  # unclear
        elif peak_mag < 5:
            probs[1] = 0.4  # single_lens
            probs[2] = 0.2  # binary_lens
            probs[3] = 0.2  # planetary
        else:
            probs[2] = 0.4  # binary_lens
            probs[1] = 0.3  # single_lens

        return probs / np.sum(probs)
