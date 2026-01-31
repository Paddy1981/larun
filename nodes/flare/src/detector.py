"""
Stellar Flare Detector Node (FLARE-001)

TinyML-optimized detector for stellar flares in short time windows.
Uses causal convolutions for real-time streaming analysis.

Model size: 32KB (INT8 quantized)
Input: 256-point light curve window
Output: Flare classification + energy estimate

Flare Classes:
- no_flare: No significant brightening
- weak_flare: Minor flare (E < 10^31 erg)
- moderate_flare: Moderate flare (E ~ 10^31-32 erg)
- strong_flare: Strong flare (E ~ 10^32-33 erg)
- superflare: Superflare (E > 10^33 erg)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class FlareNode(BaseNode):
    """
    Stellar flare detection and characterization.

    Designed for real-time monitoring of stellar activity,
    particularly on M-dwarfs and other active stars.
    """

    CLASSES = [
        'no_flare',
        'weak_flare',
        'moderate_flare',
        'strong_flare',
        'superflare',
    ]

    # Energy thresholds (in log10 erg)
    ENERGY_THRESHOLDS = {
        'weak_flare': 30,      # 10^30 erg
        'moderate_flare': 31,  # 10^31 erg
        'strong_flare': 32,    # 10^32 erg
        'superflare': 33,      # 10^33 erg
    }

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 256
        self.tflite_path = self.model_path / 'detector.tflite'
        self.weights_path = self.model_path / 'weights.npz'
        self._use_tflite = self.tflite_path.exists()

    def load_model(self) -> None:
        """Load the flare detection model."""
        if self._use_tflite:
            self._load_tflite()
        else:
            self._model = FlareStatisticalDetector()

    def _load_tflite(self) -> None:
        """Load TFLite model."""
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(
                model_path=str(self.tflite_path)
            )
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(
                    model_path=str(self.tflite_path)
                )
            except ImportError:
                self._model = FlareStatisticalDetector()
                return

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """
        Preprocess light curve window for flare detection.

        Uses robust normalization (median/MAD) to handle outliers.
        """
        data = np.asarray(raw_input).flatten()

        # Handle NaN
        if np.any(np.isnan(data)):
            nans = np.isnan(data)
            if np.all(nans):
                data = np.zeros(self.input_length)
            else:
                indices = np.arange(len(data))
                data[nans] = np.interp(indices[nans], indices[~nans], data[~nans])

        # Resample to target length
        if len(data) != self.input_length:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, self.input_length)
            data = np.interp(x_new, x_old, data)

        # Robust normalization (median/MAD)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad > 1e-10:
            data = (data - median) / (mad * 1.4826)  # Scale MAD to std
        else:
            data = data - median

        return data.reshape(self.input_length, 1).astype(np.float32)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        if self._interpreter is not None:
            input_data = np.expand_dims(preprocessed_data, axis=0)
            input_dtype = self._input_details[0]['dtype']

            if input_dtype == np.int8:
                scale, zero_point = self._input_details[0]['quantization']
                input_data = (input_data / scale + zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(input_dtype)

            self._interpreter.set_tensor(
                self._input_details[0]['index'],
                input_data
            )
            self._interpreter.invoke()

            output = self._interpreter.get_tensor(
                self._output_details[0]['index']
            )

            if self._output_details[0]['dtype'] == np.int8:
                scale, zero_point = self._output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale

            return output[0]

        elif self._model is not None:
            input_data = np.expand_dims(preprocessed_data, axis=0)
            return self._model.predict(input_data)[0]
        else:
            raise RuntimeError("No model loaded")

    def postprocess(self, model_output: np.ndarray,
                    raw_input: np.ndarray) -> NodeResult:
        """Convert model output to NodeResult with flare characterization."""
        probabilities = self._softmax(model_output)
        class_idx = np.argmax(probabilities)
        classification = self.CLASSES[class_idx]
        confidence = float(probabilities[class_idx])

        prob_dict = {
            cls: float(probabilities[i])
            for i, cls in enumerate(self.CLASSES)
        }

        detections = []
        metadata = {}

        # Analyze the light curve for flare characteristics
        flare_analysis = self._analyze_flare(raw_input)
        metadata['flare_analysis'] = flare_analysis

        # Add detection if flare found
        if classification != 'no_flare' and confidence > 0.5:
            detections.append({
                'type': 'stellar_flare',
                'classification': classification,
                'confidence': confidence,
                **flare_analysis
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

    def _analyze_flare(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """Characterize the flare from the light curve."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {'detected': False}

        # Baseline (quiescent) flux
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        sigma = mad * 1.4826

        # Find flare peak
        peak_idx = np.argmax(data)
        peak_flux = data[peak_idx]
        peak_amplitude = (peak_flux - median) / median if median > 0 else 0

        # SNR of peak
        snr = (peak_flux - median) / sigma if sigma > 0 else 0

        # Check if this is actually a flare (positive excursion)
        if snr < 3:
            return {
                'detected': False,
                'peak_snr': float(snr),
            }

        # Find flare start and end (where flux exceeds 3-sigma)
        threshold = median + 3 * sigma
        above_threshold = data > threshold

        # Find contiguous flare region containing the peak
        flare_start = peak_idx
        flare_end = peak_idx

        while flare_start > 0 and above_threshold[flare_start - 1]:
            flare_start -= 1

        while flare_end < len(data) - 1 and above_threshold[flare_end + 1]:
            flare_end += 1

        # Flare duration (as fraction of window)
        duration_points = flare_end - flare_start + 1
        duration_fraction = duration_points / len(data)

        # Rise and decay times
        rise_time = (peak_idx - flare_start) / len(data)
        decay_time = (flare_end - peak_idx) / len(data)

        # Equivalent duration (area under flare / quiescent flux)
        flare_flux = data[flare_start:flare_end + 1] - median
        equivalent_duration = np.sum(flare_flux) / median if median > 0 else 0

        # Estimate relative energy (proportional to equivalent duration)
        # This is a rough approximation without absolute calibration
        relative_energy = equivalent_duration

        return {
            'detected': True,
            'peak_amplitude': float(peak_amplitude),
            'peak_snr': float(snr),
            'flare_start_idx': int(flare_start),
            'flare_peak_idx': int(peak_idx),
            'flare_end_idx': int(flare_end),
            'duration_fraction': float(duration_fraction),
            'duration_points': int(duration_points),
            'rise_time_fraction': float(rise_time),
            'decay_time_fraction': float(decay_time),
            'equivalent_duration': float(equivalent_duration),
            'relative_energy': float(relative_energy),
            'impulsive': rise_time < decay_time * 0.5,  # Fast rise, slow decay
        }


class FlareStatisticalDetector:
    """Fallback statistical flare detector."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Detect flares using statistical thresholds."""
        x = np.asarray(data).flatten()

        # Handle NaN
        valid_mask = ~np.isnan(x)
        if not np.any(valid_mask):
            return np.array([1.0, 0, 0, 0, 0])  # no_flare

        x = x[valid_mask]

        if len(x) == 0:
            return np.array([1.0, 0, 0, 0, 0])  # no_flare

        # Robust statistics
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        sigma = mad * 1.4826 if mad > 0 else np.std(x)

        # Peak detection
        peak = np.max(x)
        snr = (peak - median) / sigma if sigma > 0 else 0

        probs = np.zeros(5)

        # Classification based on SNR
        if snr < 3:
            probs[0] = 0.9  # no_flare
        elif snr < 5:
            probs[1] = 0.6  # weak_flare
            probs[0] = 0.3
        elif snr < 10:
            probs[2] = 0.6  # moderate_flare
            probs[1] = 0.3
        elif snr < 20:
            probs[3] = 0.6  # strong_flare
            probs[2] = 0.3
        else:
            probs[4] = 0.7  # superflare
            probs[3] = 0.2

        probs = probs / np.sum(probs)
        return np.array([probs])  # Return as batch
