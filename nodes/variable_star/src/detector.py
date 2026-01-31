"""
Variable Star Classifier Node (VSTAR-001)

TinyML-optimized CNN for classifying variable stars from phase-folded
light curves.

Model size: 72KB (INT8 quantized)
Input: 512-point phase-folded light curve
Output: 7-class variable star classification

Variable Star Classes:
- Cepheid: Classical and Type II Cepheids
- RR Lyrae: RRab, RRc, RRd subtypes
- Delta Scuti: High-frequency pulsators
- Eclipsing Binary: EA, EB, EW types
- Rotational: Starspots, BY Dra variables
- Irregular: Long-period, semi-regular
- Constant: No significant variability
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class VariableStarNode(BaseNode):
    """
    Variable star classification using TinyML CNN.

    Classifies variable stars from phase-folded light curves,
    identifying the type of variability based on light curve shape.
    """

    CLASSES = [
        'cepheid',
        'rr_lyrae',
        'delta_scuti',
        'eclipsing_binary',
        'rotational',
        'irregular',
        'constant',
    ]

    # Typical period ranges for each class (days)
    PERIOD_RANGES = {
        'cepheid': (1.0, 100.0),
        'rr_lyrae': (0.2, 1.0),
        'delta_scuti': (0.02, 0.3),
        'eclipsing_binary': (0.1, 1000.0),
        'rotational': (0.5, 50.0),
        'irregular': (10.0, 1000.0),
        'constant': (None, None),
    }

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_length = 512
        self.tflite_path = self.model_path / 'classifier.tflite'
        self.weights_path = self.model_path / 'weights.npz'
        self._use_tflite = self.tflite_path.exists()

    def load_model(self) -> None:
        """Load the classification model."""
        if self._use_tflite:
            self._load_tflite()
        else:
            # Use statistical fallback
            self._model = VariableStarStatisticalClassifier()

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
                self._model = VariableStarStatisticalClassifier()
                return

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """
        Preprocess light curve for classification.

        If time series provided, assumes it's already phase-folded.
        Normalizes and resamples to 512 points.
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

        # Z-score normalization
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-10:
            data = (data - mean) / std
        else:
            data = np.zeros_like(data)

        return data.reshape(self.input_length, 1).astype(np.float32)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed data."""
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
        """Convert model output to NodeResult."""
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

        # Analyze the phase-folded curve
        curve_features = self._analyze_phase_curve(raw_input)
        metadata['curve_features'] = curve_features

        # Add detection if significant variability found
        if classification != 'constant' and confidence > 0.5:
            detections.append({
                'type': 'variable_star',
                'classification': classification,
                'confidence': confidence,
                **curve_features
            })

            # Add subtype hints
            subtype_hints = self._get_subtype_hints(
                classification, curve_features
            )
            if subtype_hints:
                metadata['subtype_hints'] = subtype_hints

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
        """Apply softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _analyze_phase_curve(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """Extract features from the phase-folded curve."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {'n_points': 0}

        # Basic statistics
        amplitude = np.ptp(data)
        mean = np.mean(data)
        std = np.std(data)

        # Asymmetry (skewness of brightness)
        skewness = self._skewness(data)

        # Find primary minimum/maximum
        min_idx = np.argmin(data)
        max_idx = np.argmax(data)

        # Phase of minimum (0-1)
        min_phase = min_idx / len(data)
        max_phase = max_idx / len(data)

        # Rise time vs fall time
        if max_phase > min_phase:
            rise_time = max_phase - min_phase
        else:
            rise_time = 1 - min_phase + max_phase
        fall_time = 1 - rise_time

        # Fourier features (simple approximation)
        # Ratio of amplitudes in different harmonics
        fft = np.abs(np.fft.rfft(data - mean))
        if len(fft) > 2:
            r21 = fft[2] / (fft[1] + 1e-10)  # 2nd/1st harmonic
            r31 = fft[3] / (fft[1] + 1e-10) if len(fft) > 3 else 0
        else:
            r21 = r31 = 0

        return {
            'n_points': len(data),
            'amplitude': float(amplitude),
            'mean': float(mean),
            'std': float(std),
            'skewness': float(skewness),
            'min_phase': float(min_phase),
            'max_phase': float(max_phase),
            'rise_fraction': float(rise_time),
            'r21': float(r21),
            'r31': float(r31),
        }

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 3) / n)

    def _get_subtype_hints(self, classification: str,
                          features: Dict[str, Any]) -> Dict[str, Any]:
        """Get hints about variable star subtype."""
        hints = {}

        if classification == 'rr_lyrae':
            # RRab have sawtooth shape, RRc more sinusoidal
            if features.get('rise_fraction', 0.5) < 0.35:
                hints['likely_subtype'] = 'RRab'
                hints['description'] = 'Fast rise, slow decline - typical RRab'
            else:
                hints['likely_subtype'] = 'RRc'
                hints['description'] = 'Symmetric - typical RRc'

        elif classification == 'cepheid':
            if features.get('amplitude', 0) > 0.5:
                hints['likely_subtype'] = 'Classical Cepheid'
            else:
                hints['likely_subtype'] = 'Type II Cepheid'

        elif classification == 'eclipsing_binary':
            # EA: clear flat portions, EB: continuous variation
            if features.get('r21', 0) > 0.3:
                hints['likely_subtype'] = 'EW (contact)'
            else:
                hints['likely_subtype'] = 'EA (detached)'

        return hints


class VariableStarStatisticalClassifier:
    """Fallback statistical classifier for variable stars."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Classify based on light curve features."""
        x = np.asarray(data).flatten()

        # Handle NaN
        valid_mask = ~np.isnan(x)
        if not np.any(valid_mask):
            return np.array([[0, 0, 0, 0, 0, 0, 1]])  # constant

        x = x[valid_mask]

        if len(x) == 0:
            return np.array([[0, 0, 0, 0, 0, 0, 1]])  # constant

        amplitude = np.ptp(x)
        std = np.std(x)
        skew = self._skewness(x)

        # Fourier analysis
        mean = np.mean(x)
        fft = np.abs(np.fft.rfft(x - mean))
        r21 = fft[2] / (fft[1] + 1e-10) if len(fft) > 2 else 0

        probs = np.zeros(7)

        # Very low variability = constant
        if std < 0.01:
            probs[6] = 0.9
            probs = probs / np.sum(probs)
            return np.array([probs])

        # High asymmetry (sawtooth) suggests RR Lyrae or Cepheid
        if skew < -0.3:
            probs[0] = 0.3  # cepheid
            probs[1] = 0.4  # rr_lyrae

        # Symmetric with double wave suggests eclipsing binary
        elif r21 > 0.3:
            probs[3] = 0.6  # eclipsing_binary

        # Sinusoidal suggests rotational or delta scuti
        elif amplitude < 0.1 and std < 0.05:
            probs[2] = 0.3  # delta_scuti
            probs[4] = 0.4  # rotational

        # Otherwise irregular
        else:
            probs[5] = 0.4  # irregular
            probs[4] = 0.3  # rotational

        probs = probs / np.sum(probs)
        return np.array([probs])  # Return as batch

    def _skewness(self, x: np.ndarray) -> float:
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.sum(((x - mean) / std) ** 3) / n)
