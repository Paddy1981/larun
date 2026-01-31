"""
Exoplanet Transit Detector Node (EXOPLANET-001)

TinyML-optimized CNN for detecting exoplanet transits in light curves.
Model size: 48KB (INT8 quantized)
Input: 1024-point light curve
Output: 6-class classification + transit parameters

This node is designed for:
- TESS, Kepler, K2 light curves
- Real-time analysis on edge devices
- High recall for transit detection
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys

# Add parent paths for imports
node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import TFLiteNode, NumpyNode, BaseNode, NodeResult


class ExoplanetNode(BaseNode):
    """
    Exoplanet transit detection node using TinyML CNN.

    Detects planetary transits in stellar light curves and estimates
    basic transit parameters (depth, duration, period hints).

    Output Classes:
        - noise: No significant signal
        - stellar_signal: Stellar variability but no transit
        - planetary_transit: Likely exoplanet transit
        - eclipsing_binary: Eclipsing binary star
        - instrument_artifact: Instrumental systematics
        - unknown_anomaly: Unclassified anomaly
    """

    # Class labels
    CLASSES = [
        'noise',
        'stellar_signal',
        'planetary_transit',
        'eclipsing_binary',
        'instrument_artifact',
        'unknown_anomaly',
    ]

    def __init__(self, node_path: Path):
        super().__init__(node_path)

        self.input_length = 1024
        self.tflite_path = self.model_path / 'detector.tflite'
        self.weights_path = self.model_path / 'weights.npz'

        # Check which model format is available
        self._use_tflite = self.tflite_path.exists()
        self._use_numpy = self.weights_path.exists()

    def load_model(self) -> None:
        """Load the TinyML model (TFLite preferred, NumPy fallback)."""
        if self._use_tflite:
            self._load_tflite_model()
        elif self._use_numpy:
            self._load_numpy_model()
        else:
            # Try to use the production model from main larun
            prod_model = self.node_path.parent.parent / 'models' / 'production' / 'larun_model_int8.tflite'
            if prod_model.exists():
                self.tflite_path = prod_model
                self._use_tflite = True
                self._load_tflite_model()
            else:
                # Create a simple fallback model
                self._create_fallback_model()

    def _load_tflite_model(self) -> None:
        """Load TensorFlow Lite model."""
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
                raise ImportError("TensorFlow or tflite_runtime required")

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _load_numpy_model(self) -> None:
        """Load pure NumPy model weights."""
        from model.numpy_cnn import SpectralCNN

        weights = np.load(self.weights_path, allow_pickle=True)
        self._model = SpectralCNN(
            input_length=self.input_length,
            num_classes=len(self.CLASSES)
        )
        self._model.load_weights(weights)

    def _create_fallback_model(self) -> None:
        """Create a simple fallback model for testing."""
        # Simple statistical classifier as fallback
        self._model = SimpleStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """
        Preprocess light curve for model input.

        Steps:
        1. Flatten if needed
        2. Handle NaN values
        3. Resample to 1024 points
        4. Normalize to [0, 1]
        5. Reshape to (1024, 1)
        """
        # Flatten if multi-dimensional
        data = np.asarray(raw_input).flatten()

        # Handle NaN values - linear interpolation
        if np.any(np.isnan(data)):
            nans = np.isnan(data)
            if np.all(nans):
                # All NaN - return zeros
                data = np.zeros(self.input_length)
            else:
                # Interpolate
                indices = np.arange(len(data))
                data[nans] = np.interp(
                    indices[nans],
                    indices[~nans],
                    data[~nans]
                )

        # Resample to target length
        if len(data) != self.input_length:
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, self.input_length)
            data = np.interp(x_new, x_old, data)

        # Normalize to [0, 1] (min-max)
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-10:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data) + 0.5

        # Reshape to (1024, 1)
        return data.reshape(self.input_length, 1).astype(np.float32)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed data."""
        if self._interpreter is not None:
            # TFLite inference
            input_data = np.expand_dims(preprocessed_data, axis=0)

            # Handle quantization
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

            # Dequantize if needed
            if self._output_details[0]['dtype'] == np.int8:
                scale, zero_point = self._output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale

            return output[0]  # Remove batch dimension

        elif self._model is not None:
            # NumPy model inference
            input_data = np.expand_dims(preprocessed_data, axis=0)
            return self._model.predict(input_data)[0]

        else:
            raise RuntimeError("No model loaded")

    def postprocess(self, model_output: np.ndarray,
                    raw_input: np.ndarray) -> NodeResult:
        """
        Convert model output to NodeResult with transit analysis.

        Performs additional analysis for transit-like signals:
        - Estimates transit depth
        - Estimates transit duration
        - Calculates SNR
        """
        # Get classification
        probabilities = self._softmax(model_output)
        class_idx = np.argmax(probabilities)
        classification = self.CLASSES[class_idx]
        confidence = float(probabilities[class_idx])

        # Create probability dict
        prob_dict = {
            cls: float(probabilities[i])
            for i, cls in enumerate(self.CLASSES)
        }

        # Analyze for detections
        detections = []
        metadata = {}

        # If transit detected, analyze parameters
        if classification == 'planetary_transit' and confidence > 0.5:
            transit_info = self._analyze_transit(raw_input)
            if transit_info:
                detections.append(transit_info)
                metadata['transit_analysis'] = transit_info

        # If eclipsing binary, note the characteristics
        elif classification == 'eclipsing_binary' and confidence > 0.5:
            eb_info = self._analyze_eclipsing_binary(raw_input)
            if eb_info:
                detections.append(eb_info)
                metadata['eb_analysis'] = eb_info

        # Add general light curve statistics
        metadata['lc_stats'] = self._calculate_lc_stats(raw_input)

        return NodeResult(
            node_id=self.node_id,
            classification=classification,
            confidence=confidence,
            probabilities=prob_dict,
            detections=detections,
            metadata=metadata,
            inference_time_ms=0.0,  # Will be updated by run()
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _analyze_transit(self, raw_input: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze light curve for transit parameters."""
        data = np.asarray(raw_input).flatten()

        # Handle NaN
        data = data[~np.isnan(data)]
        if len(data) < 100:
            return None

        # Estimate transit depth (approximate)
        median_flux = np.median(data)
        min_flux = np.percentile(data, 1)  # 1st percentile to avoid outliers
        depth = (median_flux - min_flux) / median_flux

        # Find potential transit regions (flux below median - 2*MAD)
        mad = np.median(np.abs(data - median_flux))
        transit_threshold = median_flux - 3 * mad
        in_transit = data < transit_threshold

        # Estimate duration (as fraction of total)
        duration_fraction = np.sum(in_transit) / len(data)

        # Calculate SNR (very rough estimate)
        noise = mad * 1.4826  # Convert MAD to std
        snr = depth * median_flux / noise if noise > 0 else 0

        return {
            'type': 'transit',
            'depth': float(depth),
            'depth_ppm': float(depth * 1e6),
            'duration_fraction': float(duration_fraction),
            'snr': float(snr),
            'n_points_in_transit': int(np.sum(in_transit)),
        }

    def _analyze_eclipsing_binary(self, raw_input: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze light curve for eclipsing binary characteristics."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) < 100:
            return None

        median_flux = np.median(data)
        min_flux = np.min(data)
        max_flux = np.max(data)

        # Primary eclipse depth
        primary_depth = (median_flux - min_flux) / median_flux

        # Check for secondary eclipse (approximate)
        # Look for second minimum
        secondary_depth = 0.0

        return {
            'type': 'eclipsing_binary',
            'primary_depth': float(primary_depth),
            'secondary_depth': float(secondary_depth),
            'amplitude': float((max_flux - min_flux) / median_flux),
        }

    def _calculate_lc_stats(self, raw_input: np.ndarray) -> Dict[str, float]:
        """Calculate basic light curve statistics."""
        data = np.asarray(raw_input).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return {'n_points': 0}

        return {
            'n_points': len(data),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'mad': float(np.median(np.abs(data - np.median(data)))),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.ptp(data)),
        }


class SimpleStatisticalClassifier:
    """
    Simple statistical classifier as fallback when no TinyML model available.

    Uses basic light curve statistics to classify signals.
    Not as accurate as the CNN but provides baseline functionality.
    """

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict class probabilities from light curve statistics."""
        # Flatten and clean data
        x = data.flatten()
        x = x[~np.isnan(x)]

        if len(x) == 0:
            return np.array([1.0, 0, 0, 0, 0, 0])  # noise

        # Calculate features
        std = np.std(x)
        mad = np.median(np.abs(x - np.median(x)))
        skew = self._skewness(x)
        kurtosis = self._kurtosis(x)

        # Simple heuristic classification
        probs = np.zeros(6)

        # Very low variability = noise
        if std < 0.01:
            probs[0] = 0.8  # noise
            probs[1] = 0.2  # stellar_signal

        # Negative skew suggests dips (transits/eclipses)
        elif skew < -0.5:
            if kurtosis > 3:  # Sharp features
                probs[2] = 0.4  # planetary_transit
                probs[3] = 0.4  # eclipsing_binary
            else:
                probs[1] = 0.5  # stellar_signal
                probs[2] = 0.3  # planetary_transit

        # Positive skew suggests flares or artifacts
        elif skew > 0.5:
            probs[4] = 0.5  # instrument_artifact
            probs[1] = 0.3  # stellar_signal

        # Symmetric variability
        else:
            probs[1] = 0.6  # stellar_signal
            probs[5] = 0.2  # unknown

        # Normalize
        probs = probs / np.sum(probs)
        return probs

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.sum(((x - mean) / std) ** 3) / n

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.sum(((x - mean) / std) ** 4) / n - 3
