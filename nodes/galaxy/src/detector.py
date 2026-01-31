"""
Galaxy Morphology Classifier Node (GALAXY-001)

Classifies galaxy morphology from RGB image cutouts.

Model size: 88KB (INT8 quantized)
Input: 64x64x3 RGB image
Output: Galaxy type classification
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

node_path = Path(__file__).parent.parent.parent.parent
if str(node_path / 'src') not in sys.path:
    sys.path.insert(0, str(node_path / 'src'))

from nodes.base import BaseNode, NodeResult


class GalaxyNode(BaseNode):
    """Galaxy morphology classification."""

    CLASSES = [
        'elliptical',
        'spiral',
        'barred_spiral',
        'irregular',
        'merger',
        'edge_on',
        'unknown',
    ]

    def __init__(self, node_path: Path):
        super().__init__(node_path)
        self.input_shape = (64, 64, 3)
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
                self._model = GalaxyStatisticalClassifier()
        else:
            self._model = GalaxyStatisticalClassifier()

    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """Preprocess galaxy image."""
        data = np.asarray(raw_input)

        # Handle various input shapes
        if data.ndim == 1:
            # Assume flattened 64x64x3
            side = int(np.sqrt(len(data) / 3))
            if side * side * 3 == len(data):
                data = data.reshape(side, side, 3)
            else:
                # Assume grayscale, replicate to 3 channels
                side = int(np.sqrt(len(data)))
                data = data.reshape(side, side)
                data = np.stack([data, data, data], axis=-1)

        if data.ndim == 2:
            # Grayscale to RGB
            data = np.stack([data, data, data], axis=-1)

        # Resize to 64x64 if needed
        if data.shape[:2] != (64, 64):
            # Simple resize using interpolation
            from scipy import ndimage
            zoom_factors = (64 / data.shape[0], 64 / data.shape[1], 1)
            data = ndimage.zoom(data, zoom_factors, order=1)

        # Normalize to [0, 1]
        data = data.astype(np.float32)
        if data.max() > 1:
            data = data / 255.0

        return data

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

        # Analyze image properties
        img_props = self._analyze_image(raw_input)

        metadata = {
            'morphology': classification,
            'image_properties': img_props,
        }

        detections = [{
            'type': 'galaxy_morphology',
            'classification': classification,
            **img_props
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

    def _analyze_image(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """Extract basic image properties."""
        data = np.asarray(raw_input)

        if data.ndim < 2:
            return {}

        # Average brightness
        brightness = np.mean(data)

        # Central concentration (ratio of central to outer flux)
        h, w = data.shape[:2]
        center_size = h // 4
        center = data[h//2-center_size:h//2+center_size,
                      w//2-center_size:w//2+center_size]
        concentration = np.mean(center) / (np.mean(data) + 1e-10)

        # Asymmetry (difference when rotated 180)
        rotated = np.rot90(np.rot90(data))
        asymmetry = np.mean(np.abs(data - rotated)) / (np.mean(data) + 1e-10)

        return {
            'brightness': float(brightness),
            'concentration': float(concentration),
            'asymmetry': float(asymmetry),
        }


class GalaxyStatisticalClassifier:
    """Fallback statistical classifier."""

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Classify based on image statistics."""
        data = np.asarray(data)

        probs = np.zeros(7)

        if data.size == 0:
            probs[6] = 1.0
            return probs

        # Calculate concentration
        if data.ndim >= 2:
            h, w = data.shape[:2]
            center_size = max(1, h // 4)
            center = data[h//2-center_size:h//2+center_size,
                          w//2-center_size:w//2+center_size]
            concentration = np.mean(center) / (np.mean(data) + 1e-10)
        else:
            concentration = 1.0

        # High concentration = likely elliptical
        if concentration > 1.5:
            probs[0] = 0.5  # elliptical
            probs[5] = 0.2  # edge_on
        elif concentration > 1.2:
            probs[1] = 0.4  # spiral
            probs[2] = 0.3  # barred_spiral
        else:
            probs[3] = 0.3  # irregular
            probs[4] = 0.2  # merger
            probs[6] = 0.2  # unknown

        return probs / np.sum(probs)
