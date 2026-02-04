"""
Specialized TinyML Models for Astronomical Analysis
====================================================
8 specialized CNN models for different astronomical tasks.
Each model is optimized for TinyML deployment (<100KB).
"""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class ModelSpec:
    """Specification for a specialized model."""
    node_id: str
    name: str
    input_shape: Tuple[int, ...]
    output_classes: List[str]
    description: str


# Model Specifications
MODEL_SPECS = {
    "EXOPLANET-001": ModelSpec(
        node_id="EXOPLANET-001",
        name="ExoplanetDetector",
        input_shape=(1024, 1),
        output_classes=["noise", "stellar_signal", "planetary_transit",
                       "eclipsing_binary", "instrument_artifact", "unknown_anomaly"],
        description="Exoplanet transit detection in light curves"
    ),
    "VSTAR-001": ModelSpec(
        node_id="VSTAR-001",
        name="VariableStarClassifier",
        input_shape=(512, 1),
        output_classes=["cepheid", "rr_lyrae", "delta_scuti",
                       "eclipsing_binary", "rotational", "irregular", "constant"],
        description="Variable star classification"
    ),
    "FLARE-001": ModelSpec(
        node_id="FLARE-001",
        name="FlareDetector",
        input_shape=(256, 1),
        output_classes=["no_flare", "weak_flare", "moderate_flare",
                       "strong_flare", "superflare"],
        description="Stellar flare detection"
    ),
    "ASTERO-001": ModelSpec(
        node_id="ASTERO-001",
        name="AsteroseismologyAnalyzer",
        input_shape=(512, 1),
        output_classes=["no_oscillation", "solar_like", "red_giant",
                       "delta_scuti", "gamma_dor", "hybrid"],
        description="Asteroseismology oscillation detection"
    ),
    "SUPERNOVA-001": ModelSpec(
        node_id="SUPERNOVA-001",
        name="SupernovaDetector",
        input_shape=(128, 1),
        output_classes=["no_transient", "sn_ia", "sn_ii", "sn_ibc",
                       "kilonova", "tde", "other_transient"],
        description="Supernova and transient detection"
    ),
    "GALAXY-001": ModelSpec(
        node_id="GALAXY-001",
        name="GalaxyClassifier",
        input_shape=(64, 64, 1),  # Grayscale image
        output_classes=["elliptical", "spiral", "barred_spiral",
                       "irregular", "merger", "edge_on", "unknown"],
        description="Galaxy morphology classification"
    ),
    "SPECTYPE-001": ModelSpec(
        node_id="SPECTYPE-001",
        name="SpectralTypeClassifier",
        input_shape=(8,),
        output_classes=["O", "B", "A", "F", "G", "K", "M", "L"],
        description="Stellar spectral type classification"
    ),
    "MICROLENS-001": ModelSpec(
        node_id="MICROLENS-001",
        name="MicrolensingDetector",
        input_shape=(512, 1),
        output_classes=["no_event", "single_lens", "binary_lens",
                       "planetary", "parallax", "unclear"],
        description="Microlensing event detection"
    ),
}


class BaseNumpyModel:
    """Base class for NumPy-based neural network models."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.weights: Dict[str, np.ndarray] = {}
        self.input_shape = spec.input_shape
        self.num_classes = len(spec.output_classes)
        self.class_labels = spec.output_classes

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _conv1d(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """1D convolution with same padding."""
        batch_size, seq_len, in_channels = x.shape
        kernel_size, _, out_channels = weights.shape
        pad = kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        output = np.zeros((batch_size, seq_len, out_channels), dtype=np.float32)
        for i in range(seq_len):
            window = x_padded[:, i:i+kernel_size, :]
            for j in range(out_channels):
                output[:, i, j] = np.sum(window * weights[:, :, j], axis=(1, 2)) + bias[j]
        return output

    def _conv2d(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """2D convolution with same padding."""
        batch_size, h, w, in_channels = x.shape
        kh, kw, _, out_channels = weights.shape
        pad_h, pad_w = kh // 2, kw // 2
        x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        output = np.zeros((batch_size, h, w, out_channels), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                window = x_padded[:, i:i+kh, j:j+kw, :]
                for k in range(out_channels):
                    output[:, i, j, k] = np.sum(window * weights[:, :, :, k], axis=(1, 2, 3)) + bias[k]
        return output

    def _maxpool1d(self, x: np.ndarray, pool_size: int) -> np.ndarray:
        batch_size, seq_len, channels = x.shape
        out_len = seq_len // pool_size
        output = np.zeros((batch_size, out_len, channels), dtype=np.float32)
        for i in range(out_len):
            output[:, i, :] = np.max(x[:, i*pool_size:(i+1)*pool_size, :], axis=1)
        return output

    def _maxpool2d(self, x: np.ndarray, pool_size: int) -> np.ndarray:
        batch_size, h, w, channels = x.shape
        out_h, out_w = h // pool_size, w // pool_size
        output = np.zeros((batch_size, out_h, out_w, channels), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                window = x[:, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :]
                output[:, i, j, :] = np.max(window, axis=(1, 2))
        return output

    def _global_avgpool(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 3:
            return np.mean(x, axis=1)
        else:
            return np.mean(x, axis=(1, 2))

    def _dense(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.dot(x, weights) + bias

    def _batchnorm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.forward(x)
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return predictions, confidences

    def predict_labels(self, x: np.ndarray) -> List[Tuple[str, float]]:
        preds, confs = self.predict(x)
        return [(self.class_labels[p], float(c)) for p, c in zip(preds, confs)]

    def save(self, filepath: str):
        np.savez(filepath, **self.weights,
                 _spec_node_id=self.spec.node_id,
                 _class_labels=np.array(self.class_labels))

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.weights = {k: data[k] for k in data.files if not k.startswith('_')}

    def get_model_size(self) -> Dict[str, Any]:
        total_params = sum(w.size for w in self.weights.values())
        return {
            "node_id": self.spec.node_id,
            "total_parameters": total_params,
            "size_float32_kb": total_params * 4 / 1024,
            "size_int8_kb": total_params / 1024,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes
        }

    def export_tflite_weights(self, filepath: str):
        """Export weights in a format suitable for TFLite conversion."""
        export_data = {
            "node_id": self.spec.node_id,
            "name": self.spec.name,
            "input_shape": list(self.input_shape),
            "output_classes": self.class_labels,
            "weights": {k: v.tolist() for k, v in self.weights.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(export_data, f)


class ExoplanetDetector(BaseNumpyModel):
    """
    EXOPLANET-001: CNN for detecting planetary transits in light curves.
    Input: 1024-point light curve
    Output: 6 classes (noise, stellar_signal, planetary_transit, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["EXOPLANET-001"])
        self._build()

    def _build(self):
        np.random.seed(42)
        # Conv layers: 16 -> 32 -> 64 filters
        self.weights["conv1_w"] = np.random.randn(7, 1, 16).astype(np.float32) * np.sqrt(2/7)
        self.weights["conv1_b"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(16, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(16, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(5, 16, 32).astype(np.float32) * np.sqrt(2/80)
        self.weights["conv2_b"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(32, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(32, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(3, 32, 64).astype(np.float32) * np.sqrt(2/96)
        self.weights["conv3_b"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(64, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(64, dtype=np.float32)

        # Dense layers
        self.weights["fc1_w"] = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc1_b"] = np.zeros(32, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(32, self.num_classes).astype(np.float32) * np.sqrt(2/32)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        # Block 1
        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        # Block 2
        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        # Block 3
        x = self._conv1d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        # Global pooling + Dense
        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class VariableStarClassifier(BaseNumpyModel):
    """
    VSTAR-001: CNN for classifying variable star types.
    Input: 512-point periodic light curve
    Output: 7 classes (cepheid, rr_lyrae, delta_scuti, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["VSTAR-001"])
        self._build()

    def _build(self):
        np.random.seed(43)
        # Deeper network for complex periodic patterns
        self.weights["conv1_w"] = np.random.randn(11, 1, 24).astype(np.float32) * np.sqrt(2/11)
        self.weights["conv1_b"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(24, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(24, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(7, 24, 48).astype(np.float32) * np.sqrt(2/168)
        self.weights["conv2_b"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(48, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(48, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(5, 48, 64).astype(np.float32) * np.sqrt(2/240)
        self.weights["conv3_b"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(64, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(64, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(64, 48).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc1_b"] = np.zeros(48, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(48, self.num_classes).astype(np.float32) * np.sqrt(2/48)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class FlareDetector(BaseNumpyModel):
    """
    FLARE-001: CNN for detecting stellar flares.
    Input: 256-point light curve segment
    Output: 5 classes (no_flare, weak, moderate, strong, superflare)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["FLARE-001"])
        self._build()

    def _build(self):
        np.random.seed(44)
        # Compact network for quick flare detection
        self.weights["conv1_w"] = np.random.randn(5, 1, 16).astype(np.float32) * np.sqrt(2/5)
        self.weights["conv1_b"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(16, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(16, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(3, 16, 32).astype(np.float32) * np.sqrt(2/48)
        self.weights["conv2_b"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(32, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(32, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(32, 24).astype(np.float32) * np.sqrt(2/32)
        self.weights["fc1_b"] = np.zeros(24, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(24, self.num_classes).astype(np.float32) * np.sqrt(2/24)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class AsteroseismologyAnalyzer(BaseNumpyModel):
    """
    ASTERO-001: CNN for analyzing stellar oscillations.
    Input: 512-point power spectrum
    Output: 6 classes (no_oscillation, solar_like, red_giant, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["ASTERO-001"])
        self._build()

    def _build(self):
        np.random.seed(45)
        self.weights["conv1_w"] = np.random.randn(9, 1, 20).astype(np.float32) * np.sqrt(2/9)
        self.weights["conv1_b"] = np.zeros(20, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(20, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(20, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(20, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(20, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(7, 20, 40).astype(np.float32) * np.sqrt(2/140)
        self.weights["conv2_b"] = np.zeros(40, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(40, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(40, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(40, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(40, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(5, 40, 64).astype(np.float32) * np.sqrt(2/200)
        self.weights["conv3_b"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(64, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(64, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc1_b"] = np.zeros(32, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(32, self.num_classes).astype(np.float32) * np.sqrt(2/32)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class SupernovaDetector(BaseNumpyModel):
    """
    SUPERNOVA-001: CNN for detecting supernovae and transients.
    Input: 128-point light curve segment
    Output: 7 classes (no_transient, sn_ia, sn_ii, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["SUPERNOVA-001"])
        self._build()

    def _build(self):
        np.random.seed(46)
        # Wider network for diverse transient types
        self.weights["conv1_w"] = np.random.randn(5, 1, 32).astype(np.float32) * np.sqrt(2/5)
        self.weights["conv1_b"] = np.zeros(32, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(32, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(32, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(32, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(32, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(3, 32, 64).astype(np.float32) * np.sqrt(2/96)
        self.weights["conv2_b"] = np.zeros(64, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(64, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(64, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(64, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(64, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(3, 64, 96).astype(np.float32) * np.sqrt(2/192)
        self.weights["conv3_b"] = np.zeros(96, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(96, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(96, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(96, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(96, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(96, 48).astype(np.float32) * np.sqrt(2/96)
        self.weights["fc1_b"] = np.zeros(48, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(48, self.num_classes).astype(np.float32) * np.sqrt(2/48)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 2)

        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 2)

        x = self._conv1d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 2)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class GalaxyClassifier(BaseNumpyModel):
    """
    GALAXY-001: 2D CNN for galaxy morphology classification.
    Input: 64x64 grayscale image
    Output: 7 classes (elliptical, spiral, barred_spiral, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["GALAXY-001"])
        self._build()

    def _build(self):
        np.random.seed(47)
        # 2D CNN for image classification
        self.weights["conv1_w"] = np.random.randn(3, 3, 1, 16).astype(np.float32) * np.sqrt(2/9)
        self.weights["conv1_b"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(16, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(16, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(16, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(3, 3, 16, 32).astype(np.float32) * np.sqrt(2/144)
        self.weights["conv2_b"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(32, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(32, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(32, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(3, 3, 32, 64).astype(np.float32) * np.sqrt(2/288)
        self.weights["conv3_b"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(64, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(64, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(64, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(64, 48).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc1_b"] = np.zeros(48, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(48, self.num_classes).astype(np.float32) * np.sqrt(2/48)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[:, :, :, np.newaxis]

        x = self._conv2d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool2d(x, 2)

        x = self._conv2d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool2d(x, 2)

        x = self._conv2d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool2d(x, 2)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class SpectralTypeClassifier(BaseNumpyModel):
    """
    SPECTYPE-001: MLP for stellar spectral type classification.
    Input: 8 photometric features
    Output: 8 classes (O, B, A, F, G, K, M, L)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["SPECTYPE-001"])
        self._build()

    def _build(self):
        np.random.seed(48)
        # Simple MLP for tabular data
        self.weights["fc1_w"] = np.random.randn(8, 32).astype(np.float32) * np.sqrt(2/8)
        self.weights["fc1_b"] = np.zeros(32, dtype=np.float32)

        self.weights["fc2_w"] = np.random.randn(32, 64).astype(np.float32) * np.sqrt(2/32)
        self.weights["fc2_b"] = np.zeros(64, dtype=np.float32)

        self.weights["fc3_w"] = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc3_b"] = np.zeros(32, dtype=np.float32)

        self.weights["out_w"] = np.random.randn(32, self.num_classes).astype(np.float32) * np.sqrt(2/32)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[np.newaxis, :]

        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)

        x = self._dense(x, self.weights["fc2_w"], self.weights["fc2_b"])
        x = self._relu(x)

        x = self._dense(x, self.weights["fc3_w"], self.weights["fc3_b"])
        x = self._relu(x)

        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


class MicrolensingDetector(BaseNumpyModel):
    """
    MICROLENS-001: CNN for detecting microlensing events.
    Input: 512-point light curve
    Output: 6 classes (no_event, single_lens, binary_lens, etc.)
    """

    def __init__(self):
        super().__init__(MODEL_SPECS["MICROLENS-001"])
        self._build()

    def _build(self):
        np.random.seed(49)
        # Multi-scale convolutions for different event durations
        self.weights["conv1_w"] = np.random.randn(15, 1, 24).astype(np.float32) * np.sqrt(2/15)
        self.weights["conv1_b"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_gamma"] = np.ones(24, dtype=np.float32)
        self.weights["bn1_beta"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_mean"] = np.zeros(24, dtype=np.float32)
        self.weights["bn1_var"] = np.ones(24, dtype=np.float32)

        self.weights["conv2_w"] = np.random.randn(9, 24, 48).astype(np.float32) * np.sqrt(2/216)
        self.weights["conv2_b"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_gamma"] = np.ones(48, dtype=np.float32)
        self.weights["bn2_beta"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_mean"] = np.zeros(48, dtype=np.float32)
        self.weights["bn2_var"] = np.ones(48, dtype=np.float32)

        self.weights["conv3_w"] = np.random.randn(5, 48, 72).astype(np.float32) * np.sqrt(2/240)
        self.weights["conv3_b"] = np.zeros(72, dtype=np.float32)
        self.weights["bn3_gamma"] = np.ones(72, dtype=np.float32)
        self.weights["bn3_beta"] = np.zeros(72, dtype=np.float32)
        self.weights["bn3_mean"] = np.zeros(72, dtype=np.float32)
        self.weights["bn3_var"] = np.ones(72, dtype=np.float32)

        self.weights["fc1_w"] = np.random.randn(72, 48).astype(np.float32) * np.sqrt(2/72)
        self.weights["fc1_b"] = np.zeros(48, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(48, self.num_classes).astype(np.float32) * np.sqrt(2/48)
        self.weights["out_b"] = np.zeros(self.num_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        x = self._conv1d(x, self.weights["conv1_w"], self.weights["conv1_b"])
        x = self._batchnorm(x, self.weights["bn1_gamma"], self.weights["bn1_beta"],
                          self.weights["bn1_mean"], self.weights["bn1_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv2_w"], self.weights["conv2_b"])
        x = self._batchnorm(x, self.weights["bn2_gamma"], self.weights["bn2_beta"],
                          self.weights["bn2_mean"], self.weights["bn2_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._conv1d(x, self.weights["conv3_w"], self.weights["conv3_b"])
        x = self._batchnorm(x, self.weights["bn3_gamma"], self.weights["bn3_beta"],
                          self.weights["bn3_mean"], self.weights["bn3_var"])
        x = self._relu(x)
        x = self._maxpool1d(x, 4)

        x = self._global_avgpool(x)
        x = self._dense(x, self.weights["fc1_w"], self.weights["fc1_b"])
        x = self._relu(x)
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])
        return self._softmax(x)


# Model Registry
MODEL_CLASSES = {
    "EXOPLANET-001": ExoplanetDetector,
    "VSTAR-001": VariableStarClassifier,
    "FLARE-001": FlareDetector,
    "ASTERO-001": AsteroseismologyAnalyzer,
    "SUPERNOVA-001": SupernovaDetector,
    "GALAXY-001": GalaxyClassifier,
    "SPECTYPE-001": SpectralTypeClassifier,
    "MICROLENS-001": MicrolensingDetector,
}


def get_model(node_id: str) -> BaseNumpyModel:
    """Factory function to get a model by node ID."""
    if node_id not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {node_id}. Available: {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[node_id]()


def list_models() -> List[Dict[str, Any]]:
    """List all available models with their specifications."""
    return [
        {
            "node_id": spec.node_id,
            "name": spec.name,
            "input_shape": spec.input_shape,
            "num_classes": len(spec.output_classes),
            "classes": spec.output_classes,
            "description": spec.description
        }
        for spec in MODEL_SPECS.values()
    ]


if __name__ == "__main__":
    print("Testing all 8 specialized models...\n")

    for node_id, ModelClass in MODEL_CLASSES.items():
        model = ModelClass()
        spec = MODEL_SPECS[node_id]

        # Create test input
        if len(spec.input_shape) == 2:
            x = np.random.randn(2, *spec.input_shape).astype(np.float32)
        elif len(spec.input_shape) == 3:
            x = np.random.randn(2, *spec.input_shape).astype(np.float32)
        else:
            x = np.random.randn(2, spec.input_shape[0]).astype(np.float32)

        # Test forward pass
        preds, confs = model.predict(x)
        size = model.get_model_size()

        print(f"{node_id} ({spec.name})")
        print(f"  Input: {spec.input_shape} -> Output: {len(spec.output_classes)} classes")
        print(f"  Size: {size['size_float32_kb']:.1f} KB (float32), {size['size_int8_kb']:.1f} KB (int8)")
        print(f"  Test prediction: {spec.output_classes[preds[0]]} ({confs[0]:.2%})")
        print()
