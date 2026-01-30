#!/usr/bin/env python3
"""
TinyML Spectral CNN - Pure NumPy Implementation
================================================
Lightweight CNN that works without TensorFlow for edge deployment.
Exports to C header for embedded systems.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import json


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    type: str
    name: str
    params: Dict[str, Any]


class NumpyCNN:
    """
    Pure NumPy CNN implementation for spectral analysis.
    Designed for TinyML deployment on microcontrollers.
    """
    
    CLASSIFICATION_LABELS = [
        "noise",
        "stellar_signal", 
        "planetary_transit",
        "eclipsing_binary",
        "instrument_artifact",
        "unknown_anomaly"
    ]
    
    def __init__(self, input_shape: Tuple[int, int] = (1024, 1), num_classes: int = 6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights: Dict[str, np.ndarray] = {}
        self.layer_configs: List[LayerConfig] = []
        self._build_architecture()
        
    def _build_architecture(self):
        """Define the model architecture."""
        self.layer_configs = [
            LayerConfig("conv1d", "conv1", {"filters": 16, "kernel_size": 7, "padding": "same"}),
            LayerConfig("batchnorm", "bn1", {}),
            LayerConfig("relu", "relu1", {}),
            LayerConfig("maxpool1d", "pool1", {"pool_size": 4}),
            
            LayerConfig("conv1d", "conv2", {"filters": 32, "kernel_size": 5, "padding": "same"}),
            LayerConfig("batchnorm", "bn2", {}),
            LayerConfig("relu", "relu2", {}),
            LayerConfig("maxpool1d", "pool2", {"pool_size": 4}),
            
            LayerConfig("conv1d", "conv3", {"filters": 64, "kernel_size": 3, "padding": "same"}),
            LayerConfig("batchnorm", "bn3", {}),
            LayerConfig("relu", "relu3", {}),
            LayerConfig("maxpool1d", "pool3", {"pool_size": 4}),
            
            LayerConfig("global_avgpool", "gap", {}),
            LayerConfig("dense", "fc1", {"units": 32}),
            LayerConfig("relu", "relu_fc", {}),
            LayerConfig("dropout", "dropout", {"rate": 0.3}),
            LayerConfig("dense", "output", {"units": self.num_classes}),
            LayerConfig("softmax", "softmax", {}),
        ]
    
    def initialize_weights(self, seed: int = 42):
        """Initialize weights with He initialization."""
        np.random.seed(seed)
        
        in_channels = self.input_shape[1]
        
        for layer in self.layer_configs:
            if layer.type == "conv1d":
                filters = layer.params["filters"]
                kernel_size = layer.params["kernel_size"]
                # He initialization
                std = np.sqrt(2.0 / (kernel_size * in_channels))
                self.weights[f"{layer.name}_w"] = np.random.randn(
                    kernel_size, in_channels, filters
                ).astype(np.float32) * std
                self.weights[f"{layer.name}_b"] = np.zeros(filters, dtype=np.float32)
                in_channels = filters
                
            elif layer.type == "batchnorm":
                # Batch norm parameters (gamma, beta, running_mean, running_var)
                self.weights[f"{layer.name}_gamma"] = np.ones(in_channels, dtype=np.float32)
                self.weights[f"{layer.name}_beta"] = np.zeros(in_channels, dtype=np.float32)
                self.weights[f"{layer.name}_mean"] = np.zeros(in_channels, dtype=np.float32)
                self.weights[f"{layer.name}_var"] = np.ones(in_channels, dtype=np.float32)
                
            elif layer.type == "maxpool1d":
                in_channels = in_channels  # unchanged
                
            elif layer.type == "dense":
                units = layer.params["units"]
                if layer.name == "fc1":
                    # Calculate input size after pooling
                    spatial_size = self.input_shape[0] // (4 * 4 * 4)  # After 3 pooling layers
                    in_features = spatial_size * in_channels if layer.name == "fc1" else in_channels
                    # For global avg pool, in_features = in_channels
                    in_features = in_channels
                else:
                    in_features = 32  # From fc1
                    
                std = np.sqrt(2.0 / in_features)
                self.weights[f"{layer.name}_w"] = np.random.randn(
                    in_features, units
                ).astype(np.float32) * std
                self.weights[f"{layer.name}_b"] = np.zeros(units, dtype=np.float32)
                in_channels = units
    
    def _conv1d(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray, padding: str = "same") -> np.ndarray:
        """1D convolution with padding."""
        batch_size, seq_len, in_channels = x.shape
        kernel_size, _, out_channels = weights.shape
        
        if padding == "same":
            pad = kernel_size // 2
            x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        else:
            x_padded = x
            
        output = np.zeros((batch_size, seq_len, out_channels), dtype=np.float32)
        
        for i in range(seq_len):
            window = x_padded[:, i:i+kernel_size, :]  # (batch, kernel, in_ch)
            for j in range(out_channels):
                output[:, i, j] = np.sum(window * weights[:, :, j], axis=(1, 2)) + bias[j]
        
        return output
    
    def _batchnorm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Batch normalization (inference mode)."""
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _maxpool1d(self, x: np.ndarray, pool_size: int) -> np.ndarray:
        """1D max pooling."""
        batch_size, seq_len, channels = x.shape
        out_len = seq_len // pool_size
        output = np.zeros((batch_size, out_len, channels), dtype=np.float32)
        
        for i in range(out_len):
            window = x[:, i*pool_size:(i+1)*pool_size, :]
            output[:, i, :] = np.max(window, axis=1)
        
        return output
    
    def _global_avgpool(self, x: np.ndarray) -> np.ndarray:
        """Global average pooling."""
        return np.mean(x, axis=1)
    
    def _dense(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Dense/fully connected layer."""
        return np.dot(x, weights) + bias
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _dropout(self, x: np.ndarray, rate: float, training: bool = False) -> np.ndarray:
        """Dropout (no-op during inference)."""
        return x
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the network."""
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        
        for layer in self.layer_configs:
            if layer.type == "conv1d":
                x = self._conv1d(
                    x, 
                    self.weights[f"{layer.name}_w"],
                    self.weights[f"{layer.name}_b"],
                    layer.params.get("padding", "same")
                )
            elif layer.type == "batchnorm":
                x = self._batchnorm(
                    x,
                    self.weights[f"{layer.name}_gamma"],
                    self.weights[f"{layer.name}_beta"],
                    self.weights[f"{layer.name}_mean"],
                    self.weights[f"{layer.name}_var"]
                )
            elif layer.type == "relu":
                x = self._relu(x)
            elif layer.type == "maxpool1d":
                x = self._maxpool1d(x, layer.params["pool_size"])
            elif layer.type == "global_avgpool":
                x = self._global_avgpool(x)
            elif layer.type == "dense":
                x = self._dense(
                    x,
                    self.weights[f"{layer.name}_w"],
                    self.weights[f"{layer.name}_b"]
                )
            elif layer.type == "softmax":
                x = self._softmax(x)
            elif layer.type == "dropout":
                x = self._dropout(x, layer.params["rate"], training)
        
        return x
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class and confidence for input samples.
        
        Returns:
            predictions: Class indices
            confidences: Confidence scores
        """
        probs = self.forward(x)
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return predictions, confidences
    
    def predict_class_name(self, x: np.ndarray) -> List[Tuple[str, float]]:
        """Predict with class names."""
        preds, confs = self.predict(x)
        return [(self.CLASSIFICATION_LABELS[p], c) for p, c in zip(preds, confs)]
    
    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.001) -> float:
        """
        Simple training step using gradient descent.
        Uses numerical gradients for simplicity.
        """
        # Forward pass
        probs = self.forward(x, training=True)
        
        # Cross-entropy loss
        eps = 1e-7
        y_one_hot = np.eye(self.num_classes)[y]
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=-1))
        
        # Simple gradient descent with numerical gradients
        # (For production, use backpropagation)
        delta = 1e-5
        for name, weight in self.weights.items():
            if 'mean' in name or 'var' in name:
                continue  # Skip running stats
            
            grad = np.zeros_like(weight)
            flat_weight = weight.flatten()
            flat_grad = grad.flatten()
            
            # Subsample for speed (only update ~10% of weights per step)
            indices = np.random.choice(len(flat_weight), min(100, len(flat_weight)), replace=False)
            
            for idx in indices:
                orig = flat_weight[idx]
                
                flat_weight[idx] = orig + delta
                self.weights[name] = flat_weight.reshape(weight.shape)
                loss_plus = -np.mean(np.sum(y_one_hot * np.log(self.forward(x) + eps), axis=-1))
                
                flat_weight[idx] = orig - delta
                self.weights[name] = flat_weight.reshape(weight.shape)
                loss_minus = -np.mean(np.sum(y_one_hot * np.log(self.forward(x) + eps), axis=-1))
                
                flat_weight[idx] = orig
                flat_grad[idx] = (loss_plus - loss_minus) / (2 * delta)
            
            self.weights[name] = flat_weight.reshape(weight.shape)
            
            # Update weights
            self.weights[name] -= learning_rate * flat_grad.reshape(weight.shape)
        
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 10, batch_size: int = 32, 
            learning_rate: float = 0.001,
            validation_split: float = 0.2,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Uses a simplified training approach suitable for embedded systems.
        """
        # Initialize weights if not done
        if not self.weights:
            self.initialize_weights()
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            # Train on batches
            epoch_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                loss = self.train_step(batch_x, batch_y, learning_rate)
                epoch_losses.append(loss)
            
            # Calculate metrics
            train_loss = np.mean(epoch_losses)
            train_preds, _ = self.predict(X_train)
            train_acc = np.mean(train_preds == y_train)
            
            val_probs = self.forward(X_val)
            val_loss = -np.mean(np.sum(np.eye(self.num_classes)[y_val] * np.log(val_probs + 1e-7), axis=-1))
            val_preds, _ = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        return history
    
    def save(self, filepath: str):
        """Save model weights to file."""
        np.savez(filepath, **self.weights)
        
    def load(self, filepath: str):
        """Load model weights from file."""
        data = np.load(filepath)
        self.weights = {key: data[key] for key in data.files}
    
    def export_to_c_header(self, filepath: str):
        """Export model weights to C header file for embedded deployment."""
        with open(filepath, 'w') as f:
            f.write("// AstroTinyML Model Weights\n")
            f.write("// Auto-generated - DO NOT EDIT\n\n")
            f.write("#ifndef ASTRO_TINYML_WEIGHTS_H\n")
            f.write("#define ASTRO_TINYML_WEIGHTS_H\n\n")
            f.write("#include <stdint.h>\n\n")
            
            # Quantize to int8
            for name, weight in self.weights.items():
                if 'mean' in name or 'var' in name:
                    continue
                    
                # Quantize
                scale = np.max(np.abs(weight)) / 127.0
                quantized = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
                
                f.write(f"// {name} - shape: {weight.shape}\n")
                f.write(f"static const float {name}_scale = {scale}f;\n")
                f.write(f"static const int8_t {name}[] = {{\n")
                
                flat = quantized.flatten()
                for i, val in enumerate(flat):
                    if i % 16 == 0:
                        f.write("    ")
                    f.write(f"{val:4d},")
                    if i % 16 == 15 or i == len(flat) - 1:
                        f.write("\n")
                
                f.write("};\n\n")
            
            f.write("#endif // ASTRO_TINYML_WEIGHTS_H\n")
        
        print(f"Exported C header to {filepath}")
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size statistics."""
        total_params = sum(w.size for w in self.weights.values())
        total_bytes = total_params * 4  # float32
        quantized_bytes = total_params  # int8
        
        return {
            "total_parameters": total_params,
            "size_float32_kb": total_bytes / 1024,
            "size_int8_kb": quantized_bytes / 1024,
            "layers": len(self.layer_configs),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes
        }


class SimpleClassifier:
    """
    Simple statistical classifier for quick detection.
    Complements the CNN for edge deployment.
    """
    
    def __init__(self):
        self.class_stats = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Learn class statistics."""
        for class_id in np.unique(y):
            mask = y == class_id
            class_data = X[mask]
            
            self.class_stats[class_id] = {
                "mean": np.mean(class_data, axis=0),
                "std": np.std(class_data, axis=0),
                "var": np.var(class_data, axis=0),
                "min": np.min(class_data, axis=0),
                "max": np.max(class_data, axis=0),
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Mahalanobis-like distance."""
        if not self.class_stats:
            raise ValueError("Model not fitted")
        
        distances = np.zeros((len(X), len(self.class_stats)))
        
        for class_id, stats in self.class_stats.items():
            diff = X - stats["mean"]
            # Use variance-weighted distance
            dist = np.sum(diff**2 / (stats["var"] + 1e-7), axis=-1)
            distances[:, class_id] = dist
        
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    # Quick test
    print("Testing NumpyCNN...")
    
    model = NumpyCNN(input_shape=(1024, 1), num_classes=6)
    model.initialize_weights()
    
    # Test forward pass
    x = np.random.randn(4, 1024, 1).astype(np.float32)
    output = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be 1.0): {output.sum(axis=1)}")
    
    # Model size
    size = model.get_model_size()
    print(f"\nModel size:")
    print(f"  Parameters: {size['total_parameters']:,}")
    print(f"  Float32: {size['size_float32_kb']:.1f} KB")
    print(f"  INT8: {size['size_int8_kb']:.1f} KB")
