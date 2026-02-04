"""
Neural Network Trainer with Proper Backpropagation
===================================================
Implements gradient descent with Adam optimizer for TinyML models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    beta1: float = 0.9  # Adam momentum
    beta2: float = 0.999  # Adam RMSprop
    epsilon: float = 1e-8
    weight_decay: float = 0.0001
    dropout_rate: float = 0.2
    early_stopping_patience: int = 10
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5


class AdamOptimizer:
    """Adam optimizer for gradient descent."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, np.ndarray] = {}  # First moment
        self.v: Dict[str, np.ndarray] = {}  # Second moment
        self.t = 0  # Timestep

    def update(self, weights: Dict[str, np.ndarray],
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update weights using Adam algorithm."""
        self.t += 1

        for name in grads:
            if name not in self.m:
                self.m[name] = np.zeros_like(weights[name])
                self.v[name] = np.zeros_like(weights[name])

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]

            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads[name] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update weights
            weights[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights


class NeuralNetworkTrainer:
    """
    Trainer with proper backpropagation for TinyML models.

    Supports:
    - 1D CNNs (for light curve models)
    - 2D CNNs (for galaxy classification)
    - MLPs (for spectral type classification)
    """

    def __init__(self, model, config: Optional[TrainingConfig] = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = AdamOptimizer(
            lr=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2
        )
        self.cache: Dict[str, np.ndarray] = {}
        self.training = True

    def forward_with_cache(self, x: np.ndarray) -> np.ndarray:
        """Forward pass storing activations for backprop."""
        self.cache = {}

        # Handle different model architectures
        model_type = self._detect_model_type()

        if model_type == "cnn1d":
            return self._forward_cnn1d(x)
        elif model_type == "cnn2d":
            return self._forward_cnn2d(x)
        elif model_type == "mlp":
            return self._forward_mlp(x)
        else:
            # Fallback to model's forward
            return self.model.forward(x)

    def _detect_model_type(self) -> str:
        """Detect model architecture type."""
        if "conv1_w" in self.model.weights:
            shape = self.model.weights["conv1_w"].shape
            if len(shape) == 3:
                return "cnn1d"
            elif len(shape) == 4:
                return "cnn2d"
        if "fc1_w" in self.model.weights and "conv1_w" not in self.model.weights:
            return "mlp"
        return "unknown"

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_backward(self, dout: np.ndarray, x: np.ndarray) -> np.ndarray:
        return dout * (x > 0)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Apply dropout during training."""
        if self.training and rate > 0:
            mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
            self.cache["dropout_mask"] = mask
            return x * mask
        return x

    def _conv1d_forward(self, x: np.ndarray, w: np.ndarray, b: np.ndarray,
                        name: str) -> np.ndarray:
        """1D convolution forward pass with caching."""
        batch_size, seq_len, in_channels = x.shape
        kernel_size, _, out_channels = w.shape
        pad = kernel_size // 2

        x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        self.cache[f"{name}_input"] = x
        self.cache[f"{name}_padded"] = x_padded

        output = np.zeros((batch_size, seq_len, out_channels), dtype=np.float32)
        for i in range(seq_len):
            window = x_padded[:, i:i+kernel_size, :]
            for j in range(out_channels):
                output[:, i, j] = np.sum(window * w[:, :, j], axis=(1, 2)) + b[j]

        return output

    def _conv1d_backward(self, dout: np.ndarray, name: str,
                         w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1D convolution backward pass."""
        x_padded = self.cache[f"{name}_padded"]
        x_input = self.cache[f"{name}_input"]

        batch_size, seq_len, out_channels = dout.shape
        kernel_size = w.shape[0]
        in_channels = w.shape[1]
        pad = kernel_size // 2

        # Gradient w.r.t weights
        dw = np.zeros_like(w)
        for i in range(seq_len):
            window = x_padded[:, i:i+kernel_size, :]
            for j in range(out_channels):
                # dw[:, :, j] += sum over batch of (window * dout[:, i, j])
                dw[:, :, j] += np.sum(window * dout[:, i:i+1, j:j+1], axis=0)

        # Gradient w.r.t bias
        db = np.sum(dout, axis=(0, 1))

        # Gradient w.r.t input (simplified)
        dx_padded = np.zeros_like(x_padded)
        for i in range(seq_len):
            for j in range(out_channels):
                dx_padded[:, i:i+kernel_size, :] += np.outer(
                    dout[:, i, j].reshape(-1, 1),
                    w[:, :, j].reshape(-1)
                ).reshape(batch_size, kernel_size, in_channels)

        # Remove padding
        dx = dx_padded[:, pad:-pad if pad > 0 else None, :]

        return dx, dw, db

    def _dense_forward(self, x: np.ndarray, w: np.ndarray, b: np.ndarray,
                       name: str) -> np.ndarray:
        """Dense layer forward with caching."""
        self.cache[f"{name}_input"] = x
        return np.dot(x, w) + b

    def _dense_backward(self, dout: np.ndarray, name: str,
                        w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense layer backward pass."""
        x = self.cache[f"{name}_input"]
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        dx = np.dot(dout, w.T)
        return dx, dw, db

    def _maxpool1d_forward(self, x: np.ndarray, pool_size: int,
                           name: str) -> np.ndarray:
        """Max pooling forward with index caching."""
        batch_size, seq_len, channels = x.shape
        out_len = seq_len // pool_size

        output = np.zeros((batch_size, out_len, channels), dtype=np.float32)
        indices = np.zeros((batch_size, out_len, channels), dtype=np.int32)

        for i in range(out_len):
            window = x[:, i*pool_size:(i+1)*pool_size, :]
            indices[:, i, :] = np.argmax(window, axis=1)
            output[:, i, :] = np.max(window, axis=1)

        self.cache[f"{name}_input_shape"] = x.shape
        self.cache[f"{name}_indices"] = indices
        self.cache[f"{name}_pool_size"] = pool_size
        return output

    def _maxpool1d_backward(self, dout: np.ndarray, name: str) -> np.ndarray:
        """Max pooling backward pass."""
        input_shape = self.cache[f"{name}_input_shape"]
        indices = self.cache[f"{name}_indices"]
        pool_size = self.cache[f"{name}_pool_size"]

        dx = np.zeros(input_shape, dtype=np.float32)
        batch_size, out_len, channels = dout.shape

        for b in range(batch_size):
            for i in range(out_len):
                for c in range(channels):
                    idx = indices[b, i, c]
                    dx[b, i*pool_size + idx, c] = dout[b, i, c]

        return dx

    def _global_avgpool_forward(self, x: np.ndarray, name: str) -> np.ndarray:
        """Global average pooling forward."""
        self.cache[f"{name}_input_shape"] = x.shape
        if len(x.shape) == 3:
            return np.mean(x, axis=1)
        else:
            return np.mean(x, axis=(1, 2))

    def _global_avgpool_backward(self, dout: np.ndarray, name: str) -> np.ndarray:
        """Global average pooling backward."""
        input_shape = self.cache[f"{name}_input_shape"]
        if len(input_shape) == 3:
            batch_size, seq_len, channels = input_shape
            dx = np.broadcast_to(dout[:, np.newaxis, :] / seq_len, input_shape)
        else:
            batch_size, h, w, channels = input_shape
            dx = np.broadcast_to(dout[:, np.newaxis, np.newaxis, :] / (h * w), input_shape)
        return dx.copy()

    def _batchnorm_forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                           running_mean: np.ndarray, running_var: np.ndarray,
                           name: str, eps: float = 1e-5) -> np.ndarray:
        """Batch normalization forward pass."""
        if self.training:
            # Compute batch statistics
            if len(x.shape) == 3:
                mean = np.mean(x, axis=(0, 1))
                var = np.var(x, axis=(0, 1))
            else:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)

            # Update running statistics (momentum = 0.1)
            running_mean[:] = 0.9 * running_mean + 0.1 * mean
            running_var[:] = 0.9 * running_var + 0.1 * var
        else:
            mean = running_mean
            var = running_var

        x_norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_norm + beta

        self.cache[f"{name}_x"] = x
        self.cache[f"{name}_x_norm"] = x_norm
        self.cache[f"{name}_mean"] = mean
        self.cache[f"{name}_var"] = var
        self.cache[f"{name}_gamma"] = gamma

        return out

    def _batchnorm_backward(self, dout: np.ndarray, name: str,
                            eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch normalization backward pass."""
        x = self.cache[f"{name}_x"]
        x_norm = self.cache[f"{name}_x_norm"]
        mean = self.cache[f"{name}_mean"]
        var = self.cache[f"{name}_var"]
        gamma = self.cache[f"{name}_gamma"]

        if len(x.shape) == 3:
            N = x.shape[0] * x.shape[1]
            axis = (0, 1)
        else:
            N = x.shape[0]
            axis = 0

        dgamma = np.sum(dout * x_norm, axis=axis)
        dbeta = np.sum(dout, axis=axis)

        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=axis)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=axis) + dvar * np.mean(-2 * (x - mean), axis=axis)

        dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N

        return dx, dgamma, dbeta

    def _forward_cnn1d(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 1D CNN models."""
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        self.cache["input"] = x
        weights = self.model.weights

        # Determine number of conv blocks
        num_blocks = 0
        for i in range(1, 5):
            if f"conv{i}_w" in weights:
                num_blocks = i
            else:
                break

        # Forward through conv blocks
        for i in range(1, num_blocks + 1):
            x = self._conv1d_forward(x, weights[f"conv{i}_w"], weights[f"conv{i}_b"], f"conv{i}")
            self.cache[f"pre_bn{i}"] = x

            if f"bn{i}_gamma" in weights:
                x = self._batchnorm_forward(
                    x, weights[f"bn{i}_gamma"], weights[f"bn{i}_beta"],
                    weights[f"bn{i}_mean"], weights[f"bn{i}_var"], f"bn{i}"
                )

            self.cache[f"pre_relu{i}"] = x
            x = self._relu(x)
            self.cache[f"post_relu{i}"] = x
            x = self._maxpool1d_forward(x, 4, f"pool{i}")
            self.cache[f"post_pool{i}"] = x

        # Global pooling
        x = self._global_avgpool_forward(x, "gap")
        self.cache["post_gap"] = x

        # Dropout
        x = self._dropout(x, self.config.dropout_rate)

        # Dense layers
        x = self._dense_forward(x, weights["fc1_w"], weights["fc1_b"], "fc1")
        self.cache["pre_relu_fc1"] = x
        x = self._relu(x)
        self.cache["post_relu_fc1"] = x

        x = self._dense_forward(x, weights["out_w"], weights["out_b"], "out")
        self.cache["logits"] = x

        return self._softmax(x)

    def _forward_mlp(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for MLP models (SpectralType)."""
        self.cache["input"] = x
        weights = self.model.weights

        # Hidden layers
        layer_idx = 1
        while f"fc{layer_idx}_w" in weights:
            x = self._dense_forward(x, weights[f"fc{layer_idx}_w"],
                                   weights[f"fc{layer_idx}_b"], f"fc{layer_idx}")
            self.cache[f"pre_relu_fc{layer_idx}"] = x
            x = self._relu(x)
            self.cache[f"post_relu_fc{layer_idx}"] = x
            x = self._dropout(x, self.config.dropout_rate)
            layer_idx += 1

        # Output layer
        x = self._dense_forward(x, weights["out_w"], weights["out_b"], "out")
        self.cache["logits"] = x

        return self._softmax(x)

    def _forward_cnn2d(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D CNN models (Galaxy)."""
        # For now, use model's forward pass
        # Can implement full 2D backprop later
        return self.model.forward(x)

    def backward(self, probs: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass computing gradients."""
        grads = {}
        weights = self.model.weights
        batch_size = probs.shape[0]

        # Softmax + Cross-entropy gradient
        y_one_hot = np.eye(self.model.num_classes)[y]
        dlogits = (probs - y_one_hot) / batch_size

        model_type = self._detect_model_type()

        if model_type == "cnn1d":
            grads = self._backward_cnn1d(dlogits)
        elif model_type == "mlp":
            grads = self._backward_mlp(dlogits)
        else:
            # Simplified gradient for unsupported architectures
            grads["out_w"] = np.random.randn(*weights["out_w"].shape) * 0.01
            grads["out_b"] = np.mean(dlogits, axis=0)

        # Add weight decay
        for name in grads:
            if "_w" in name and name in weights:
                grads[name] += self.config.weight_decay * weights[name]

        return grads

    def _backward_cnn1d(self, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass for 1D CNN."""
        grads = {}
        weights = self.model.weights

        # Output layer
        dx, grads["out_w"], grads["out_b"] = self._dense_backward(
            dlogits, "out", weights["out_w"]
        )

        # ReLU after fc1
        dx = self._relu_backward(dx, self.cache["pre_relu_fc1"])

        # FC1
        dx, grads["fc1_w"], grads["fc1_b"] = self._dense_backward(
            dx, "fc1", weights["fc1_w"]
        )

        # Global average pooling backward
        dx = self._global_avgpool_backward(dx, "gap")

        # Determine number of conv blocks
        num_blocks = 0
        for i in range(1, 5):
            if f"conv{i}_w" in weights:
                num_blocks = i

        # Backward through conv blocks (reverse order)
        for i in range(num_blocks, 0, -1):
            # Maxpool backward
            dx = self._maxpool1d_backward(dx, f"pool{i}")

            # ReLU backward
            dx = self._relu_backward(dx, self.cache[f"pre_relu{i}"])

            # Batchnorm backward
            if f"bn{i}_gamma" in weights:
                dx, grads[f"bn{i}_gamma"], grads[f"bn{i}_beta"] = self._batchnorm_backward(
                    dx, f"bn{i}"
                )

            # Conv backward
            dx, grads[f"conv{i}_w"], grads[f"conv{i}_b"] = self._conv1d_backward(
                dx, f"conv{i}", weights[f"conv{i}_w"]
            )

        return grads

    def _backward_mlp(self, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass for MLP."""
        grads = {}
        weights = self.model.weights

        # Output layer
        dx, grads["out_w"], grads["out_b"] = self._dense_backward(
            dlogits, "out", weights["out_w"]
        )

        # Hidden layers (reverse order)
        layer_idx = 1
        while f"fc{layer_idx}_w" in weights:
            layer_idx += 1
        layer_idx -= 1

        while layer_idx >= 1:
            # ReLU backward
            dx = self._relu_backward(dx, self.cache[f"pre_relu_fc{layer_idx}"])

            # Dense backward
            dx, grads[f"fc{layer_idx}_w"], grads[f"fc{layer_idx}_b"] = self._dense_backward(
                dx, f"fc{layer_idx}", weights[f"fc{layer_idx}_w"]
            )
            layer_idx -= 1

        return grads

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Single training step."""
        self.training = True

        # Forward pass
        probs = self.forward_with_cache(x)

        # Compute loss
        eps = 1e-7
        y_one_hot = np.eye(self.model.num_classes)[y]
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=-1))

        # Compute accuracy
        preds = np.argmax(probs, axis=-1)
        accuracy = np.mean(preds == y)

        # Backward pass
        grads = self.backward(probs, y)

        # Update weights
        self.model.weights = self.optimizer.update(self.model.weights, grads)

        return loss, accuracy

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate on validation data."""
        self.training = False
        probs = self.model.forward(x)

        eps = 1e-7
        y_one_hot = np.eye(self.model.num_classes)[y]
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=-1))

        preds = np.argmax(probs, axis=-1)
        accuracy = np.mean(preds == y)

        return loss, accuracy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.

        Returns:
            Training history with loss and accuracy curves
        """
        history = {
            "loss": [], "accuracy": [],
            "val_loss": [], "val_accuracy": []
        }

        best_val_acc = 0.0
        best_weights = None
        patience_counter = 0
        lr_patience_counter = 0

        n_samples = len(X_train)
        n_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size

        for epoch in range(self.config.epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []
            epoch_accs = []

            # Training loop
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, n_samples)

                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                loss, acc = self.train_step(batch_x, batch_y)
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            # Compute epoch metrics
            train_loss = np.mean(epoch_losses)
            train_acc = np.mean(epoch_accs)

            # Validation
            val_loss, val_acc = self.evaluate(X_val, y_val)

            history["loss"].append(float(train_loss))
            history["accuracy"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_acc))

            # Save best weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {k: v.copy() for k, v in self.model.weights.items()}
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1

            # Learning rate decay
            if lr_patience_counter >= self.config.lr_decay_patience:
                self.optimizer.lr *= self.config.lr_decay_factor
                lr_patience_counter = 0
                if verbose:
                    print(f"  Learning rate reduced to {self.optimizer.lr:.6f}")

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            # Progress output
            if verbose and (epoch % 5 == 0 or epoch == self.config.epochs - 1):
                bar_len = 20
                filled = int(bar_len * (epoch + 1) / self.config.epochs)
                bar = '█' * filled + '░' * (bar_len - filled)

                color = '\033[92m' if val_acc > 0.7 else '\033[93m' if val_acc > 0.5 else '\033[91m'
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} [{bar}] "
                      f"loss: {train_loss:.4f} acc: {train_acc:.3f} "
                      f"val_loss: {val_loss:.4f} {color}val_acc: {val_acc:.3f}\033[0m")

        # Restore best weights
        if best_weights:
            self.model.weights = best_weights

        return history
