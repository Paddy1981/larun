"""
TinyML Spectral Analysis Model
==============================
Lightweight neural network optimized for edge deployment and spectral data analysis.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

logger = logging.getLogger(__name__)


class SpectralCNN:
    """
    Convolutional Neural Network for spectral data analysis.
    Optimized for TinyML deployment with INT8 quantization.
    """
    
    CLASSIFICATION_LABELS = [
        "noise",
        "stellar_signal", 
        "planetary_transit",
        "eclipsing_binary",
        "instrument_artifact",
        "unknown_anomaly"
    ]
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1024, 1),
        num_classes: int = 6,
        use_lstm: bool = False,
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_lstm = use_lstm
        self.model_config = model_config or {}
        self.model: Optional[keras.Model] = None
        self.tflite_model: Optional[bytes] = None
        
    def build_model(self) -> keras.Model:
        """
        Build the CNN model architecture.
        
        Architecture designed for:
        - Small model size (< 100KB quantized)
        - Fast inference (< 10ms on Cortex-M4)
        - Good accuracy on spectral classification
        """
        inputs = layers.Input(shape=self.input_shape, name="spectral_input")
        
        if self.use_lstm:
            # Hybrid CNN-LSTM Architecture (from train_real_data.py)
            x = layers.Conv1D(filters=32, kernel_size=16, activation='relu')(inputs)
            x = layers.MaxPooling1D(pool_size=4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv1D(filters=64, kernel_size=8, activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.LSTM(64, return_sequences=False)(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            
            outputs = layers.Dense(self.num_classes, activation='softmax', name="classification")(x)
            
            self.model = keras.Model(inputs=inputs, outputs=outputs, name="SpectralHybrid")
            
            logger.info(f"Built hybrid model with {self.model.count_params():,} parameters")
            return self.model
        
        # Initial convolution block
        x = layers.Conv1D(
            filters=16, 
            kernel_size=7, 
            padding="same",
            activation=None,
            name="conv1"
        )(inputs)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.ReLU(name="relu1")(x)
        x = layers.MaxPooling1D(pool_size=4, name="pool1")(x)
        
        # Second convolution block
        x = layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            activation=None,
            name="conv2"
        )(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.ReLU(name="relu2")(x)
        x = layers.MaxPooling1D(pool_size=4, name="pool2")(x)
        
        # Third convolution block - depthwise separable for efficiency
        x = layers.SeparableConv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation=None,
            name="conv3"
        )(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.ReLU(name="relu3")(x)
        x = layers.MaxPooling1D(pool_size=4, name="pool3")(x)
        
        # Fourth convolution block
        x = layers.SeparableConv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation=None,
            name="conv4"
        )(x)
        x = layers.BatchNormalization(name="bn4")(x)
        x = layers.ReLU(name="relu4")(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name="global_pool")(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.3, name="dropout")(x)
        
        # Dense classification head
        x = layers.Dense(32, activation="relu", name="dense1")(x)
        outputs = layers.Dense(
            self.num_classes, 
            activation="softmax", 
            name="classification"
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="SpectralCNN")
        
        logger.info(f"Built model with {self.model.count_params():,} parameters")
        
        return self.model
    
    def build_anomaly_detector(self) -> keras.Model:
        """
        Build an autoencoder-based anomaly detector.
        Detects unusual spectral patterns that don't match known categories.
        """
        inputs = layers.Input(shape=self.input_shape, name="spectral_input")
        
        # Encoder
        x = layers.Conv1D(16, 7, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling1D(4)(x)
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(4)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        encoded = layers.MaxPooling1D(4)(x)
        
        # Bottleneck
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(encoded)
        
        # Decoder
        x = layers.UpSampling1D(4)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(4)(x)
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(4)(x)
        x = layers.Conv1D(16, 7, padding="same", activation="relu")(x)
        
        # Output
        decoded = layers.Conv1D(1, 3, padding="same", activation="linear")(x)
        
        self.anomaly_model = keras.Model(inputs=inputs, outputs=decoded, name="AnomalyDetector")
        
        return self.anomaly_model
    
    def compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: str = "sparse_categorical_crossentropy"
    ):
        """Compile the model with specified optimizer and loss."""
        if self.model is None:
            self.build_model()
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy", keras.metrics.SparseCategoricalAccuracy(name="sparse_acc")]
        )
        
        logger.info(f"Compiled model with {optimizer}, lr={learning_rate}")
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        checkpoint_dir: str = "models/checkpoints"
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train: Training data of shape (n_samples, input_shape)
            y_train: Training labels of shape (n_samples,)
            x_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile()
        
        # Ensure correct shape
        if len(x_train.shape) == 2:
            x_train = x_train[..., np.newaxis]
        if x_val is not None and len(x_val.shape) == 2:
            x_val = x_val[..., np.newaxis]
        
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),
                save_best_only=True,
                monitor="val_loss"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(checkpoint_dir, "logs"),
                histogram_freq=1
            )
        ]
        
        validation_data = (x_val, y_val) if x_val is not None else None
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info(f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return history
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            x: Input data of shape (n_samples, input_shape) or (input_shape,)
            
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Handle single sample
        if len(x.shape) == 1:
            x = x[np.newaxis, ..., np.newaxis]
        elif len(x.shape) == 2:
            x = x[..., np.newaxis]
        
        probabilities = self.model.predict(x, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence
    
    def predict_with_labels(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """
        Make predictions with human-readable labels.
        
        Returns:
            List of dictionaries with prediction details
        """
        predictions, confidence = self.predict(x)
        
        results = []
        for pred, conf in zip(predictions, confidence):
            results.append({
                "class_id": int(pred),
                "class_label": self.CLASSIFICATION_LABELS[pred],
                "confidence": float(conf),
                "is_high_confidence": conf >= 0.8
            })
        
        return results
    
    def convert_to_tflite(
        self,
        quantize: bool = True,
        representative_data: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Convert model to TensorFlow Lite format for edge deployment.
        
        Args:
            quantize: Whether to apply INT8 quantization
            representative_data: Data for quantization calibration
            output_path: Path to save the .tflite file
            
        Returns:
            TFLite model as bytes
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable Select TF Ops (needed for some LSTM operations)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_data is not None:
                # Ensure correct shape
                if len(representative_data.shape) == 2:
                    representative_data = representative_data[..., np.newaxis]
                
                def representative_dataset():
                    for i in range(min(100, len(representative_data))):
                        yield [representative_data[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        self.tflite_model = converter.convert()
        
        # Log model size
        size_kb = len(self.tflite_model) / 1024
        logger.info(f"TFLite model size: {size_kb:.2f} KB")
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(self.tflite_model)
            logger.info(f"Saved TFLite model to {output_path}")
        
        return self.tflite_model
    
    def export_for_edge(
        self,
        output_dir: str = "models/tflite",
        include_c_header: bool = True
    ):
        """
        Export model for edge deployment.
        
        Creates:
        - .tflite file
        - C header file (optional, for embedded deployment)
        - Model metadata JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to TFLite
        tflite_path = output_dir / "spectral_model.tflite"
        self.convert_to_tflite(quantize=True, output_path=str(tflite_path))
        
        # Create C header for embedded systems
        if include_c_header:
            self._create_c_header(output_dir / "spectral_model.h")
        
        # Save metadata
        metadata = {
            "model_name": "SpectralCNN",
            "version": "1.0.0",
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "class_labels": self.CLASSIFICATION_LABELS,
            "quantized": True,
            "model_size_kb": len(self.tflite_model) / 1024
        }
        
        import json
        with open(output_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported model to {output_dir}")
    
    def _create_c_header(self, output_path: Path):
        """Create C header file with model data for embedded deployment."""
        if self.tflite_model is None:
            return
        
        # Convert to C array
        hex_array = ", ".join(f"0x{b:02x}" for b in self.tflite_model)
        
        header_content = f'''/*
 * Auto-generated TFLite model for AstroTinyML
 * Model: SpectralCNN v1.0
 * Size: {len(self.tflite_model)} bytes
 */

#ifndef SPECTRAL_MODEL_H
#define SPECTRAL_MODEL_H

#include <stdint.h>

const uint8_t spectral_model_tflite[] = {{
    {hex_array}
}};

const unsigned int spectral_model_tflite_len = {len(self.tflite_model)};

#endif // SPECTRAL_MODEL_H
'''
        
        with open(output_path, "w") as f:
            f.write(header_content)
    
    def save(self, path: str):
        """Save the Keras model."""
        if self.model:
            self.model.save(path)
            logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load a saved Keras model."""
        self.model = keras.models.load_model(path)
        logger.info(f"Loaded model from {path}")
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


class TFLiteInference:
    """
    TensorFlow Lite inference engine for edge deployment.
    """
    
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]
        
        logger.info(f"Loaded TFLite model. Input shape: {self.input_shape}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            x: Input data matching model's expected shape
            
        Returns:
            Model output (class probabilities)
        """
        # Prepare input
        if len(x.shape) == 1:
            x = x[np.newaxis, ..., np.newaxis]
        elif len(x.shape) == 2:
            x = x[..., np.newaxis]
        
        # Quantize input if needed
        if self.input_dtype == np.int8:
            input_scale, input_zero = self.input_details[0]["quantization"]
            x = (x / input_scale + input_zero).astype(np.int8)
        else:
            x = x.astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        # Dequantize output if needed
        if self.output_details[0]["dtype"] == np.int8:
            output_scale, output_zero = self.output_details[0]["quantization"]
            output = (output.astype(np.float32) - output_zero) * output_scale
        
        return output
    
    def benchmark(self, n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        # Generate random input
        input_shape = self.input_shape.copy()
        input_shape[0] = 1  # Single sample
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.predict(test_input)
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.predict(test_input)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "iterations": n_iterations
        }


# Factory function
def create_model(config: Dict[str, Any]) -> SpectralCNN:
    """Create a SpectralCNN model from configuration."""
    model = SpectralCNN(
        input_shape=tuple(config.get("input_shape", [1024, 1])),
        num_classes=len(config.get("output_classes", SpectralCNN.CLASSIFICATION_LABELS)),
        use_lstm=config.get("use_lstm", False),
        model_config=config
    )
    model.build_model()
    model.compile(
        learning_rate=config.get("training", {}).get("learning_rate", 0.001)
    )
    return model


if __name__ == "__main__":
    # Demo: build and export model
    model = SpectralCNN()
    model.build_model()
    model.compile()
    model.summary()
    
    # Generate synthetic data for testing
    x_test = np.random.randn(10, 1024, 1).astype(np.float32)
    
    # Test prediction
    preds, conf = model.predict(x_test)
    print(f"Predictions: {preds}")
    print(f"Confidence: {conf}")
    
    # Export for edge
    model.export_for_edge("models/tflite", include_c_header=True)
