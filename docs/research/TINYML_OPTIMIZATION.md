# TinyML Optimization - Research Documentation

## Overview

This document covers techniques for optimizing machine learning models for edge deployment on resource-constrained devices, essential for LARUN's embedded astronomy applications.

---

## 1. TinyML Constraints

### Target Hardware Specifications

| Device | RAM | Flash | Clock | Power |
|--------|-----|-------|-------|-------|
| Arduino Nano 33 BLE | 256 KB | 1 MB | 64 MHz | ~20 mW |
| ESP32 | 520 KB | 4 MB | 240 MHz | ~100 mW |
| Raspberry Pi Pico | 264 KB | 2 MB | 133 MHz | ~50 mW |
| STM32F4 | 192 KB | 1 MB | 168 MHz | ~100 mW |
| Raspberry Pi Zero | 512 MB | SD | 1 GHz | ~500 mW |

### LARUN Model Requirements

| Constraint | Target | Reason |
|------------|--------|--------|
| Model Size | <100 KB | Flash memory limits |
| RAM Usage | <50 KB | Working memory |
| Inference Time | <10 ms | Real-time analysis |
| Parameters | <100,000 | Size constraint |
| Operations | <10M | Speed constraint |

---

## 2. Model Quantization

### Quantization Types

```
Full Precision (FP32): 32 bits per parameter
├── Half Precision (FP16): 16 bits → 2× smaller
├── INT8: 8 bits → 4× smaller
├── INT4: 4 bits → 8× smaller (experimental)
└── Binary: 1 bit → 32× smaller (limited accuracy)
```

### Post-Training Quantization (PTQ)

```python
import tensorflow as tf

def quantize_model_int8(model, representative_data):
    """
    Convert Keras model to INT8 TFLite.
    
    Args:
        model: Trained Keras model
        representative_data: Sample data for calibration
    
    Returns:
        quantized_tflite: INT8 quantized model bytes
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for calibration
    def representative_dataset():
        for sample in representative_data[:100]:
            yield [sample[np.newaxis, ...].astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Full INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    quantized_tflite = converter.convert()
    
    # Report size
    original_size = model.count_params() * 4  # FP32 bytes
    quantized_size = len(quantized_tflite)
    print(f"Original: {original_size/1024:.1f} KB")
    print(f"Quantized: {quantized_size/1024:.1f} KB")
    print(f"Compression: {original_size/quantized_size:.1f}×")
    
    return quantized_tflite


def quantize_model_dynamic(model):
    """
    Dynamic range quantization (simpler, no calibration data needed).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    return converter.convert()
```

### Quantization-Aware Training (QAT)

```python
import tensorflow_model_optimization as tfmot

def create_qat_model(model):
    """
    Apply quantization-aware training for better INT8 accuracy.
    """
    # Apply quantization to all layers
    quantize_model = tfmot.quantization.keras.quantize_model(model)
    
    # Compile with same optimizer
    quantize_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return quantize_model


def train_with_qat(model, train_data, val_data, epochs=10):
    """
    Fine-tune with quantization awareness.
    """
    qat_model = create_qat_model(model)
    
    qat_model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    return converter.convert()
```

### Accuracy vs Size Tradeoff

| Method | Size Reduction | Accuracy Loss | Use Case |
|--------|---------------|---------------|----------|
| FP16 | 2× | ~0% | GPU inference |
| Dynamic INT8 | 4× | 1-3% | Quick deployment |
| Full INT8 (PTQ) | 4× | 1-5% | Standard edge |
| INT8 (QAT) | 4× | <1% | Accuracy-critical |
| INT4 | 8× | 5-10% | Extreme constraint |

---

## 3. Model Pruning

### Weight Pruning

```python
import tensorflow_model_optimization as tfmot

def apply_pruning(model, target_sparsity=0.5):
    """
    Prune model weights to achieve target sparsity.
    
    Args:
        model: Keras model
        target_sparsity: Fraction of weights to zero (0.5 = 50% sparse)
    
    Returns:
        pruned_model: Model with pruning applied
    """
    # Pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=1000
        )
    }
    
    # Apply to model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        **pruning_params
    )
    
    return pruned_model


def fine_tune_pruned_model(pruned_model, train_data, epochs=10):
    """
    Fine-tune pruned model to recover accuracy.
    """
    # Add pruning callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
    ]
    
    pruned_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    pruned_model.fit(
        train_data,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Strip pruning wrappers for deployment
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    return final_model
```

### Structured Pruning (Filter Pruning)

```python
def prune_filters(model, layer_name, num_filters_to_prune):
    """
    Remove entire filters from convolutional layer.
    More efficient than weight pruning for inference.
    """
    # Get layer weights
    layer = model.get_layer(layer_name)
    weights, biases = layer.get_weights()
    
    # Calculate filter importance (L1 norm)
    filter_importance = np.sum(np.abs(weights), axis=(0, 1, 2))
    
    # Find least important filters
    prune_indices = np.argsort(filter_importance)[:num_filters_to_prune]
    keep_indices = np.argsort(filter_importance)[num_filters_to_prune:]
    
    # Create new smaller weights
    new_weights = weights[:, :, :, keep_indices]
    new_biases = biases[keep_indices]
    
    return new_weights, new_biases, keep_indices
```

---

## 4. Knowledge Distillation

### Teacher-Student Training

```python
def create_student_model(input_shape, num_classes):
    """
    Create small student model for distillation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(8, 7, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(16, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


class DistillationTrainer:
    """
    Train student model to mimic teacher model.
    """
    
    def __init__(self, teacher, student, temperature=3.0, alpha=0.5):
        """
        Args:
            teacher: Large, accurate model
            student: Small model to train
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for distillation loss vs hard labels
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, y_true, y_pred, teacher_logits):
        """
        Combined loss: hard labels + soft teacher predictions.
        """
        # Hard label loss
        hard_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Soft label loss (KL divergence with temperature)
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature)
        soft_student = tf.nn.softmax(y_pred / self.temperature)
        soft_loss = tf.keras.losses.KLDivergence()(soft_teacher, soft_student)
        soft_loss *= self.temperature**2  # Scale by T²
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
    
    def train_step(self, x, y_true):
        """
        Single training step with distillation.
        """
        # Get teacher predictions (no gradient)
        teacher_logits = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Student predictions
            student_pred = self.student(x, training=True)
            
            # Distillation loss
            loss = self.distillation_loss(y_true, student_pred, teacher_logits)
        
        # Update student
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        return loss
```

---

## 5. Architecture Optimization

### Efficient Convolution Alternatives

```python
def depthwise_separable_conv1d(x, filters, kernel_size):
    """
    Depthwise separable convolution: more efficient than standard conv.
    
    Standard Conv: O(K × C_in × C_out)
    Depthwise Sep: O(K × C_in + C_in × C_out)
    
    Typically 8-9× fewer operations.
    """
    # Depthwise: apply filter to each channel separately
    x = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Pointwise: 1×1 conv to mix channels
    x = tf.keras.layers.Conv1D(filters, 1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x


def create_efficient_spectral_cnn(input_shape=(1024, 1), num_classes=6):
    """
    Efficient CNN for spectral classification.
    Uses depthwise separable convolutions.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial conv
    x = tf.keras.layers.Conv1D(8, 7, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    
    # Depthwise separable blocks
    x = depthwise_separable_conv1d(x, 16, 5)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    
    x = depthwise_separable_conv1d(x, 32, 3)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

### Neural Architecture Search (NAS)

```python
def nas_search_space():
    """
    Define search space for LARUN spectral models.
    """
    return {
        'num_layers': [2, 3, 4],
        'filters': [8, 16, 32],
        'kernel_sizes': [3, 5, 7, 11],
        'pooling': [2, 4],
        'dense_units': [8, 16, 32],
        'use_separable': [True, False],
    }


def evaluate_architecture(config, train_data, val_data, max_params=100000):
    """
    Evaluate a single architecture configuration.
    """
    # Build model from config
    model = build_model_from_config(config)
    
    # Check parameter count
    if model.count_params() > max_params:
        return {'accuracy': 0, 'params': model.count_params(), 'valid': False}
    
    # Quick train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=5, verbose=0)
    
    # Evaluate
    _, accuracy = model.evaluate(val_data, verbose=0)
    
    return {
        'accuracy': accuracy,
        'params': model.count_params(),
        'valid': True
    }
```

---

## 6. Memory Optimization

### Tensor Arena Sizing

```python
def estimate_tensor_arena_size(model, input_shape):
    """
    Estimate TFLite Micro tensor arena size.
    
    The tensor arena holds intermediate activations during inference.
    """
    # Get model info
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Calculate peak memory
    max_memory = 0
    current_memory = 0
    
    for tensor in tensor_details:
        size = np.prod(tensor['shape']) * np.dtype(tensor['dtype']).itemsize
        current_memory += size
        max_memory = max(max_memory, current_memory)
    
    # Add 20% buffer
    recommended_arena = int(max_memory * 1.2)
    
    print(f"Recommended tensor arena: {recommended_arena} bytes ({recommended_arena/1024:.1f} KB)")
    
    return recommended_arena
```

### Memory-Efficient Inference

```python
def tflite_inference(model_content, input_data):
    """
    Run inference with TFLite interpreter.
    """
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output


def benchmark_inference(model_content, input_data, num_runs=100):
    """
    Benchmark inference speed.
    """
    import time
    
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }
```

---

## 7. TFLite Micro for Microcontrollers

### C++ Integration

```cpp
// Example TFLite Micro inference code for Arduino/ESP32

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include model (converted to C array)
#include "spectral_model.h"

// Tensor arena (adjust size based on model)
constexpr int kTensorArenaSize = 50 * 1024;  // 50 KB
uint8_t tensor_arena[kTensorArenaSize];

// Setup
tflite::MicroMutableOpResolver<6> resolver;
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddRelu();
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddReshape();

const tflite::Model* model = tflite::GetModel(spectral_model_data);

tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
interpreter.AllocateTensors();

// Inference
TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// Copy input data
memcpy(input->data.int8, input_data, input_size);

// Run inference
interpreter.Invoke();

// Get output
int8_t* output_data = output->data.int8;
```

### Arduino Deployment

```python
def export_for_arduino(model_content, model_name="spectral_model"):
    """
    Export TFLite model as C array for Arduino.
    """
    # Convert to C array format
    c_array = ", ".join([f"0x{b:02x}" for b in model_content])
    
    header = f"""
// Auto-generated by LARUN TinyML
// Model: {model_name}
// Size: {len(model_content)} bytes

#ifndef {model_name.upper()}_H
#define {model_name.upper()}_H

const unsigned char {model_name}_data[] = {{
    {c_array}
}};
const int {model_name}_data_len = {len(model_content)};

#endif // {model_name.upper()}_H
"""
    
    return header
```

---

## 8. Optimization Workflow

### Complete Pipeline

```python
def optimize_model_for_edge(keras_model, train_data, target_size_kb=100):
    """
    Complete optimization pipeline for edge deployment.
    
    Steps:
    1. Pruning (if needed)
    2. Quantization-aware training
    3. INT8 quantization
    4. Validation
    """
    print("=" * 50)
    print("LARUN TinyML Optimization Pipeline")
    print("=" * 50)
    
    original_size = keras_model.count_params() * 4 / 1024
    print(f"Original model size: {original_size:.1f} KB")
    
    # Step 1: Check if pruning needed
    if original_size > target_size_kb * 4:
        print("\n1. Applying pruning...")
        sparsity = 1 - (target_size_kb * 4 / original_size)
        keras_model = apply_pruning(keras_model, min(sparsity, 0.7))
        keras_model = fine_tune_pruned_model(keras_model, train_data)
    else:
        print("\n1. Pruning not needed")
    
    # Step 2: Quantization-aware training
    print("\n2. Quantization-aware training...")
    qat_model = create_qat_model(keras_model)
    qat_model.fit(train_data, epochs=5)
    
    # Step 3: INT8 quantization
    print("\n3. Converting to INT8 TFLite...")
    
    def rep_dataset():
        for x, y in train_data.take(100):
            yield [x]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Step 4: Validation
    final_size = len(tflite_model) / 1024
    print(f"\nFinal model size: {final_size:.1f} KB")
    print(f"Compression ratio: {original_size/final_size:.1f}×")
    
    if final_size <= target_size_kb:
        print(f"✓ Target met! ({final_size:.1f} KB <= {target_size_kb} KB)")
    else:
        print(f"✗ Target not met. Consider more pruning or smaller architecture.")
    
    return tflite_model
```

---

## 9. LARUN Model Specifications

### Recommended Architectures

| Model | Purpose | Size | Accuracy | Inference |
|-------|---------|------|----------|-----------|
| LarunNet-Spectral | Light curve classification | 48 KB | 92% | 3 ms |
| LarunNet-Galaxy | Galaxy morphology | 95 KB | 88% | 8 ms |
| LarunNet-Transit | Transit detection | 32 KB | 95% | 2 ms |
| LarunNet-Star | Spectral type | 24 KB | 90% | 1 ms |

### Model Card Template

```yaml
# LARUN Model Card

model_name: LarunNet-Spectral
version: 1.0
task: Light curve classification
classes: [noise, stellar_signal, planetary_transit, eclipsing_binary, instrument_artifact, unknown_anomaly]

architecture:
  type: 1D CNN
  layers: 4
  parameters: 12,000
  
performance:
  accuracy: 0.92
  f1_macro: 0.89
  inference_time_ms: 3
  
size:
  keras_h5: 192 KB
  tflite_fp32: 48 KB
  tflite_int8: 12 KB
  
training:
  dataset: NASA Exoplanet Archive + MAST
  samples: 10,000
  epochs: 100
  optimizer: Adam
  
hardware:
  tested_on: [Arduino Nano 33 BLE, ESP32, Raspberry Pi Pico]
  tensor_arena: 20 KB
  
license: MIT
```

---

## References

1. Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Workshop.
3. Han, S., et al. (2015). "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." ICLR.
4. David, R., et al. (2021). "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems." MLSys.
5. Banbury, C., et al. (2021). "MLPerf Tiny Benchmark." NeurIPS Datasets and Benchmarks.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*
