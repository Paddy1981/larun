#!/usr/bin/env python3
"""
Training Script for LARUN Federated Multi-Model TinyML Nodes

Creates specialized TinyML models (<100KB each) for each analysis node.
Uses TensorFlow/Keras with INT8 quantization for edge deployment.

Usage:
    python train_nodes.py --node VSTAR-001      # Train single node
    python train_nodes.py --all                 # Train all nodes
    python train_nodes.py --node FLARE-001 --epochs 50

Each node has specific:
- Input shape
- Output classes
- Training data source
- Model architecture optimized for size
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ============================================================================
# Model Architectures (Optimized for <100KB)
# ============================================================================

def create_1d_cnn_model(input_length: int, num_classes: int,
                        filters: list = [16, 32, 32],
                        kernel_size: int = 5,
                        dense_units: int = 32) -> Model:
    """
    Create a 1D CNN for time-series classification.
    Optimized for small model size.

    Target size: ~30-70KB with INT8 quantization
    """
    inputs = layers.Input(shape=(input_length, 1))

    x = inputs
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, padding='same', activation='relu')(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def create_2d_cnn_model(input_shape: Tuple[int, int, int],
                        num_classes: int) -> Model:
    """
    Create a 2D CNN for image classification (Galaxy morphology).
    Optimized for 64x64x3 images, <100KB model.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def create_mlp_model(input_dim: int, num_classes: int) -> Model:
    """
    Create a simple MLP for tabular data (Spectral type from photometry).
    Very small model for 8-dimensional input.
    """
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# ============================================================================
# Synthetic Data Generators (For demonstration - replace with real data)
# ============================================================================

def generate_variable_star_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic phase-folded light curves for variable star types.

    Classes: cepheid, rr_lyrae, delta_scuti, eclipsing_binary, rotational, irregular, constant
    """
    n_points = 512
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 7)
        phase = np.linspace(0, 1, n_points)

        if label == 0:  # Cepheid - asymmetric periodic
            amplitude = np.random.uniform(0.3, 0.8)
            rise_time = np.random.uniform(0.2, 0.4)
            lc = amplitude * (1 - np.abs(phase - rise_time) / max(rise_time, 1 - rise_time))

        elif label == 1:  # RR Lyrae - sawtooth
            amplitude = np.random.uniform(0.4, 1.0)
            rise = np.random.uniform(0.1, 0.3)
            lc = np.where(phase < rise,
                         amplitude * phase / rise,
                         amplitude * (1 - (phase - rise) / (1 - rise)))

        elif label == 2:  # Delta Scuti - low amplitude, high freq
            amplitude = np.random.uniform(0.01, 0.05)
            freq = np.random.uniform(2, 5)
            lc = amplitude * np.sin(2 * np.pi * freq * phase)

        elif label == 3:  # Eclipsing binary - two dips
            depth1 = np.random.uniform(0.1, 0.4)
            depth2 = np.random.uniform(0.05, depth1)
            width = np.random.uniform(0.05, 0.1)
            lc = 1.0 - depth1 * np.exp(-(phase - 0.0)**2 / width**2)
            lc -= depth2 * np.exp(-(phase - 0.5)**2 / width**2)

        elif label == 4:  # Rotational - sinusoidal
            amplitude = np.random.uniform(0.02, 0.1)
            lc = amplitude * np.sin(2 * np.pi * phase)

        elif label == 5:  # Irregular - random walk
            lc = np.cumsum(np.random.randn(n_points) * 0.01)
            lc = lc - np.mean(lc)

        else:  # Constant
            lc = np.zeros(n_points)

        # Add noise
        lc += np.random.randn(n_points) * 0.01
        lc = (lc - np.mean(lc)) / (np.std(lc) + 1e-6)

        X.append(lc)
        y.append(label)

    return np.array(X).reshape(-1, n_points, 1), np.array(y)


def generate_flare_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic flare light curves.

    Classes: no_flare, weak_flare, moderate_flare, strong_flare, superflare
    """
    n_points = 256
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 5)

        # Base quiescent flux
        lc = np.ones(n_points) + np.random.randn(n_points) * 0.002

        if label > 0:  # Flare present
            # Flare parameters scale with class
            amplitude = [0, 0.01, 0.05, 0.2, 1.0][label] * np.random.uniform(0.8, 1.2)
            peak_idx = np.random.randint(n_points // 4, 3 * n_points // 4)
            rise_time = np.random.randint(2, 10)
            decay_time = np.random.randint(10, 50)

            # Create flare profile (fast rise, exponential decay)
            for i in range(n_points):
                if i < peak_idx:
                    if i >= peak_idx - rise_time:
                        lc[i] += amplitude * (i - (peak_idx - rise_time)) / rise_time
                else:
                    lc[i] += amplitude * np.exp(-(i - peak_idx) / decay_time)

        # Normalize
        median = np.median(lc)
        mad = np.median(np.abs(lc - median))
        lc = (lc - median) / (mad * 1.4826 + 1e-6)

        X.append(lc)
        y.append(label)

    return np.array(X).reshape(-1, n_points, 1), np.array(y)


def generate_asteroseismo_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic power spectra for asteroseismology.

    Classes: no_oscillation, solar_like, red_giant, delta_scuti, gamma_dor, hybrid
    """
    n_points = 512
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 6)
        freq = np.linspace(0, 100, n_points)  # Arbitrary frequency units

        # Base noise spectrum
        power = np.random.exponential(0.1, n_points)

        if label == 1:  # Solar-like (nu_max ~ 100-3000 uHz)
            nu_max = np.random.uniform(30, 70)
            delta_nu = np.random.uniform(3, 10)
            envelope = np.exp(-(freq - nu_max)**2 / (10)**2)
            for n in range(5):
                power += envelope * np.exp(-(freq - (nu_max + n * delta_nu))**2 / 1**2)

        elif label == 2:  # Red giant (nu_max ~ 10-100 uHz)
            nu_max = np.random.uniform(10, 30)
            delta_nu = np.random.uniform(1, 3)
            envelope = np.exp(-(freq - nu_max)**2 / (5)**2)
            for n in range(5):
                power += 2 * envelope * np.exp(-(freq - (nu_max + n * delta_nu))**2 / 0.5**2)

        elif label == 3:  # Delta Scuti (high frequency)
            for _ in range(np.random.randint(3, 8)):
                f = np.random.uniform(50, 90)
                power += np.random.uniform(0.5, 2) * np.exp(-(freq - f)**2 / 0.5**2)

        elif label == 4:  # Gamma Dor (low frequency)
            for _ in range(np.random.randint(2, 5)):
                f = np.random.uniform(5, 20)
                power += np.random.uniform(0.5, 1.5) * np.exp(-(freq - f)**2 / 1**2)

        elif label == 5:  # Hybrid
            # Both low and high frequency
            for _ in range(np.random.randint(2, 4)):
                f = np.random.uniform(5, 20)
                power += np.random.uniform(0.3, 1) * np.exp(-(freq - f)**2 / 1**2)
            for _ in range(np.random.randint(2, 4)):
                f = np.random.uniform(50, 80)
                power += np.random.uniform(0.3, 1) * np.exp(-(freq - f)**2 / 0.5**2)

        # Log and normalize
        power = np.log10(power + 1e-6)
        power = (power - np.mean(power)) / (np.std(power) + 1e-6)

        X.append(power)
        y.append(label)

    return np.array(X).reshape(-1, n_points, 1), np.array(y)


def generate_transient_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic transient light curves.

    Classes: no_transient, sn_ia, sn_ii, sn_ibc, kilonova, tde, other_transient
    """
    n_points = 128
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 7)
        t = np.linspace(0, 100, n_points)  # days

        # Baseline
        lc = np.ones(n_points)

        if label == 0:  # No transient
            lc += np.random.randn(n_points) * 0.01

        elif label == 1:  # SN Ia - characteristic rise and decline
            t0 = np.random.uniform(20, 40)
            peak_mag = np.random.uniform(1.5, 2.5)
            rise_time = np.random.uniform(15, 20)
            decline_rate = np.random.uniform(0.02, 0.04)

            for i, ti in enumerate(t):
                if ti < t0:
                    lc[i] += peak_mag * np.exp(-(t0 - ti) / rise_time)
                else:
                    lc[i] += peak_mag * np.exp(-(ti - t0) * decline_rate)

        elif label == 2:  # SN II - plateau
            t0 = np.random.uniform(20, 40)
            peak_mag = np.random.uniform(1.0, 2.0)
            plateau_length = np.random.uniform(50, 80)

            for i, ti in enumerate(t):
                if ti < t0:
                    lc[i] += peak_mag * (ti / t0)
                elif ti < t0 + plateau_length:
                    lc[i] += peak_mag * 0.9
                else:
                    lc[i] += peak_mag * 0.9 * np.exp(-(ti - t0 - plateau_length) / 20)

        elif label == 3:  # SN Ibc - fast decline
            t0 = np.random.uniform(20, 40)
            peak_mag = np.random.uniform(1.2, 2.0)

            for i, ti in enumerate(t):
                if ti < t0:
                    lc[i] += peak_mag * np.exp(-(t0 - ti) / 10)
                else:
                    lc[i] += peak_mag * np.exp(-(ti - t0) / 15)

        elif label == 4:  # Kilonova - very fast
            t0 = np.random.uniform(20, 40)
            peak_mag = np.random.uniform(0.5, 1.5)

            for i, ti in enumerate(t):
                if ti < t0:
                    lc[i] += peak_mag * np.exp(-(t0 - ti) / 1)
                else:
                    lc[i] += peak_mag * np.exp(-(ti - t0) / 3)

        elif label == 5:  # TDE - slow rise, power law decline
            t0 = np.random.uniform(30, 50)
            peak_mag = np.random.uniform(1.0, 2.0)

            for i, ti in enumerate(t):
                if ti < t0:
                    lc[i] += peak_mag * (ti / t0)**2
                else:
                    lc[i] += peak_mag * (t0 / (ti + 1))**1.5

        else:  # Other transient
            t0 = np.random.uniform(20, 50)
            peak_mag = np.random.uniform(0.5, 1.5)
            lc += peak_mag * np.exp(-(t - t0)**2 / np.random.uniform(50, 200))

        # Add noise
        lc += np.random.randn(n_points) * 0.02

        # Normalize
        lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc) + 1e-6)

        X.append(lc)
        y.append(label)

    return np.array(X).reshape(-1, n_points, 1), np.array(y)


def generate_spectral_type_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic photometric data for spectral type classification.

    Classes: O, B, A, F, G, K, M, L
    Input: [BP-RP, G, J, H, K, W1, W2, parallax]
    """
    X = []
    y = []

    # Approximate color-magnitude relations
    # [BP-RP mean, BP-RP std, G mean, G std]
    type_params = {
        0: (-0.3, 0.1, -5, 2),    # O
        1: (-0.1, 0.1, -2, 2),    # B
        2: (0.1, 0.1, 1, 2),      # A
        3: (0.4, 0.1, 3, 2),      # F
        4: (0.7, 0.1, 5, 2),      # G
        5: (1.1, 0.15, 7, 2),     # K
        6: (2.0, 0.3, 10, 2),     # M
        7: (3.5, 0.5, 15, 2),     # L
    }

    for _ in range(n_samples):
        label = np.random.randint(0, 8)
        params = type_params[label]

        bp_rp = np.random.normal(params[0], params[1])
        G = np.random.normal(params[2], params[3])

        # Other photometry roughly correlated
        J = G + np.random.normal(-1.5 * bp_rp, 0.2)
        H = J + np.random.normal(-0.3, 0.1)
        K = H + np.random.normal(-0.1, 0.05)
        W1 = K + np.random.normal(0, 0.1)
        W2 = W1 + np.random.normal(0, 0.1)
        parallax = np.random.exponential(5)  # mas

        X.append([bp_rp, G, J, H, K, W1, W2, parallax])
        y.append(label)

    # Normalize
    X = np.array(X)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    return X.astype(np.float32), np.array(y)


def generate_microlensing_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic microlensing light curves.

    Classes: no_event, single_lens, binary_lens, planetary, parallax, unclear
    """
    n_points = 512
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 6)
        t = np.linspace(-50, 50, n_points)  # Centered on t0

        # Baseline magnification = 1
        mag = np.ones(n_points)

        if label >= 1:  # Some event
            t0 = np.random.uniform(-10, 10)
            tE = np.random.uniform(10, 50)  # Einstein crossing time
            u0 = np.random.uniform(0.1, 1.0)  # Impact parameter

            # Paczynski curve
            u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
            mag_base = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

            if label == 1:  # Single lens
                mag = mag_base

            elif label == 2:  # Binary lens - add caustic crossing
                mag = mag_base.copy()
                # Add spike at random location
                spike_t = t0 + np.random.uniform(-0.5, 0.5) * tE
                spike_idx = np.argmin(np.abs(t - spike_t))
                spike_width = np.random.randint(2, 10)
                mag[max(0, spike_idx-spike_width):min(n_points, spike_idx+spike_width)] *= np.random.uniform(1.5, 3)

            elif label == 3:  # Planetary - small deviation
                mag = mag_base.copy()
                # Add small bump
                bump_t = t0 + np.random.uniform(-0.3, 0.3) * tE
                bump_width = tE * 0.1
                mag += 0.1 * np.exp(-(t - bump_t)**2 / bump_width**2)

            elif label == 4:  # Parallax - asymmetric
                # Add asymmetry
                mag = mag_base * (1 + 0.1 * np.tanh((t - t0) / (tE * 0.5)))

            else:  # Unclear
                mag = mag_base * (1 + np.random.randn(n_points) * 0.1)

        # Add noise
        mag += np.random.randn(n_points) * 0.02

        # Normalize to baseline = 1
        baseline = np.median(mag[:n_points//4])
        mag = mag / baseline

        X.append(mag)
        y.append(label)

    return np.array(X).reshape(-1, n_points, 1), np.array(y)


def generate_galaxy_data(n_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic galaxy images.

    Classes: elliptical, spiral, barred_spiral, irregular, merger, edge_on, unknown
    """
    size = 64
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, 7)
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Center coordinates
        cx, cy = size // 2, size // 2

        # Create coordinate grids
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))

        if label == 0:  # Elliptical
            # Elliptical galaxy - smooth, concentrated
            a = np.random.uniform(10, 20)
            b = np.random.uniform(5, a)
            angle = np.random.uniform(0, np.pi)

            x_rot = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            y_rot = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)

            r = np.sqrt((x_rot / a)**2 + (y_rot / b)**2)
            brightness = np.exp(-r)
            img[:, :, 0] = brightness * 0.9
            img[:, :, 1] = brightness * 0.8
            img[:, :, 2] = brightness * 0.6

        elif label in [1, 2]:  # Spiral / Barred spiral
            # Central bulge
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            bulge = np.exp(-r / 5)

            # Spiral arms
            theta = np.arctan2(yy - cy, xx - cx)
            arm_factor = 2 if label == 1 else 3
            spiral = np.sin(arm_factor * theta + r / 5) * np.exp(-r / 20)
            spiral = np.maximum(0, spiral)

            # Bar for barred spiral
            if label == 2:
                bar = np.exp(-((xx - cx)**2 / 100 + (yy - cy)**2 / 10))
                bulge += bar * 0.5

            brightness = bulge + spiral * 0.5
            img[:, :, 0] = brightness * 0.8
            img[:, :, 1] = brightness * 0.7
            img[:, :, 2] = brightness * np.maximum(0, spiral) * 2

        elif label == 3:  # Irregular
            # Random blobs
            for _ in range(np.random.randint(3, 7)):
                bx = np.random.randint(10, size - 10)
                by = np.random.randint(10, size - 10)
                bs = np.random.uniform(3, 8)
                blob = np.exp(-((xx - bx)**2 + (yy - by)**2) / bs**2)
                color = np.random.rand(3) * 0.5 + 0.5
                for c in range(3):
                    img[:, :, c] += blob * color[c]

        elif label == 4:  # Merger
            # Two overlapping galaxies
            for offset in [(-8, -5), (8, 5)]:
                r = np.sqrt((xx - cx - offset[0])**2 + (yy - cy - offset[1])**2)
                galaxy = np.exp(-r / 8)
                img[:, :, 0] += galaxy * 0.6
                img[:, :, 1] += galaxy * 0.5
                img[:, :, 2] += galaxy * 0.4
            # Tidal tails
            tails = np.exp(-((xx - cx)**2 / 300 + (yy - cy)**2 / 50))
            img[:, :, :] += tails[:, :, None] * 0.2

        elif label == 5:  # Edge-on
            # Thin disk
            disk = np.exp(-np.abs(yy - cy) / 2) * np.exp(-np.abs(xx - cx) / 15)
            # Bulge
            bulge = np.exp(-((xx - cx)**2 + (yy - cy)**2) / 25)
            brightness = disk + bulge * 0.5
            img[:, :, 0] = brightness * 0.8
            img[:, :, 1] = brightness * 0.7
            img[:, :, 2] = brightness * 0.5

        else:  # Unknown
            # Random noise pattern
            img = np.random.rand(size, size, 3) * 0.3
            # Add some structure
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            img += np.exp(-r / 15)[:, :, None] * 0.5

        # Add noise
        img += np.random.randn(size, size, 3) * 0.05
        img = np.clip(img, 0, 1)

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)


# ============================================================================
# Training Functions
# ============================================================================

def quantize_model(model: Model, representative_data: np.ndarray,
                   output_path: Path) -> int:
    """
    Convert Keras model to INT8 quantized TFLite.

    Returns the model size in KB.
    """
    def representative_dataset():
        for i in range(min(100, len(representative_data))):
            yield [representative_data[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) // 1024
    return size_kb


def train_node(node_id: str, epochs: int = 30, batch_size: int = 32,
               verbose: int = 1) -> Dict[str, Any]:
    """
    Train a specific node's model.

    Returns training history and model info.
    """
    base_path = Path(__file__).parent / 'nodes'

    # Node configurations
    configs = {
        'VSTAR-001': {
            'folder': 'variable_star',
            'model_file': 'classifier.tflite',
            'data_fn': generate_variable_star_data,
            'model_fn': lambda: create_1d_cnn_model(512, 7, [16, 32], 5, 32),
            'input_shape': (512, 1),
        },
        'FLARE-001': {
            'folder': 'flare',
            'model_file': 'detector.tflite',
            'data_fn': generate_flare_data,
            'model_fn': lambda: create_1d_cnn_model(256, 5, [16, 24], 5, 24),
            'input_shape': (256, 1),
        },
        'ASTERO-001': {
            'folder': 'asteroseismo',
            'model_file': 'analyzer.tflite',
            'data_fn': generate_asteroseismo_data,
            'model_fn': lambda: create_1d_cnn_model(512, 6, [16, 32], 7, 32),
            'input_shape': (512, 1),
        },
        'SUPERNOVA-001': {
            'folder': 'supernova',
            'model_file': 'detector.tflite',
            'data_fn': generate_transient_data,
            'model_fn': lambda: create_1d_cnn_model(128, 7, [16, 32, 32], 5, 32),
            'input_shape': (128, 1),
        },
        'SPECTYPE-001': {
            'folder': 'spectral_type',
            'model_file': 'classifier.tflite',
            'data_fn': generate_spectral_type_data,
            'model_fn': lambda: create_mlp_model(8, 8),
            'input_shape': (8,),
        },
        'MICROLENS-001': {
            'folder': 'microlensing',
            'model_file': 'detector.tflite',
            'data_fn': generate_microlensing_data,
            'model_fn': lambda: create_1d_cnn_model(512, 6, [16, 32], 7, 32),
            'input_shape': (512, 1),
        },
        'GALAXY-001': {
            'folder': 'galaxy',
            'model_file': 'classifier.tflite',
            'data_fn': generate_galaxy_data,
            'model_fn': lambda: create_2d_cnn_model((64, 64, 3), 7),
            'input_shape': (64, 64, 3),
        },
    }

    if node_id not in configs:
        raise ValueError(f"Unknown node: {node_id}. Available: {list(configs.keys())}")

    config = configs[node_id]
    node_path = base_path / config['folder']
    model_path = node_path / 'model' / config['model_file']

    print(f"\n{'='*60}")
    print(f" Training {node_id}")
    print(f"{'='*60}")

    # Generate data
    print("\nGenerating training data...")
    X, y = config['data_fn']()
    print(f"  Data shape: {X.shape}")
    print(f"  Classes: {np.unique(y)}")

    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create model
    print("\nCreating model...")
    model = config['model_fn']()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ]
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {accuracy:.2%}")

    # Quantize and save
    print("\nQuantizing to INT8 TFLite...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    size_kb = quantize_model(model, X_train, model_path)

    print(f"  Model saved to: {model_path}")
    print(f"  Model size: {size_kb}KB")

    if size_kb > 100:
        print(f"  WARNING: Model exceeds 100KB limit!")
    else:
        print(f"  Model is within 100KB limit")

    return {
        'node_id': node_id,
        'accuracy': float(accuracy),
        'model_size_kb': size_kb,
        'epochs_trained': len(history.history['loss']),
        'model_path': str(model_path),
    }


def train_all_nodes(epochs: int = 30) -> None:
    """Train all node models."""
    nodes = [
        'VSTAR-001',
        'FLARE-001',
        'ASTERO-001',
        'SUPERNOVA-001',
        'SPECTYPE-001',
        'MICROLENS-001',
        'GALAXY-001',
    ]

    results = []
    for node_id in nodes:
        try:
            result = train_node(node_id, epochs=epochs)
            results.append(result)
        except Exception as e:
            print(f"\nError training {node_id}: {e}")
            results.append({'node_id': node_id, 'error': str(e)})

    # Summary
    print("\n" + "=" * 60)
    print(" Training Summary")
    print("=" * 60)

    total_size = 0
    for r in results:
        if 'error' in r:
            print(f"  {r['node_id']}: FAILED - {r['error']}")
        else:
            status = "" if r['model_size_kb'] <= 100 else ""
            print(f"  {status} {r['node_id']}: {r['accuracy']:.1%} accuracy, {r['model_size_kb']}KB")
            total_size += r['model_size_kb']

    print(f"\n  Total model size: {total_size}KB")

    # Save results
    results_path = Path(__file__).parent / 'nodes' / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train LARUN TinyML nodes')
    parser.add_argument('--node', '-n', help='Node ID to train (e.g., VSTAR-001)')
    parser.add_argument('--all', '-a', action='store_true', help='Train all nodes')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--list', '-l', action='store_true', help='List available nodes')

    args = parser.parse_args()

    available_nodes = [
        'VSTAR-001', 'FLARE-001', 'ASTERO-001', 'SUPERNOVA-001',
        'SPECTYPE-001', 'MICROLENS-001', 'GALAXY-001'
    ]

    if args.list:
        print("\nAvailable nodes for training:")
        for node in available_nodes:
            print(f"  - {node}")
        return

    if args.all:
        train_all_nodes(epochs=args.epochs)
    elif args.node:
        train_node(args.node, epochs=args.epochs, batch_size=args.batch_size)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_nodes.py --node VSTAR-001")
        print("  python train_nodes.py --all --epochs 50")


if __name__ == '__main__':
    main()
