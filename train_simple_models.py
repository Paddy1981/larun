#!/usr/bin/env python3
"""
Simplified Training for GALAXY-001 and SUPERNOVA-001
=====================================================
Uses feature extraction for faster, more effective training.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any

from src.model.specialized_models import get_model, MODEL_SPECS
from src.model.data_generators import DatasetConfig, GalaxyDataGenerator, SupernovaDataGenerator
from src.model.trainer import NeuralNetworkTrainer, TrainingConfig


def extract_galaxy_features(image: np.ndarray) -> np.ndarray:
    """Extract statistical features from galaxy image for classification."""
    features = []

    # Global statistics
    features.append(image.mean())
    features.append(image.std())
    features.append(image.max())
    features.append(np.median(image))

    # Central vs outer brightness
    center = image.shape[0] // 2
    inner_mask = np.zeros_like(image, dtype=bool)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    inner_mask = r < center / 2
    outer_mask = r > center / 2

    features.append(image[inner_mask].mean())  # Central brightness
    features.append(image[outer_mask].mean())  # Outer brightness
    features.append(image[inner_mask].mean() / (image[outer_mask].mean() + 1e-8))  # Concentration

    # Asymmetry (flip and compare)
    flipped_lr = np.fliplr(image)
    flipped_ud = np.flipud(image)
    features.append(np.mean(np.abs(image - flipped_lr)))  # LR asymmetry
    features.append(np.mean(np.abs(image - flipped_ud)))  # UD asymmetry

    # Radial profile (brightness at different radii)
    radii = [8, 16, 24, 32]
    for radius in radii:
        ring_mask = (r >= radius - 4) & (r < radius + 4)
        if ring_mask.any():
            features.append(image[ring_mask].mean())
        else:
            features.append(0)

    # Angular features (for spiral detection)
    theta = np.arctan2(y - center, x - center)
    for angle_idx in range(4):
        angle_min = angle_idx * np.pi / 2 - np.pi
        angle_max = (angle_idx + 1) * np.pi / 2 - np.pi
        sector_mask = (theta >= angle_min) & (theta < angle_max)
        features.append(image[sector_mask].mean())

    # Edge detection (simple gradient magnitude)
    gy, gx = np.gradient(image)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    features.append(gradient_mag.mean())
    features.append(gradient_mag.max())
    features.append(gradient_mag[inner_mask].mean())

    # Elongation (using moments)
    y_coords, x_coords = np.mgrid[:image.shape[0], :image.shape[1]]
    total = image.sum() + 1e-8
    cx = (x_coords * image).sum() / total
    cy = (y_coords * image).sum() / total
    mu20 = ((x_coords - cx)**2 * image).sum() / total
    mu02 = ((y_coords - cy)**2 * image).sum() / total
    mu11 = ((x_coords - cx) * (y_coords - cy) * image).sum() / total

    features.append(mu20)
    features.append(mu02)
    features.append(mu11)
    features.append((mu20 - mu02) / (mu20 + mu02 + 1e-8))  # Elongation

    return np.array(features, dtype=np.float32)


def extract_supernova_features(lc: np.ndarray) -> np.ndarray:
    """Extract features from supernova light curve."""
    features = []

    # Global statistics
    features.append(lc.mean())
    features.append(lc.std())
    features.append(lc.max())
    features.append(lc.min())
    features.append(lc.max() - lc.min())  # Amplitude

    # Peak characteristics
    peak_idx = np.argmax(lc)
    features.append(peak_idx / len(lc))  # Peak position (normalized)
    features.append(lc[peak_idx])  # Peak value

    # Rise and decline rates
    if peak_idx > 5:
        rise_rate = (lc[peak_idx] - lc[0]) / peak_idx
        features.append(rise_rate)
    else:
        features.append(0)

    if peak_idx < len(lc) - 5:
        decline_rate = (lc[peak_idx] - lc[-1]) / (len(lc) - peak_idx)
        features.append(decline_rate)
    else:
        features.append(0)

    # Time above thresholds
    threshold_90 = lc.min() + 0.9 * (lc.max() - lc.min())
    threshold_50 = lc.min() + 0.5 * (lc.max() - lc.min())
    features.append(np.sum(lc > threshold_90) / len(lc))
    features.append(np.sum(lc > threshold_50) / len(lc))

    # Derivatives
    diff1 = np.diff(lc)
    diff2 = np.diff(diff1)
    features.append(diff1.mean())
    features.append(diff1.std())
    features.append(diff1.max())
    features.append(diff1.min())
    features.append(diff2.mean())
    features.append(diff2.std())

    # Segment statistics
    n_segments = 4
    seg_len = len(lc) // n_segments
    for i in range(n_segments):
        seg = lc[i * seg_len:(i + 1) * seg_len]
        features.append(seg.mean())
        features.append(seg.std())

    # Smoothness
    smoothness = np.mean(np.abs(diff2))
    features.append(smoothness)

    return np.array(features, dtype=np.float32)


class SimpleGalaxyClassifier:
    """Feature-based galaxy classifier using extracted features."""

    def __init__(self):
        self.weights = {}
        n_features = 24  # Number of extracted features (counted from extract_galaxy_features)
        n_classes = 7

        np.random.seed(42)
        # Simple 2-layer MLP
        self.weights["fc1_w"] = np.random.randn(n_features, 64).astype(np.float32) * np.sqrt(2/n_features)
        self.weights["fc1_b"] = np.zeros(64, dtype=np.float32)
        self.weights["fc2_w"] = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2/64)
        self.weights["fc2_b"] = np.zeros(32, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(32, n_classes).astype(np.float32) * np.sqrt(2/32)
        self.weights["out_b"] = np.zeros(n_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Layer 1
        h = x @ self.weights["fc1_w"] + self.weights["fc1_b"]
        h = np.maximum(0, h)  # ReLU

        # Layer 2
        h = h @ self.weights["fc2_w"] + self.weights["fc2_b"]
        h = np.maximum(0, h)

        # Output
        logits = h @ self.weights["out_w"] + self.weights["out_b"]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.forward(x)
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)
        return preds, confs

    def save(self, path: str):
        np.savez(path, **self.weights)

    def load(self, path: str):
        data = np.load(path)
        self.weights = {k: data[k] for k in data.files}

    def get_model_size(self) -> Dict[str, Any]:
        total = sum(w.size for w in self.weights.values())
        return {"total_parameters": total, "size_int8_kb": total / 1024}


class SimpleSupernovaClassifier:
    """Feature-based supernova classifier."""

    def __init__(self):
        self.weights = {}
        n_features = 26  # Number of extracted features (counted from extract_supernova_features)
        n_classes = 4  # Simplified: no_transient, sn_type_i, sn_type_ii, other

        np.random.seed(43)
        # Simple 2-layer MLP
        self.weights["fc1_w"] = np.random.randn(n_features, 48).astype(np.float32) * np.sqrt(2/n_features)
        self.weights["fc1_b"] = np.zeros(48, dtype=np.float32)
        self.weights["fc2_w"] = np.random.randn(48, 24).astype(np.float32) * np.sqrt(2/48)
        self.weights["fc2_b"] = np.zeros(24, dtype=np.float32)
        self.weights["out_w"] = np.random.randn(24, n_classes).astype(np.float32) * np.sqrt(2/24)
        self.weights["out_b"] = np.zeros(n_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.weights["fc1_w"] + self.weights["fc1_b"]
        h = np.maximum(0, h)
        h = h @ self.weights["fc2_w"] + self.weights["fc2_b"]
        h = np.maximum(0, h)
        logits = h @ self.weights["out_w"] + self.weights["out_b"]
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.forward(x)
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)
        return preds, confs

    def save(self, path: str):
        np.savez(path, **self.weights)

    def load(self, path: str):
        data = np.load(path)
        self.weights = {k: data[k] for k in data.files}

    def get_model_size(self) -> Dict[str, Any]:
        total = sum(w.size for w in self.weights.values())
        return {"total_parameters": total, "size_int8_kb": total / 1024}


def train_simple_classifier(model, X_train, y_train, X_val, y_val,
                            epochs=100, lr=0.01, batch_size=32):
    """Train a simple classifier with SGD."""
    n_samples = len(X_train)
    best_val_acc = 0
    best_weights = None

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Mini-batch training
        total_loss = 0
        correct = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            probs = model.forward(X_batch)

            # Loss (cross-entropy)
            eps = 1e-8
            y_onehot = np.eye(probs.shape[-1])[y_batch]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=-1))
            total_loss += loss * len(X_batch)

            # Accuracy
            preds = np.argmax(probs, axis=-1)
            correct += np.sum(preds == y_batch)

            # Backward pass (simplified gradient descent)
            grad_logits = probs - y_onehot

            # Output layer gradients
            h2 = X_batch @ model.weights["fc1_w"] + model.weights["fc1_b"]
            h2 = np.maximum(0, h2)
            h2 = h2 @ model.weights["fc2_w"] + model.weights["fc2_b"]
            h2 = np.maximum(0, h2)

            grad_out_w = h2.T @ grad_logits / len(X_batch)
            grad_out_b = grad_logits.mean(axis=0)

            # FC2 gradients
            grad_h2 = grad_logits @ model.weights["out_w"].T
            grad_h2 = grad_h2 * (h2 > 0)

            h1 = X_batch @ model.weights["fc1_w"] + model.weights["fc1_b"]
            h1 = np.maximum(0, h1)

            grad_fc2_w = h1.T @ grad_h2 / len(X_batch)
            grad_fc2_b = grad_h2.mean(axis=0)

            # FC1 gradients
            grad_h1 = grad_h2 @ model.weights["fc2_w"].T
            grad_h1 = grad_h1 * (h1 > 0)

            grad_fc1_w = X_batch.T @ grad_h1 / len(X_batch)
            grad_fc1_b = grad_h1.mean(axis=0)

            # Update weights
            model.weights["out_w"] -= lr * grad_out_w
            model.weights["out_b"] -= lr * grad_out_b
            model.weights["fc2_w"] -= lr * grad_fc2_w
            model.weights["fc2_b"] -= lr * grad_fc2_b
            model.weights["fc1_w"] -= lr * grad_fc1_w
            model.weights["fc1_b"] -= lr * grad_fc1_b

        # Validation
        val_probs = model.forward(X_val)
        val_preds = np.argmax(val_probs, axis=-1)
        val_acc = np.mean(val_preds == y_val)

        train_acc = correct / n_samples

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.copy() for k, v in model.weights.items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Restore best weights
    if best_weights:
        model.weights = best_weights

    return best_val_acc


def main():
    output_dir = Path("models/trained")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ============================================================
    # SUPERNOVA-001 with simplified classes
    # ============================================================
    print("\n" + "=" * 60)
    print("Training SUPERNOVA-001 (simplified)")
    print("=" * 60)

    np.random.seed(42)
    generator = SupernovaDataGenerator(n_points=128)
    n_samples = 6000

    X_list = []
    y_list = []

    # Simplified classes: 0=no_transient, 1=sn_type_i (Ia,Ibc), 2=sn_type_ii, 3=other (kilonova,tde,other)
    samples_per_class = n_samples // 4

    for _ in range(samples_per_class):
        # No transient
        lc = 1.0 + np.random.normal(0, 0.02, 128)
        X_list.append(extract_supernova_features(lc))
        y_list.append(0)

        # Type I (Ia or Ibc)
        peak = np.random.uniform(0.3, 0.8)
        t_rise = np.random.uniform(0.08, 0.25)
        t_decline = np.random.uniform(0.15, 0.45)
        lc = generator.generate_sn_ia(peak, t_rise, t_decline)
        lc += np.random.normal(0, 0.02, 128)
        X_list.append(extract_supernova_features(lc))
        y_list.append(1)

        # Type II
        peak = np.random.uniform(0.2, 0.6)
        plateau = np.random.uniform(0.3, 0.5)
        lc = generator.generate_sn_ii(peak, plateau)
        lc += np.random.normal(0, 0.02, 128)
        X_list.append(extract_supernova_features(lc))
        y_list.append(2)

        # Other (kilonova, tde, etc.)
        if np.random.random() > 0.5:
            peak = np.random.uniform(0.2, 0.5)
            lc = generator.generate_kilonova(peak)
        else:
            peak = np.random.uniform(0.3, 0.6)
            lc = generator.generate_tde(peak)
        lc += np.random.normal(0, 0.02, 128)
        X_list.append(extract_supernova_features(lc))
        y_list.append(3)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    # Shuffle and split
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    n_val = int(len(X) * 0.2)
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Features: {X.shape[1]}")

    start = time.time()
    sn_model = SimpleSupernovaClassifier()
    best_acc = train_simple_classifier(sn_model, X_train, y_train, X_val, y_val,
                                        epochs=150, lr=0.05, batch_size=64)
    sn_time = time.time() - start

    sn_model.save(str(output_dir / "SUPERNOVA-001_weights.npz"))
    size_info = sn_model.get_model_size()

    results["SUPERNOVA-001"] = {
        "accuracy": best_acc,
        "size_kb": size_info["size_int8_kb"],
        "training_time": sn_time,
        "classes": ["no_transient", "sn_type_i", "sn_type_ii", "other"]
    }

    print(f"\nSUPERNOVA-001: {best_acc*100:.1f}% accuracy ({sn_time:.1f}s)")

    # ============================================================
    # GALAXY-001 with feature extraction
    # ============================================================
    print("\n" + "=" * 60)
    print("Training GALAXY-001 (feature-based)")
    print("=" * 60)

    np.random.seed(42)
    gal_generator = GalaxyDataGenerator(image_size=64)
    n_samples = 7000

    X_list = []
    y_list = []

    samples_per_class = n_samples // 7

    for _ in range(samples_per_class):
        # Elliptical
        img = gal_generator.generate_elliptical(np.random.uniform(0.1, 0.7), np.random.uniform(0.5, 1.5))
        X_list.append(extract_galaxy_features(img))
        y_list.append(0)

        # Spiral
        img = gal_generator.generate_spiral(np.random.randint(2, 5), np.random.uniform(0.3, 0.8))
        X_list.append(extract_galaxy_features(img))
        y_list.append(1)

        # Barred spiral
        img = gal_generator.generate_barred_spiral()
        X_list.append(extract_galaxy_features(img))
        y_list.append(2)

        # Irregular
        img = gal_generator.generate_irregular()
        X_list.append(extract_galaxy_features(img))
        y_list.append(3)

        # Merger
        img = gal_generator.generate_merger()
        X_list.append(extract_galaxy_features(img))
        y_list.append(4)

        # Edge-on
        img = gal_generator.generate_edge_on()
        X_list.append(extract_galaxy_features(img))
        y_list.append(5)

        # Unknown/noise
        img = np.random.uniform(0, 0.3, (64, 64))
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.randint(10, 54, 2)
            yg, xg = np.ogrid[:64, :64]
            blob = 0.3 * np.exp(-((xg - cx)**2 + (yg - cy)**2) / np.random.uniform(20, 50))
            img = img + blob
        X_list.append(extract_galaxy_features(img))
        y_list.append(6)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    # Shuffle and split
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    n_val = int(len(X) * 0.2)
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Features: {X.shape[1]}")

    start = time.time()
    gal_model = SimpleGalaxyClassifier()
    best_acc = train_simple_classifier(gal_model, X_train, y_train, X_val, y_val,
                                        epochs=150, lr=0.05, batch_size=64)
    gal_time = time.time() - start

    gal_model.save(str(output_dir / "GALAXY-001_weights.npz"))
    size_info = gal_model.get_model_size()

    results["GALAXY-001"] = {
        "accuracy": best_acc,
        "size_kb": size_info["size_int8_kb"],
        "training_time": gal_time,
        "classes": ["elliptical", "spiral", "barred_spiral", "irregular", "merger", "edge_on", "unknown"]
    }

    print(f"\nGALAXY-001: {best_acc*100:.1f}% accuracy ({gal_time:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    for model_id, res in results.items():
        print(f"{model_id}: {res['accuracy']*100:.1f}% - {res['size_kb']:.1f}KB")

    with open(output_dir / "improved_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
