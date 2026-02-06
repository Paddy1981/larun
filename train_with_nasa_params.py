#!/usr/bin/env python3
"""
Train EXOPLANET-001 Using Real NASA Exoplanet Parameters
=========================================================
Uses actual transit parameters from confirmed exoplanets to generate
realistic synthetic light curves for training.

This bridges the gap between fully synthetic and fully real data by:
1. Using real transit depths, periods, and durations from NASA archive
2. Generating synthetic light curves with these exact parameters
3. Adding realistic noise levels based on Kepler/TESS data

Usage:
    python train_with_nasa_params.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_simplified import extract_lightcurve_features, SimpleClassifier, train_classifier


def load_nasa_parameters(params_file: str = "data/real_exoplanets/nasa_planet_parameters.json") -> Dict:
    """Load real exoplanet parameters from NASA archive."""
    with open(params_file) as f:
        return json.load(f)


def generate_realistic_transit(depth_ppt: float, duration_hours: float,
                               period_days: float, n_points: int = 1024,
                               total_days: float = 30.0) -> np.ndarray:
    """
    Generate a realistic transit light curve using real parameters.

    Args:
        depth_ppt: Transit depth in parts per thousand (0.001 = 0.1% = 1000 ppm)
        duration_hours: Transit duration in hours
        period_days: Orbital period in days
        n_points: Number of data points
        total_days: Total observation time

    Returns:
        Normalized flux array
    """
    # Convert to fractional units
    depth = depth_ppt / 100.0  # ppt to fraction (1 ppt = 0.01 = 1%)
    duration = duration_hours / 24.0 / total_days  # hours to fractional time
    period = period_days / total_days  # days to fractional time

    # Time array
    t = np.linspace(0, 1, n_points)
    flux = np.ones(n_points)

    # Add transits at regular intervals
    n_transits = max(1, int(1.0 / period)) if period > 0 else 1

    for i in range(n_transits):
        transit_center = period * (i + 0.5)
        if transit_center > 1.0:
            break

        # Create transit shape (box + limb darkening approximation)
        half_duration = duration / 2

        for j in range(n_points):
            dist = abs(t[j] - transit_center)
            if dist < half_duration:
                # Trapezoidal ingress/egress
                ingress_duration = half_duration * 0.15
                if dist < half_duration - ingress_duration:
                    flux[j] -= depth
                else:
                    # Linear ingress/egress
                    frac = (dist - (half_duration - ingress_duration)) / ingress_duration
                    flux[j] -= depth * (1 - frac)

    return flux.astype(np.float32)


def generate_realistic_eclipsing_binary(n_points: int = 1024) -> np.ndarray:
    """Generate realistic eclipsing binary light curve."""
    t = np.linspace(0, 1, n_points)
    flux = np.ones(n_points)

    # Primary and secondary eclipse
    period = np.random.uniform(0.05, 0.3)  # Short periods for EBs
    primary_depth = np.random.uniform(0.1, 0.5)  # Deep primary
    secondary_depth = np.random.uniform(0.02, primary_depth * 0.5)  # Shallower secondary

    n_periods = int(1.0 / period)

    for i in range(n_periods):
        # Primary eclipse
        center1 = period * i + period * 0.25
        half_dur = period * 0.08
        for j in range(n_points):
            if abs(t[j] - center1) < half_dur:
                flux[j] -= primary_depth * (1 - abs(t[j] - center1) / half_dur)

        # Secondary eclipse
        center2 = period * i + period * 0.75
        for j in range(n_points):
            if abs(t[j] - center2) < half_dur:
                flux[j] -= secondary_depth * (1 - abs(t[j] - center2) / half_dur)

    return flux.astype(np.float32)


def generate_training_data(nasa_params: Dict, n_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data using real NASA parameters.

    Classes:
        0: no_transit (quiet stars)
        1: transit (real exoplanet parameters)
        2: eclipsing_binary (false positives)
    """
    print("Generating training data with real NASA parameters...")

    planets = nasa_params['planets']
    n_planets = len(planets)

    X_list = []
    y_list = []

    n_per_class = n_samples // 3

    # Class 0: No transit (quiet field stars)
    print(f"  Generating {n_per_class} quiet star light curves...")
    for i in range(n_per_class):
        # Realistic Kepler/TESS noise levels: 20-200 ppm
        noise_level = np.random.uniform(0.0001, 0.001)  # 100-1000 ppm
        lc = np.ones(1024) + np.random.normal(0, noise_level, 1024)

        # Occasionally add low-frequency stellar variability
        if np.random.random() < 0.3:
            freq = np.random.uniform(0.5, 3)
            amp = np.random.uniform(0.0001, 0.0005)
            t = np.linspace(0, 1, 1024)
            lc += amp * np.sin(2 * np.pi * freq * t)

        X_list.append(extract_lightcurve_features(lc))
        y_list.append(0)

    # Class 1: Real exoplanet transits
    print(f"  Generating {n_per_class} transit light curves with real parameters...")
    for i in range(n_per_class):
        # Sample from real planets
        planet = planets[i % n_planets]

        depth = planet['pl_trandep']  # ppt
        duration = planet['pl_trandur']  # hours
        period = planet['pl_orbper']  # days

        # Handle missing/invalid values
        if depth <= 0 or np.isnan(depth):
            depth = np.random.uniform(0.1, 2.0)
        if duration <= 0 or np.isnan(duration):
            duration = np.random.uniform(1.5, 6.0)
        if period <= 0 or np.isnan(period):
            period = np.random.uniform(1, 30)

        lc = generate_realistic_transit(depth, duration, period)

        # Add realistic noise (50-300 ppm)
        noise_level = np.random.uniform(0.00005, 0.0003)
        lc += np.random.normal(0, noise_level, len(lc))

        X_list.append(extract_lightcurve_features(lc))
        y_list.append(1)

    # Class 2: Eclipsing binaries (false positives)
    print(f"  Generating {n_per_class} eclipsing binary light curves...")
    for i in range(n_per_class):
        lc = generate_realistic_eclipsing_binary()

        # Add noise
        noise_level = np.random.uniform(0.0001, 0.0005)
        lc += np.random.normal(0, noise_level, len(lc))

        X_list.append(extract_lightcurve_features(lc))
        y_list.append(2)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


def main():
    print("=" * 60)
    print("TRAINING EXOPLANET-001 WITH NASA PARAMETERS")
    print("=" * 60)

    # Load NASA parameters
    params_file = "data/real_exoplanets/nasa_planet_parameters.json"
    if not Path(params_file).exists():
        print(f"ERROR: {params_file} not found")
        print("Run: python train_real_exoplanets.py --download first")
        return

    nasa_params = load_nasa_parameters(params_file)
    print(f"\nLoaded {nasa_params['count']} real exoplanet parameters from NASA")
    print(f"  Transit depth: {nasa_params['statistics']['depth_ppm']['mean']:.2f} +/- {nasa_params['statistics']['depth_ppm']['std']:.2f} ppt")
    print(f"  Period: {nasa_params['statistics']['period_days']['mean']:.1f} +/- {nasa_params['statistics']['period_days']['std']:.1f} days")
    print(f"  Duration: {nasa_params['statistics']['duration_hours']['mean']:.1f} +/- {nasa_params['statistics']['duration_hours']['std']:.1f} hours")

    # Generate training data
    n_samples = 9000
    X, y = generate_training_data(nasa_params, n_samples)

    print(f"\n  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")

    # Compute normalization
    norm_mean = X.mean(axis=0)
    norm_std = X.std(axis=0)
    X_normalized = (X - norm_mean) / (norm_std + 1e-8)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_normalized = X_normalized[indices]
    y = y[indices]

    n_val = int(len(X) * 0.2)
    X_train, y_train = X_normalized[n_val:], y[n_val:]
    X_val, y_val = X_normalized[:n_val], y[:n_val]

    print(f"\n  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")

    # Class distribution
    labels = ["no_transit", "transit", "eclipsing_binary"]
    print("\n  Class distribution:")
    for i in range(3):
        print(f"    {labels[i]}: {(y_train == i).sum()} train, {(y_val == i).sum()} val")

    # Train
    print(f"\n  Training for 150 epochs...")
    model = SimpleClassifier(n_features=X.shape[1], n_classes=3, hidden1=64, hidden2=32)

    best_acc = train_classifier(
        model, X_train, y_train, X_val, y_val,
        epochs=150, lr=0.05, batch_size=64
    )

    # Set normalization and labels
    model.set_normalization(norm_mean, norm_std)
    model.set_class_labels(labels)

    # Save
    output_dir = Path("models/trained")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "EXOPLANET-001_nasa_weights.npz"
    model.save(str(model_path))

    print(f"\n  Model saved to: {model_path}")
    print(f"  Best validation accuracy: {best_acc*100:.1f}%")

    # Per-class accuracy
    print("\n  Per-class validation accuracy:")
    val_preds, _, _ = model.predict_from_raw(X[:n_val])
    for i in range(3):
        mask = y[:n_val] == i
        if mask.sum() > 0:
            acc = (val_preds[mask] == i).mean()
            print(f"    {labels[i]}: {acc*100:.1f}%")

    # Save results
    results = {
        "model": "EXOPLANET-001 (NASA parameters)",
        "accuracy": float(best_acc),
        "n_samples": len(X),
        "n_planets_used": nasa_params['count'],
        "features": int(X.shape[1]),
        "classes": labels,
        "training_type": "realistic_synthetic_from_nasa",
    }

    results_path = output_dir / "nasa_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Accuracy: {best_acc*100:.1f}%")
    print(f"  Model: {model_path}")
    print(f"  Used {nasa_params['count']} real planet parameters from NASA")


if __name__ == "__main__":
    main()
