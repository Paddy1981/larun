#!/usr/bin/env python3
"""
AstroTinyML Demo
================
Demonstrates the complete pipeline with synthetic data.
No NASA API access required for this demo.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def generate_synthetic_transit(
    n_points: int = 1024,
    depth: float = 0.01,
    duration_fraction: float = 0.05,
    noise_level: float = 0.002
) -> tuple:
    """Generate a synthetic transit light curve."""
    time = np.linspace(0, 10, n_points)  # 10 days
    flux = np.ones(n_points)
    
    # Add transit
    transit_center = 5.0
    transit_half_duration = duration_fraction * 10 / 2
    
    in_transit = np.abs(time - transit_center) < transit_half_duration
    flux[in_transit] -= depth
    
    # Add ingress/egress
    for i, t in enumerate(time):
        dist = abs(t - transit_center)
        if transit_half_duration <= dist < transit_half_duration * 1.2:
            flux[i] -= depth * (1 - (dist - transit_half_duration) / (transit_half_duration * 0.2))
    
    # Add noise
    flux += np.random.normal(0, noise_level, n_points)
    
    # Add slight stellar variability
    flux += 0.001 * np.sin(2 * np.pi * time / 2.5)
    
    return time, flux


def generate_synthetic_eclipsing_binary(
    n_points: int = 1024,
    depth1: float = 0.15,
    depth2: float = 0.08,
    period: float = 3.0,
    noise_level: float = 0.003
) -> tuple:
    """Generate a synthetic eclipsing binary light curve."""
    time = np.linspace(0, 10, n_points)
    flux = np.ones(n_points)
    
    phase = (time % period) / period
    
    # Primary eclipse
    eclipse1_center = 0.0
    eclipse1_width = 0.05
    in_eclipse1 = np.abs(phase) < eclipse1_width
    flux[in_eclipse1] -= depth1
    
    # Secondary eclipse
    eclipse2_center = 0.5
    eclipse2_width = 0.04
    in_eclipse2 = np.abs(phase - eclipse2_center) < eclipse2_width
    flux[in_eclipse2] -= depth2
    
    # Add noise
    flux += np.random.normal(0, noise_level, n_points)
    
    return time, flux


def generate_noise_only(
    n_points: int = 1024,
    noise_level: float = 0.003
) -> tuple:
    """Generate pure noise (no signal)."""
    time = np.linspace(0, 10, n_points)
    flux = np.ones(n_points) + np.random.normal(0, noise_level, n_points)
    return time, flux


def generate_stellar_signal(
    n_points: int = 1024,
    amplitude: float = 0.02,
    period: float = 1.5,
    noise_level: float = 0.002
) -> tuple:
    """Generate stellar variability signal."""
    time = np.linspace(0, 10, n_points)
    flux = np.ones(n_points)
    
    # Sinusoidal variability
    flux += amplitude * np.sin(2 * np.pi * time / period)
    
    # Add some harmonics
    flux += amplitude * 0.3 * np.sin(4 * np.pi * time / period)
    
    # Add noise
    flux += np.random.normal(0, noise_level, n_points)
    
    return time, flux


def main():
    print("=" * 60)
    print("AstroTinyML Demo - Synthetic Data")
    print("=" * 60)
    
    # Create output directories
    Path("data/demo").mkdir(parents=True, exist_ok=True)
    Path("models/demo").mkdir(parents=True, exist_ok=True)
    Path("reports/demo").mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    print("\n1. Generating synthetic training data...")
    
    np.random.seed(42)
    
    X_train = []
    y_train = []
    
    n_samples_per_class = 200
    
    # Class 0: Noise
    print("   - Generating noise samples...")
    for _ in range(n_samples_per_class):
        _, flux = generate_noise_only(noise_level=np.random.uniform(0.002, 0.005))
        X_train.append(flux)
        y_train.append(0)
    
    # Class 1: Stellar signal
    print("   - Generating stellar signals...")
    for _ in range(n_samples_per_class):
        _, flux = generate_stellar_signal(
            amplitude=np.random.uniform(0.01, 0.05),
            period=np.random.uniform(0.5, 5.0)
        )
        X_train.append(flux)
        y_train.append(1)
    
    # Class 2: Planetary transit
    print("   - Generating planetary transits...")
    for _ in range(n_samples_per_class):
        _, flux = generate_synthetic_transit(
            depth=np.random.uniform(0.001, 0.02),
            duration_fraction=np.random.uniform(0.02, 0.1)
        )
        X_train.append(flux)
        y_train.append(2)
    
    # Class 3: Eclipsing binary
    print("   - Generating eclipsing binaries...")
    for _ in range(n_samples_per_class):
        _, flux = generate_synthetic_eclipsing_binary(
            depth1=np.random.uniform(0.1, 0.3),
            depth2=np.random.uniform(0.05, 0.15)
        )
        X_train.append(flux)
        y_train.append(3)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"   Generated {len(X_train)} training samples")
    
    # Save training data
    np.savez("data/demo/training_data.npz", X_train=X_train, y_train=y_train)
    
    # Build and train model
    print("\n2. Building and training model...")
    
    from src.model.spectral_cnn import SpectralCNN
    
    model = SpectralCNN(input_shape=(1024, 1), num_classes=4)  # Match training data classes
    model.build_model()
    model.compile(learning_rate=0.001)
    
    # Reshape for training
    X_train_reshaped = X_train[..., np.newaxis]
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_reshaped, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_tr)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Train (reduced epochs for demo)
    history = model.model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    print(f"   Final accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"   Final val accuracy: {history.history['val_accuracy'][-1]:.2%}")
    
    # Export model
    print("\n3. Exporting TinyML model...")
    model.export_for_edge("models/demo")
    print("   Model exported to models/demo/")
    
    # Run detection on test samples
    print("\n4. Running detection on test samples...")
    
    from src.detector.detector import SpectralDetector, Detection, DetectionBatch
    
    detector = SpectralDetector(model, {
        "thresholds": {
            "transit_depth_min": 0.0001,
            "duration_min_hours": 0.5,
            "snr_min": 5.0
        }
    })
    
    # Generate test data
    test_data = []
    
    # 5 test transits
    for i in range(5):
        t, f = generate_synthetic_transit(depth=np.random.uniform(0.005, 0.02))
        test_data.append((f, t, f"TEST_TRANSIT_{i+1}"))
    
    # 3 test eclipsing binaries
    for i in range(3):
        t, f = generate_synthetic_eclipsing_binary()
        test_data.append((f, t, f"TEST_EB_{i+1}"))
    
    # 2 noise samples
    for i in range(2):
        t, f = generate_noise_only()
        test_data.append((f, t, f"TEST_NOISE_{i+1}"))
    
    batch = detector.detect_batch(test_data, batch_id="DEMO_BATCH")
    
    print(f"\n   Results:")
    print(f"   - Total processed: {len(batch.detections)}")
    print(f"   - Significant detections: {len(batch.significant_detections)}")
    print(f"   - Transit candidates: {len(batch.transit_candidates)}")
    
    print("\n   Detection details:")
    for d in batch.detections:
        conf_str = f"{d.confidence:.1%}"
        sig_str = "✓" if d.is_significant else " "
        print(f"   [{sig_str}] {d.object_id:20s} → {d.classification:20s} ({conf_str})")
    
    # Generate report
    print("\n5. Generating NASA-compatible report...")
    
    from src.reporter.report_generator import NASAReportGenerator, ReportConfig
    
    report_config = ReportConfig(
        title="AstroTinyML Demo Report",
        institution="AstroTinyML Demo",
        contact_email="demo@astrotinyml.org",
        data_source="Synthetic Data"
    )
    
    reporter = NASAReportGenerator(report_config, output_dir="reports/demo")
    
    output_files = reporter.generate_report(
        batch,
        calibration_metrics={
            "timestamp": "2024-01-15T10:00:00",
            "accuracy": history.history['val_accuracy'][-1],
            "drift_detected": False,
            "reference_count": len(X_train)
        },
        output_formats=["html", "json", "csv"]
    )
    
    print("\n   Generated files:")
    for fmt, path in output_files.items():
        print(f"   - {fmt}: {path}")
    
    # Create visualization
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sample transit
    t, f = generate_synthetic_transit(depth=0.01)
    axes[0, 0].plot(t, f, 'b-', linewidth=0.5)
    axes[0, 0].set_title("Planetary Transit")
    axes[0, 0].set_xlabel("Time (days)")
    axes[0, 0].set_ylabel("Normalized Flux")
    
    # Sample eclipsing binary
    t, f = generate_synthetic_eclipsing_binary()
    axes[0, 1].plot(t, f, 'r-', linewidth=0.5)
    axes[0, 1].set_title("Eclipsing Binary")
    axes[0, 1].set_xlabel("Time (days)")
    axes[0, 1].set_ylabel("Normalized Flux")
    
    # Training history
    axes[1, 0].plot(history.history['accuracy'], label='Train')
    axes[1, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[1, 0].set_title("Training History")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    
    # Detection summary
    classes = list(batch.summary.get("classification_counts", {}).keys())
    counts = list(batch.summary.get("classification_counts", {}).values())
    if classes:
        axes[1, 1].bar(classes, counts, color=['gray', 'blue', 'green', 'red', 'orange', 'purple'][:len(classes)])
        axes[1, 1].set_title("Detection Summary")
        axes[1, 1].set_xlabel("Classification")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("reports/demo/visualization.png", dpi=150)
    print("   Saved visualization to reports/demo/visualization.png")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated report in reports/demo/")
    print("2. Check the TinyML model in models/demo/")
    print("3. Run with real NASA data using: python main.py --mode full --target Kepler-186")
    print()


if __name__ == "__main__":
    main()
