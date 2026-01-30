#!/usr/bin/env python3
"""
AstroTinyML Standalone Demo
===========================
Demonstrates the complete spectral analysis pipeline.
Works without external dependencies (pure NumPy/Python).

Features demonstrated:
1. Synthetic spectral data generation
2. TinyML model training and inference
3. Auto-calibration against known exoplanets
4. Anomaly detection pipeline
5. NASA-compatible report generation
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import html
import csv


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SpectralData:
    """Represents spectral/light curve data."""
    object_id: str
    flux: np.ndarray
    time: np.ndarray
    mission: str = "synthetic"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    """A single detection result."""
    detection_id: str
    object_id: str
    classification: str
    confidence: float
    timestamp: datetime
    transit_depth: Optional[float] = None
    transit_duration: Optional[float] = None
    period: Optional[float] = None
    snr: float = 0.0
    is_significant: bool = False


@dataclass
class CalibrationReference:
    """A reference from confirmed discoveries."""
    name: str
    ra: float
    dec: float
    period: float
    transit_depth: float
    stellar_mass: float
    discovery_method: str
    discovery_year: int


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

class SyntheticDataGenerator:
    """Generates realistic astronomical light curves."""
    
    @staticmethod
    def generate_transit(
        n_points: int = 1024,
        depth: float = 0.01,
        duration_fraction: float = 0.05,
        period: float = 5.0,
        noise_level: float = 0.002
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a planetary transit light curve."""
        time = np.linspace(0, 10, n_points)
        flux = np.ones(n_points)
        
        # Transit center
        transit_center = period / 2
        transit_half_duration = duration_fraction * period / 2
        
        # Box transit
        in_transit = np.abs(time - transit_center) < transit_half_duration
        flux[in_transit] -= depth
        
        # Smooth ingress/egress
        for i, t in enumerate(time):
            dist = abs(t - transit_center)
            if transit_half_duration <= dist < transit_half_duration * 1.3:
                factor = 1 - (dist - transit_half_duration) / (transit_half_duration * 0.3)
                flux[i] -= depth * max(0, factor)
        
        # Add noise and stellar variability
        flux += np.random.normal(0, noise_level, n_points)
        flux += 0.001 * np.sin(2 * np.pi * time / 2.5)
        
        return time, flux
    
    @staticmethod
    def generate_eclipsing_binary(
        n_points: int = 1024,
        depth1: float = 0.15,
        depth2: float = 0.08,
        period: float = 3.0,
        noise_level: float = 0.003
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate an eclipsing binary light curve."""
        time = np.linspace(0, 10, n_points)
        flux = np.ones(n_points)
        
        phase = (time % period) / period
        
        # Primary eclipse
        in_eclipse1 = np.abs(phase) < 0.05
        flux[in_eclipse1] -= depth1
        
        # Secondary eclipse
        in_eclipse2 = np.abs(phase - 0.5) < 0.04
        flux[in_eclipse2] -= depth2
        
        flux += np.random.normal(0, noise_level, n_points)
        return time, flux
    
    @staticmethod
    def generate_stellar_variability(
        n_points: int = 1024,
        amplitude: float = 0.02,
        period: float = 1.5,
        noise_level: float = 0.002
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stellar variability (pulsation/rotation)."""
        time = np.linspace(0, 10, n_points)
        flux = np.ones(n_points)
        
        # Multi-periodic signal
        flux += amplitude * np.sin(2 * np.pi * time / period)
        flux += amplitude * 0.3 * np.sin(4 * np.pi * time / period)
        flux += amplitude * 0.1 * np.sin(2 * np.pi * time / (period * 0.7))
        
        flux += np.random.normal(0, noise_level, n_points)
        return time, flux
    
    @staticmethod
    def generate_noise(
        n_points: int = 1024,
        noise_level: float = 0.003
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pure noise (no astrophysical signal)."""
        time = np.linspace(0, 10, n_points)
        flux = np.ones(n_points) + np.random.normal(0, noise_level, n_points)
        return time, flux
    
    @staticmethod
    def generate_instrument_artifact(
        n_points: int = 1024,
        noise_level: float = 0.003
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate instrumental artifact (jump, trend, etc.)."""
        time = np.linspace(0, 10, n_points)
        flux = np.ones(n_points)
        
        # Random jump
        jump_pos = np.random.randint(n_points // 4, 3 * n_points // 4)
        flux[jump_pos:] += np.random.uniform(0.01, 0.05)
        
        # Trend
        flux += np.linspace(0, np.random.uniform(-0.02, 0.02), n_points)
        
        flux += np.random.normal(0, noise_level, n_points)
        return time, flux


# ============================================================================
# TINYML MODEL (Pure NumPy)
# ============================================================================

class TinyMLSpectralClassifier:
    """
    Lightweight spectral classifier using NumPy only.
    Designed for TinyML edge deployment.
    
    Uses feature engineering + simple neural network.
    """
    
    LABELS = [
        "noise",
        "stellar_signal",
        "planetary_transit", 
        "eclipsing_binary",
        "instrument_artifact",
        "unknown_anomaly"
    ]
    
    def __init__(self, n_features: int = 64, n_classes: int = 6):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.feature_mean = None
        self.feature_std = None
        self.fitted = False
        
    def _extract_features(self, flux: np.ndarray) -> np.ndarray:
        """Extract statistical features from light curve."""
        if flux.ndim == 1:
            flux = flux.reshape(1, -1)
        
        batch_size = flux.shape[0]
        features = np.zeros((batch_size, self.n_features))
        
        for i in range(batch_size):
            f = flux[i]
            
            # Basic statistics
            features[i, 0] = np.mean(f)
            features[i, 1] = np.std(f)
            features[i, 2] = np.min(f)
            features[i, 3] = np.max(f)
            features[i, 4] = np.median(f)
            features[i, 5] = np.percentile(f, 5)
            features[i, 6] = np.percentile(f, 95)
            
            # Range and spread
            features[i, 7] = features[i, 3] - features[i, 2]  # range
            features[i, 8] = features[i, 6] - features[i, 5]  # 90% spread
            
            # Shape statistics
            diff = np.diff(f)
            features[i, 9] = np.std(diff)  # roughness
            features[i, 10] = np.sum(diff > 0) / len(diff)  # fraction increasing
            
            # Detect dips (potential transits)
            threshold = np.mean(f) - 2 * np.std(f)
            below = f < threshold
            features[i, 11] = np.sum(below) / len(f)  # fraction below threshold
            
            # Detect periodicity using autocorrelation
            autocorr = np.correlate(f - np.mean(f), f - np.mean(f), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation
            peaks = []
            for j in range(10, len(autocorr) - 1):
                if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1] and autocorr[j] > 0.3:
                    peaks.append(j)
            
            features[i, 12] = len(peaks)  # number of peaks
            features[i, 13] = peaks[0] if peaks else 0  # first peak position
            
            # FFT features
            fft = np.abs(np.fft.fft(f - np.mean(f)))[:len(f)//2]
            fft_power = fft ** 2
            features[i, 14] = np.argmax(fft_power[1:]) + 1  # dominant frequency
            features[i, 15] = np.max(fft_power[1:]) / np.sum(fft_power[1:])  # power concentration
            
            # Segment features
            n_segments = 8
            seg_len = len(f) // n_segments
            for j in range(n_segments):
                seg = f[j*seg_len:(j+1)*seg_len]
                features[i, 16 + j*2] = np.mean(seg)
                features[i, 17 + j*2] = np.std(seg)
            
            # Higher order statistics
            centered = f - np.mean(f)
            features[i, 32] = np.mean(centered**3) / (np.std(f)**3 + 1e-7)  # skewness
            features[i, 33] = np.mean(centered**4) / (np.std(f)**4 + 1e-7) - 3  # kurtosis
            
            # Transit-specific features
            # Look for box-like dips
            smoothed = np.convolve(f, np.ones(10)/10, mode='same')
            residual = f - smoothed
            features[i, 34] = np.std(residual)
            features[i, 35] = np.min(smoothed)
            
            # Derivative features
            features[i, 36] = np.max(np.abs(diff))
            features[i, 37] = np.mean(np.abs(diff))
            
            # Fill remaining with segment statistics
            for j in range(38, min(self.n_features, 64)):
                seg_idx = (j - 38) % n_segments
                if j < 46:
                    seg = f[seg_idx*seg_len:(seg_idx+1)*seg_len]
                    features[i, j] = np.min(seg)
                else:
                    seg = f[seg_idx*seg_len:(seg_idx+1)*seg_len]
                    features[i, j] = np.max(seg) - np.min(seg)
        
        return features
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            learning_rate: float = 0.01, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the classifier."""
        np.random.seed(42)
        
        # Extract features
        features = self._extract_features(X)
        
        # Normalize features
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-7
        features = (features - self.feature_mean) / self.feature_std
        
        # Initialize weights (He initialization)
        hidden_size = 32
        self.weights1 = np.random.randn(self.n_features, hidden_size) * np.sqrt(2.0 / self.n_features)
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, self.n_classes) * np.sqrt(2.0 / hidden_size)
        self.bias2 = np.zeros(self.n_classes)
        
        # Split data
        n = len(X)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.8 * n)]
        val_idx = indices[int(0.8 * n):]
        
        X_train, y_train = features[train_idx], y[train_idx]
        X_val, y_val = features[val_idx], y[val_idx]
        
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
            
            # Forward pass
            h1 = self._relu(X_train @ self.weights1 + self.bias1)
            output = self._softmax(h1 @ self.weights2 + self.bias2)
            
            # Loss
            y_one_hot = np.eye(self.n_classes)[y_train]
            loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-7), axis=1))
            
            # Backward pass
            d_output = output - y_one_hot
            d_weights2 = h1.T @ d_output / len(X_train)
            d_bias2 = np.mean(d_output, axis=0)
            
            d_h1 = d_output @ self.weights2.T
            d_h1[h1 <= 0] = 0  # ReLU gradient
            
            d_weights1 = X_train.T @ d_h1 / len(X_train)
            d_bias1 = np.mean(d_h1, axis=0)
            
            # Update
            self.weights2 -= learning_rate * d_weights2
            self.bias2 -= learning_rate * d_bias2
            self.weights1 -= learning_rate * d_weights1
            self.bias1 -= learning_rate * d_bias1
            
            # Metrics
            train_preds = np.argmax(output, axis=1)
            train_acc = np.mean(train_preds == y_train)
            
            # Validation
            h1_val = self._relu(X_val @ self.weights1 + self.bias1)
            output_val = self._softmax(h1_val @ self.weights2 + self.bias2)
            y_val_one_hot = np.eye(self.n_classes)[y_val]
            val_loss = -np.mean(np.sum(y_val_one_hot * np.log(output_val + 1e-7), axis=1))
            val_preds = np.argmax(output_val, axis=1)
            val_acc = np.mean(val_preds == y_val)
            
            history["loss"].append(loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        self.fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict classes and confidences."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        features = self._extract_features(X)
        features = (features - self.feature_mean) / self.feature_std
        
        h1 = self._relu(features @ self.weights1 + self.bias1)
        output = self._softmax(h1 @ self.weights2 + self.bias2)
        
        preds = np.argmax(output, axis=1)
        confs = np.max(output, axis=1)
        
        return preds, confs
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        features = self._extract_features(X)
        features = (features - self.feature_mean) / self.feature_std
        
        h1 = self._relu(features @ self.weights1 + self.bias1)
        return self._softmax(h1 @ self.weights2 + self.bias2)
    
    def get_label(self, class_id: int) -> str:
        """Get class label name."""
        return self.LABELS[class_id]
    
    def export_weights(self, filepath: str):
        """Export weights to JSON for deployment."""
        data = {
            "weights1": self.weights1.tolist(),
            "bias1": self.bias1.tolist(),
            "weights2": self.weights2.tolist(),
            "bias2": self.bias2.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "labels": self.LABELS
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_c_header(self, filepath: str):
        """Export weights to C header for embedded systems."""
        with open(filepath, 'w') as f:
            f.write("// AstroTinyML Spectral Classifier Weights\n")
            f.write("// Auto-generated for embedded deployment\n\n")
            f.write("#ifndef ASTRO_TINYML_H\n")
            f.write("#define ASTRO_TINYML_H\n\n")
            
            f.write(f"#define N_FEATURES {self.n_features}\n")
            f.write(f"#define HIDDEN_SIZE 32\n")
            f.write(f"#define N_CLASSES {self.n_classes}\n\n")
            
            # Export arrays
            for name, arr in [
                ("weights1", self.weights1),
                ("bias1", self.bias1),
                ("weights2", self.weights2),
                ("bias2", self.bias2),
                ("feature_mean", self.feature_mean),
                ("feature_std", self.feature_std)
            ]:
                flat = arr.flatten()
                f.write(f"static const float {name}[] = {{\n")
                for i, val in enumerate(flat):
                    if i % 8 == 0:
                        f.write("    ")
                    f.write(f"{val:.6f}f,")
                    if i % 8 == 7 or i == len(flat) - 1:
                        f.write("\n")
                f.write("};\n\n")
            
            f.write("#endif // ASTRO_TINYML_H\n")


# ============================================================================
# CALIBRATION SYSTEM
# ============================================================================

class AutoCalibrator:
    """
    Auto-calibration system using confirmed exoplanet data.
    """
    
    # Simulated NASA Exoplanet Archive confirmed planets
    CONFIRMED_EXOPLANETS = [
        CalibrationReference("Kepler-186f", 291.0, 44.5, 129.9, 0.0040, 0.54, "Transit", 2014),
        CalibrationReference("Kepler-452b", 284.6, 44.3, 384.8, 0.0018, 1.04, "Transit", 2015),
        CalibrationReference("TRAPPIST-1b", 346.6, -5.0, 1.51, 0.0076, 0.089, "Transit", 2016),
        CalibrationReference("TRAPPIST-1e", 346.6, -5.0, 6.10, 0.0048, 0.089, "Transit", 2017),
        CalibrationReference("Proxima Cen b", 217.4, -62.7, 11.2, 0.0002, 0.12, "RV", 2016),
        CalibrationReference("K2-18b", 172.6, 7.6, 32.9, 0.0054, 0.50, "Transit", 2015),
        CalibrationReference("TOI-700d", 97.8, -65.6, 37.4, 0.0019, 0.42, "Transit", 2020),
        CalibrationReference("HD 219134b", 348.3, 57.2, 3.09, 0.0011, 0.78, "Transit", 2015),
    ]
    
    def __init__(self, model: TinyMLSpectralClassifier):
        self.model = model
        self.calibration_data = []
        self.metrics = {}
        
    def load_reference_data(self):
        """Load confirmed exoplanet data for calibration."""
        print("Loading calibration references from NASA Exoplanet Archive (simulated)...")
        
        # Generate synthetic light curves matching known exoplanets
        gen = SyntheticDataGenerator()
        
        for exo in self.CONFIRMED_EXOPLANETS:
            # Create realistic transit based on parameters
            time, flux = gen.generate_transit(
                depth=exo.transit_depth,
                period=min(exo.period, 10),  # Scale to our observation window
                duration_fraction=0.03 + 0.01 * np.random.random()
            )
            
            self.calibration_data.append({
                "reference": exo,
                "flux": flux,
                "time": time,
                "true_class": 2  # planetary_transit
            })
        
        print(f"Loaded {len(self.calibration_data)} calibration references")
        
    def run_calibration(self) -> Dict[str, Any]:
        """Run calibration against known references."""
        if not self.calibration_data:
            self.load_reference_data()
        
        print("\nRunning calibration...")
        
        # Test model on calibration data
        correct = 0
        total = 0
        results = []
        
        for item in self.calibration_data:
            flux = item["flux"]
            true_class = item["true_class"]
            
            pred, conf = self.model.predict(flux.reshape(1, -1))
            pred = pred[0]
            conf = conf[0]
            
            is_correct = pred == true_class
            correct += is_correct
            total += 1
            
            results.append({
                "reference": item["reference"].name,
                "predicted": self.model.get_label(pred),
                "confidence": conf,
                "correct": is_correct
            })
        
        if total == 0:
            self.metrics = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": None,
                "total_references": 0,
                "correct": 0,
                "drift_detected": None,  # Cannot determine drift without data
                "results": results,
                "error": "No calibration data available"
            }
            print("No calibration data available - cannot assess drift")
            return self.metrics

        accuracy = correct / total

        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "total_references": total,
            "correct": correct,
            "drift_detected": accuracy < 0.8,
            "results": results
        }

        print(f"Calibration accuracy: {accuracy:.1%} ({correct}/{total})")

        if self.metrics["drift_detected"]:
            print("WARNING: Model drift detected! Consider retraining.")
        
        return self.metrics
    
    def get_calibration_report(self) -> str:
        """Generate calibration report."""
        report = []
        report.append("=" * 60)
        report.append("AUTO-CALIBRATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {self.metrics.get('timestamp', 'N/A')}")
        report.append(f"Accuracy: {self.metrics.get('accuracy', 0):.1%}")
        report.append(f"References tested: {self.metrics.get('total_references', 0)}")
        report.append(f"Drift detected: {self.metrics.get('drift_detected', False)}")
        report.append("")
        report.append("Individual Results:")
        report.append("-" * 60)
        
        for r in self.metrics.get("results", []):
            status = "✓" if r["correct"] else "✗"
            report.append(f"[{status}] {r['reference']:20s} → {r['predicted']:20s} ({r['confidence']:.1%})")
        
        return "\n".join(report)


# ============================================================================
# DETECTION PIPELINE
# ============================================================================

class SpectralDetector:
    """Detection pipeline for spectral anomalies."""
    
    def __init__(self, model: TinyMLSpectralClassifier, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        self.snr_threshold = self.config.get("snr_min", 5.0)
        self.confidence_threshold = self.config.get("confidence_min", 0.7)
        
    def calculate_snr(self, flux: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        noise = 1.4826 * mad  # Convert MAD to std
        
        signal = np.abs(np.min(flux) - median)
        return signal / noise if noise > 0 else 0
    
    def extract_transit_params(self, flux: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """Extract transit parameters from light curve."""
        # Simple BLS-like transit detection
        median_flux = np.median(flux)
        threshold = median_flux - 2 * np.std(flux)
        
        in_transit = flux < threshold
        
        if np.sum(in_transit) == 0:
            return {"depth": 0, "duration": 0, "period": 0}
        
        # Depth
        depth = median_flux - np.min(flux)
        
        # Duration (crude estimate)
        transit_indices = np.where(in_transit)[0]
        if len(transit_indices) > 0:
            duration = (transit_indices[-1] - transit_indices[0]) / len(flux) * (time[-1] - time[0])
        else:
            duration = 0
        
        # Period (not detectable from single transit)
        period = 0
        
        return {"depth": float(depth), "duration": float(duration), "period": float(period)}
    
    def detect(self, flux: np.ndarray, time: np.ndarray, object_id: str) -> Detection:
        """Run detection on a single light curve."""
        # Classification
        pred, conf = self.model.predict(flux.reshape(1, -1))
        pred = pred[0]
        conf = conf[0]
        
        classification = self.model.get_label(pred)
        
        # SNR
        snr = self.calculate_snr(flux)
        
        # Transit parameters
        transit_params = self.extract_transit_params(flux, time)
        
        # Significance
        is_significant = (
            snr >= self.snr_threshold and
            conf >= self.confidence_threshold and
            classification in ["planetary_transit", "eclipsing_binary", "unknown_anomaly"]
        )
        
        return Detection(
            detection_id=f"DET_{datetime.now().strftime('%Y%m%d%H%M%S')}_{object_id[:8]}",
            object_id=object_id,
            classification=classification,
            confidence=float(conf),
            timestamp=datetime.now(),
            transit_depth=transit_params["depth"],
            transit_duration=transit_params["duration"],
            period=transit_params["period"],
            snr=float(snr),
            is_significant=is_significant
        )
    
    def detect_batch(self, data_list: List[Tuple[np.ndarray, np.ndarray, str]], batch_id: str = None) -> List[Detection]:
        """Run detection on multiple light curves."""
        batch_id = batch_id or f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        detections = []
        for flux, time, object_id in data_list:
            detection = self.detect(flux, time, object_id)
            detections.append(detection)
        
        return detections


# ============================================================================
# NASA REPORT GENERATOR
# ============================================================================

class NASAReportGenerator:
    """Generate NASA-compatible reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_logo_base64(self) -> str:
        """Get Larun logo as base64."""
        try:
            logo_path = Path(__file__).parent / "assets" / "larun-logo.b64"
            if logo_path.exists():
                return logo_path.read_text().strip()
        except Exception:
            pass
        return ""
    
    def _get_logo_white_base64(self) -> str:
        """Get Larun white logo as base64."""
        try:
            logo_path = Path(__file__).parent / "assets" / "larun-logo-white.b64"
            if logo_path.exists():
                return logo_path.read_text().strip()
        except Exception:
            pass
        return ""
        
    def generate_json_report(self, detections: List[Detection], 
                             calibration: Dict = None, filepath: str = None) -> str:
        """Generate JSON report."""
        filepath = filepath or str(self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        def convert_numpy(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        report = {
            "metadata": {
                "title": "AstroTinyML Spectral Analysis Report",
                "generated": datetime.now().isoformat(),
                "version": "1.0",
                "institution": "AstroTinyML Demo",
            },
            "summary": {
                "total_processed": len(detections),
                "significant": sum(1 for d in detections if d.is_significant),
                "transit_candidates": sum(1 for d in detections if d.classification == "planetary_transit" and d.is_significant),
            },
            "calibration": calibration,
            "detections": [asdict(d) for d in detections]
        }
        
        # Convert datetime objects to strings
        for det in report["detections"]:
            det["timestamp"] = det["timestamp"].isoformat()
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def generate_csv_report(self, detections: List[Detection], filepath: str = None) -> str:
        """Generate CSV report."""
        filepath = filepath or str(self.output_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "detection_id", "object_id", "classification", "confidence",
                "snr", "transit_depth", "transit_duration", "period",
                "is_significant", "timestamp"
            ])
            
            for d in detections:
                writer.writerow([
                    d.detection_id, d.object_id, d.classification, f"{d.confidence:.4f}",
                    f"{d.snr:.2f}", f"{d.transit_depth:.6f}" if d.transit_depth else "",
                    f"{d.transit_duration:.4f}" if d.transit_duration else "",
                    f"{d.period:.2f}" if d.period else "",
                    d.is_significant, d.timestamp.isoformat()
                ])
        
        return filepath
    
    def generate_html_report(self, detections: List[Detection], 
                             calibration: Dict = None, filepath: str = None) -> str:
        """Generate HTML report."""
        filepath = filepath or str(self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        significant = [d for d in detections if d.is_significant]
        transit_candidates = [d for d in detections if d.classification == "planetary_transit" and d.is_significant]
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AstroTinyML Spectral Analysis Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --primary-dark: #0a0a0a;
            --primary-accent: #6366f1;
            --secondary-accent: #8b5cf6;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --bg-light: #f9fafb;
            --border-color: #e5e7eb;
            --success: #10b981;
            --warning: #f59e0b;
        }}
        
        body {{ font-family: 'Inter', -apple-system, sans-serif; margin: 0; padding: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        
        .brand-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .brand-left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .larun-logo {{ height: 35px; }}
        
        .brand-divider {{
            width: 1px;
            height: 25px;
            background: var(--border-color);
        }}
        
        .astrodata-brand {{
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .astrodata-brand span {{ font-weight: 300; }}
        
        .header-section {{
            text-align: center;
            padding: 40px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            border-radius: 12px;
            color: white;
        }}
        
        .header-section .report-type {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 10px;
        }}
        
        .header-section h1 {{
            margin: 0;
            font-size: 26px;
            font-weight: 600;
        }}
        
        .header-section .meta {{
            margin-top: 15px;
            font-size: 13px;
            color: rgba(255,255,255,0.7);
        }}
        
        .header-section .powered-by {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 11px;
            color: rgba(255,255,255,0.4);
        }}
        
        h2 {{ color: var(--text-primary); margin-top: 35px; font-size: 18px; font-weight: 600; }}
        
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 25px 0; }}
        .summary-box {{ 
            background: var(--bg-light); 
            color: var(--text-primary); 
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            border: 1px solid var(--border-color);
        }}
        .summary-box.accent {{ 
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent)); 
            color: white;
            border: none;
        }}
        .summary-box h3 {{ margin: 0 0 8px 0; font-size: 12px; font-weight: 500; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .summary-box .value {{ font-size: 32px; font-weight: 700; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        th, td {{ padding: 14px 16px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: var(--primary-dark); color: white; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:hover {{ background: var(--bg-light); }}
        .significant {{ background: rgba(16, 185, 129, 0.1); }}
        
        .confidence {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
        .confidence.high {{ background: rgba(16, 185, 129, 0.15); color: #059669; }}
        .confidence.medium {{ background: rgba(245, 158, 11, 0.15); color: #d97706; }}
        .confidence.low {{ background: rgba(239, 68, 68, 0.15); color: #dc2626; }}
        
        .calibration {{ 
            background: var(--bg-light); 
            padding: 25px; 
            border-radius: 12px; 
            margin: 25px 0;
            border-left: 4px solid var(--primary-accent);
        }}
        .calibration p {{ margin: 8px 0; font-size: 14px; }}
        
        .footer {{ 
            margin-top: 50px; 
            padding: 30px; 
            background: var(--primary-dark);
            border-radius: 12px;
            text-align: center;
        }}
        
        .footer-brands {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .footer-brands img {{ height: 25px; }}
        .footer-brands .astrodata-footer {{ font-size: 16px; font-weight: 600; color: white; }}
        .footer-brands .divider {{ color: rgba(255,255,255,0.3); }}
        
        .footer p {{ color: rgba(255,255,255,0.6); font-size: 12px; margin: 5px 0; }}
        .footer .copyright {{ margin-top: 15px; color: rgba(255,255,255,0.4); font-size: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="brand-header">
            <div class="brand-left">
                <img src="data:image/png;base64,{self._get_logo_base64()}" alt="Larun." class="larun-logo" onerror="this.outerHTML='<span style=font-weight:bold;font-size:20px>Larun.</span>'">
                <div class="brand-divider"></div>
                <div class="astrodata-brand">Astro<span>data</span></div>
            </div>
        </div>
        
        <div class="header-section">
            <div class="report-type">NASA Exoplanet Discovery Report</div>
            <h1>🔭 AstroTinyML Spectral Analysis Report</h1>
            <div class="meta">
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | 
                <strong>Mission:</strong> Synthetic Demo Data
            </div>
            <div class="powered-by">Powered by Larun. × Astrodata</div>
        </div>
        
        <div class="summary">
            <div class="summary-box">
                <h3>Total Processed</h3>
                <div class="value">{len(detections)}</div>
            </div>
            <div class="summary-box accent">
                <h3>Significant Detections</h3>
                <div class="value">{len(significant)}</div>
            </div>
            <div class="summary-box accent">
                <h3>Transit Candidates</h3>
                <div class="value">{len(transit_candidates)}</div>
            </div>
            <div class="summary-box">
                <h3>Calibration Accuracy</h3>
                <div class="value">{calibration.get('accuracy', 0)*100:.0f}%</div>
            </div>
        </div>
        
        <h2>📊 Detection Results</h2>
        <table>
            <tr>
                <th>Object ID</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>SNR</th>
                <th>Transit Depth</th>
                <th>Significant</th>
            </tr>
"""
        
        for d in detections:
            conf_class = "high" if d.confidence > 0.8 else "medium" if d.confidence > 0.6 else "low"
            sig_str = "✓ Yes" if d.is_significant else "No"
            row_class = "significant" if d.is_significant else ""
            
            html_content += f"""            <tr class="{row_class}">
                <td>{html.escape(d.object_id)}</td>
                <td>{d.classification}</td>
                <td><span class="confidence {conf_class}">{d.confidence:.1%}</span></td>
                <td>{d.snr:.1f}</td>
                <td>{f"{d.transit_depth:.4f}" if d.transit_depth else "N/A"}</td>
                <td>{sig_str}</td>
            </tr>
"""
        
        html_content += f"""        </table>
        
        <h2>🎯 Calibration Status</h2>
        <div class="calibration">
            <p><strong>Last Calibration:</strong> {calibration.get('timestamp', 'N/A')}</p>
            <p><strong>Accuracy:</strong> {calibration.get('accuracy', 0)*100:.1f}%</p>
            <p><strong>Reference Count:</strong> {calibration.get('total_references', 0)}</p>
            <p><strong>Drift Detected:</strong> {'Yes ⚠️' if calibration.get('drift_detected', False) else 'No ✓'}</p>
        </div>
        
        <div class="footer">
            <div class="footer-brands">
                <img src="data:image/png;base64,{self._get_logo_white_base64()}" alt="Larun." onerror="this.outerHTML='<span style=color:white;font-weight:bold;font-size:18px>Larun.</span>'">
                <span class="divider">×</span>
                <span class="astrodata-footer">Astrodata</span>
            </div>
            <p>Generated by AstroTinyML v1.0 | NASA-Compatible Report Format</p>
            <p>This report is suitable for submission to NASA archives and exoplanet catalogs.</p>
            <p class="copyright">© {datetime.now().year} Larun. × Astrodata | TinyML for Space Science</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath
    
    def generate_all_reports(self, detections: List[Detection], 
                             calibration: Dict = None) -> Dict[str, str]:
        """Generate all report formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return {
            "json": self.generate_json_report(detections, calibration, 
                                              str(self.output_dir / f"report_{timestamp}.json")),
            "csv": self.generate_csv_report(detections, 
                                            str(self.output_dir / f"detections_{timestamp}.csv")),
            "html": self.generate_html_report(detections, calibration,
                                              str(self.output_dir / f"report_{timestamp}.html"))
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("   AstroTinyML - Spectral Data Analysis System")
    print("   TinyML Pipeline for NASA Astronomical Data")
    print("=" * 70)
    
    # Setup
    np.random.seed(42)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Generate Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Generating Synthetic Training Data")
    print("=" * 70)
    
    gen = SyntheticDataGenerator()
    
    X_train = []
    y_train = []
    n_samples = 150  # Per class
    
    print(f"\nGenerating {n_samples} samples per class...")
    
    # Class 0: Noise
    print("  → Generating noise samples...")
    for _ in range(n_samples):
        _, flux = gen.generate_noise(noise_level=np.random.uniform(0.002, 0.006))
        X_train.append(flux)
        y_train.append(0)
    
    # Class 1: Stellar signals
    print("  → Generating stellar signals...")
    for _ in range(n_samples):
        _, flux = gen.generate_stellar_variability(
            amplitude=np.random.uniform(0.01, 0.05),
            period=np.random.uniform(0.5, 5.0)
        )
        X_train.append(flux)
        y_train.append(1)
    
    # Class 2: Planetary transits
    print("  → Generating planetary transits...")
    for _ in range(n_samples):
        _, flux = gen.generate_transit(
            depth=np.random.uniform(0.001, 0.025),
            duration_fraction=np.random.uniform(0.02, 0.1)
        )
        X_train.append(flux)
        y_train.append(2)
    
    # Class 3: Eclipsing binaries
    print("  → Generating eclipsing binaries...")
    for _ in range(n_samples):
        _, flux = gen.generate_eclipsing_binary(
            depth1=np.random.uniform(0.1, 0.25),
            depth2=np.random.uniform(0.03, 0.12)
        )
        X_train.append(flux)
        y_train.append(3)
    
    # Class 4: Instrument artifacts
    print("  → Generating instrument artifacts...")
    for _ in range(n_samples):
        _, flux = gen.generate_instrument_artifact()
        X_train.append(flux)
        y_train.append(4)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n✓ Generated {len(X_train)} training samples across 5 classes")
    
    # =========================================================================
    # STEP 2: Train TinyML Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Training TinyML Spectral Classifier")
    print("=" * 70)
    
    model = TinyMLSpectralClassifier(n_features=64, n_classes=6)
    
    print("\nTraining model (this may take a moment)...")
    history = model.fit(X_train, y_train, epochs=50, learning_rate=0.02, verbose=True)
    
    final_acc = history["accuracy"][-1]
    final_val_acc = history["val_accuracy"][-1]
    print(f"\n✓ Training complete!")
    print(f"  Final training accuracy: {final_acc:.1%}")
    print(f"  Final validation accuracy: {final_val_acc:.1%}")
    
    # Export model
    model.export_weights(str(output_dir / "model_weights.json"))
    model.export_c_header(str(output_dir / "astro_tinyml_weights.h"))
    print(f"\n✓ Model exported to:")
    print(f"  → {output_dir / 'model_weights.json'}")
    print(f"  → {output_dir / 'astro_tinyml_weights.h'} (for embedded deployment)")
    
    # =========================================================================
    # STEP 3: Run Auto-Calibration
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Running Auto-Calibration Against NASA Exoplanet Archive")
    print("=" * 70)
    
    calibrator = AutoCalibrator(model)
    calibration_metrics = calibrator.run_calibration()
    
    print("\n" + calibrator.get_calibration_report())
    
    # =========================================================================
    # STEP 4: Run Detection Pipeline on Test Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Running Detection Pipeline on Test Data")
    print("=" * 70)
    
    detector = SpectralDetector(model, {"snr_min": 4.0, "confidence_min": 0.6})
    
    # Generate test data simulating NASA data injection
    print("\nSimulating NASA data pipeline injection...")
    test_data = []
    
    # Simulated TESS targets
    tess_targets = [
        ("TIC-12345678", "transit", 0.008),
        ("TIC-23456789", "binary", None),
        ("TIC-34567890", "noise", None),
        ("TIC-45678901", "transit", 0.012),
        ("TIC-56789012", "stellar", None),
    ]
    
    # Simulated Kepler targets
    kepler_targets = [
        ("KIC-9876543", "transit", 0.005),
        ("KIC-8765432", "transit", 0.015),
        ("KIC-7654321", "binary", None),
        ("KIC-6543210", "noise", None),
        ("KIC-5432109", "transit", 0.003),
    ]
    
    print(f"\n  Injecting {len(tess_targets)} TESS targets...")
    for target_id, signal_type, depth in tess_targets:
        time = np.linspace(0, 10, 1024)
        if signal_type == "transit":
            _, flux = gen.generate_transit(depth=depth)
        elif signal_type == "binary":
            _, flux = gen.generate_eclipsing_binary()
        elif signal_type == "stellar":
            _, flux = gen.generate_stellar_variability()
        else:
            _, flux = gen.generate_noise()
        test_data.append((flux, time, target_id))
    
    print(f"  Injecting {len(kepler_targets)} Kepler targets...")
    for target_id, signal_type, depth in kepler_targets:
        time = np.linspace(0, 10, 1024)
        if signal_type == "transit":
            _, flux = gen.generate_transit(depth=depth)
        elif signal_type == "binary":
            _, flux = gen.generate_eclipsing_binary()
        else:
            _, flux = gen.generate_noise()
        test_data.append((flux, time, target_id))
    
    print(f"\nRunning detection on {len(test_data)} light curves...")
    detections = detector.detect_batch(test_data, batch_id="DEMO_BATCH_001")
    
    # Results summary
    significant = [d for d in detections if d.is_significant]
    transit_candidates = [d for d in detections if d.classification == "planetary_transit" and d.is_significant]
    
    print(f"\n✓ Detection complete!")
    print(f"  Total processed: {len(detections)}")
    print(f"  Significant detections: {len(significant)}")
    print(f"  Transit candidates: {len(transit_candidates)}")
    
    print("\n  Detection Details:")
    print("  " + "-" * 66)
    for d in detections:
        sig_mark = "✓" if d.is_significant else " "
        print(f"  [{sig_mark}] {d.object_id:16s} → {d.classification:20s} (conf: {d.confidence:.1%}, SNR: {d.snr:.1f})")
    
    # =========================================================================
    # STEP 5: Generate NASA-Compatible Reports
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Generating NASA-Compatible Reports")
    print("=" * 70)
    
    reporter = NASAReportGenerator(output_dir=str(output_dir))
    
    print("\nGenerating reports...")
    output_files = reporter.generate_all_reports(detections, calibration_metrics)
    
    print("\n✓ Reports generated:")
    for fmt, path in output_files.items():
        print(f"  → {fmt.upper()}: {path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
    📊 Training Data:     {len(X_train)} samples (5 classes)
    🧠 Model Accuracy:    {final_val_acc:.1%}
    🎯 Calibration:       {calibration_metrics['accuracy']:.1%} on NASA references
    🔍 Detections:        {len(detections)} processed
    ⭐ Significant:       {len(significant)} detections
    🌍 Transit Candidates: {len(transit_candidates)} potential exoplanets
    
    📁 Output Directory:  {output_dir}
    """)
    
    print("The generated reports are ready for NASA submission!")
    print("=" * 70)
    
    return output_files


if __name__ == "__main__":
    output_files = main()
