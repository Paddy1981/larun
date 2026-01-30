"""
Auto-Calibration System
=======================
Self-calibrating system based on previously verified astronomical discoveries.
Uses confirmed exoplanets and known phenomena for continuous model improvement.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

logger = logging.getLogger(__name__)


class CalibrationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for calibration data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, datetime):
            return {"__datetime__": True, "iso": obj.isoformat()}
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def calibration_json_decoder(obj: Dict) -> Any:
    """Custom JSON decoder for calibration data types."""
    if "__numpy__" in obj:
        return np.array(obj["data"], dtype=obj["dtype"])
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["iso"])
    return obj


@dataclass
class CalibrationReference:
    """A reference data point for calibration."""
    object_id: str
    category: str  # e.g., "planetary_transit", "eclipsing_binary"
    spectral_features: np.ndarray
    verified: bool = True
    discovery_date: Optional[datetime] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0


@dataclass
class CalibrationMetrics:
    """Metrics from a calibration run."""
    timestamp: datetime
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class CalibrationDatabase:
    """
    Database for storing calibration references.
    Uses confirmed discoveries as ground truth for calibration.
    """
    
    def __init__(self, db_path: str = "data/calibration/calibration_db.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.references: Dict[str, CalibrationReference] = {}
        self.metrics_history: List[CalibrationMetrics] = []

        self._load()

    def _load(self):
        """Load database from disk using safe JSON deserialization."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f, object_hook=calibration_json_decoder)

                    # Reconstruct CalibrationReference objects
                    refs = data.get("references", {})
                    self.references = {
                        k: CalibrationReference(**v) for k, v in refs.items()
                    }

                    # Reconstruct CalibrationMetrics objects
                    metrics = data.get("metrics_history", [])
                    self.metrics_history = [
                        CalibrationMetrics(**m) for m in metrics
                    ]

                logger.info(f"Loaded {len(self.references)} calibration references")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in calibration database: {e}")
            except (KeyError, TypeError) as e:
                logger.error(f"Error reconstructing calibration data: {e}")

    def _save(self):
        """Save database to disk using safe JSON serialization."""
        data = {
            "references": {k: asdict(v) for k, v in self.references.items()},
            "metrics_history": [asdict(m) for m in self.metrics_history]
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, cls=CalibrationJSONEncoder, indent=2)
    
    def add_reference(self, ref: CalibrationReference):
        """Add a calibration reference."""
        self.references[ref.object_id] = ref
        self._save()
        logger.info(f"Added calibration reference: {ref.object_id}")
    
    def add_references_from_exoplanet_archive(self, df: pd.DataFrame):
        """
        Add calibration references from NASA Exoplanet Archive data.
        
        Args:
            df: DataFrame from exoplanet archive with transit parameters
        """
        for _, row in df.iterrows():
            try:
                # Create synthetic spectral features based on transit parameters
                features = self._create_transit_features(row)
                
                ref = CalibrationReference(
                    object_id=row.get("pl_name", f"planet_{_}"),
                    category="planetary_transit",
                    spectral_features=features,
                    verified=True,
                    source="NASA_Exoplanet_Archive",
                    metadata={
                        "hostname": row.get("hostname"),
                        "period_days": row.get("pl_orbper"),
                        "transit_depth_ppm": row.get("pl_trandep"),
                        "transit_duration_hours": row.get("pl_trandur"),
                        "discovery_year": row.get("disc_year")
                    }
                )
                self.references[ref.object_id] = ref
                
            except Exception as e:
                logger.warning(f"Error processing {row.get('pl_name')}: {e}")
        
        self._save()
        logger.info(f"Added {len(df)} exoplanet references to calibration database")
    
    def _create_transit_features(self, row: pd.Series, n_points: int = 1024) -> np.ndarray:
        """
        Create synthetic transit light curve from parameters.
        
        This creates a representative transit signal based on:
        - Transit depth
        - Transit duration
        - Period
        """
        # Get parameters with defaults
        depth = row.get("pl_trandep", 100) / 1e6  # Convert ppm to fraction
        duration = row.get("pl_trandur", 3)  # hours
        period = row.get("pl_orbper", 10)  # days
        
        if pd.isna(depth) or pd.isna(duration) or pd.isna(period):
            depth = 0.001
            duration = 3
            period = 10
        
        # Create time array (assuming we see one transit)
        t = np.linspace(0, period * 24, n_points)  # hours
        
        # Create transit signal
        flux = np.ones(n_points)
        
        # Transit centered at t=0
        transit_start = period * 24 / 2 - duration / 2
        transit_end = period * 24 / 2 + duration / 2
        
        # Simple box transit model (can be replaced with more sophisticated model)
        in_transit = (t >= transit_start) & (t <= transit_end)
        flux[in_transit] = 1.0 - depth
        
        # Add slight ingress/egress
        ingress_time = duration * 0.1
        for i, ti in enumerate(t):
            if transit_start - ingress_time <= ti < transit_start:
                flux[i] = 1.0 - depth * (ti - (transit_start - ingress_time)) / ingress_time
            elif transit_end < ti <= transit_end + ingress_time:
                flux[i] = 1.0 - depth * (1 - (ti - transit_end) / ingress_time)
        
        # Add realistic noise
        noise_level = depth * 0.1  # 10% of transit depth
        flux += np.random.normal(0, noise_level, n_points)
        
        return flux.astype(np.float32)
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data from calibration references.
        
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        label_map = {
            "noise": 0,
            "stellar_signal": 1,
            "planetary_transit": 2,
            "eclipsing_binary": 3,
            "instrument_artifact": 4,
            "unknown_anomaly": 5
        }
        
        for ref in self.references.values():
            if ref.verified:
                features.append(ref.spectral_features)
                labels.append(label_map.get(ref.category, 5))
        
        return np.array(features), np.array(labels)
    
    def get_references_by_category(self, category: str) -> List[CalibrationReference]:
        """Get all references for a specific category."""
        return [r for r in self.references.values() if r.category == category]
    
    def add_metrics(self, metrics: CalibrationMetrics):
        """Add calibration metrics to history."""
        self.metrics_history.append(metrics)
        self._save()
    
    def get_recent_metrics(self, n: int = 10) -> List[CalibrationMetrics]:
        """Get most recent calibration metrics."""
        return self.metrics_history[-n:]


class AutoCalibrator:
    """
    Automatic calibration system for the spectral analysis model.
    
    Features:
    - Uses confirmed discoveries as ground truth
    - Detects model drift over time
    - Provides recommendations for retraining
    - Supports incremental calibration
    """
    
    def __init__(
        self,
        model,  # SpectralCNN instance
        database: CalibrationDatabase,
        config: Dict[str, Any]
    ):
        self.model = model
        self.database = database
        self.config = config
        
        # Thresholds
        self.confidence_high = config.get("confidence", {}).get("high", 0.95)
        self.confidence_medium = config.get("confidence", {}).get("medium", 0.80)
        self.max_drift = config.get("adaptive", {}).get("max_drift", 0.1)
        
        # Tracking
        self.baseline_accuracy: Optional[float] = None
        self.last_calibration: Optional[datetime] = None
    
    def run_calibration(self) -> CalibrationMetrics:
        """
        Run a full calibration cycle.

        Steps:
        1. Get training data from verified references
        2. Evaluate model on calibration set
        3. Compute metrics and detect drift
        4. Generate recommendations

        Returns:
            CalibrationMetrics with results

        Raises:
            ValueError: If model is not built/trained
        """
        # Verify model is ready for inference
        if not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError(
                "Model must be built and trained before calibration. "
                "Call model.build_model() and model.train() first."
            )

        logger.info("Starting calibration run...")
        
        # Get calibration data
        X_cal, y_cal = self.database.get_training_data()
        
        if len(X_cal) == 0:
            logger.warning("No calibration data available")
            return CalibrationMetrics(
                timestamp=datetime.now(),
                accuracy=0.0,
                precision={},
                recall={},
                f1_score={},
                confusion_matrix=np.array([]),
                recommendations=["No calibration data. Fetch confirmed discoveries first."]
            )
        
        # Ensure correct shape
        if len(X_cal.shape) == 2:
            X_cal = X_cal[..., np.newaxis]
        
        # Get predictions
        y_pred, confidence = self.model.predict(X_cal)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_cal)
        
        # Per-class metrics
        labels = ["noise", "stellar_signal", "planetary_transit", 
                  "eclipsing_binary", "instrument_artifact", "unknown_anomaly"]
        
        report = classification_report(y_cal, y_pred, output_dict=True, zero_division=0)
        
        precision = {}
        recall = {}
        f1 = {}
        
        for i, label in enumerate(labels):
            if str(i) in report:
                precision[label] = report[str(i)]["precision"]
                recall[label] = report[str(i)]["recall"]
                f1[label] = report[str(i)]["f1-score"]
        
        # Confusion matrix
        cm = confusion_matrix(y_cal, y_pred)
        
        # Detect drift
        drift_detected = False
        drift_magnitude = 0.0
        
        if self.baseline_accuracy is not None:
            drift_magnitude = abs(accuracy - self.baseline_accuracy)
            if drift_magnitude > self.max_drift:
                drift_detected = True
                logger.warning(f"Model drift detected: {drift_magnitude:.4f}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy, precision, recall, f1, drift_detected, confidence
        )
        
        # Create metrics object
        metrics = CalibrationMetrics(
            timestamp=datetime.now(),
            accuracy=float(accuracy),
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            recommendations=recommendations
        )
        
        # Store metrics
        self.database.add_metrics(metrics)
        
        # Update baseline if first run
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
        
        self.last_calibration = datetime.now()
        
        logger.info(f"Calibration complete. Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def _generate_recommendations(
        self,
        accuracy: float,
        precision: Dict[str, float],
        recall: Dict[str, float],
        f1: Dict[str, float],
        drift_detected: bool,
        confidence: np.ndarray
    ) -> List[str]:
        """Generate recommendations based on calibration results."""
        recommendations = []
        
        # Overall accuracy check
        if accuracy < 0.8:
            recommendations.append(
                f"Overall accuracy ({accuracy:.2%}) is below threshold. "
                "Consider retraining with more data."
            )
        
        # Check per-class performance
        for label, f1_score in f1.items():
            if f1_score < 0.7:
                recommendations.append(
                    f"Poor performance on '{label}' (F1={f1_score:.2f}). "
                    f"Add more training examples for this class."
                )
        
        # Drift check
        if drift_detected:
            recommendations.append(
                "Model drift detected. Consider incremental retraining "
                "with recent verified discoveries."
            )
        
        # Confidence distribution
        low_confidence = np.mean(confidence < self.confidence_medium)
        if low_confidence > 0.3:
            recommendations.append(
                f"{low_confidence:.1%} of predictions have low confidence. "
                "Model may need more diverse training data."
            )
        
        # Check for class imbalance issues
        if precision.get("planetary_transit", 0) < recall.get("planetary_transit", 0) * 0.8:
            recommendations.append(
                "High false positive rate for planetary transits. "
                "Add more negative examples to training set."
            )
        
        if not recommendations:
            recommendations.append("Model is well-calibrated. No action needed.")
        
        return recommendations
    
    def incremental_update(
        self,
        new_discoveries: List[Tuple[np.ndarray, str, Dict[str, Any]]]
    ):
        """
        Perform incremental model update with newly verified discoveries.
        
        Args:
            new_discoveries: List of (features, category, metadata) tuples
        """
        logger.info(f"Performing incremental update with {len(new_discoveries)} new discoveries")
        
        # Add to database
        for features, category, metadata in new_discoveries:
            ref = CalibrationReference(
                object_id=metadata.get("object_id", f"discovery_{datetime.now().timestamp()}"),
                category=category,
                spectral_features=features,
                verified=True,
                discovery_date=datetime.now(),
                source="user_verified",
                metadata=metadata
            )
            self.database.add_reference(ref)
        
        # Get updated training data
        X, y = self.database.get_training_data()
        
        if len(X) < 100:
            logger.warning("Insufficient data for incremental training. Need at least 100 samples.")
            return
        
        # Perform light fine-tuning
        if len(X.shape) == 2:
            X = X[..., np.newaxis]
        
        # Use a lower learning rate for fine-tuning
        learning_rate = self.config.get("adaptive", {}).get("learning_rate", 0.0001)
        
        self.model.model.optimizer.learning_rate.assign(learning_rate)
        
        # Fine-tune for a few epochs
        history = self.model.model.fit(
            X, y,
            epochs=5,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        logger.info("Incremental update complete")
        
        # Run calibration to verify
        self.run_calibration()
    
    def compute_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for predictions using bootstrap.
        
        Returns:
            Dictionary mapping class names to (lower, upper) confidence bounds
        """
        intervals = {}
        
        for class_id in range(6):
            class_mask = predictions == class_id
            if not np.any(class_mask):
                continue
            
            class_conf = confidence[class_mask]
            
            # Bootstrap
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(class_conf, size=len(class_conf), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            lower = np.percentile(bootstrap_means, 2.5)
            upper = np.percentile(bootstrap_means, 97.5)
            
            class_name = self.model.CLASSIFICATION_LABELS[class_id]
            intervals[class_name] = (lower, upper)
        
        return intervals
    
    def check_needs_calibration(self) -> bool:
        """Check if calibration is needed based on time and metrics."""
        
        # Check time since last calibration
        update_interval = self.config.get("update_interval_hours", 24)
        if self.last_calibration is None:
            return True
        
        time_since = datetime.now() - self.last_calibration
        if time_since > timedelta(hours=update_interval):
            return True
        
        # Check recent metrics for drift
        recent_metrics = self.database.get_recent_metrics(5)
        if any(m.drift_detected for m in recent_metrics):
            return True
        
        return False
    
    def get_calibration_report(self) -> str:
        """Generate a human-readable calibration report."""
        recent_metrics = self.database.get_recent_metrics(1)
        if not recent_metrics:
            return "No calibration data available. Run calibration first."
        
        metrics = recent_metrics[0]
        
        report_lines = [
            "=" * 60,
            "CALIBRATION REPORT",
            "=" * 60,
            f"Timestamp: {metrics.timestamp.isoformat()}",
            f"Overall Accuracy: {metrics.accuracy:.2%}",
            "",
            "Per-Class Performance:",
            "-" * 40,
        ]
        
        for label in self.model.CLASSIFICATION_LABELS:
            if label in metrics.f1_score:
                report_lines.append(
                    f"  {label:20s}: "
                    f"P={metrics.precision[label]:.2f} "
                    f"R={metrics.recall[label]:.2f} "
                    f"F1={metrics.f1_score[label]:.2f}"
                )
        
        report_lines.extend([
            "",
            f"Drift Detected: {'Yes' if metrics.drift_detected else 'No'}",
            f"Drift Magnitude: {metrics.drift_magnitude:.4f}",
            "",
            "Recommendations:",
            "-" * 40,
        ])
        
        for rec in metrics.recommendations:
            report_lines.append(f"  • {rec}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


# CLI interface
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Auto-Calibration System")
    parser.add_argument("--reference", type=str, help="Path to reference data CSV")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--report", action="store_true", help="Generate calibration report")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = CalibrationDatabase()
    
    # Load reference data if provided
    if args.reference:
        df = pd.read_csv(args.reference)
        db.add_references_from_exoplanet_archive(df)
        print(f"Loaded {len(df)} references from {args.reference}")
    
    # Initialize model (would normally load from checkpoint)
    from src.model.spectral_cnn import SpectralCNN
    model = SpectralCNN()
    model.build_model()
    model.compile()
    
    # Initialize calibrator
    calibrator = AutoCalibrator(model, db, config.get("calibration", {}))
    
    # Run calibration
    metrics = calibrator.run_calibration()
    
    if args.report:
        print(calibrator.get_calibration_report())
    else:
        print(f"Calibration accuracy: {metrics.accuracy:.2%}")
        print(f"Drift detected: {metrics.drift_detected}")
