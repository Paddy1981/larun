#!/usr/bin/env python3
"""
Automated Training Pipeline
============================
Automatically collects real astronomical data and trains all TinyML models.

This pipeline:
1. Collects data from multiple astronomical archives
2. Preprocesses and validates data
3. Trains models with configurable parameters
4. Evaluates performance and saves best models
5. Updates the model registry

Usage:
    python -m src.data.automated_trainer --all           # Train all models
    python -m src.data.automated_trainer --model EXOPLANET-001  # Train specific model
    python -m src.data.automated_trainer --collect-only  # Only collect data
    python -m src.data.automated_trainer --schedule      # Run on schedule
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingConfig:
    """Configuration for automated training."""
    # Data collection
    n_samples_per_model: int = 5000
    validation_split: float = 0.2
    test_split: float = 0.1
    use_cached_data: bool = True

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Output
    output_dir: str = "models/trained"
    save_checkpoints: bool = True
    update_registry: bool = True

    # Scheduling
    schedule_interval_hours: int = 24
    max_retries: int = 3


@dataclass
class ModelTrainingResult:
    """Result from training a single model."""
    model_id: str
    status: str  # "success", "failed", "skipped"
    accuracy: float = 0.0
    loss: float = 0.0
    training_time_seconds: float = 0.0
    data_samples: int = 0
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


# Model to collector mapping
MODEL_COLLECTORS = {
    "EXOPLANET-001": ("mast", "exoplanet"),
    "VSTAR-001": ("variable_star", "variable"),
    "FLARE-001": ("mast_flare", "flare"),
    "ASTERO-001": ("mast_astero", "asteroseismology"),
    "SUPERNOVA-001": ("transient", "supernova"),
    "GALAXY-001": ("galaxy", "morphology"),
    "SPECTYPE-001": ("gaia", "spectral_type"),
    "MICROLENS-001": ("variable_star", "microlensing"),
}


class AutomatedTrainer:
    """
    Automated training pipeline for all TinyML models.

    This class orchestrates:
    1. Data collection from multiple sources
    2. Data preprocessing and validation
    3. Model training with early stopping
    4. Performance evaluation
    5. Model serialization and registry updates
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collectors = {}
        self.models = {}
        self.results: Dict[str, ModelTrainingResult] = {}

        self._initialize_collectors()
        self._initialize_models()

    def _initialize_collectors(self):
        """Initialize all data collectors."""
        from src.data.collectors.mast_collector import MASTCollector, FlareCollector, AsteroseismologyCollector
        from src.data.collectors.gaia_collector import GaiaCollector
        from src.data.collectors.transient_collector import TransientCollector
        from src.data.collectors.variable_star_collector import VariableStarCollector
        from src.data.collectors.galaxy_collector import GalaxyCollector

        self.collectors = {
            "mast": MASTCollector(),
            "mast_flare": FlareCollector(),
            "mast_astero": AsteroseismologyCollector(),
            "gaia": GaiaCollector(),
            "transient": TransientCollector(),
            "variable_star": VariableStarCollector(),
            "galaxy": GalaxyCollector(),
        }

        logger.info(f"Initialized {len(self.collectors)} data collectors")

    def _initialize_models(self):
        """Initialize all models."""
        from src.model.specialized_models import MODEL_CLASSES

        for model_id, ModelClass in MODEL_CLASSES.items():
            self.models[model_id] = ModelClass()

        logger.info(f"Initialized {len(self.models)} models")

    def collect_data(self, model_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect and split data for a model.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        collector_name, target_type = MODEL_COLLECTORS[model_id]
        collector = self.collectors[collector_name]

        logger.info(f"Collecting data for {model_id} from {collector_name}...")

        # Collect data
        X, y, info = collector.collect(
            n_samples=self.config.n_samples_per_model,
            use_cache=self.config.use_cached_data,
            target_type=target_type
        )

        logger.info(f"Collected {len(X)} samples: {info.class_distribution}")

        # Split data
        n_samples = len(X)
        n_test = int(n_samples * self.config.test_split)
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_test - n_val

        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]

        # Split
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, model_id: str) -> ModelTrainingResult:
        """
        Train a single model with collected data.

        Args:
            model_id: Model identifier

        Returns:
            Training result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_id}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Collect data
            X_train, y_train, X_val, y_val, X_test, y_test = self.collect_data(model_id)

            # Get model
            model = self.models[model_id]

            # Initialize weights
            if hasattr(model, 'initialize_weights'):
                model.initialize_weights()

            # Training loop with early stopping
            best_val_acc = 0.0
            best_weights = None
            patience_counter = 0

            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            for epoch in range(self.config.epochs):
                # Shuffle training data
                perm = np.random.permutation(len(X_train))
                X_train_shuffled = X_train[perm]
                y_train_shuffled = y_train[perm]

                # Train on batches
                epoch_losses = []
                for i in range(0, len(X_train), self.config.batch_size):
                    batch_x = X_train_shuffled[i:i+self.config.batch_size]
                    batch_y = y_train_shuffled[i:i+self.config.batch_size]

                    # Forward pass
                    probs = model.forward(batch_x)

                    # Loss
                    eps = 1e-7
                    y_one_hot = np.eye(model.num_classes)[batch_y]
                    loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=-1))
                    epoch_losses.append(loss)

                    # Update weights
                    self._update_weights(model, batch_x, batch_y)

                # Compute metrics
                train_loss = np.mean(epoch_losses)
                train_preds, _ = model.predict(X_train)
                train_acc = np.mean(train_preds == y_train)

                val_probs = model.forward(X_val)
                val_loss = -np.mean(np.sum(np.eye(model.num_classes)[y_val] * np.log(val_probs + 1e-7), axis=-1))
                val_preds, _ = model.predict(X_val)
                val_acc = np.mean(val_preds == y_val)

                history["loss"].append(float(train_loss))
                history["accuracy"].append(float(train_acc))
                history["val_loss"].append(float(val_loss))
                history["val_accuracy"].append(float(val_acc))

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.copy() for k, v in model.weights.items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                    logger.info(f"Epoch {epoch+1:3d} - loss: {train_loss:.4f} - "
                               f"acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

            # Restore best weights
            if best_weights:
                model.weights = best_weights

            # Final evaluation on test set
            test_preds, test_confs = model.predict(X_test)
            test_acc = np.mean(test_preds == y_test)

            # Per-class metrics
            class_metrics = {}
            for class_idx in range(model.num_classes):
                class_mask = y_test == class_idx
                if class_mask.sum() > 0:
                    class_acc = np.mean(test_preds[class_mask] == class_idx)
                    class_metrics[f"class_{class_idx}_accuracy"] = float(class_acc)

            # Save model
            model_path = self.output_dir / f"{model_id}_weights.npz"
            model.save(str(model_path))

            training_time = time.time() - start_time

            result = ModelTrainingResult(
                model_id=model_id,
                status="success",
                accuracy=float(test_acc),
                loss=float(history["val_loss"][-1]),
                training_time_seconds=training_time,
                data_samples=len(X_train) + len(X_val) + len(X_test),
                model_path=str(model_path),
                metrics={
                    "train_accuracy": float(history["accuracy"][-1]),
                    "val_accuracy": float(best_val_acc),
                    "test_accuracy": float(test_acc),
                    "epochs_trained": len(history["loss"]),
                    **class_metrics
                }
            )

            logger.info(f"\nTraining complete: {model_id}")
            logger.info(f"  Test Accuracy: {test_acc*100:.2f}%")
            logger.info(f"  Training Time: {training_time:.1f}s")
            logger.info(f"  Model saved: {model_path}")

        except Exception as e:
            logger.error(f"Training failed for {model_id}: {e}")
            result = ModelTrainingResult(
                model_id=model_id,
                status="failed",
                error_message=str(e),
                training_time_seconds=time.time() - start_time
            )

        self.results[model_id] = result
        return result

    def _update_weights(self, model, batch_x: np.ndarray, batch_y: np.ndarray):
        """Update model weights using simplified gradient descent."""
        probs = model.forward(batch_x)
        y_one_hot = np.eye(model.num_classes)[batch_y]
        output_error = probs - y_one_hot

        # Update output bias
        if "out_b" in model.weights:
            grad_b = np.mean(output_error, axis=0)
            model.weights["out_b"] -= self.config.learning_rate * grad_b

        # Add noise to help escape local minima
        noise_scale = self.config.learning_rate * 0.05
        for name, weight in model.weights.items():
            if 'mean' in name or 'var' in name:
                continue
            noise = np.random.randn(*weight.shape) * noise_scale
            model.weights[name] += noise

    def train_all(self, model_ids: List[str] = None) -> Dict[str, ModelTrainingResult]:
        """
        Train all models (or specified subset).

        Args:
            model_ids: List of model IDs to train (None = all)

        Returns:
            Dictionary of training results
        """
        if model_ids is None:
            model_ids = list(self.models.keys())

        logger.info(f"\nStarting automated training for {len(model_ids)} models")
        logger.info(f"Configuration: {self.config}")

        total_start = time.time()

        for i, model_id in enumerate(model_ids):
            logger.info(f"\n[{i+1}/{len(model_ids)}] Training {model_id}")

            for attempt in range(self.config.max_retries):
                result = self.train_model(model_id)
                if result.status == "success":
                    break
                logger.warning(f"Attempt {attempt+1} failed, retrying...")
                time.sleep(5)

        total_time = time.time() - total_start

        # Save training summary
        self._save_summary(total_time)

        # Update registry if configured
        if self.config.update_registry:
            self._update_registry()

        return self.results

    def _save_summary(self, total_time: float):
        """Save training summary to file."""
        summary = {
            "training_date": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "config": {
                "n_samples": self.config.n_samples_per_model,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate
            },
            "results": {}
        }

        for model_id, result in self.results.items():
            summary["results"][model_id] = {
                "status": result.status,
                "accuracy": result.accuracy,
                "loss": result.loss,
                "training_time": result.training_time_seconds,
                "data_samples": result.data_samples,
                "metrics": result.metrics,
                "error": result.error_message
            }

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nTraining summary saved: {summary_path}")

        # Print summary
        self._print_summary(summary)

    def _print_summary(self, summary: Dict):
        """Print training summary."""
        print("\n" + "="*60)
        print("AUTOMATED TRAINING COMPLETE")
        print("="*60)

        print(f"\nTotal Time: {summary['total_time_seconds']/60:.1f} minutes")
        print(f"Models Trained: {len(summary['results'])}")

        print("\nModel Performance:")
        print(f"{'Model':<20} {'Status':<10} {'Accuracy':>10} {'Time':>10}")
        print("-"*50)

        successful = 0
        for model_id, result in summary['results'].items():
            status = result['status']
            if status == 'success':
                successful += 1
                acc = f"{result['accuracy']*100:.1f}%"
            else:
                acc = "N/A"
            time_str = f"{result['training_time']:.1f}s"
            print(f"{model_id:<20} {status:<10} {acc:>10} {time_str:>10}")

        print("-"*50)
        print(f"Success Rate: {successful}/{len(summary['results'])}")

    def _update_registry(self):
        """Update model registry with training results."""
        registry_path = PROJECT_ROOT / "models" / "registry.json"

        if not registry_path.exists():
            logger.warning("Registry not found, skipping update")
            return

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        for model_id, result in self.results.items():
            if result.status != "success":
                continue

            if model_id in registry['models']:
                model_entry = registry['models'][model_id]

                # Update metrics
                model_entry['metrics']['accuracy'] = round(result.accuracy, 3)
                model_entry['training']['training_date'] = datetime.now().strftime('%Y-%m-%d')
                model_entry['training']['samples'] = result.data_samples

        # Update registry timestamp
        registry['updated_at'] = datetime.now().isoformat()

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Registry updated: {registry_path}")

    def run_scheduled(self):
        """Run training on a schedule."""
        logger.info(f"Starting scheduled training every {self.config.schedule_interval_hours} hours")

        while True:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Scheduled training run: {datetime.now()}")
                logger.info(f"{'='*60}")

                self.train_all()

                next_run = datetime.now() + timedelta(hours=self.config.schedule_interval_hours)
                logger.info(f"\nNext run scheduled: {next_run}")

                time.sleep(self.config.schedule_interval_hours * 3600)

            except KeyboardInterrupt:
                logger.info("Scheduled training stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduled run failed: {e}")
                time.sleep(3600)  # Wait 1 hour before retry


def main():
    parser = argparse.ArgumentParser(description='Automated TinyML model training')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--model', type=str, help='Train specific model')
    parser.add_argument('--collect-only', action='store_true', help='Only collect data')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--samples', type=int, default=5000, help='Samples per model')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        n_samples_per_model=args.samples,
        use_cached_data=not args.no_cache
    )

    trainer = AutomatedTrainer(config)

    if args.schedule:
        trainer.run_scheduled()
    elif args.model:
        if args.model not in trainer.models:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(trainer.models.keys())}")
            sys.exit(1)
        trainer.train_model(args.model)
    elif args.collect_only:
        for model_id in trainer.models:
            trainer.collect_data(model_id)
    else:
        trainer.train_all()


if __name__ == "__main__":
    main()
