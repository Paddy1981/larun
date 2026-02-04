#!/usr/bin/env python3
"""
LARUN TinyML Model Training
============================
Training script with proper backpropagation and Adam optimizer.
No TensorFlow required - pure NumPy implementation.

Features:
- Proper gradient computation via backpropagation
- Adam optimizer for fast convergence
- Early stopping and learning rate decay
- Dropout regularization
- Batch normalization

Usage:
    python train_models.py                        # Train all models
    python train_models.py --model EXOPLANET-001  # Train specific model
    python train_models.py --epochs 50            # Custom epochs
    python train_models.py --list                 # List available models
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Direct imports to avoid TensorFlow dependencies
from src.model.specialized_models import (
    MODEL_CLASSES, MODEL_SPECS, get_model, BaseNumpyModel
)
from src.model.data_generators import (
    DATA_GENERATORS, DatasetConfig, get_generator
)
from src.model.trainer import NeuralNetworkTrainer, TrainingConfig


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class TinyMLTrainer:
    """Lightweight trainer for TinyML astronomical models."""

    def __init__(self, output_dir: str = "models/trained", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}

    def log(self, msg: str, end="\n"):
        if self.verbose:
            print(msg, end=end)

    def print_banner(self):
        """Print training banner."""
        self.log(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                                ║
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗               ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║               ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║               ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║               ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║               ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝               ║
║                                                                ║
║            TinyML Model Training System                        ║
║         Democratizing Space Discovery                          ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")

    def train_model(self, model_id: str, epochs: int = 50,
                    batch_size: int = 32, n_samples: int = 5000,
                    learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train a single TinyML model using proper backpropagation.

        Args:
            model_id: Model identifier (e.g., "EXOPLANET-001")
            epochs: Number of training epochs
            batch_size: Training batch size
            n_samples: Number of training samples
            learning_rate: Learning rate

        Returns:
            Training results dictionary
        """
        self.log(f"\n{'='*60}")
        self.log(f"{Colors.BOLD}Training: {model_id}{Colors.ENDC}")
        self.log(f"{'='*60}")

        start_time = time.time()

        # Get model and spec
        model = get_model(model_id)
        spec = MODEL_SPECS[model_id]

        self.log(f"\n{Colors.CYAN}Model: {spec.name}{Colors.ENDC}")
        self.log(f"Description: {spec.description[:60]}...")
        self.log(f"Input shape: {spec.input_shape}")
        self.log(f"Output classes: {len(spec.output_classes)}")

        # Generate training data
        self.log(f"\n{Colors.YELLOW}Generating {n_samples} synthetic samples...{Colors.ENDC}")
        generator = get_generator(model_id)
        config = DatasetConfig(n_samples=n_samples, noise_level=0.02, seed=42)
        X, y = generator.generate_dataset(config)

        self.log(f"Data shape: X={X.shape}, y={y.shape}")

        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        self.log(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

        # Split data (80% train, 20% validation)
        n_val = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.log(f"Train: {len(X_train)}, Validation: {len(X_val)}")

        # Configure trainer with proper backpropagation
        train_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=0.2,
            weight_decay=0.0001,
            early_stopping_patience=15,
            lr_decay_patience=7,
            lr_decay_factor=0.5
        )

        # Create trainer with Adam optimizer
        trainer = NeuralNetworkTrainer(model, train_config)

        self.log(f"\n{Colors.YELLOW}Training with Adam optimizer and backpropagation...{Colors.ENDC}")
        self.log(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
        self.log(f"Dropout: {train_config.dropout_rate}, Weight decay: {train_config.weight_decay}\n")

        # Train model
        history = trainer.fit(X_train, y_train, X_val, y_val, verbose=self.verbose)

        # Final evaluation
        self.log(f"\n{Colors.GREEN}Training complete!{Colors.ENDC}")

        final_preds, final_confs = model.predict(X_val)
        final_acc = np.mean(final_preds == y_val)
        best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else final_acc

        # Per-class accuracy
        class_metrics = {}
        for idx, class_name in enumerate(spec.output_classes):
            mask = y_val == idx
            if mask.sum() > 0:
                class_acc = np.mean(final_preds[mask] == idx)
                class_metrics[class_name] = float(class_acc)

        # Save model
        model_path = self.output_dir / f"{model_id}_weights.npz"
        model.save(str(model_path))
        self.log(f"Model saved: {model_path}")

        # Get model size
        size_info = model.get_model_size()
        training_time = time.time() - start_time

        results = {
            "model_id": model_id,
            "name": spec.name,
            "training_time_seconds": round(training_time, 2),
            "epochs": len(history["loss"]),  # Actual epochs (may be less due to early stopping)
            "n_samples": n_samples,
            "final_accuracy": round(float(final_acc), 4),
            "best_val_accuracy": round(float(best_val_acc), 4),
            "class_metrics": class_metrics,
            "model_size_kb": round(size_info["size_int8_kb"], 2),
            "total_parameters": size_info["total_parameters"],
            "model_path": str(model_path)
        }

        self.results[model_id] = results
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print training summary."""
        self.log(f"""
{Colors.BOLD}{'─'*50}
SUMMARY: {results['model_id']}
{'─'*50}{Colors.ENDC}
  Training Time: {results['training_time_seconds']:.1f}s
  Final Accuracy: {Colors.GREEN}{results['final_accuracy']*100:.1f}%{Colors.ENDC}
  Model Size: {results['model_size_kb']:.1f} KB
  Parameters: {results['total_parameters']:,}

  Per-class accuracy:""")

        for cls, acc in results['class_metrics'].items():
            color = Colors.GREEN if acc >= 0.7 else Colors.YELLOW if acc >= 0.5 else Colors.RED
            self.log(f"    {cls:20} {color}{acc*100:5.1f}%{Colors.ENDC}")

    def train_all(self, epochs: int = 50, n_samples: int = 5000) -> Dict[str, Dict]:
        """Train all 8 TinyML models."""
        self.print_banner()
        self.log(f"{Colors.BOLD}Training all 8 TinyML models...{Colors.ENDC}")

        total_start = time.time()

        for i, model_id in enumerate(MODEL_CLASSES.keys()):
            self.log(f"\n[{i+1}/8] ", end="")
            try:
                self.train_model(model_id, epochs=epochs, n_samples=n_samples)
            except Exception as e:
                self.log(f"{Colors.RED}Error training {model_id}: {e}{Colors.ENDC}")

        total_time = time.time() - total_start

        # Save summary
        summary = {
            "training_date": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 2),
            "epochs": epochs,
            "samples_per_model": n_samples,
            "models": self.results
        }

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self._print_final_summary(summary)

        return self.results

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary for all models."""
        self.log(f"""
{Colors.BOLD}{'═'*60}
              ALL MODELS TRAINED SUCCESSFULLY!
{'═'*60}{Colors.ENDC}

  Total Time: {summary['total_time_seconds']/60:.1f} minutes
  Models Trained: {len(summary['models'])}

{Colors.BOLD}Performance Summary:{Colors.ENDC}
  {'Model':<18} {'Accuracy':>10} {'Size (KB)':>10} {'Parameters':>12}
  {'-'*18} {'-'*10} {'-'*10} {'-'*12}""")

        total_size = 0
        total_params = 0

        for model_id, r in summary['models'].items():
            acc = r['final_accuracy'] * 100
            size = r['model_size_kb']
            params = r['total_parameters']
            total_size += size
            total_params += params

            color = Colors.GREEN if acc >= 70 else Colors.YELLOW if acc >= 50 else Colors.RED
            self.log(f"  {model_id:<18} {color}{acc:>9.1f}%{Colors.ENDC} {size:>10.1f} {params:>12,}")

        self.log(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*12}")
        self.log(f"  {'TOTAL':<18} {'':>10} {total_size:>10.1f} {total_params:>12,}")

        self.log(f"""
{Colors.CYAN}Output: {self.output_dir}{Colors.ENDC}
{Colors.GREEN}Models ready for deployment!{Colors.ENDC}
""")


def list_models():
    """List all available models."""
    print(f"\n{Colors.BOLD}Available TinyML Models:{Colors.ENDC}\n")
    print(f"  {'Model ID':<18} {'Name':<35} {'Classes':>8}")
    print(f"  {'-'*18} {'-'*35} {'-'*8}")

    for model_id, spec in MODEL_SPECS.items():
        print(f"  {model_id:<18} {spec.name[:35]:<35} {len(spec.output_classes):>8}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Train LARUN TinyML astronomical models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py                          # Train all models
  python train_models.py --model EXOPLANET-001    # Train specific model
  python train_models.py --epochs 100 --samples 10000  # Custom training
  python train_models.py --list                   # List available models
"""
    )
    parser.add_argument('--model', type=str, help='Train specific model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples (default: 5000)')
    parser.add_argument('--output', type=str, default='models/trained', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    trainer = TinyMLTrainer(output_dir=args.output, verbose=not args.quiet)

    if args.model:
        if args.model not in MODEL_CLASSES:
            print(f"{Colors.RED}Unknown model: {args.model}{Colors.ENDC}")
            print(f"Available: {list(MODEL_CLASSES.keys())}")
            sys.exit(1)
        trainer.train_model(args.model, epochs=args.epochs, n_samples=args.samples)
    else:
        trainer.train_all(epochs=args.epochs, n_samples=args.samples)


if __name__ == "__main__":
    main()
