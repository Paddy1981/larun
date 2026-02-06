#!/usr/bin/env python3
"""
Parallel Model Training
=======================
Train all 8 TinyML models in parallel for faster execution.
"""

import multiprocessing as mp
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.model.specialized_models import MODEL_CLASSES, MODEL_SPECS, get_model
from src.model.data_generators import DatasetConfig, get_generator
from src.model.trainer import NeuralNetworkTrainer, TrainingConfig


def train_single_model(args) -> Dict[str, Any]:
    """Train a single model (for multiprocessing)."""
    model_id, epochs, n_samples = args

    try:
        print(f"[{model_id}] Starting training...")
        start_time = time.time()

        # Get model and spec
        model = get_model(model_id)
        spec = MODEL_SPECS[model_id]

        # Generate data
        generator = get_generator(model_id)
        config = DatasetConfig(n_samples=n_samples, noise_level=0.02, seed=42)
        X, y = generator.generate_dataset(config)

        # Split data
        n_val = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Training config
        train_config = TrainingConfig(
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            dropout_rate=0.2,
            weight_decay=0.0001,
            early_stopping_patience=10,
            lr_decay_patience=5,
            lr_decay_factor=0.5
        )

        # Train
        trainer = NeuralNetworkTrainer(model, train_config)
        history = trainer.fit(X_train, y_train, X_val, y_val, verbose=False)

        # Evaluate
        preds, confs = model.predict(X_val)
        accuracy = float(np.mean(preds == y_val))
        best_acc = max(history["val_accuracy"]) if history["val_accuracy"] else accuracy

        # Save model
        output_dir = Path("models/trained")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_id}_weights.npz"
        model.save(str(model_path))

        training_time = time.time() - start_time
        size_info = model.get_model_size()

        result = {
            "model_id": model_id,
            "name": spec.name,
            "accuracy": round(accuracy, 4),
            "best_accuracy": round(best_acc, 4),
            "epochs_trained": len(history["loss"]),
            "training_time": round(training_time, 1),
            "size_kb": round(size_info["size_int8_kb"], 1),
            "parameters": size_info["total_parameters"],
            "status": "success"
        }

        print(f"[{model_id}] Done! Accuracy: {accuracy*100:.1f}% in {training_time:.1f}s")
        return result

    except Exception as e:
        print(f"[{model_id}] Error: {str(e)}")
        return {
            "model_id": model_id,
            "status": "error",
            "error": str(e)
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel model training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL MODEL TRAINING")
    print("=" * 60)
    print(f"Models: {len(MODEL_CLASSES)}")
    print(f"Epochs: {args.epochs}, Samples: {args.samples}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    start_time = time.time()

    # Prepare training args for each model
    training_args = [
        (model_id, args.epochs, args.samples)
        for model_id in MODEL_CLASSES.keys()
    ]

    # Train in parallel
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(train_single_model, training_args)

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print(f"{'Model':<18} {'Accuracy':>10} {'Size':>10} {'Time':>10}")
    print("-" * 50)

    successful = []
    for r in results:
        if r["status"] == "success":
            successful.append(r)
            print(f"{r['model_id']:<18} {r['accuracy']*100:>9.1f}% {r['size_kb']:>9.1f}KB {r['training_time']:>9.1f}s")
        else:
            print(f"{r['model_id']:<18} {'ERROR':>10} - {r.get('error', 'Unknown')[:30]}")

    # Save results
    summary = {
        "total_time": total_time,
        "epochs": args.epochs,
        "samples": args.samples,
        "models": results
    }

    with open("models/trained/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Successful: {len(successful)}/{len(results)} models")
    print(f"Results saved to: models/trained/training_summary.json")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
