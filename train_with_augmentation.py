#!/usr/bin/env python3
"""
LARUN Training with Data Augmentation
======================================
Retrain the model using data augmentation to improve accuracy.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LARUN Training with Data Augmentation                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Load existing training data
    data_path = Path("models/real/real_training_data.npz")
    if not data_path.exists():
        data_path = Path("data/real/training_data.npz")

    if not data_path.exists():
        print("❌ No training data found. Run /train first.")
        return

    print(f"Loading training data from {data_path}...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    print(f"Original data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    # Import augmentation
    from augmentation import LightCurveAugmenter, AugmentationConfig, compute_class_weights

    # Configure augmentation
    config = AugmentationConfig(
        noise_level=0.002,
        time_shift_max=0.15,
        flux_scale_range=(0.95, 1.05),
        enable_noise=True,
        enable_time_shift=True,
        enable_flux_scale=True,
        enable_dropout=False
    )

    augmenter = LightCurveAugmenter(config)

    # Use BALANCED augmentation (key fix for class imbalance!)
    print("\nApplying CLASS-BALANCED augmentation...")
    print("(Oversampling minority classes to match majority class)")
    X_aug, y_aug = augmenter.augment_batch_balanced(X, y)

    print(f"Balanced data: {X_aug.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y_aug)}")

    # Compute class weights for extra protection
    class_weights = compute_class_weights(y_aug)
    print(f"Class weights: {class_weights}")

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Build model
    print("\nBuilding model...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Improved model with batch normalization and residual-like connections
    inputs = keras.Input(shape=(X_train.shape[1], 1))

    # Block 1
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='larun_augmented')
    model.summary()

    # Compile with learning rate schedule
    initial_lr = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            'models/real/astro_tinyml_augmented.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    # Train
    print("\nTraining with augmented data...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,  # Use class weights for balanced learning
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✓ Final Validation Loss: {val_loss:.4f}")

    # Save model
    model.save('models/real/astro_tinyml_augmented.h5')
    print(f"\n✓ Model saved to: models/real/astro_tinyml_augmented.h5")

    # Export to TFLite
    print("\nExporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('models/real/astro_tinyml_augmented.tflite', 'wb') as f:
        f.write(tflite_model)

    # Quantized version
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant = converter.convert()

    with open('models/real/astro_tinyml_augmented_int8.tflite', 'wb') as f:
        f.write(tflite_quant)

    print(f"✓ TFLite model: {len(tflite_model)/1024:.1f} KB")
    print(f"✓ Quantized model: {len(tflite_quant)/1024:.1f} KB")

    # Compare with original
    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("=" * 60)

    original_path = Path("models/real/astro_tinyml_real.h5")
    if original_path.exists():
        original_model = keras.models.load_model(original_path)
        orig_loss, orig_acc = original_model.evaluate(X_val, y_val, verbose=0)
        print(f"Original model accuracy: {orig_acc*100:.2f}%")
        print(f"Augmented model accuracy: {val_acc*100:.2f}%")
        improvement = (val_acc - orig_acc) * 100
        if improvement > 0:
            print(f"✓ Improvement: +{improvement:.2f}%")
        else:
            print(f"  Difference: {improvement:.2f}%")

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
