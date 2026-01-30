#!/usr/bin/env python3
"""
LARUN Training with Class Weights
==================================
Train with class weights to handle imbalance WITHOUT augmentation.

The original model at 88.51% is actually quite good given limited data.
This script tests if class weights alone can improve minority class performance.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LARUN Training with Class Weights (No Augmentation)        ║")
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

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    # Compute class weights
    from augmentation import compute_class_weights
    class_weights = compute_class_weights(y)
    print(f"\nClass weights: {class_weights}")

    # Split data (use original distribution)
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Train distribution: {np.bincount(y_train)}")
    print(f"Val distribution: {np.bincount(y_val)}")

    # Build model - SIMPLER architecture to avoid overfitting
    print("\nBuilding simplified model...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(X_train.shape[1], 1))

    # Simpler model with more regularization
    x = layers.Conv1D(16, 7, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(32, 5, padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='softmax')(x)  # 5 classes

    model = keras.Model(inputs, outputs, name='larun_weighted')
    model.summary()

    # Compile with lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=30,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            monitor='val_loss'
        ),
    ]

    # Train with class weights
    print("\nTraining with class weights...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,  # Smaller batch for small dataset
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✓ Final Validation Loss: {val_loss:.4f}")

    # Save model
    model.save('models/real/astro_tinyml_weighted.h5')
    print(f"\n✓ Model saved to: models/real/astro_tinyml_weighted.h5")

    # Export to TFLite
    print("\nExporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('models/real/astro_tinyml_weighted.tflite', 'wb') as f:
        f.write(tflite_model)

    # Quantized version
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant = converter.convert()

    with open('models/real/astro_tinyml_weighted_int8.tflite', 'wb') as f:
        f.write(tflite_quant)

    print(f"✓ TFLite model: {len(tflite_model)/1024:.1f} KB")
    print(f"✓ Quantized model: {len(tflite_quant)/1024:.1f} KB")

    # Per-class accuracy analysis
    print("\n" + "=" * 60)
    print("PER-CLASS ANALYSIS")
    print("=" * 60)

    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    class_names = ['Confirmed', 'False Positive', 'Candidate', 'Binary', 'Variable']

    for i, name in enumerate(class_names):
        mask = y_val == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_val[mask])
            print(f"  Class {i} ({name}): {class_acc*100:.1f}% ({np.sum(mask)} samples)")

    # Compare with original
    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("=" * 60)

    original_path = Path("models/real/astro_tinyml_real.h5")
    if original_path.exists():
        try:
            original_model = keras.models.load_model(original_path, compile=False)
            orig_pred = original_model.predict(X_val, verbose=0)
            orig_classes = np.argmax(orig_pred, axis=1)
            orig_acc = np.mean(orig_classes == y_val)
            print(f"Original model accuracy: {orig_acc*100:.2f}%")
            print(f"Weighted model accuracy: {val_acc*100:.2f}%")
            improvement = (val_acc - orig_acc) * 100
            if improvement > 0:
                print(f"✓ Improvement: +{improvement:.2f}%")
            else:
                print(f"  Difference: {improvement:.2f}%")
        except Exception as e:
            print(f"Could not load original model: {e}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("To significantly improve accuracy, we need MORE TRAINING DATA.")
    print("Options:")
    print("  1. Use Google Colab notebook to fetch more light curves")
    print("  2. Download from MAST archive directly")
    print("  3. Use the /train command with more target stars")
    print("\nCurrent dataset has only 108 samples total - too small for deep learning.")
    print("Target: 1000+ samples per class for robust training.")

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
