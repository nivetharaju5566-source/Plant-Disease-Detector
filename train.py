"""
train.py - Training Pipeline for Plant Disease Detection CNN
RISE Internship - Project 8

Usage:
    python train.py                        # default settings
    python train.py --model mobilenet      # use transfer learning
    python train.py --epochs 20 --lr 0.001
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from model import create_custom_cnn, create_transfer_learning_model, unfreeze_for_fine_tuning
from utils.data_utils import (
    get_data_generators, check_dataset,
    plot_training_history, evaluate_model
)

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR    = 'data'
OUTPUT_DIR  = 'outputs'
MODEL_PATH  = 'models/plant_disease_model.h5'
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32


def parse_args():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection CNN')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'mobilenet'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size (square)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune MobileNet after initial training (mobilenet only)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    return parser.parse_args()


def create_callbacks(model_path):
    """Create training callbacks."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    callbacks = [
        # Save best model based on val_accuracy
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        # Stop if no improvement for 5 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir='logs',
            histogram_freq=1
        ),
    ]
    return callbacks


def save_training_report(history, metrics, model_type, args):
    """Save training summary to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        'model_type': model_type,
        'epochs_trained': len(history.history['accuracy']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'config': vars(args)
    }
    if metrics:
        for k, v in metrics.items():
            report[f'eval_{k}'] = float(v)

    report_path = os.path.join(OUTPUT_DIR, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Training report saved to {report_path}")
    return report


def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)

    print("=" * 60)
    print("🌿 Plant Disease Detection - Training Pipeline")
    print("=" * 60)
    print(f"  Model     : {args.model}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Img Size  : {img_size}")
    print(f"  LR        : {args.lr}")
    print(f"  Augment   : {not args.no_augment}")
    print("=" * 60)

    # ── 1. Dataset validation ────────────────────────────────────────────────
    counts = check_dataset(DATA_DIR)
    if counts is None or sum(counts.values()) == 0:
        print("\n❌ ERROR: No images found in data/ folder.")
        print("   Please add images to data/healthy/ and data/diseased/")
        print("   Download dataset from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        return

    # ── 2. Data generators ───────────────────────────────────────────────────
    print("\n📦 Creating data generators...")
    train_gen, val_gen, class_indices = get_data_generators(
        DATA_DIR,
        img_size=img_size,
        batch_size=args.batch_size,
        augment=not args.no_augment
    )
    print(f"   Class mapping: {class_indices}")
    print(f"   Train samples : {train_gen.samples}")
    print(f"   Val samples   : {val_gen.samples}")

    # ── 3. Model creation ────────────────────────────────────────────────────
    print(f"\n🏗️  Building {args.model} model...")
    base_model = None
    if args.model == 'mobilenet':
        model, base_model = create_transfer_learning_model(
            input_shape=(*img_size, 3), num_classes=1
        )
    else:
        model = create_custom_cnn(input_shape=(*img_size, 3), num_classes=1)

    model.summary()

    # ── 4. Training ──────────────────────────────────────────────────────────
    callbacks = create_callbacks(MODEL_PATH)
    print(f"\n🚀 Starting training for {args.epochs} epochs...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── 5. Fine-tuning (MobileNet only) ─────────────────────────────────────
    if args.fine_tune and args.model == 'mobilenet' and base_model is not None:
        print("\n🔧 Fine-tuning MobileNetV2 top layers...")
        model = unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100, learning_rate=1e-5)
        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        # Extend history for plotting
        for key in history.history:
            history.history[key].extend(fine_tune_history.history.get(key, []))

    # ── 6. Evaluation ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics, y_true, y_pred = evaluate_model(model, val_gen)

    # ── 7. Plot results ──────────────────────────────────────────────────────
    plot_training_history(history, save_path=os.path.join(OUTPUT_DIR, 'training_history.png'))

    # ── 8. Save final model ──────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    # Also save as SavedModel format (better for deployment)
    saved_model_path = MODEL_PATH.replace('.h5', '_savedmodel')
    model.save(saved_model_path)
    print(f"✅ SavedModel format saved to {saved_model_path}")

    # ── 9. Report ────────────────────────────────────────────────────────────
    report = save_training_report(history, metrics, args.model, args)

    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"   Best Val Accuracy : {report['best_val_accuracy']:.4f}")
    print(f"   Model saved to    : {MODEL_PATH}")
    print(f"   Next step: streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
