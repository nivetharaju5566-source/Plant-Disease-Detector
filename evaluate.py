"""
evaluate.py - Standalone Model Evaluation Script
RISE Internship - Project 8: Plant Disease Detection

Usage:
    python evaluate.py                            # evaluate on data/
    python evaluate.py --data_dir my_test_data/
    python evaluate.py --image path/to/leaf.jpg   # single image
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.data_utils import (
    get_data_generators, preprocess_single_image,
    plot_confusion_matrix, check_dataset
)


MODEL_PATH = 'models/plant_disease_model.h5'
DATA_DIR   = 'data'
OUTPUT_DIR = 'outputs'


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Plant Disease Detection Model')
    p.add_argument('--model', default=MODEL_PATH, help='Path to .h5 model file')
    p.add_argument('--data_dir', default=DATA_DIR, help='Data directory for batch eval')
    p.add_argument('--image', default=None, help='Single image path for quick test')
    p.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    return p.parse_args()


def predict_single(model, img_path, threshold=0.5):
    """Predict disease for a single image and display result."""
    arr = preprocess_single_image(img_path)
    raw = float(model.predict(arr, verbose=0)[0][0])

    is_healthy = raw > threshold
    confidence = raw if is_healthy else (1 - raw)
    label = "HEALTHY 🌱" if is_healthy else "DISEASED 🍂"

    # Display
    img = Image.open(img_path).resize((256, 256))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.array(img))
    color = '#2e7d32' if is_healthy else '#e65100'
    ax.set_title(f"{label}\nConfidence: {confidence:.2%}", fontsize=14,
                 fontweight='bold', color=color, pad=12)
    ax.axis('off')

    border = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                     fill=False, edgecolor=color, linewidth=5)
    ax.add_patch(border)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'single_prediction.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n🌿 Prediction for: {img_path}")
    print(f"   Result     : {label}")
    print(f"   Confidence : {confidence:.4f} ({confidence:.2%})")
    print(f"   Raw output : {raw:.6f}")
    print(f"   Saved to   : {out_path}")
    return label, confidence


def batch_evaluate(model, data_dir, threshold=0.5):
    """Evaluate model on entire dataset and show metrics."""
    print(f"\n📦 Loading data from: {data_dir}")
    check_dataset(data_dir)

    _, val_gen, class_indices = get_data_generators(data_dir, val_split=0.3)
    print(f"   Class indices: {class_indices}")

    print("\n🔍 Running predictions...")
    val_gen.reset()
    preds_raw = model.predict(val_gen, verbose=1)
    y_pred = (preds_raw.flatten() > threshold).astype(int)
    y_true = val_gen.classes

    # Metrics
    results = model.evaluate(val_gen, verbose=0)
    print("\n📊 Evaluation Metrics:")
    print("=" * 40)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:15s}: {val:.4f}")

    # Confusion matrix
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred,
                          save_path=os.path.join(OUTPUT_DIR, 'eval_confusion_matrix.png'))

    # Find worst predictions (most confident wrong answers)
    wrong_idx = np.where(y_pred != y_true)[0]
    print(f"\n  Wrong predictions: {len(wrong_idx)} / {len(y_true)}")


def main():
    args = parse_args()

    print("=" * 60)
    print("🌿 Plant Disease Detection - Evaluation")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Run `python train.py` first.")
        return

    print(f"📂 Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print("✅ Model loaded successfully")

    if args.image:
        predict_single(model, args.image, args.threshold)
    else:
        batch_evaluate(model, args.data_dir, args.threshold)


if __name__ == '__main__':
    main()
