"""
utils/data_utils.py - Data Preprocessing & Augmentation Utilities
RISE Internship - Project 8: Plant Disease Detection
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASSES = ['diseased', 'healthy']   # folder names in data/


# ─── Data Generators ────────────────────────────────────────────────────────

def get_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                        val_split=0.2, augment=True):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        data_dir: Root data directory containing class subfolders
        img_size: Tuple (height, width)
        batch_size: Batch size for training
        val_split: Fraction for validation
        augment: Whether to apply augmentation to training set
    
    Returns:
        train_gen, val_gen, class_indices
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=val_split,
            # Augmentation techniques
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=val_split
        )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen, train_gen.class_indices


def preprocess_single_image(img_path, img_size=IMG_SIZE):
    """
    Preprocess a single image for inference.
    
    Args:
        img_path: Path to image or PIL Image or numpy array
        img_size: Target size tuple
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    if isinstance(img_path, str):
        img = Image.open(img_path).convert('RGB')
    elif isinstance(img_path, np.ndarray):
        img = Image.fromarray(img_path).convert('RGB')
    else:
        img = img_path.convert('RGB')

    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)   # add batch dim
    return arr


def check_dataset(data_dir):
    """
    Validate dataset structure and report class balance.
    
    Returns:
        dict with class counts and balance info
    """
    counts = {}
    total = 0

    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return None

    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            counts[cls] = len(imgs)
            total += len(imgs)

    print("\n📊 Dataset Summary:")
    print("=" * 40)
    for cls, cnt in counts.items():
        pct = (cnt / total * 100) if total > 0 else 0
        print(f"  {cls:15s}: {cnt:5d} images  ({pct:.1f}%)")
    print(f"  {'TOTAL':15s}: {total:5d} images")
    print("=" * 40)

    if total < 100:
        print("⚠️  WARNING: Very small dataset. Consider collecting more images.")
    elif total < 500:
        print("⚠️  Dataset is small. Data augmentation is important.")
    else:
        print("✅ Dataset size looks good!")

    return counts


# ─── Visualization ──────────────────────────────────────────────────────────

def plot_sample_images(data_dir, n=8, figsize=(16, 4)):
    """Display sample images from each class."""
    fig, axes = plt.subplots(2, n // 2, figsize=figsize)
    fig.suptitle("Sample Dataset Images", fontsize=14, fontweight='bold')

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for row, cls in enumerate(classes[:2]):
        cls_dir = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:n // 2]
        for col, fname in enumerate(imgs):
            img = Image.open(os.path.join(cls_dir, fname)).resize((128, 128))
            ax = axes[row][col] if len(classes) > 1 else axes[col]
            ax.imshow(np.array(img))
            ax.set_title(cls.capitalize(), color='green' if cls == 'healthy' else 'red',
                         fontweight='bold', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Sample images saved to outputs/sample_images.png")


def plot_training_history(history, save_path='outputs/training_history.png'):
    """Plot training & validation accuracy/loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History - Plant Disease Detection CNN',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('accuracy', 'val_accuracy', 'Accuracy', 'royalblue', 'cornflowerblue'),
        ('loss', 'val_loss', 'Loss', 'tomato', 'lightsalmon'),
        ('precision', 'val_precision', 'Precision & Recall', 'mediumseagreen', 'lightgreen'),
    ]

    for ax, (train_key, val_key, title, tc, vc) in zip(axes, metrics):
        if train_key in history.history:
            epochs = range(1, len(history.history[train_key]) + 1)
            ax.plot(epochs, history.history[train_key], color=tc,
                    linewidth=2, marker='o', markersize=4, label=f'Train {title}')
            if val_key in history.history:
                ax.plot(epochs, history.history[val_key], color=vc,
                        linewidth=2, linestyle='--', marker='s', markersize=4,
                        label=f'Val {title}')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Optionally overlay recall if available
        if train_key == 'precision' and 'recall' in history.history:
            ax.plot(epochs, history.history['recall'], color='darkorange',
                    linewidth=2, marker='^', markersize=4, label='Train Recall')
            if 'val_recall' in history.history:
                ax.plot(epochs, history.history['val_recall'], color='moccasin',
                        linewidth=2, linestyle='--', marker='v', markersize=4,
                        label='Val Recall')
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          save_path='outputs/confusion_matrix.png'):
    """Plot a styled confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if class_names is None:
        class_names = ['Diseased', 'Healthy']

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, annot_kws={"size": 14, "weight": "bold"})
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Plant Disease Detection',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"✅ Confusion matrix saved to {save_path}")


def evaluate_model(model, val_gen, threshold=0.5):
    """
    Evaluate model and compute precision, recall, F1.
    Returns dict of metrics.
    """
    print("\n🔍 Evaluating model on validation set...")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = (preds.flatten() > threshold).astype(int)
    y_true = val_gen.classes

    results = model.evaluate(val_gen, verbose=0)
    metric_names = model.metrics_names

    print("\n📊 Evaluation Results:")
    print("=" * 40)
    metrics = {}
    for name, val in zip(metric_names, results):
        print(f"  {name:15s}: {val:.4f}")
        metrics[name] = val

    plot_confusion_matrix(y_true, y_pred)
    return metrics, y_true, y_pred
