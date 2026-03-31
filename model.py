"""
model.py - CNN Model Architecture for Plant Disease Detection
RISE Internship - Project 8
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2


def create_custom_cnn(input_shape=(128, 128, 3), num_classes=1):
    """
    Custom CNN model for binary plant disease classification.
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: 1 for binary (healthy/diseased), >1 for multi-class
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        # Classifier Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax'),
    ], name="PlantDiseaseCNN")

    loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model


def create_transfer_learning_model(input_shape=(128, 128, 3), num_classes=1):
    """
    Transfer learning model using MobileNetV2 (lightweight, mobile-friendly).
    Great for production deployment on mobile/edge devices.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze base layers initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="PlantDiseaseMobileNetV2")

    loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100, learning_rate=1e-5):
    """
    Unfreeze top layers of the base model for fine-tuning.
    Call this after initial training converges.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    print(f"Fine-tuning: {len(model.trainable_variables)} trainable variables")
    return model


def get_model_summary(model):
    """Print model summary with parameter count."""
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Approximate Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
