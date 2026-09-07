"""Supervised image models: a small custom CNN built for single-channel input,
and a transfer-learning path over ResNet50/EfficientNetB0/VGG16 (all three
expect 3-channel input, so we replicate the grayscale channel rather than
retraining a stem from scratch — not worth the extra params for this dataset size).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_BACKBONES = {
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
}


def build_custom_cnn(img_size: int = 224, channels: int = 1) -> models.Model:
    """A from-scratch CNN sized for a few thousand training patches, not
    ImageNet — four conv blocks is already plenty before we start overfitting.
    """
    inputs = layers.Input(shape=(img_size, img_size, channels))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])
    return model


@tf.keras.utils.register_keras_serializable(package="cellscan")
class ReplicateChannels(layers.Layer):
    """Grayscale -> 3ch inside the graph, so preprocessing stays 1-channel
    everywhere else and only the model itself knows it needs 3.

    Registered for serialization on purpose — a plain unregistered
    subclass round-trips fine within one process but fails to reload with
    "Unknown layer: 'ReplicateChannels'" the moment you save the model and
    load it back in a fresh process (which is every real deployment path:
    train_image.py saves it, image_service.py loads it later). Found this
    the hard way generating throwaway demo weights for the dashboard —
    it would have bitten the first real trained model too.
    """
    def call(self, x):
        return tf.repeat(x, repeats=3, axis=-1)


@tf.keras.utils.register_keras_serializable(package="cellscan")
class PreprocessForBackbone(layers.Layer):
    """Applies the ImageNet preprocessing a given backbone expects
    (channel-wise mean subtraction, RGB<->BGR reordering, etc — different
    per architecture). A plain `layers.Lambda(lambda t: preprocess_fn(...))`
    has the same serialization problem as the unregistered ReplicateChannels
    above (closures over an external function aren't reconstructable from
    config), so this looks the function up by name at call time instead of
    capturing it.
    """
    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name

    def call(self, x):
        _, preprocess_fn = _BACKBONES[self.backbone_name]
        return preprocess_fn(x * 255.0)

    def get_config(self):
        config = super().get_config()
        config["backbone_name"] = self.backbone_name
        return config


def build_transfer_model(backbone_name: str, img_size: int = 224, fine_tune_last_n: int = 20) -> models.Model:
    if backbone_name not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone_name}, choose from {list(_BACKBONES)}")
    backbone_cls, _ = _BACKBONES[backbone_name]

    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = ReplicateChannels()(inputs)
    x = PreprocessForBackbone(backbone_name)(x)

    base = backbone_cls(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False
    if fine_tune_last_n:
        for layer in base.layers[-fine_tune_last_n:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"transfer_{backbone_name}")
    model.compile(
        optimizer=optimizers.Adam(1e-4),  # lower LR than the custom CNN — we're fine-tuning pretrained weights
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def default_callbacks(checkpoint_path: Path, patience: int = 6) -> list:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_recall", mode="max", save_best_only=True),
    ]


def train(model: models.Model, X_train, y_train, X_val, y_val, checkpoint_path: Path, epochs: int = 40, batch_size: int = 32, augmentor=None):
    """augmentor: an ImageDataGenerator (see image_preprocessing.make_augmentor) —
    when given, trains off its .flow() instead of the raw arrays directly.
    Only the training split gets augmented; X_val/y_val are always evaluated
    as-is, since validation is supposed to measure performance on real,
    un-augmented data.
    """
    class_weights = get_class_weights(y_train)
    logger.info("Training %s with class weights %s%s", model.name, class_weights, " (augmented)" if augmentor else "")
    if augmentor is not None:
        history = model.fit(
            augmentor.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            class_weight=class_weights,
            callbacks=default_callbacks(checkpoint_path),
            verbose=2,
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=default_callbacks(checkpoint_path),
            verbose=2,
        )
    return history
