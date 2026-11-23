"""
MobileNetV2 binary classifier (Healthy vs Heart Rot) with ImageNet backbone weights.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

CLASS_NAMES = ("Healthy", "Heart_Rot")


def build_binary_mobilenetv2(
    *,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    trainable_base: bool = False,
) -> keras.Model:
    """
    Binary classification head on MobileNetV2 (weights='imagenet', include_top=False).
    Output: softmax over 2 classes [Healthy, Heart_Rot].
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="probs")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_heart_rot_binary")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_or_build_model(weights_path: str | None = None) -> tuple[keras.Model, bool]:
    """
    Load full saved model if path exists; otherwise build architecture with ImageNet backbone
    and optionally load only top weights if a weights file is provided for the full model.

    Returns (model, weights_loaded).
    """
    env_path = weights_path or os.environ.get("MODEL_WEIGHTS_PATH", "")
    path = Path(env_path) if env_path else None

    if path and path.is_file():
        try:
            model = keras.models.load_model(str(path), compile=False)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model, True
        except Exception:
            model = build_binary_mobilenetv2()
            try:
                model.load_weights(str(path))
                return model, True
            except Exception:
                pass

    return build_binary_mobilenetv2(), False


def infer(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """
    Single forward pass without `predict()` batching overhead — suitable for lightweight
    MobileNetV2 deployment on modest hardware.
    """
    out = model(batch, training=False)
    return out.numpy()
