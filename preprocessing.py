"""
OpenCV preprocessing for MobileNetV2 inference (224x224, normalization, optional luminance correction).
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

TARGET_SIZE = (224, 224)


def preprocess_image(
    image_bgr: np.ndarray,
    *,
    use_histogram_equalization: bool = False,
    normalize_mode: str = "mobilenet",
) -> np.ndarray:
    """
    Mandatory preprocessing pipeline for model input.

    Parameters
    ----------
    image_bgr :
        uint8 image in BGR order (OpenCV default).
    use_histogram_equalization :
        If True, apply histogram equalization on the L channel (LAB) to test luminance correction.
    normalize_mode :
        "mobilenet" — scale inputs as required by `keras.applications.mobilenet_v2.preprocess_input`
        (recommended with ImageNet backbone weights).
        "div255" — divide RGB by 255.0 only (simple [0,1] scaling).

    Returns
    -------
    np.ndarray
        float32 array of shape (1, 224, 224, 3) ready for `model.predict`.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Invalid image array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    work = image_bgr.copy()

    if use_histogram_equalization:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge([l_eq, a, b])
        work = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    resized = cv2.resize(work, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    if normalize_mode == "mobilenet":
        batch = np.expand_dims(rgb, axis=0)
        batch = preprocess_input(batch)
    elif normalize_mode == "div255":
        batch = np.expand_dims(rgb / 255.0, axis=0)
    else:
        raise ValueError("normalize_mode must be 'mobilenet' or 'div255'.")

    return batch


def decode_upload_bytes(file_bytes: bytes) -> np.ndarray | None:
    """Decode raw upload bytes to BGR image; returns None if decode fails."""
    if not file_bytes:
        return None
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def preprocess_from_bytes(
    file_bytes: bytes,
    *,
    use_histogram_equalization: bool = False,
    normalize_mode: str = "mobilenet",
) -> np.ndarray:
    """Decode noisy uploads safely; raises ValueError if image cannot be read."""
    image = decode_upload_bytes(file_bytes)
    if image is None:
        raise ValueError("Could not decode image. Supported formats include JPEG and PNG.")
    return preprocess_image(
        image,
        use_histogram_equalization=use_histogram_equalization,
        normalize_mode=normalize_mode,
    )
