"""Image prediction + Grad-CAM, framework-agnostic (see tabular_service.py
for why — this is the twin of that module for the image pipeline).
"""
from __future__ import annotations

import base64
import threading

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CFG = load_config()
IMAGE_MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"] / "image"

_cache: dict = {}
_cache_lock = threading.Lock()


def load_image_model(prefer: str = "transfer"):
    """Returns (model, model_key, is_demo) or (None, None, None) only if
    even the in-memory fallback build fails (missing tensorflow, etc).

    Resolution order: real trained checkpoint -> demo weights saved on disk
    by scripts/generate_demo_weights.py -> build the demo model fresh in
    memory. The last step means the image pipeline is never a hard dead end
    even before that script has been run once — same convenience trade-off
    tabular_service.py makes with its untrained-RandomForest fallback, and
    same caveat: every one of these fallback tiers gets flagged is_demo=True
    so nothing downstream can mistake it for a real result.
    """
    if "model" in _cache:
        return _cache["model"]

    # Without this lock, a request landing while the startup warmup thread
    # is still loading this same model would race it and build a second,
    # independent copy — briefly doubling memory for a model load that's
    # already the heaviest thing this process does. Cheap on the fast path
    # (the cache check above already returns before ever touching the lock
    # once warmup has finished), and only serializes the one-time load.
    with _cache_lock:
        if "model" in _cache:  # re-check: another thread may have finished while we waited for the lock
            return _cache["model"]
        return _load_image_model_uncached(prefer)


def _load_image_model_uncached(prefer: str):
    backbone = CFG["image"]["transfer_backbone"]
    real_candidates = [
        (IMAGE_MODELS_DIR / f"transfer_{backbone}_final.h5", f"transfer_{backbone}"),
        (IMAGE_MODELS_DIR / "custom_cnn_final.h5", "custom_cnn"),
    ]
    if prefer == "custom":
        real_candidates.reverse()

    import tensorflow as tf

    # importing this registers the custom ReplicateChannels/PreprocessForBackbone
    # layers with Keras's serialization registry — required before load_model()
    # can deserialize a transfer-learning checkpoint, even though nothing here
    # calls into image_cnn directly on this path
    from src.models import image_cnn  # noqa: F401

    for path, key in real_candidates:
        if path.exists():
            result = (tf.keras.models.load_model(path), key, False)
            _cache["model"] = result
            logger.info("Loaded trained image model: %s", key)
            return result

    demo_path = IMAGE_MODELS_DIR / f"demo_untrained_transfer_{backbone}.h5"
    if demo_path.exists():
        result = (tf.keras.models.load_model(demo_path), f"transfer_{backbone}", True)
        _cache["model"] = result
        logger.warning("No trained checkpoint found — loaded DEMO weights (%s, untrained). Predictions are not meaningful.", demo_path.name)
        return result

    logger.warning(
        "No trained checkpoint or demo weights found — building an ImageNet-only DEMO model "
        "in memory (run scripts/generate_demo_weights.py to persist this instead of rebuilding every restart)."
    )
    from src.models import image_cnn

    img_cfg = CFG["image"]
    model = image_cnn.build_transfer_model(backbone, img_cfg["img_size"], img_cfg["fine_tune_last_n_layers"])
    result = (model, f"transfer_{backbone}", True)
    _cache["model"] = result
    return result


def gray_float_to_png_base64(img_float: np.ndarray) -> str:
    """img_float: (H, W) or (H, W, 1) in [0, 1] -> base64-encoded PNG string."""
    img = img_float[..., 0] if img_float.ndim == 3 else img_float
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_uint8)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def bgr_to_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def predict(preprocessed_img: np.ndarray) -> dict | None:
    from src.explainability.gradcam import explain_prediction
    from src.feature_engineering.texture_features import extract_classical_features, pixel_intensity_histogram

    model, model_key, is_demo = load_image_model()
    if model is None:
        return None

    batch = preprocessed_img[np.newaxis, ..., np.newaxis] if preprocessed_img.ndim == 2 else preprocessed_img[np.newaxis, ...]
    proba = float(model.predict(batch, verbose=0)[0, 0])
    predicted_class = "Malignant" if proba >= 0.5 else "Benign"
    overlay = explain_prediction(model, model_key, preprocessed_img)

    # real per-image GLCM/edge features, not a canned example — computed
    # straight off the actual preprocessed pixels the model just scored
    texture_features = extract_classical_features(preprocessed_img)
    pixel_histogram = pixel_intensity_histogram(preprocessed_img)

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "preprocessed_png_base64": gray_float_to_png_base64(preprocessed_img),
        "gradcam_png_base64": bgr_to_png_base64(overlay),
        "model_key": model_key,
        "texture_features": texture_features,
        "pixel_histogram": pixel_histogram,
        "is_demo": is_demo,
    }
