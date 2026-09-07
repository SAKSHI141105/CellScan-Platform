"""Preprocessing for the histopathology image pipeline (BreakHis / Kaggle IDC patches).

Expects the standard Kaggle IDC layout after extraction:
    data/raw/histopathology/0/*.png   (benign / non-IDC patches)
    data/raw/histopathology/1/*.png   (malignant / IDC patches)
or a benign/malignant folder split for BreakHis — see README for the download
step. We don't ship the images; they're a few GB and licensed for
research redistribution only through the original sources.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif"}


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization — plain global equalization
    blows out the nuclei detail that actually distinguishes IDC from normal tissue.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    return clahe.apply(img)


def denoise(img: np.ndarray, method: str = "median", kernel: int = 3) -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    if method == "median":
        return cv2.medianBlur(img, kernel)
    raise ValueError(f"unknown denoise method: {method}")


def normalize(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def preprocess_single(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Full preprocessing chain for one image — this is what both training and
    the dashboard's single-image upload path call, so they can never drift apart.
    """
    cfg = cfg or load_config()["image"]
    gray = to_grayscale(img)
    resized = resize(gray, cfg["img_size"])
    enhanced = apply_clahe(resized, cfg["clahe_clip_limit"], cfg["clahe_tile_grid"])
    cleaned = denoise(enhanced, cfg["denoise_method"], cfg["denoise_kernel"])
    return normalize(cleaned)


def preprocess_from_path(path: str | Path, cfg: dict | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image at {path}")
    return preprocess_single(img, cfg)


def discover_image_paths(image_root: Path) -> tuple[list[Path], list[int]]:
    """Walks class-labelled subfolders. Accepts either {0,1} or {benign,malignant}
    naming since BreakHis and the Kaggle IDC set use different conventions.
    """
    label_dirs = {
        "0": 0, "benign": 0, "non_idc": 0,
        "1": 1, "malignant": 1, "idc": 1,
    }
    paths, labels = [], []
    for sub in sorted(image_root.iterdir()):
        if not sub.is_dir():
            continue
        label = label_dirs.get(sub.name.lower())
        if label is None:
            logger.warning("Skipping unrecognized class folder: %s", sub.name)
            continue
        for f in sub.rglob("*"):
            if f.suffix.lower() in VALID_EXTENSIONS:
                paths.append(f)
                labels.append(label)
    logger.info("Discovered %d images under %s", len(paths), image_root)
    return paths, labels


def build_dataset_arrays(image_root: Path, cfg: dict | None = None, limit: int | None = None):
    """Loads + preprocesses everything into memory. Fine for BreakHis-scale (a few
    thousand patches); for the full IDC set (~277k patches) switch to the
    tf.data generator in build_tf_dataset instead — this will OOM.
    """
    cfg = cfg or load_config()["image"]
    paths, labels = discover_image_paths(image_root)
    if limit:
        paths, labels = paths[:limit], labels[:limit]

    X = np.zeros((len(paths), cfg["img_size"], cfg["img_size"], 1), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i, ..., 0] = preprocess_from_path(p, cfg)
    y = np.array(labels, dtype=np.int32)
    return X, y


def make_augmentor(cfg: dict | None = None):
    """Thin wrapper around Keras' ImageDataGenerator — the classical route
    (rotate/flip/zoom) rather than a custom albumentations pipeline, mainly
    because it plugs straight into model.fit without extra glue code.

    Deliberately does NOT use `brightness_range`, despite it being in
    config.yaml's augmentation block — Keras' brightness transform round-trips
    through PIL internally and does a raw cast to uint8 assuming the array is
    already in [0, 255]. Our preprocessing pipeline normalizes to [0, 1]
    (image_preprocessing.normalize), so every pixel value is <1 and gets
    truncated to 0 — every augmented image comes out solid black. Confirmed
    empirically (batch min/max/mean all exactly 0.0 with brightness_range on,
    matching the real data's range with it off) after this silently collapsed
    a real training run to a trivial majority-class predictor: the model
    trains on blank images with real labels attached, which looks exactly
    like "not learning" rather than an obvious crash. The rotation/flip/zoom
    transforms below are plain affine ops with no such round-trip and are
    unaffected.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    cfg = cfg or load_config()["image"]
    aug = cfg["augmentation"]
    return ImageDataGenerator(
        rotation_range=aug["rotation_range"],
        horizontal_flip=aug["horizontal_flip"],
        vertical_flip=aug["vertical_flip"],
        zoom_range=aug["zoom_range"],
        fill_mode="reflect",
    )
