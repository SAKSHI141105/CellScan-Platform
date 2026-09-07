"""End-to-end image training run. Requires the histopathology dataset to
already be extracted under config.paths.image_root (see README) — this
script will raise a clear error and bail out rather than silently doing
nothing if that folder is empty, since that's the most common setup mistake.

Trains: custom CNN, one transfer-learning backbone (config-selected), runs
the unsupervised clustering + autoencoder anomaly detection, and saves
everything needed for the dashboard's Grad-CAM panel.
"""
from __future__ import annotations

import gc

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data_preprocessing.image_preprocessing import build_dataset_arrays, make_augmentor
from src.models import image_cnn
from src.models.autoencoder import anomaly_threshold, build_conv_autoencoder, fit_on_benign_only, reconstruction_error
from src.models.image_clustering import extract_combined_features, run_image_clustering
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger
from src.utils.metrics import classification_report_dict

logger = get_logger(__name__)


def main():
    cfg = load_config()
    img_cfg = cfg["image"]
    image_root = PROJECT_ROOT / cfg["paths"]["image_root"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "image"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not image_root.exists() or not any(image_root.iterdir()):
        raise FileNotFoundError(
            f"No images found under {image_root}. Download BreakHis or the Kaggle IDC "
            "patch dataset and extract it there first — see README's 'Image dataset setup' section."
        )

    X, y = build_dataset_arrays(image_root, img_cfg, limit=None)
    logger.info("Loaded %d images, class balance: %s", len(y), dict(zip(*np.unique(y, return_counts=True))))

    # --- unsupervised: clustering on combined CNN + texture features ---
    # Done here, before the train/val/test split, specifically so the full
    # X array can be freed immediately after (see `del X` below) instead of
    # sitting in memory for the entire rest of this function alongside the
    # train/val/test splits — sklearn's train_test_split returns copies, not
    # views, so keeping both the original array and its splits alive at once
    # meant holding roughly two full copies of the dataset in RAM
    # simultaneously for no reason. On a several-thousand-image dataset at
    # 224x224 that's the difference between fitting comfortably and OOMing
    # partway through training.
    feature_df = extract_combined_features(X)
    cluster_out = run_image_clustering(feature_df, y, cfg["clustering"]["kmeans_k"], cfg["clustering"]["dbscan_eps"], cfg["clustering"]["dbscan_min_samples"])
    for name, res in cluster_out["cluster_results"].items():
        logger.info("[image clustering: %s] silhouette=%.3f ari=%.3f nmi=%.3f", name, res.get("silhouette", float("nan")), res.get("ari", float("nan")), res.get("nmi", float("nan")))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    del X, feature_df
    gc.collect()

    # rotation/flip/zoom/brightness on the training split only — validation
    # and test stay on real, un-augmented images so metrics reflect actual
    # generalization, not the model's ability to recognize its own augments
    augmentor = make_augmentor(img_cfg)

    # --- supervised: custom CNN ---
    custom_cnn = image_cnn.build_custom_cnn(img_cfg["img_size"], img_cfg["channels"])
    image_cnn.train(
        custom_cnn, X_train, y_train, X_val, y_val,
        checkpoint_path=models_dir / "custom_cnn.h5",
        epochs=img_cfg["epochs"], batch_size=img_cfg["batch_size"], augmentor=augmentor,
    )
    custom_metrics = classification_report_dict(y_test, (custom_cnn.predict(X_test, verbose=0) > 0.5).astype(int), custom_cnn.predict(X_test, verbose=0).ravel())
    logger.info("Custom CNN test metrics: %s", custom_metrics)
    custom_cnn.save(models_dir / "custom_cnn_final.h5")
    del custom_cnn
    tf.keras.backend.clear_session()  # releases the graph/session memory gc.collect() alone won't
    gc.collect()

    # --- supervised: transfer learning ---
    transfer_model = image_cnn.build_transfer_model(img_cfg["transfer_backbone"], img_cfg["img_size"], img_cfg["fine_tune_last_n_layers"])
    image_cnn.train(
        transfer_model, X_train, y_train, X_val, y_val,
        checkpoint_path=models_dir / f"transfer_{img_cfg['transfer_backbone']}.h5",
        epochs=img_cfg["epochs"], batch_size=img_cfg["batch_size"], augmentor=augmentor,
    )
    transfer_metrics = classification_report_dict(y_test, (transfer_model.predict(X_test, verbose=0) > 0.5).astype(int), transfer_model.predict(X_test, verbose=0).ravel())
    logger.info("Transfer model (%s) test metrics: %s", img_cfg["transfer_backbone"], transfer_metrics)
    transfer_model.save(models_dir / f"transfer_{img_cfg['transfer_backbone']}_final.h5")
    del transfer_model
    tf.keras.backend.clear_session()
    gc.collect()

    # --- unsupervised: autoencoder anomaly detection, trained on benign only ---
    autoencoder = build_conv_autoencoder(img_cfg["img_size"], img_cfg["channels"])
    fit_on_benign_only(autoencoder, X_train, y_train == 0, epochs=30, batch_size=img_cfg["batch_size"])
    benign_errors = reconstruction_error(autoencoder, X_train[y_train == 0])
    threshold = anomaly_threshold(benign_errors)
    test_errors = reconstruction_error(autoencoder, X_test)
    flagged = (test_errors > threshold).astype(int)
    ae_metrics = classification_report_dict(y_test, flagged)
    logger.info("Autoencoder anomaly detection (threshold=%.5f) test metrics: %s", threshold, ae_metrics)

    autoencoder.save(models_dir / "conv_autoencoder.h5")

    logger.info("Image training run complete. Artifacts in %s", models_dir)


if __name__ == "__main__":
    main()
