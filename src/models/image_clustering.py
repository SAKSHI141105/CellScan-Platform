"""Unsupervised side of the image pipeline: CNN embeddings + classical texture
features feed the same KMeans/DBSCAN/PCA/t-SNE/UMAP toolkit we used on the
tabular data, plus the conv-autoencoder anomaly detector.

We concatenate CNN features with the GLCM/edge ones rather than picking one —
the deep features capture whatever the backbone learned on ImageNet, the
classical ones are more directly tied to nucleus texture/shape, and in
practice the combination clusters a bit more cleanly than either alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.feature_engineering.texture_features import batch_extract
from src.utils.logging_setup import get_logger
from src.utils.metrics import clustering_metrics

logger = get_logger(__name__)

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def build_cnn_feature_extractor(img_size: int = 224):
    """ResNet50 with the classification head removed, global-average-pooled —
    a fixed 2048-d embedding per image, no fine-tuning needed for this step.
    """
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(img_size, img_size, 3))
    base.trainable = False
    return base


def to_3channel(images_gray: np.ndarray) -> np.ndarray:
    return np.repeat(images_gray, 3, axis=-1)


def extract_cnn_embeddings(images_gray: np.ndarray, extractor=None, img_size: int = 224, batch_size: int = 32) -> np.ndarray:
    """Processes images_gray in chunks rather than converting the whole array
    to 3-channel + preprocessed float32 in one shot. On a several-thousand-
    image dataset, materializing that all at once means holding multiple
    full-size intermediate copies simultaneously (3x size for the channel
    repeat, another full copy for the *255.0, another for preprocess_input)
    — easily 5-10x the size of the original array at peak, which is what
    actually OOM'd this step in practice. Chunking keeps peak memory to
    roughly one batch's worth regardless of dataset size; the embeddings
    themselves (a couple thousand floats per image) are cheap to accumulate.
    """
    extractor = extractor or build_cnn_feature_extractor(img_size)
    embeddings = []
    for start in range(0, len(images_gray), batch_size):
        batch = images_gray[start : start + batch_size]
        batch_3ch = preprocess_input(to_3channel(batch) * 255.0)
        embeddings.append(extractor.predict(batch_3ch, verbose=0))
    return np.concatenate(embeddings, axis=0)


def extract_combined_features(images_gray: np.ndarray, extractor=None) -> pd.DataFrame:
    cnn_feats = extract_cnn_embeddings(images_gray, extractor)
    cnn_df = pd.DataFrame(cnn_feats, columns=[f"cnn_{i}" for i in range(cnn_feats.shape[1])])
    texture_df = batch_extract(images_gray[..., 0] if images_gray.ndim == 4 else images_gray)
    return pd.concat([cnn_df.reset_index(drop=True), texture_df.reset_index(drop=True)], axis=1)


def project(X: np.ndarray, method: str = "pca", n_components: int = 2, random_state: int = 42):
    if method == "pca":
        return PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    if method == "tsne":
        return TSNE(n_components=n_components, random_state=random_state, init="pca").fit_transform(X)
    if method == "umap":
        if not _HAS_UMAP:
            raise ImportError("umap-learn not installed — pip install umap-learn")
        return umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(X)
    raise ValueError(f"unknown projection method: {method}")


def run_image_clustering(feature_df: pd.DataFrame, y_true: np.ndarray, kmeans_k: int = 2, dbscan_eps: float = 3.0, dbscan_min_samples: int = 5) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)

    # PCA down to a manageable dimensionality first — clustering directly on
    # 2048+ raw CNN dims is slow and the distance metric gets noisy that high up
    reducer = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)

    km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42).fit_predict(X_reduced)
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_reduced)

    results = {
        "kmeans": {"labels": km, **clustering_metrics(X_reduced, km, y_true)},
        "dbscan": {"labels": db, **clustering_metrics(X_reduced, db, y_true)},
    }
    return {"cluster_results": results, "X_reduced": X_reduced, "scaler": scaler, "reducer": reducer}
