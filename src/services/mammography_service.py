"""Mammography prediction service — same shape as image_service.py
(histopathology) and tabular_service.py: model loading with a demo-weights
fallback, a predict() that returns everything the API/UI need, is_demo
threaded through every response so a demo prediction can never be mistaken
for a real one.
"""
from __future__ import annotations

import base64
import threading

import cv2
import numpy as np
import torch

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CFG = load_config()
MAMMO_CFG = CFG["mammography"]

_cache: dict = {}
_cache_lock = threading.Lock()


def _build_model(pretrained: bool) -> torch.nn.Module:
    from src.models.mammography.lesion_guided_model import LesionGuidedModel

    return LesionGuidedModel(
        backbone_name=MAMMO_CFG["backbone"],
        pretrained=pretrained,
        in_channels=1,
        num_classes=1,
        decoder_channels=MAMMO_CFG["decoder_channels"],
    )


def load_mammography_model() -> tuple[torch.nn.Module, bool]:
    """Returns (model, is_demo). Resolution order: real trained checkpoint
    (data/models/mammography/lesion_guided_resnet50.pt) -> saved demo
    weights (scripts/generate_demo_weights.py) -> built fresh in memory.
    Same three-tier fallback image_service.py uses for the histopathology
    model, for the same reason — never a hard dead end, always clearly
    tagged when it isn't a real model.
    """
    if "model" in _cache:
        return _cache["model"]

    # See image_service.load_image_model's identical lock for why: without
    # it, a request racing the startup warmup thread would build a second
    # independent model, briefly doubling memory during the heaviest load
    # this process does.
    with _cache_lock:
        if "model" in _cache:
            return _cache["model"]
        return _load_mammography_model_uncached()


def _load_mammography_model_uncached() -> tuple[torch.nn.Module, bool]:
    for rel_path in MAMMO_CFG["checkpoint_candidates"]:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu")
        model = _build_model(pretrained=False)  # weights come from the checkpoint, not ImageNet download
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        is_demo = bool(checkpoint.get("is_demo", "demo" in path.name))
        result = (model, is_demo)
        _cache["model"] = result
        logger.info("Loaded mammography checkpoint: %s (is_demo=%s)", path.name, is_demo)
        return result

    logger.warning(
        "No mammography checkpoint found on disk — building an ImageNet-only DEMO model in memory "
        "(run scripts/generate_demo_weights.py to persist this instead of rebuilding every restart)."
    )
    model = _build_model(pretrained=True)
    model.eval()
    result = (model, True)
    _cache["model"] = result
    return result


def _mask_coverage(mask_probs: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of the predicted segmentation mask above threshold — a
    genuine derived quantity from the model's own decoder output (not a
    fabricated clinical measurement), used as a rough "estimated lesion
    area" indicator in the structured explanation.
    """
    return float((mask_probs >= threshold).mean())


def _attention_concentration(heatmap: np.ndarray, top_fraction: float = 0.1) -> float:
    """What share of total Grad-CAM 'energy' sits in the hottest top_fraction
    of pixels — high means attention is localized on one region (consistent
    with a discrete mass/calcification), low means it's diffuse across the
    image. Also a real computed quantity, not a fabricated one.
    """
    flat = heatmap.flatten()
    total = flat.sum()
    if total <= 0:
        return 0.0
    n_top = max(1, int(len(flat) * top_fraction))
    top_sum = np.sort(flat)[-n_top:].sum()
    return float(top_sum / total)


def gray_float_to_png_base64(img_float: np.ndarray) -> str:
    img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_uint8)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def bgr_to_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def predict(preprocessed_img: np.ndarray) -> dict:
    from src.explainability.mammography_gradcam import GradCAM, find_last_conv, overlay_heatmap

    model, is_demo = load_mammography_model()

    tensor = torch.from_numpy(preprocessed_img).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        proba = torch.sigmoid(outputs["class_logits"])[0, 0].item()
        mask_probs = torch.sigmoid(outputs["mask_logits"])[0, 0].numpy()

    predicted_class = "Malignant" if proba >= 0.5 else "Benign"

    # Grad-CAM needs its own forward+backward pass (gradients wrt this specific
    # output), separate from the no_grad() inference above
    target_layer = find_last_conv(model.backbone.backbone)
    cam = GradCAM(model, target_layer)
    tensor_for_cam = torch.from_numpy(preprocessed_img).float().unsqueeze(0).unsqueeze(0)
    heatmap = cam(tensor_for_cam)
    overlay = overlay_heatmap(preprocessed_img, heatmap)

    from src.feature_engineering.texture_features import extract_classical_features, pixel_intensity_histogram

    texture_features = extract_classical_features(preprocessed_img)
    pixel_histogram = pixel_intensity_histogram(preprocessed_img)
    concentration = _attention_concentration(heatmap)

    explanation = {
        "predicted_probability": round(proba, 4),
        "estimated_lesion_area_fraction": round(_mask_coverage(mask_probs), 4),
        "attention_concentration": round(concentration, 4),
        "summary": (
            f"Grad-CAM attention is concentrated on a localized region (attention concentration "
            f"{concentration:.2f}), consistent with a discrete mass/calcification pattern, alongside a "
            f"GLCM contrast of {texture_features['glcm_contrast']:.1f} and homogeneity of "
            f"{texture_features['glcm_homogeneity']:.2f} in the analyzed region."
            if concentration > 0.4 else
            f"Grad-CAM attention is diffuse across the tissue (concentration {concentration:.2f}) rather than "
            f"localized on a single structure, with a GLCM contrast of {texture_features['glcm_contrast']:.1f} "
            "over the full patch."
        ),
    }

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "preprocessed_png_base64": gray_float_to_png_base64(preprocessed_img),
        "gradcam_png_base64": bgr_to_png_base64(overlay),
        "model_key": f"mammography_lesion_guided_{MAMMO_CFG['backbone']}",
        "explanation": explanation,
        "texture_features": texture_features,
        "pixel_histogram": pixel_histogram,
        "is_demo": is_demo,
    }
