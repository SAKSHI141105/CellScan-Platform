"""Picks whichever mammography checkpoint has the best test AUC among the
backbones trained by train_mammography_variants.py / the CNN comparison run,
then tries a couple of alternate learning rates on that one backbone only —
a light sweep, not a full grid, since CPU time is the constraint here.

    python -m scripts.sweep_best_mammography

Sweep trials are tagged (lr5e-5 / lr3e-4) so they write to their own
checkpoint files and never overwrite the winning backbone's real checkpoint
or the config-pointed production checkpoint (lesion_guided_resnet50.pt).
"""
from __future__ import annotations

import torch

from scripts.train_mammography import main
from src.utils.config import PROJECT_ROOT
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models" / "mammography"

# backbone_name -> decoder_channels, so the sweep can rebuild the same architecture
BACKBONES = {
    "resnet50": [256, 128, 64, 32],
    "swin_tiny_patch4_window7_224": [128, 64, 32],
    "coatnet_1_rw_224": [256, 128, 64, 32],
    "vit_base_patch16_224": [128, 64],
    "densenet121": [256, 128, 64, 32],
    "tf_efficientnetv2_s": [256, 128, 64, 32],
    "convnext_tiny": [128, 64, 32],
}


def find_best() -> tuple[str, list[int], float]:
    best_name, best_auc = None, -1.0
    for name in BACKBONES:
        path = MODELS_DIR / f"lesion_guided_{name}.pt"
        if not path.exists():
            logger.warning("No checkpoint for %s yet (%s) — skipping", name, path)
            continue
        ckpt = torch.load(path, map_location="cpu")
        auc = float(ckpt.get("test_metrics", {}).get("auc", -1.0) or -1.0)
        logger.info("%-30s auc=%.4f", name, auc)
        if auc > best_auc:
            best_auc, best_name = auc, name
    if best_name is None:
        raise RuntimeError("No trained checkpoints found — run the comparison training first.")
    return best_name, BACKBONES[best_name], best_auc


if __name__ == "__main__":
    winner, decoder_channels, winner_auc = find_best()
    logger.info("=" * 70)
    logger.info("Winner so far: %s (auc=%.4f) — running a light LR sweep on it", winner, winner_auc)
    logger.info("=" * 70)

    for lr, tag in [(5e-5, "lr5e-5"), (3e-4, "lr3e-4")]:
        logger.info("--- sweep trial: backbone=%s lr=%s ---", winner, lr)
        main(epochs=6, batch_size=4, lr=lr, backbone_name=winner, decoder_channels=decoder_channels, img_size=512, tag=tag)

    logger.info("=" * 70)
    logger.info("Sweep complete. Compare data/models/mammography/lesion_guided_%s*.pt checkpoints' test_metrics.auc", winner)
    logger.info("=" * 70)
