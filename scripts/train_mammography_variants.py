"""Trains the mammography pipeline on three additional backbones for comparison
against the existing ResNet50 checkpoint: a hierarchical vision transformer
(Swin), a CNN+Transformer hybrid (CoAtNet), and a plain ViT.

    python -m scripts.train_mammography_variants

Each run writes its own checkpoint (data/models/mammography/lesion_guided_<backbone>.pt)
and resumes from it if re-run, same as scripts/train_mammography.py.

Decoder depth differs per backbone because it's capped by how many pyramid
levels that backbone's timm features_only output actually has: Swin gives 4,
CoAtNet gives 5, plain ViT gives only 3 (it has no hierarchical downsampling,
so those 3 levels are same-resolution features from different depths rather
than a real multi-scale pyramid — an inherent limitation of plain ViT, not a
bug here).
"""
from __future__ import annotations

from scripts.train_mammography import main
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

VARIANTS = [
    ("swin_tiny_patch4_window7_224", [128, 64, 32]),
    ("coatnet_1_rw_224", [256, 128, 64, 32]),
    ("vit_base_patch16_224", [128, 64]),
]

if __name__ == "__main__":
    for backbone_name, decoder_channels in VARIANTS:
        logger.info("=" * 70)
        logger.info("Starting training run: backbone=%s decoder_channels=%s", backbone_name, decoder_channels)
        logger.info("=" * 70)
        try:
            main(epochs=12, batch_size=4, backbone_name=backbone_name, decoder_channels=decoder_channels, img_size=512)
        except Exception:
            logger.exception("Training run failed for backbone=%s — continuing with next variant", backbone_name)
