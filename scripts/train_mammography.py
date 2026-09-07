"""Trains the lesion-guided mammography model on a prepared manifest
(see scripts/prepare_mias.py) and saves a checkpoint the dashboard's
Mammography page loads directly.

    python -m scripts.prepare_mias      # once, to build the manifest
    python -m scripts.train_mammography

Deliberately trains against src/models/mammography/ (this app's port) rather
than shelling out to research/mammography_generalization/train.py. The two
model definitions are structurally identical by design, so the checkpoint is
interchangeable either way — but the research pipeline has its own venv and
its own top-level `src` package, which collides with this one's inside a
single process (see that project's README). Training here keeps it to one
environment and guarantees the state_dict lands in exactly the shape
mammography_service.load_mammography_model() expects.

Loss mirrors research/.../training/losses.py's JointLoss: classification BCE
plus a Dice+BCE lesion term applied only to samples that actually carry a
mask, so the ~2/3 of MIAS rows without lesion annotations don't get trained
toward "there is no lesion here". The one addition is a pos_weight on the
classification term — MIAS is ~16% malignant, and without it the model
converges to predicting benign for everything, which scores well on accuracy
and is useless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data_preprocessing.mammography_preprocessing import preprocess_mammogram
from src.models.mammography.lesion_guided_model import LesionGuidedModel
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mammography"


class MammoManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        import cv2

        row = self.df.iloc[idx]
        image = cv2.imread(str(DATA_ROOT / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        # same chain the API runs on an uploaded scan (preprocess_upload calls
        # straight into this), so training and inference can't drift apart
        image = preprocess_mammogram(
            image,
            img_size=self.cfg["img_size"],
            clahe_clip_limit=self.cfg["clahe_clip_limit"],
            clahe_tile_grid=tuple(self.cfg["clahe_tile_grid"]),
            denoise_method=self.cfg["denoise_method"],
            denoise_kernel=self.cfg["denoise_kernel"],
        )

        mask_rel = row["mask_path"]
        has_mask = isinstance(mask_rel, str) and mask_rel.strip() != ""
        if has_mask:
            mask = cv2.imread(str(DATA_ROOT / mask_rel), cv2.IMREAD_GRAYSCALE)
            mask = (cv2.resize(mask, image.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        else:
            mask = np.zeros_like(image, dtype=np.float32)

        if self.augment and np.random.rand() < 0.5:
            # horizontal flip only — a mammogram's left/right orientation is
            # a laterality artifact, but flipping vertically would put the
            # pectoral muscle somewhere anatomically impossible
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        return {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "has_mask": torch.tensor(has_mask),
            "label": torch.tensor(float(row["label"])),
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return 1 - (2 * intersection + eps) / (union + eps)


def joint_loss(outputs: dict, batch: dict, pos_weight: torch.Tensor, lesion_weight: float) -> torch.Tensor:
    class_loss = F.binary_cross_entropy_with_logits(
        outputs["class_logits"].squeeze(-1), batch["label"], pos_weight=pos_weight
    )
    has_mask = batch["has_mask"]
    if lesion_weight > 0 and has_mask.any():
        logits = outputs["mask_logits"][has_mask]
        targets = batch["mask"][has_mask]
        lesion = dice_loss(logits, targets).mean() + F.binary_cross_entropy_with_logits(logits, targets)
        return class_loss + lesion_weight * lesion
    return class_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        out = model(batch["image"])
        probs.append(torch.sigmoid(out["class_logits"].squeeze(-1)).cpu().numpy())
        labels.append(batch["label"].numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan"),
        "accuracy": float((preds == labels).mean()),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


FREEZE_PREFIXES = ("backbone.backbone.conv1", "backbone.backbone.bn1", "backbone.backbone.layer1", "backbone.backbone.layer2")


def freeze_early_backbone(model: nn.Module) -> tuple[int, int]:
    """Freezes the stem + first two ResNet stages, leaving layer3/layer4, the
    U-Net decoder and the classifier trainable.

    Two reasons, and the speed one is the lesser of them: with 257 training
    images, fine-tuning all ~23M backbone parameters overfits long before it
    generalizes, and early conv filters (edges, texture) transfer from
    ImageNet essentially unchanged anyway. Skipping their gradients also
    makes the backward pass materially cheaper on CPU.

    Returns (frozen, trainable) parameter-tensor counts so a naming mismatch
    shows up in the log as "froze 0" instead of silently training everything.
    """
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(FREEZE_PREFIXES):
            param.requires_grad = False
            frozen += 1
        else:
            trainable += 1
    return frozen, trainable


def main(epochs: int = 12, batch_size: int = 4, lr: float = 1e-4, lesion_weight: float = 0.5):
    cfg = load_config()
    mammo_cfg = cfg["mammography"]

    manifest = DATA_ROOT / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found — run `python -m scripts.prepare_mias` first.")

    df = pd.read_csv(manifest).fillna({"mask_path": ""})
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    logger.info(
        "Train %d (%d malignant) / test %d (%d malignant)",
        len(train_df), int(train_df.label.sum()), len(test_df), int(test_df.label.sum()),
    )

    train_loader = DataLoader(MammoManifestDataset(train_df, mammo_cfg, augment=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MammoManifestDataset(test_df, mammo_cfg), batch_size=batch_size)

    model = LesionGuidedModel(
        backbone_name=mammo_cfg["backbone"],
        pretrained=True,
        in_channels=1,
        num_classes=1,
        decoder_channels=mammo_cfg["decoder_channels"],
    )

    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "mammography"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"lesion_guided_{mammo_cfg['backbone']}.pt"
    resume_path = out_path
    resumed_auc = -1.0
    if resume_path.exists():
        # Training runs here get killed by machine sleep / session ends often
        # enough that starting from scratch every time meant never getting
        # past the first epoch or two. Picking up the previous best checkpoint
        # makes progress cumulative across runs instead of disposable.
        ckpt = torch.load(resume_path, map_location="cpu")
        if not ckpt.get("is_demo", True):
            model.load_state_dict(ckpt["model_state"])
            resumed_auc = float(ckpt.get("test_metrics", {}).get("auc", -1.0) or -1.0)
            logger.info(
                "Resuming from %s (previous best auc=%.4f, epoch %s)",
                resume_path.name, resumed_auc, ckpt.get("epoch", "?"),
            )
        else:
            logger.info("Found only demo weights at %s — starting from ImageNet init", resume_path.name)

    n_frozen, n_trainable = freeze_early_backbone(model)
    logger.info("Froze %d parameter tensors (stem + layer1/2), %d remain trainable", n_frozen, n_trainable)
    if n_frozen == 0:
        logger.warning("Freeze matched nothing — backbone parameter names may differ from %s", FREEZE_PREFIXES)

    n_pos = int(train_df.label.sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    logger.info("Classification pos_weight=%.2f (%d neg / %d pos)", pos_weight.item(), n_neg, n_pos)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_auc = resumed_auc
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["image"])
            loss = joint_loss(outputs, batch, pos_weight, lesion_weight)
            loss.backward()
            optimizer.step()
            running += loss.item()

        metrics = evaluate(model, test_loader)
        logger.info(
            "epoch %d/%d — train_loss=%.4f test_auc=%.4f acc=%.4f recall=%.4f precision=%.4f",
            epoch, epochs, running / max(len(train_loader), 1),
            metrics["auc"], metrics["accuracy"], metrics["recall"], metrics["precision"],
        )

        if metrics["auc"] == metrics["auc"] and metrics["auc"] > best_auc:  # NaN-safe
            best_auc = metrics["auc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "is_demo": False,
                    "backbone": mammo_cfg["backbone"],
                    "trained_on": "MIAS (322 scans, circular masks from centroid+radius annotations)",
                    "test_metrics": metrics,
                    "epoch": epoch,
                },
                out_path,
            )
            logger.info("  saved new best checkpoint (auc=%.4f) -> %s", best_auc, out_path)

    logger.info("Mammography training complete. Best test AUC=%.4f. Checkpoint: %s", best_auc, out_path)


if __name__ == "__main__":
    main()
