"""Sequential CNN vs Transformer vs Hybrid comparison on the approved
50-patient IDC histopathology subset (data/raw/histopathology_idc/,
data/splits/idc_patient_split.json).

Trains one model at a time (never simultaneously), on GPU (RTX 3050,
CUDA 12.4) with mixed precision, evaluates each on the SAME held-out test
set, and appends its row to results/comparison/final_image_model_comparison.csv
immediately after finishing — so a crash partway through does not lose
already-completed results.

    python -m scripts.train_idc_comparison

Checkpoint selection criterion (documented, not tuned on test): best
validation ROC-AUC, matching this project's emphasis on
recall/sensitivity + ROC-AUC over raw accuracy for a medical classifier.
"""
from __future__ import annotations

import csv
import gc
import json
import shutil
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.config import PROJECT_ROOT
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "histopathology_idc"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "idc_patient_split.json"
MODELS_DIR = PROJECT_ROOT / "models" / "image"
RESULTS_DIR = PROJECT_ROOT / "results" / "image"
COMPARISON_CSV = PROJECT_ROOT / "results" / "comparison" / "final_image_model_comparison.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_FREE_DISK_GB = 3.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def disk_free_gb() -> float:
    return shutil.disk_usage("C:/").free / 1e9


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)} | free={free/1e9:.2f}GB / total={total/1e9:.2f}GB"


class IDCDataset(Dataset):
    def __init__(self, patients: list[str], img_size: int, augment: bool):
        self.samples = []
        for p in patients:
            for cls in ("0", "1"):
                cls_dir = DATA_DIR / p / cls
                if not cls_dir.exists():
                    continue
                for f in cls_dir.glob("*.png"):
                    self.samples.append((f, int(cls)))
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))


class CustomCNN(nn.Module):
    """From-scratch 4-block CNN, no pretraining — the "traditional deep
    learning" baseline before transfer-learned architectures enter the
    comparison."""

    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x))


# name -> (timm model name or None for CustomCNN, img_size, is_pretrained_transfer)
MODEL_SPECS = [
    ("custom_cnn", None, 64, False),
    ("resnet50", "resnet50", 224, True),
    ("densenet121", "densenet121", 224, True),
    ("efficientnetv2_s", "tf_efficientnetv2_s", 224, True),
    ("convnext_tiny", "convnext_tiny", 224, True),
    ("vit_b16", "vit_base_patch16_224", 224, True),
    ("deit_s", "deit_small_patch16_224", 224, True),
    ("swin_tiny", "swin_tiny_patch4_window7_224", 224, True),
    ("hybrid_coatnet", "coatnet_1_rw_224", 224, True),  # closest real CNN+Transformer hybrid available in timm
]


def build_model(timm_name: str | None) -> nn.Module:
    if timm_name is None:
        return CustomCNN()
    return timm.create_model(timm_name, pretrained=True, num_classes=1)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sanity_check(model: nn.Module, loader: DataLoader) -> bool:
    model.train()
    batch = next(iter(loader))
    x, y = batch
    x, y = x[:2].to(DEVICE), y[:2].to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    out = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y)
    loss.backward()
    opt.step()
    assert next(model.parameters()).is_cuda == (DEVICE.type == "cuda")
    logger.info("  sanity check passed: model on %s, forward+backward+step OK, loss=%.4f", DEVICE, loss.item())
    return True


def train_one_epoch(model, loader, optimizer, pos_weight, scaler, use_amp):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, use_amp):
    model.eval()
    probs, labels = [], []
    running = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y)
        running += loss.item()
        probs.append(torch.sigmoid(out).float().cpu().numpy())
        labels.append(y.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    return running / max(len(loader), 1), probs, labels


def full_metrics(probs, labels, threshold=0.5) -> dict:
    n_nan = int(np.isnan(probs).sum())
    if n_nan:
        logger.warning("full_metrics: %d/%d probs were NaN (numerical instability) — clamped to 0.5", n_nan, len(probs))
        probs = np.nan_to_num(probs, nan=0.5)
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_plots(name: str, history: dict, probs, labels, cm: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{name} — Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{name} — Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].plot(history["train_recall"], label="train"); axes[2].plot(history["val_recall"], label="val")
    axes[2].set_title(f"{name} — Recall"); axes[2].set_xlabel("epoch"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=100); plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(labels, probs):.3f}" if len(set(labels.tolist())) > 1 else "ROC (single class)")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(f"{name} — ROC Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=100); plt.close(fig)

    prec, rec, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"PR-AUC={average_precision_score(labels, probs):.3f}" if len(set(labels.tolist())) > 1 else "PR (single class)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"{name} — Precision-Recall Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "pr_curve.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Non-IDC", "Pred IDC"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Non-IDC", "True IDC"])
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=100); plt.close(fig)


def append_comparison_row(row: dict):
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "architecture_type", "status", "accuracy", "precision", "recall",
                  "specificity", "f1", "roc_auc", "pr_auc", "training_time_sec", "parameters", "epochs", "best_epoch"]
    write_header = not COMPARISON_CSV.exists()
    with open(COMPARISON_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


ARCH_TYPE = {
    "custom_cnn": "CNN (from scratch)", "resnet50": "CNN (transfer)", "densenet121": "CNN (transfer)",
    "efficientnetv2_s": "CNN (transfer)", "convnext_tiny": "CNN (transfer)", "vit_b16": "Transformer",
    "deit_s": "Transformer", "swin_tiny": "Transformer", "hybrid_coatnet": "Hybrid CNN+Transformer",
}


def run_model(name: str, timm_name: str | None, img_size: int, pretrained: bool, split: dict, pos_weight_val: float):
    print(f"\n{'='*70}\nMODEL: {name}\n{'='*70}")
    free_gb = disk_free_gb()
    print(f"Disk free before this model: {free_gb:.2f} GB")
    print(f"GPU status: {gpu_status()}")
    if free_gb < MIN_FREE_DISK_GB:
        print(f"STOPPING before {name}: free disk ({free_gb:.2f}GB) below {MIN_FREE_DISK_GB}GB floor.")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "NOT_ATTEMPTED_DISK_FULL",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": "", "parameters": "", "epochs": "", "best_epoch": ""})
        return "STOP"

    train_ds = IDCDataset(split["train"], img_size, augment=True)
    val_ds = IDCDataset(split["val"], img_size, augment=False)
    test_ds = IDCDataset(split["test"], img_size, augment=False)
    print(f"Train/Val/Test patch counts: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    batch_size = 64 if timm_name is None else 32
    n_workers = 2

    def make_loaders(bs):
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=True),
            DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=n_workers, pin_memory=True),
        )

    try:
        model = build_model(timm_name).to(DEVICE)
    except Exception as e:
        print(f"MODEL: {name}\nSTATUS: FAILED (model creation)\nERROR: {e}\nLIKELY CAUSE: architecture/weights unavailable")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "FAILED_MODEL_CREATION",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": "", "parameters": "", "epochs": "", "best_epoch": ""})
        return "CONTINUE"

    total_params, _ = count_params(model)

    # two-stage transfer learning: freeze backbone / train head, then unfreeze and fine-tune
    stage1_epochs, stage2_epochs = (0, 10) if timm_name is None else (3, 5)
    use_amp = DEVICE.type == "cuda"

    train_loader, val_loader, test_loader = make_loaders(batch_size)
    try:
        sanity_check(model, train_loader)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        batch_size = max(8, batch_size // 2)
        print(f"  OOM on sanity check -> reducing batch size to {batch_size} and retrying")
        train_loader, val_loader, test_loader = make_loaders(batch_size)
        sanity_check(model, train_loader)
    except Exception as e:
        print(f"MODEL: {name}\nSTATUS: FAILED (sanity check)\nERROR: {e}")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "FAILED_SANITY_CHECK",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": "", "parameters": "", "epochs": "", "best_epoch": ""})
        del model; torch.cuda.empty_cache(); gc.collect()
        return "CONTINUE"

    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_recall": [], "val_recall": []}
    best_auc, best_epoch, patience, patience_ctr = -1.0, 0, 3, 0
    ckpt_dir = MODELS_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    t0 = time.time()
    try:
        epoch = 0
        # Stage 1: frozen backbone (transfer-learned models only)
        if stage1_epochs > 0:
            for p in model.parameters():
                p.requires_grad = False
            if timm_name and "resnet" in timm_name:
                model.fc.requires_grad_(True)
            else:
                for p in list(model.parameters())[-2:]:
                    p.requires_grad = True
            head_attr = None
            for attr in ("fc", "head", "classifier", "head.fc"):
                obj = model
                ok = True
                for part in attr.split("."):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        ok = False
                        break
                if ok:
                    for p in obj.parameters():
                        p.requires_grad = True
                    head_attr = attr
                    break
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            print(f"  Stage 1 (frozen backbone, head='{head_attr}'): {stage1_epochs} epochs")
            for _ in range(stage1_epochs):
                epoch += 1
                tr_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, scaler, use_amp)
                val_loss, val_probs, val_labels = evaluate(model, val_loader, use_amp)
                _, tr_probs, tr_labels = evaluate(model, train_loader, use_amp)
                tr_m, val_m = full_metrics(tr_probs, tr_labels), full_metrics(val_probs, val_labels)
                history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
                history["train_acc"].append(tr_m["accuracy"]); history["val_acc"].append(val_m["accuracy"])
                history["train_recall"].append(tr_m["recall"]); history["val_recall"].append(val_m["recall"])
                print(f"    [stage1] epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_auc={val_m['roc_auc']:.4f}")
                if val_m["roc_auc"] == val_m["roc_auc"] and val_m["roc_auc"] > best_auc:
                    best_auc, best_epoch = val_m["roc_auc"], epoch
                    torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_m}, ckpt_path)

        # Stage 2: unfreeze everything, fine-tune at lower LR (or full training for custom CNN)
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if timm_name else 1e-3)
        print(f"  Stage 2 (fine-tune all): up to {stage2_epochs} epochs, early stopping patience={patience}")
        for _ in range(stage2_epochs):
            epoch += 1
            tr_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, scaler, use_amp)
            val_loss, val_probs, val_labels = evaluate(model, val_loader, use_amp)
            _, tr_probs, tr_labels = evaluate(model, train_loader, use_amp)
            tr_m, val_m = full_metrics(tr_probs, tr_labels), full_metrics(val_probs, val_labels)
            history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_m["accuracy"]); history["val_acc"].append(val_m["accuracy"])
            history["train_recall"].append(tr_m["recall"]); history["val_recall"].append(val_m["recall"])
            print(f"    [stage2] epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_auc={val_m['roc_auc']:.4f}")
            if val_m["roc_auc"] == val_m["roc_auc"] and val_m["roc_auc"] > best_auc:
                best_auc, best_epoch = val_m["roc_auc"], epoch
                patience_ctr = 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_m}, ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"    early stopping at epoch {epoch} (no val_auc improvement for {patience} epochs)")
                    break
    except torch.cuda.OutOfMemoryError as e:
        print(f"MODEL: {name}\nSTATUS: FAILED (CUDA OOM during training)\nERROR: {e}\nLIKELY CAUSE: batch_size={batch_size} too large for 4GB VRAM at img_size={img_size}\nATTEMPTED FIX: none further after sanity-check reduction")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "FAILED_OOM",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": "", "parameters": total_params, "epochs": epoch, "best_epoch": ""})
        del model; torch.cuda.empty_cache(); gc.collect()
        return "CONTINUE"
    except Exception as e:
        print(f"MODEL: {name}\nSTATUS: FAILED (training)\nERROR: {e}\n{traceback.format_exc()}")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "FAILED_TRAINING",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": "", "parameters": total_params, "epochs": epoch, "best_epoch": ""})
        del model; torch.cuda.empty_cache(); gc.collect()
        return "CONTINUE"

    training_time = time.time() - t0

    if not ckpt_path.exists():
        print(f"MODEL: {name}\nSTATUS: FAILED (no checkpoint saved — val AUC never improved from -1)")
        append_comparison_row({"model": name, "architecture_type": ARCH_TYPE[name], "status": "FAILED_NO_CHECKPOINT",
                                "accuracy": "", "precision": "", "recall": "", "specificity": "", "f1": "",
                                "roc_auc": "", "pr_auc": "", "training_time_sec": training_time, "parameters": total_params, "epochs": epoch, "best_epoch": ""})
        del model; torch.cuda.empty_cache(); gc.collect()
        return "CONTINUE"

    # load best checkpoint, evaluate on TEST set (never used for checkpoint selection)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_probs, test_labels = evaluate(model, test_loader, use_amp)
    test_m = full_metrics(test_probs, test_labels)

    out_dir = RESULTS_DIR / name
    save_plots(name, history, test_probs, test_labels, test_m["confusion_matrix"], out_dir)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test_metrics": test_m, "best_val_auc": best_auc, "best_epoch": best_epoch,
                    "total_epochs": epoch, "training_time_sec": training_time, "parameters": total_params,
                    "batch_size": batch_size, "img_size": img_size, "history": history}, f, indent=2)

    row = {
        "model": name, "architecture_type": ARCH_TYPE[name], "status": "SUCCESS",
        "accuracy": round(test_m["accuracy"], 4), "precision": round(test_m["precision"], 4),
        "recall": round(test_m["recall"], 4), "specificity": round(test_m["specificity"], 4),
        "f1": round(test_m["f1"], 4), "roc_auc": round(test_m["roc_auc"], 4), "pr_auc": round(test_m["pr_auc"], 4),
        "training_time_sec": round(training_time, 1), "parameters": total_params, "epochs": epoch, "best_epoch": best_epoch,
    }
    append_comparison_row(row)

    print(f"\nMODEL: {name}")
    print("STATUS: SUCCESS")
    print(f"EPOCHS: {epoch} (best epoch {best_epoch})")
    print(f"TRAINING TIME: {training_time:.1f}s")
    print(f"ACCURACY: {test_m['accuracy']:.4f}")
    print(f"PRECISION: {test_m['precision']:.4f}")
    print(f"RECALL: {test_m['recall']:.4f}")
    print(f"SPECIFICITY: {test_m['specificity']:.4f}")
    print(f"F1: {test_m['f1']:.4f}")
    print(f"ROC-AUC: {test_m['roc_auc']:.4f}")
    print(f"PR-AUC: {test_m['pr_auc']:.4f}")
    print(f"CHECKPOINT: {ckpt_path}")
    print(f"REMAINING DISK: {disk_free_gb():.2f} GB")
    print(f"GPU: {gpu_status()}")

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return "CONTINUE"


def main():
    with open(SPLIT_PATH) as f:
        manifest = json.load(f)
    split = manifest["split"]
    c0 = manifest["split_summary"]["train"]["class0"]
    c1 = manifest["split_summary"]["train"]["class1"]
    pos_weight_val = c0 / max(c1, 1)
    print(f"Loaded split: train={len(split['train'])}p val={len(split['val'])}p test={len(split['test'])}p")
    print(f"Train class balance: class0={c0} class1={c1} -> pos_weight={pos_weight_val:.3f}")
    print(f"Device: {DEVICE} | {gpu_status()}")

    for name, timm_name, img_size, pretrained in MODEL_SPECS:
        result = run_model(name, timm_name, img_size, pretrained, split, pos_weight_val)
        if result == "STOP":
            print("\nStopping remaining experiments due to disk space floor.")
            break

    print("\n\n" + "=" * 70)
    print("ALL MODELS PROCESSED — see", COMPARISON_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()
