"""Full verification of the approved 50-patient IDC subset: patient/patch
counts, patient-level split (built fresh from the actual files on disk,
not reused blindly), leakage check, and an image-loading sanity check with
a few sample images per split.

    python -m scripts.verify_idc_50
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

from src.utils.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "histopathology_idc"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 35, 8, 7


def main():
    patient_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"Patient directories on disk: {len(patient_dirs)}")
    assert len(patient_dirs) == 50, f"expected 50 patient dirs, found {len(patient_dirs)}"

    counts = {}
    for pdir in patient_dirs:
        c0 = len(list((pdir / "0").glob("*.png"))) if (pdir / "0").exists() else 0
        c1 = len(list((pdir / "1").glob("*.png"))) if (pdir / "1").exists() else 0
        counts[pdir.name] = {"class0": c0, "class1": c1, "total": c0 + c1}

    total_c0 = sum(v["class0"] for v in counts.values())
    total_c1 = sum(v["class1"] for v in counts.values())
    total = total_c0 + total_c1
    print(f"Total patches: {total}  class0={total_c0} ({total_c0/total*100:.1f}%)  class1={total_c1} ({total_c1/total*100:.1f}%)")
    print()

    # rebuild the same seeded split
    patients = list(counts.keys())
    rng = random.Random(SEED)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    split = {
        "train": sorted(shuffled[:N_TRAIN]),
        "val": sorted(shuffled[N_TRAIN:N_TRAIN + N_VAL]),
        "test": sorted(shuffled[N_TRAIN + N_VAL:]),
    }

    sets = {k: set(v) for k, v in split.items()}
    ov_tv = sets["train"] & sets["val"]
    ov_tt = sets["train"] & sets["test"]
    ov_vt = sets["val"] & sets["test"]
    print("Patient overlap train/val:", len(ov_tv))
    print("Patient overlap train/test:", len(ov_tt))
    print("Patient overlap val/test:", len(ov_vt))
    assert not ov_tv and not ov_tt and not ov_vt, "LEAKAGE DETECTED"
    print()

    print(f"{'Split':<6} {'Patients':<9} {'Class0':<8} {'Class1':<8} {'Total':<8}")
    summary = {}
    for name in ("train", "val", "test"):
        c0 = sum(counts[p]["class0"] for p in split[name])
        c1 = sum(counts[p]["class1"] for p in split[name])
        summary[name] = {"patients": len(split[name]), "class0": c0, "class1": c1, "total": c0 + c1}
        print(f"{name:<6} {len(split[name]):<9} {c0:<8} {c1:<8} {c0+c1:<8}")
    print()

    # image loading sanity check — a few real samples per split, verify path/label/patient consistency
    print("=== Image loading sanity check ===")
    sample_rng = random.Random(SEED)
    for name in ("train", "val", "test"):
        print(f"\n-- {name} samples --")
        candidates = split[name][:3]
        for pid in candidates:
            for cls in ("0", "1"):
                cls_dir = DATA_DIR / pid / cls
                files = list(cls_dir.glob("*.png"))
                if not files:
                    continue
                f = sample_rng.choice(files)
                img = Image.open(f)
                print(f"  patient={pid} class={cls} split={name} file={f.name} size={img.size} mode={img.mode}")
                img.close()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_patients": len(patients),
        "total_patches": total,
        "class0_total": total_c0,
        "class1_total": total_c1,
        "seed": SEED,
        "split": split,
        "split_summary": summary,
        "leakage_check": {"train_val": len(ov_tv), "train_test": len(ov_tt), "val_test": len(ov_vt)},
        "patch_counts_per_patient": counts,
    }
    out_path = SPLITS_DIR / "idc_patient_split.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved final split manifest to {out_path}")

    free = shutil.disk_usage("C:/").free
    print(f"\nDisk free space: {free/1e9:.2f} GB")


if __name__ == "__main__":
    main()
