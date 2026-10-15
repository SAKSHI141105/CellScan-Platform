"""Builds the patient-level train/validation/test split for the IDC
histopathology subset (data/raw/histopathology_idc/), extracted from the
user-supplied IMAGE.zip (the Janowczyk/Madabhushi Breast Histopathology
Images / IDC dataset, 277,524 unique 50x50 patches across 279 patients).

Only a 12-patient / 12,149-patch subset (~4.4% of the full dataset) was
extracted — chosen for practicality (CPU/RTX-3050 training time within this
project's constraints), not for any accuracy target. This mirrors the same
honesty-over-convenience approach already used for the mammography pipeline
(MIAS, 322 scans total).

Split is patient-level and seeded: no patient's patches can appear in more
than one of train/val/test. Output: data/splits/idc_patient_split.json

    python -m scripts.prepare_idc_split
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.utils.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "histopathology_idc"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 8, 2, 2  # out of 12 patients


def scan_patients() -> dict[str, dict[str, int]]:
    counts = {}
    for patient_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        pid = patient_dir.name
        c0 = len(list((patient_dir / "0").glob("*.png"))) if (patient_dir / "0").exists() else 0
        c1 = len(list((patient_dir / "1").glob("*.png"))) if (patient_dir / "1").exists() else 0
        counts[pid] = {"class0": c0, "class1": c1, "total": c0 + c1}
    return counts


def main():
    counts = scan_patients()
    patients = list(counts.keys())
    assert len(patients) == N_TRAIN + N_VAL + N_TEST, (
        f"expected {N_TRAIN + N_VAL + N_TEST} patients, found {len(patients)} — "
        "re-run scripts/prepare_idc_split.py after re-extracting if the subset changed"
    )

    rng = random.Random(SEED)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    split = {
        "train": sorted(shuffled[:N_TRAIN]),
        "val": sorted(shuffled[N_TRAIN:N_TRAIN + N_VAL]),
        "test": sorted(shuffled[N_TRAIN + N_VAL:]),
    }

    # explicit leakage check
    sets = {k: set(v) for k, v in split.items()}
    overlaps = {
        "train/val": sorted(sets["train"] & sets["val"]),
        "train/test": sorted(sets["train"] & sets["test"]),
        "val/test": sorted(sets["val"] & sets["test"]),
    }
    for pair, ov in overlaps.items():
        assert not ov, f"LEAKAGE DETECTED: patient overlap {pair}: {ov}"

    print("Patient overlap train/val:", len(overlaps["train/val"]))
    print("Patient overlap train/test:", len(overlaps["train/test"]))
    print("Patient overlap val/test:", len(overlaps["val/test"]))
    print()

    print(f"{'Split':<6} {'Patients':<9} {'Class0':<8} {'Class1':<8} {'Total':<8}")
    summary = {}
    grand_c0 = grand_c1 = grand_total = 0
    for name in ("train", "val", "test"):
        c0 = sum(counts[p]["class0"] for p in split[name])
        c1 = sum(counts[p]["class1"] for p in split[name])
        total = c0 + c1
        grand_c0 += c0
        grand_c1 += c1
        grand_total += total
        summary[name] = {"patients": len(split[name]), "class0": c0, "class1": c1, "total": total}
        print(f"{name:<6} {len(split[name]):<9} {c0:<8} {c1:<8} {total:<8}")
    print(f"{'ALL':<6} {len(patients):<9} {grand_c0:<8} {grand_c1:<8} {grand_total:<8}")
    print()
    print(f"Full IDC dataset: 277,524 patches, 279 patients (source: user-supplied IMAGE.zip)")
    print(f"This subset: {grand_total} patches ({grand_total/277524*100:.2f}% of full dataset), {len(patients)} patients")
    print(f"Original full-dataset class balance: class0=198,738 (71.6%) class1=78,786 (28.4%)")
    print(f"This subset's class balance: class0={grand_c0} ({grand_c0/grand_total*100:.1f}%) class1={grand_c1} ({grand_c1/grand_total*100:.1f}%)")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "user-supplied IMAGE.zip (Janowczyk/Madabhushi Breast Histopathology Images / IDC dataset)",
        "full_dataset_patches": 277524,
        "full_dataset_patients": 279,
        "full_dataset_class0": 198738,
        "full_dataset_class1": 78786,
        "subset_seed_for_patient_selection": 42,
        "subset_patients": patients,
        "subset_patch_counts": counts,
        "split_seed": SEED,
        "split": split,
        "split_summary": summary,
        "leakage_check": {k: len(v) for k, v in overlaps.items()},
    }
    out_path = SPLITS_DIR / "idc_patient_split.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved split manifest to {out_path}")


if __name__ == "__main__":
    main()
