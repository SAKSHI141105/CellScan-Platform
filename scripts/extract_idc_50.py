"""Extracts the 38 additional patients needed to bring the IDC histopathology
subset from 12 patients to the approved 50-patient final split, without
touching the 12 already on disk or the original IMAGE.zip.

    python -m scripts.extract_idc_50
"""
from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path

from src.utils.config import PROJECT_ROOT

ZIP_PATH = r"C:\Users\paree\Downloads\IMAGE.zip"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "histopathology_idc"
SEED = 42
TARGET_PATIENTS = 50
EXISTING_12 = {"8913", "10258", "8916", "9254", "9176", "16568", "12895", "10262", "10259", "12810", "9177", "13023"}

IDC_PATIENTS_JSON = Path(r"C:\Users\paree\AppData\Local\Temp\claude\C--Users-paree-OneDrive-Desktop-CellScan\e3ed67ad-f255-4b45-ac60-c13535b216f0\scratchpad\idc_patients.json")

if __name__ == "__main__":
    with open(IDC_PATIENTS_JSON) as f:
        raw = json.load(f)

    patient_ids = list(raw.keys())  # same insertion order used originally
    rng = random.Random(SEED)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)

    chosen = shuffled[:TARGET_PATIENTS]
    assert set(chosen) >= EXISTING_12, "reproduced shuffle does not contain the existing 12 patients — seed/order mismatch"
    new_patients = set(chosen) - EXISTING_12
    print(f"Need to extract {len(new_patients)} new patients (of {len(chosen)} total in the 50-patient set)")

    pattern = re.compile(r"^(\d+)/([01])/(\d+_idx5_x\d+_y\d+_class[01]\.png)$")
    z = zipfile.ZipFile(ZIP_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for name in z.namelist():
        if name.startswith("IDC_regular_ps50_idx5/") or name.endswith("/"):
            continue
        m = pattern.match(name)
        if not m:
            continue
        patient, cls, fname = m.groups()
        if patient not in new_patients:
            continue
        dest_dir = OUT_DIR / patient / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname
        if dest_path.exists():
            continue
        with z.open(name) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        extracted += 1
        if extracted % 2000 == 0:
            print(f"  extracted {extracted}")

    print(f"done. extracted {extracted} new files.")
    print(f"Total patients now on disk: {len(list(OUT_DIR.iterdir()))}")
