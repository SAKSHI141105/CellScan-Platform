"""Converts the MIAS mammography dataset into the layout the mammography
pipeline expects: 512x512 PNG images, binary lesion masks, and a manifest CSV
with [image_path, mask_path, label, patient_id] — the same schema
research/mammography_generalization/src/data/dataset.py reads.

    python -m scripts.prepare_mias

MIAS is used here because it is genuinely, anonymously downloadable, unlike
CBIS-DDSM (TCIA + NBIA Data Retriever), VinDr-Mammo (PhysioNet credentialing)
or INbreast (author request) — see research/mammography_generalization/README.md.
The tradeoff is real and worth stating plainly: 322 images total with ~50
malignant cases is a *small* dataset by any modern standard, so expect a
weak-but-real model rather than a clinically meaningful one.

Two MIAS-specific conventions this script has to get right:

  1. Lesion annotations are a centroid + approximate radius, not a traced
     boundary. The masks written here are therefore filled circles — a
     deliberate approximation, not ground-truth segmentation. That's still
     useful as the auxiliary lesion-guidance signal (it tells the backbone
     roughly where to attend) but it is NOT a substitute for real pixel-level
     annotation, and any IoU-style number computed against it means less than
     the same number against CBIS-DDSM's traced ROIs.
  2. MIAS y-coordinates have their origin at the BOTTOM-left of the image,
     while every raster format (and OpenCV) puts it at the top-left. Not
     flipping this puts every mask on the wrong half of the breast, which
     would train the guidance head on confidently-wrong locations.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

MIAS_NATIVE_SIZE = 1024  # every MIAS scan is 1024x1024


def parse_info(info_path: Path) -> dict[str, dict]:
    """REFNUM BG CLASS SEVERITY X Y RADIUS — SEVERITY/X/Y/RADIUS absent for
    NORM rows, and a single refnum can appear on multiple lines when a scan
    has more than one annotated lesion.
    """
    records: dict[str, dict] = {}
    with open(info_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == "REFNUM":
                continue
            refnum = parts[0]
            rec = records.setdefault(refnum, {"malignant": False, "lesions": []})

            if len(parts) >= 4 and parts[3] in ("B", "M"):
                if parts[3] == "M":
                    rec["malignant"] = True
                # a few rows carry severity but no usable coordinates
                if len(parts) >= 7:
                    try:
                        rec["lesions"].append((int(parts[4]), int(parts[5]), int(parts[6])))
                    except ValueError:
                        logger.warning("Unparseable lesion coordinates for %s: %s", refnum, parts[4:7])
    return records


def build_mask(lesions: list[tuple[int, int, int]], out_size: int) -> np.ndarray:
    """Filled circles at each annotated lesion, unioned into one mask, drawn
    at native resolution then resized so the scale factor applies to the
    radius too.
    """
    mask = np.zeros((MIAS_NATIVE_SIZE, MIAS_NATIVE_SIZE), dtype=np.uint8)
    for x, y, radius in lesions:
        # flip y: MIAS measures from the bottom edge, images are top-down
        cv2.circle(mask, (x, MIAS_NATIVE_SIZE - y), radius, color=255, thickness=-1)
    return cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def main():
    cfg = load_config()
    img_size = cfg["mammography"]["img_size"]

    source = Path(
        r"C:\Users\paree\.cache\kagglehub\datasets\kmader\mias-mammography\versions\3"
    )
    info_path = source / "Info.txt"
    scans_dir = source / "all-mias"
    if not info_path.exists() or not scans_dir.exists():
        raise FileNotFoundError(
            f"MIAS source not found under {source}. Download it first:\n"
            "  python -c \"import kagglehub; print(kagglehub.dataset_download('kmader/mias-mammography'))\""
        )

    out_root = PROJECT_ROOT / "data" / "raw" / "mammography"
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = parse_info(info_path)
    logger.info("Parsed %d annotated reference numbers from Info.txt", len(records))

    rows = []
    n_masked = n_malignant = 0
    for pgm in sorted(scans_dir.glob("*.pgm")):
        refnum = pgm.stem
        rec = records.get(refnum, {"malignant": False, "lesions": []})

        image = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Could not read %s — skipping", pgm.name)
            continue
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        image_path = images_dir / f"{refnum}.png"
        cv2.imwrite(str(image_path), image)

        mask_rel = ""
        if rec["lesions"]:
            mask = build_mask(rec["lesions"], img_size)
            mask_path = masks_dir / f"{refnum}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_rel = str(mask_path.relative_to(out_root)).replace("\\", "/")
            n_masked += 1

        label = 1 if rec["malignant"] else 0
        n_malignant += label
        rows.append(
            {
                "image_path": str(image_path.relative_to(out_root)).replace("\\", "/"),
                "mask_path": mask_rel,
                "label": label,
                "patient_id": refnum,
            }
        )

    manifest = out_root / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "label", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Wrote %d rows to %s (%d malignant, %d benign/normal, %d with lesion masks)",
        len(rows), manifest, n_malignant, len(rows) - n_malignant, n_masked,
    )


if __name__ == "__main__":
    main()
