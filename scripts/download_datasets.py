"""Downloads the real datasets behind both pipelines and organizes them into
the layout the rest of the codebase already expects:

    data/raw/wdbc.csv                     (tabular_preprocessing.py)
    data/raw/histopathology/0/*.png       (or benign/, see image_preprocessing.py)
    data/raw/histopathology/1/*.png       (or malignant/)

Both {0,1} and {benign,malignant} label-folder names are already supported
by discover_image_paths() in src/data_preprocessing/image_preprocessing.py,
so this script doesn't invent a new layout — it just fills in the one that
already exists.

Needs a Kaggle account + API token (kagglehub reads the same ~/.kaggle/kaggle.json
the official `kaggle` CLI uses, or the KAGGLE_USERNAME/KAGGLE_KEY env vars).
Get one at https://www.kaggle.com/settings -> API -> "Create New Token".

    python -m scripts.download_datasets --target tabular
    python -m scripts.download_datasets --target histopathology --dataset breakhis
    python -m scripts.download_datasets --target histopathology --dataset idc
    python -m scripts.download_datasets --target all --dataset breakhis

No Kaggle token, or don't want to use the API? Download the zip yourself from
the Kaggle page (printed below on auth failure, or see MANUAL_DOWNLOAD_URLS)
and point this script at it instead — no API call, just extraction/sorting:

    python -m scripts.download_datasets --target histopathology --dataset breakhis --zip path/to/downloaded.zip
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

TABULAR_KAGGLE_SLUG = "uciml/breast-cancer-wisconsin-data"
MANUAL_DOWNLOAD_URLS = {
    "tabular": f"https://www.kaggle.com/datasets/{TABULAR_KAGGLE_SLUG}",
    "idc": "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
    "breakhis": "https://www.kaggle.com/datasets/ambarish/breakhis",
}

# Case-insensitive path-token buckets for sorting each dataset's (differently
# structured, differently nested) files into our flat benign/malignant
# layout. rglob-based, so nesting depth inside the source dataset doesn't
# matter — only whether one of these tokens shows up somewhere in the path.
LABEL_TOKENS = {
    "idc": {"benign": ["\\0\\", "/0/"], "malignant": ["\\1\\", "/1/"]},
    "breakhis": {"benign": ["benign"], "malignant": ["malignant"]},
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _auth_help(dataset_key: str) -> str:
    return (
        f"Kaggle authentication failed. Either:\n"
        f"  1) Get an API token: https://www.kaggle.com/settings -> API -> 'Create New Token',\n"
        f"     save the downloaded kaggle.json to ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME/KAGGLE_KEY), or\n"
        f"  2) Download the zip by hand from {MANUAL_DOWNLOAD_URLS[dataset_key]} and re-run this script with "
        f"--zip path/to/that.zip"
    )


def download_tabular(zip_path: str | None = None):
    raw_dir = PROJECT_ROOT / load_config()["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = raw_dir / "wdbc.csv"

    if zip_path:
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside {zip_path}")
            with zf.open(csv_names[0]) as src, open(out_csv, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        try:
            import kagglehub

            dataset_dir = Path(kagglehub.dataset_download(TABULAR_KAGGLE_SLUG))
        except Exception as exc:
            raise RuntimeError(_auth_help("tabular")) from exc

        csvs = list(dataset_dir.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in downloaded dataset at {dataset_dir}")
        # kagglehub caches its own copy outside the repo; move (not copy) our
        # one file out of it so we don't hold two copies of anything on disk
        shutil.move(str(csvs[0]), out_csv)

    logger.info("Tabular dataset ready at %s", out_csv)


def _sort_into_labels(source_root: Path, dest_root: Path, tokens: dict[str, list[str]]):
    dest_root.mkdir(parents=True, exist_ok=True)
    counts = {"benign": 0, "malignant": 0, "skipped": 0}

    for f in source_root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        path_str = str(f).lower()
        label = next((lbl for lbl, toks in tokens.items() if any(t in path_str for t in toks)), None)
        if label is None:
            counts["skipped"] += 1
            continue
        label_dir = dest_root / label
        label_dir.mkdir(exist_ok=True)
        dest = label_dir / f.name
        if dest.exists():  # duplicate filenames across patient folders are common in both datasets
            dest = label_dir / f"{f.parent.name}_{f.name}"
        # move, not copy — the source is either kagglehub's own cache or an
        # extracted zip we don't need afterward, and disk space is the
        # binding constraint here, not preserving that intermediate copy
        shutil.move(str(f), dest)
        counts[label] += 1

    return counts


def download_histopathology(dataset: str, zip_path: str | None = None):
    if dataset not in LABEL_TOKENS:
        raise ValueError(f"unknown dataset {dataset!r}, choose from {list(LABEL_TOKENS)}")

    image_root = PROJECT_ROOT / load_config()["paths"]["image_root"]

    if zip_path:
        import tempfile
        import zipfile

        extract_dir = Path(tempfile.mkdtemp(prefix=f"cellscan_{dataset}_"))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        source_root = extract_dir
    else:
        try:
            import kagglehub

            slug = "paultimothymooney/breast-histopathology-images" if dataset == "idc" else "ambarish/breakhis"
            source_root = Path(kagglehub.dataset_download(slug))
        except Exception as exc:
            raise RuntimeError(_auth_help(dataset)) from exc

    logger.info("Sorting %s images from %s into %s/{benign,malignant}...", dataset, source_root, image_root)
    counts = _sort_into_labels(source_root, image_root, LABEL_TOKENS[dataset])
    logger.info(
        "Done: %d benign, %d malignant, %d files skipped (unrecognized path pattern) -> %s",
        counts["benign"], counts["malignant"], counts["skipped"], image_root,
    )
    if counts["benign"] == 0 or counts["malignant"] == 0:
        logger.warning(
            "One class ended up empty — the downloaded dataset's folder layout may not match what this "
            "script expects. Check %s by hand and adjust LABEL_TOKENS in this script if needed.", source_root,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["tabular", "histopathology", "all"], default="all")
    parser.add_argument("--dataset", choices=["idc", "breakhis"], default="breakhis",
                         help="which histopathology source to use (ignored for --target tabular). "
                              "breakhis defaults smaller-on-disk than the full IDC set.")
    parser.add_argument("--zip", help="path to a manually downloaded zip, instead of calling the Kaggle API")
    args = parser.parse_args()

    if args.target in ("tabular", "all"):
        download_tabular(zip_path=args.zip if args.target == "tabular" else None)
    if args.target in ("histopathology", "all"):
        download_histopathology(args.dataset, zip_path=args.zip if args.target == "histopathology" else None)


if __name__ == "__main__":
    main()
