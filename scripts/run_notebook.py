"""Executes notebooks/breast_cancer_model_comparison.ipynb for real, cell by
cell, using the project's own venv as the kernel (no separate Jupyter kernel
install needed). Streams each cell's stdout to the console as it finishes,
and saves the notebook after every cell — so partial progress (already-
completed models' real outputs) survives a crash or interruption.

    python -m scripts.run_notebook
"""
from __future__ import annotations

import sys
import time

import nbformat
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.utils.config import PROJECT_ROOT

NB_PATH = PROJECT_ROOT / "notebooks" / "breast_cancer_model_comparison.ipynb"
KERNEL_NAME = "cellscan-venv"


def ensure_kernel_registered():
    ksm = KernelSpecManager()
    try:
        ksm.get_kernel_spec(KERNEL_NAME)
        return
    except NoSuchKernel:
        pass
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "ipykernel", "install", "--user",
        "--name", KERNEL_NAME, "--display-name", "CellScan venv",
    ])
    print(f"Registered Jupyter kernel '{KERNEL_NAME}' using {sys.executable}")


def cell_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
        elif out.get("output_type") == "error":
            parts.append("ERROR: " + "\n".join(out.get("traceback", [])))
    return "".join(parts)


def main():
    ensure_kernel_registered()
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(nb, kernel_name=KERNEL_NAME, timeout=-1, resources={"metadata": {"path": str(NB_PATH.parent)}})

    print(f"Executing {len(nb.cells)} cells from {NB_PATH}")
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            t0 = time.time()
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as e:
                print(f"\n=== CELL {i} RAISED AN ERROR ({time.time()-t0:.1f}s) ===")
                print(str(e)[:3000])
                # keep going — a later cell (or the next model's cell) may still be able to run
            text = cell_text(cell)
            if text.strip():
                print(f"\n--- cell {i} output ({time.time()-t0:.1f}s) ---")
                print(text)
            nbformat.write(nb, NB_PATH)  # save after every cell — incremental safety

    nbformat.write(nb, NB_PATH)
    print("\n\nNotebook execution complete. Saved to", NB_PATH)


if __name__ == "__main__":
    main()
