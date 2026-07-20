from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Thesis_v3_with_extra_graphs_tables.ipynb",
    ROOT / "notebooks" / "train_models_colab.ipynb",
]

OLD = """PROJECT_ROOT = REPO_DIR
DRIVE_ROOT = Path("/content/drive/MyDrive/thesis-work")"""
NEW = """DRIVE_ROOT = Path("/content/drive/MyDrive/thesis-work")
PROJECT_ROOT = DRIVE_ROOT if (DRIVE_ROOT / "Datasets").exists() else REPO_DIR"""


for path in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if OLD in source:
            source = source.replace(OLD, NEW)
            source = source.replace(
                "status = [\n",
                'print("Dataset root:", PROJECT_ROOT)\nstatus = [\n',
            )
            cell["source"] = source.splitlines(keepends=True)
            changed = True
    if not changed:
        raise RuntimeError(f"Expected Colab path block was not found in {path}.")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Finalized {path}")
