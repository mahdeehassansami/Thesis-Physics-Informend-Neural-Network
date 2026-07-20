from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Thesis_v3_with_extra_graphs_tables.ipynb",
    ROOT / "notebooks" / "train_models_colab.ipynb",
]

OLD_WORKSPACE = 'DRIVE_WORKSPACE = Path("/content/drive/MyDrive/thesis-work")'
NEW_WORKSPACE = '''DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/Upload"),
    Path("/content/drive/MyDrive/thesis-work"),
]
DRIVE_WORKSPACE = next(
    (
        candidate
        for candidate in DRIVE_CANDIDATES
        if (candidate / "pyproject.toml").exists()
    ),
    DRIVE_CANDIDATES[0],
)'''
OLD_ROOT = 'DRIVE_ROOT = Path("/content/drive/MyDrive/thesis-work")'
NEW_ROOT = "DRIVE_ROOT = DRIVE_WORKSPACE"


for path in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = json.dumps(notebook)
    if "MyDrive/Upload" not in source:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            code = code.replace(OLD_WORKSPACE, NEW_WORKSPACE)
            code = code.replace(OLD_ROOT, NEW_ROOT)
            cell["source"] = code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Upload-finalized {path}")
