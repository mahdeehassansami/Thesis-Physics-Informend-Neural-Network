from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Thesis_v3_with_extra_graphs_tables.ipynb",
    ROOT / "notebooks" / "train_models_colab.ipynb",
]

OLD = '''REPO_DIR = Path("/content/thesis-work")
if not (REPO_DIR / ".git").exists():
    subprocess.run(["git", "clone", REPOSITORY_URL, str(REPO_DIR)], check=True)'''
NEW = '''DRIVE_WORKSPACE = Path("/content/drive/MyDrive/thesis-work")
REPO_DIR = (
    DRIVE_WORKSPACE
    if (DRIVE_WORKSPACE / "pyproject.toml").exists()
    else Path("/content/thesis-work")
)
if not (REPO_DIR / "pyproject.toml").exists():
    subprocess.run(["git", "clone", REPOSITORY_URL, str(REPO_DIR)], check=True)'''


for path in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if OLD in source:
            source = source.replace(OLD, NEW)
            cell["source"] = source.splitlines(keepends=True)
            found = True
        elif NEW in source:
            found = True
    if not found:
        raise RuntimeError(f"Expected repository setup block was not found in {path}.")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Source-finalized {path}")
