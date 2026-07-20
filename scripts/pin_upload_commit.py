from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"
NOTEBOOKS = [
    UPLOAD / "Thesis_v3_with_extra_graphs_tables.ipynb",
    UPLOAD / "notebooks" / "train_models_colab.ipynb",
]
MANIFEST = UPLOAD / "UPLOAD_PACKAGE_MANIFEST.json"
COMMIT_PATTERN = re.compile(
    r'EXPECTED_COMMIT = "(?:PASTE_40_CHARACTER_COMMIT_SHA|[0-9a-f]{40})"'
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(commit: str) -> None:
    commit = commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full 40-character lowercase Git commit SHA.")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    if not UPLOAD.is_dir() or UPLOAD.resolve().parent != ROOT.resolve():
        raise RuntimeError(f"Unsafe or missing Upload directory: {UPLOAD.resolve()}")

    for notebook in NOTEBOOKS:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        replacements = 0
        for cell in document["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            updated, cell_replacements = COMMIT_PATTERN.subn(
                f'EXPECTED_COMMIT = "{commit}"', source
            )
            if cell_replacements:
                cell["source"] = updated.splitlines(keepends=True)
                replacements += cell_replacements
        if replacements != 1:
            raise RuntimeError(
                f"Expected one commit assignment in {notebook}; found {replacements}."
            )
        notebook.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["expected_commit_state"] = f"Pinned to pushed Git commit {commit}."
    for record in manifest["files"]:
        path = UPLOAD / record["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned Upload notebooks and manifest to {commit}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/pin_upload_commit.py <40-char SHA>")
    main(sys.argv[1])
