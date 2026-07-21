from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"
CACHE_SOURCE = (
    ROOT
    / "data"
    / "processed_features"
    / "publication"
    / "exp006"
    / "controlled_synthetic_features.csv"
)
METADATA_SOURCE = CACHE_SOURCE.with_name("controlled_synthetic_metadata.json")
NOTEBOOK_SOURCE = ROOT / "notebooks" / "train_models_colab.ipynb"
INSTRUCTIONS_SOURCE = ROOT / "UPLOAD_INSTRUCTIONS.md"
EXPECTED_CACHE_SHA256 = (
    "3199282d5abf674538797b41dc97240825cf6ec80853dffb5c9f8ca4f45bfdae"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    resolved = UPLOAD.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != "Upload":
        raise RuntimeError(f"Refusing unsafe Upload target: {resolved}")
    for source in (CACHE_SOURCE, METADATA_SOURCE, NOTEBOOK_SOURCE, INSTRUCTIONS_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)
    observed_cache_sha = sha256(CACHE_SOURCE)
    if observed_cache_sha != EXPECTED_CACHE_SHA256:
        raise RuntimeError(
            f"Controlled cache changed: {observed_cache_sha} != {EXPECTED_CACHE_SHA256}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Build the EXP-007 Upload only from a clean committed worktree.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"Unexpected Git commit: {commit}")

    if UPLOAD.exists():
        shutil.rmtree(UPLOAD)
    cache = UPLOAD / "feature_cache"
    output = UPLOAD / "experiment_outputs_exp007"
    cache.mkdir(parents=True)
    output.mkdir(parents=True)
    shutil.copy2(NOTEBOOK_SOURCE, UPLOAD / "train_models_colab.ipynb")
    shutil.copy2(INSTRUCTIONS_SOURCE, UPLOAD / "UPLOAD_INSTRUCTIONS.md")
    shutil.copy2(CACHE_SOURCE, cache / CACHE_SOURCE.name)
    shutil.copy2(METADATA_SOURCE, cache / METADATA_SOURCE.name)
    (UPLOAD / "expected_commit.txt").write_text(commit + "\n", encoding="utf-8")

    inventory = []
    for path in sorted(UPLOAD.rglob("*")):
        if path.is_file() and path.name != "UPLOAD_PACKAGE_MANIFEST.json":
            inventory.append(
                {
                    "relative_path": path.relative_to(UPLOAD).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-007",
        "run_id": "exp007_synthetic_credibility_feasibility",
        "expected_commit": commit,
        "notebook": "train_models_colab.ipynb",
        "feature_cache": "feature_cache/controlled_synthetic_features.csv",
        "feature_cache_sha256": observed_cache_sha,
        "empty_output_directory": "experiment_outputs_exp007",
        "sha_editing_required": False,
        "file_count_excluding_manifest": len(inventory),
        "files": inventory,
    }
    (UPLOAD / "UPLOAD_PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fresh EXP-007 Upload package prepared at {UPLOAD}")
    print(f"Pinned pushed-source candidate: {commit}")


if __name__ == "__main__":
    main()
