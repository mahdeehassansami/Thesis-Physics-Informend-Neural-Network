from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"
CONFIG_SOURCE = ROOT / "configs" / "experiment.yaml"
NOTEBOOK_SOURCE = ROOT / "notebooks" / "train_models_colab.ipynb"
INSTRUCTIONS_SOURCE = ROOT / "UPLOAD_INSTRUCTIONS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_SOURCE.read_text(encoding="utf-8"))
    if config.get("experiment", {}).get("id") != "EXP-007A":
        raise RuntimeError("The active configuration is not EXP-007A.")
    cache_source = ROOT / config["data"]["feature_cache"]
    metadata_source = ROOT / config["data"]["metadata_file"]
    expected_cache_sha256 = config["data"]["expected_feature_cache_sha256"]
    if expected_cache_sha256 == "PENDING_AFTER_FROZEN_SIMULATION":
        raise RuntimeError("EXP-007A cache identity has not been finalized.")
    resolved = UPLOAD.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != "Upload":
        raise RuntimeError(f"Refusing unsafe Upload target: {resolved}")
    for source in (cache_source, metadata_source, NOTEBOOK_SOURCE, INSTRUCTIONS_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)
    observed_cache_sha = sha256(cache_source)
    if observed_cache_sha != expected_cache_sha256:
        raise RuntimeError(
            f"EXP-007A cache changed: {observed_cache_sha} != {expected_cache_sha256}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Build the EXP-007A Upload only from a clean committed worktree.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"Unexpected Git commit: {commit}")
    upstream = subprocess.run(
        ["git", "rev-parse", "@{upstream}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream != commit:
        raise RuntimeError(f"Push EXP-007A before building Upload: HEAD={commit}, upstream={upstream}")

    if UPLOAD.exists():
        shutil.rmtree(UPLOAD)
    cache = UPLOAD / "feature_cache"
    output = UPLOAD / "experiment_outputs_exp007a"
    cache.mkdir(parents=True)
    output.mkdir(parents=True)
    shutil.copy2(NOTEBOOK_SOURCE, UPLOAD / "train_models_colab.ipynb")
    shutil.copy2(INSTRUCTIONS_SOURCE, UPLOAD / "UPLOAD_INSTRUCTIONS.md")
    shutil.copy2(cache_source, cache / cache_source.name)
    shutil.copy2(metadata_source, cache / metadata_source.name)
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
        "experiment_id": "EXP-007A",
        "run_id": "exp007a_counterfactual_physics_harm",
        "expected_commit": commit,
        "notebook": "train_models_colab.ipynb",
        "feature_cache": "feature_cache/multicondition_features.csv",
        "feature_cache_sha256": observed_cache_sha,
        "empty_output_directory": "experiment_outputs_exp007a",
        "sha_editing_required": False,
        "file_count_excluding_manifest": len(inventory),
        "files": inventory,
    }
    (UPLOAD / "UPLOAD_PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fresh EXP-007A Upload package prepared at {UPLOAD}")
    print(f"Pinned pushed-source candidate: {commit}")


if __name__ == "__main__":
    main()
