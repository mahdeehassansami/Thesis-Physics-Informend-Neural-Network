from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((UPLOAD / "UPLOAD_PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["experiment_id"] != "EXP-007B":
        raise ValueError("Upload manifest is not EXP-007B.")
    if manifest["run_id"] != config["experiment"]["run_id"]:
        raise ValueError("Upload run identity differs from the active configuration.")
    commit = manifest["expected_commit"]
    if len(commit) != 40 or (UPLOAD / "expected_commit.txt").read_text(
        encoding="utf-8"
    ).strip() != commit:
        raise ValueError("Upload commit pin is missing or malformed.")
    cache = UPLOAD / manifest["feature_cache"]
    metadata = UPLOAD / manifest["metadata"]
    if sha256(cache) != config["data"]["expected_feature_cache_sha256"]:
        raise ValueError("Upload feature cache hash changed.")
    if sha256(metadata) != config["data"]["expected_metadata_sha256"]:
        raise ValueError("Upload metadata hash changed.")
    output = UPLOAD / manifest["empty_output_directory"]
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("EXP-007B recovery directory must begin empty.")
    for record in manifest["files"]:
        path = UPLOAD / record["relative_path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Upload inventory mismatch: {path}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Upload file hash mismatch: {path}")
    notebook = json.loads((UPLOAD / manifest["notebook"]).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    required = ["EXP-007B", "run_exp7b_experiment", "expected_commit.txt", "nvidia-smi"]
    missing = [value for value in required if value not in source]
    if missing or "class " in code or "def " in code:
        raise ValueError(f"Upload notebook controller is invalid: missing={missing}")
    print(
        json.dumps(
            {
                "status": "passed",
                "experiment_id": "EXP-007B",
                "commit": commit,
                "files": len(manifest["files"]),
                "cache_sha256": manifest["feature_cache_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
