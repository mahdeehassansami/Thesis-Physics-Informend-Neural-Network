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
    assert manifest["experiment_id"] == "EXP-007A"
    assert manifest["run_id"] == config["experiment"]["run_id"]
    assert manifest["sha_editing_required"] is False
    assert len(manifest["expected_commit"]) == 40
    assert (UPLOAD / "expected_commit.txt").read_text(encoding="utf-8").strip() == manifest[
        "expected_commit"
    ]
    cache = UPLOAD / manifest["feature_cache"]
    assert cache.is_file()
    assert sha256(cache) == config["data"]["expected_feature_cache_sha256"]
    assert sha256(cache) == manifest["feature_cache_sha256"]
    metadata = UPLOAD / manifest["metadata"]
    assert metadata.is_file()
    assert sha256(metadata) == config["data"]["expected_metadata_sha256"]
    assert sha256(metadata) == manifest["metadata_sha256"]
    assert (UPLOAD / manifest["empty_output_directory"]).is_dir()
    assert not any((UPLOAD / manifest["empty_output_directory"]).iterdir())
    for record in manifest["files"]:
        path = UPLOAD / record["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == int(record["bytes"])
        assert sha256(path) == record["sha256"]
    notebook = json.loads((UPLOAD / manifest["notebook"]).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "EXP-007A" in "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_exp7a_experiment" in code
    assert "expected_commit.txt" in code
    assert "nvidia-smi" in code
    assert "class " not in code and "def " not in code
    print(
        json.dumps(
            {
                "status": "passed",
                "experiment_id": "EXP-007A",
                "commit": manifest["expected_commit"],
                "files": len(manifest["files"]),
                "cache_sha256": manifest["feature_cache_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
