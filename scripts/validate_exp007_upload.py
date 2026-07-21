from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not UPLOAD.is_dir() or UPLOAD.resolve().parent != ROOT.resolve():
        raise RuntimeError(f"Unsafe or missing Upload directory: {UPLOAD.resolve()}")
    manifest_path = UPLOAD / "UPLOAD_PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "EXP-007"
    assert manifest["sha_editing_required"] is False
    local_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert manifest["expected_commit"] == local_commit
    assert (UPLOAD / "expected_commit.txt").read_text(encoding="utf-8").strip() == local_commit
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/master"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    assert remote == local_commit
    expected_files = {record["relative_path"] for record in manifest["files"]}
    observed_files = {
        path.relative_to(UPLOAD).as_posix()
        for path in UPLOAD.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert observed_files == expected_files
    for record in manifest["files"]:
        path = UPLOAD / record["relative_path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
    output = UPLOAD / manifest["empty_output_directory"]
    assert output.is_dir() and not any(output.iterdir())
    notebook = json.loads((UPLOAD / manifest["notebook"]).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "expected_commit.txt" in source
    assert "PASTE_40_CHARACTER_COMMIT_SHA" not in source
    assert "EXP-007" in source and "run_exp7_experiment" in source
    cache = UPLOAD / manifest["feature_cache"]
    assert sha256(cache) == manifest["feature_cache_sha256"]
    print(
        json.dumps(
            {
                "status": "valid",
                "experiment_id": "EXP-007",
                "commit": local_commit,
                "files": len(observed_files) + 1,
                "bytes": sum(path.stat().st_size for path in UPLOAD.rglob("*") if path.is_file()),
                "remote_master_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
