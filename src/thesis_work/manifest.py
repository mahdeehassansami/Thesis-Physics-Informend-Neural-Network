from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from thesis_work.config import ProjectPaths, expected_cols
from thesis_work.ims import list_snapshots, load_snapshot, resolve_dataset_source


PACKAGE_NAMES = [
    "deepxde",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    "torch",
    "tqdm",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256_file(path),
    }


def directory_hashes(path: Path, pattern: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {item.name: file_manifest(item) for item in sorted(path.glob(pattern)) if item.is_file()}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def raw_data_manifest(paths: ProjectPaths) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for dataset in ("1st_test", "2nd_test", "3rd_test"):
        source = resolve_dataset_source(paths.raw_data, dataset)
        files = list_snapshots(source)
        first_shape = None
        if files:
            first_shape = list(load_snapshot(source, files[0], expected_cols(dataset)).shape)
        rows[dataset] = {
            "source": str(source),
            "snapshots": len(files),
            "first_file": files[0] if files else None,
            "last_file": files[-1] if files else None,
            "first_snapshot_shape": first_shape,
        }
    return rows


def write_run_manifest(
    paths: ProjectPaths,
    *,
    command: str,
    options: dict[str, Any],
    seeds: dict[str, int],
    filename: str = "run_manifest.json",
) -> Path:
    paths.ensure_output_dirs()
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "options": options,
        "seeds": seeds,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "raw_data": raw_data_manifest(paths),
        "feature_cache_hashes": directory_hashes(paths.processed_features, "*.csv"),
        "result_table_hashes": directory_hashes(paths.tables, "*.csv"),
    }
    output = paths.tables / filename
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved manifest: {output}")
    return output
