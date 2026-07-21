from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_work.exp7a_harm_credibility import (
    finalize_exp7a_artifacts,
    load_exp7a_config,
    run_exp7a_experiment,
    validate_exp7a_config,
    validate_exp7a_runtime,
)


def load_exp7b_config(path: str | Path) -> dict[str, Any]:
    return load_exp7a_config(path)


def validate_exp7b_config(
    config: dict[str, Any],
    project_root: str | Path,
    feature_path: str | Path | None = None,
) -> dict[str, Any]:
    return validate_exp7a_config(config, project_root, feature_path)


def validate_exp7b_runtime(
    config: dict[str, Any], project_root: str | Path, feature_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return validate_exp7a_runtime(config, project_root, feature_path)


def run_exp7b_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    feature_path: str | Path,
    output_root: str | Path,
    recovery_root: str | Path | None = None,
) -> dict[str, Any]:
    return run_exp7a_experiment(
        config,
        project_root,
        feature_path,
        output_root,
        recovery_root=recovery_root,
    )


def finalize_exp7b_artifacts(root: str | Path) -> Path:
    return finalize_exp7a_artifacts(root)
