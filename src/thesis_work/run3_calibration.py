from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from thesis_work.experiment_runner import (
    _comparison_summary,
    evaluate_model,
    run_dataset_experiment,
    train_one_model,
)
from thesis_work.multi_dataset import (
    enabled_dataset_configs,
    load_or_extract_dataset,
    prepare_sequence_dataset,
)
from thesis_work.sequence_models import build_model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_command(arguments: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_state(project_root: Path) -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], project_root)
    status = _run_command(["git", "status", "--porcelain"], project_root)
    return {
        "commit": commit,
        "available": commit is not None,
        "dirty": bool(status) if status is not None else None,
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if cuda_available
            else None
        ),
        "cudnn_version": torch.backends.cudnn.version(),
    }


def validate_run3_runtime(config: dict[str, Any]) -> dict[str, Any]:
    environment = _environment()
    runtime = config.get("runtime", {})
    if runtime.get("require_cuda", True) and not environment["cuda_available"]:
        raise RuntimeError("Run 3 requires a CUDA GPU; select a Colab GPU runtime.")
    required_name = runtime.get("required_gpu_name_contains")
    if required_name and required_name.lower() not in str(
        environment.get("gpu_name", "")
    ).lower():
        raise RuntimeError(
            f"Run 3 requires a GPU containing {required_name!r}; "
            f"detected {environment.get('gpu_name')!r}."
        )
    return environment


def _write_source_manifest(project_root: Path, output_root: Path) -> tuple[str, int]:
    candidates = [
        project_root / "pyproject.toml",
        project_root / "requirements-colab.txt",
        project_root / "Thesis_v3_with_extra_graphs_tables.ipynb",
        *sorted((project_root / "configs").glob("*.json")),
        *sorted((project_root / "src" / "thesis_work").glob("*.py")),
    ]
    rows = []
    for path in candidates:
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(project_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = pd.DataFrame(rows).sort_values("relative_path")
    manifest.to_csv(output_root / "source_manifest.csv", index=False)
    combined = hashlib.sha256(
        "\n".join(
            f"{row.relative_path}:{row.sha256}"
            for row in manifest.itertuples(index=False)
        ).encode()
    ).hexdigest()
    return combined, len(manifest)


def _artifact_inventory(output_root: Path) -> pd.DataFrame:
    excluded = {"run_manifest.json", "artifact_inventory.csv"}
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return pd.DataFrame(rows)


def _value_slug(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _training_seeds(config: dict[str, Any]) -> list[int]:
    training = config["training"]
    explicit = [int(value) for value in training.get("seeds", [])]
    if explicit:
        if len(explicit) != int(training.get("seed_repeats", len(explicit))):
            raise ValueError("training.seeds and seed_repeats disagree.")
        return explicit
    repeats = int(training.get("seed_repeats", 1))
    base = int(config.get("seed", 42))
    stride = int(training.get("seed_stride", 1000))
    return [base + index * stride for index in range(repeats)]


def _best_epoch_physics_diagnostics(
    history: pd.DataFrame, weights: dict[str, float]
) -> dict[str, float]:
    best_index = int(history["validation_mse"].idxmin())
    best = history.loc[best_index]
    weighted_data = float(best.get("weighted_data", np.nan))
    physics_columns = [
        f"weighted_{name}"
        for name, weight in weights.items()
        if name != "data" and float(weight) != 0.0
    ]
    weighted_physics = float(
        sum(float(best.get(column, 0.0)) for column in physics_columns)
    )
    diagnostic_history = history.loc[:best_index].dropna(
        subset=["gradient_data_weighted", "gradient_physics_weighted"],
        how="any",
    )
    gradient_data = float("nan")
    gradient_physics = float("nan")
    if not diagnostic_history.empty:
        diagnostic = diagnostic_history.iloc[-1]
        gradient_data = float(diagnostic["gradient_data_weighted"])
        gradient_physics = float(diagnostic["gradient_physics_weighted"])
    return {
        "best_epoch": int(best["epoch"]),
        "best_validation_mse": float(best["validation_mse"]),
        "weighted_data_loss": weighted_data,
        "weighted_physics_loss": weighted_physics,
        "weighted_physics_to_data_ratio": weighted_physics
        / max(abs(weighted_data), 1e-12),
        "gradient_data_weighted": gradient_data,
        "gradient_physics_weighted": gradient_physics,
        "gradient_physics_to_data_ratio": gradient_physics
        / max(abs(gradient_data), 1e-12),
    }


def _load_checkpoint_model(
    checkpoint_path: Path,
    input_dim: int,
    model_config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        checkpoint["model_name"],
        input_dim=input_dim,
        model_config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def run_strong_pinn_validation_grid(
    prepared,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration = config["calibration"]
    if not calibration.get("enabled", True):
        return pd.DataFrame(), pd.DataFrame(), {}
    if dataset_config["name"] != calibration["dataset"]:
        return pd.DataFrame(), pd.DataFrame(), {}

    model_name = calibration.get("model", "strong_pinn")
    model_config = config["models"][model_name]
    base_profile = calibration["base_weight_profile"]
    base_weights = copy.deepcopy(config["weight_profiles"][base_profile])
    paris_values = [float(value) for value in calibration["paris_weights"]]
    miner_values = [float(value) for value in calibration["miner_weights"]]
    seeds = _training_seeds(config)
    calibration_root = output_root / dataset_config["name"] / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for paris_weight, miner_weight in itertools.product(paris_values, miner_values):
        profile = (
            f"strong_paris_{_value_slug(paris_weight)}"
            f"_miner_{_value_slug(miner_weight)}"
        )
        weights = copy.deepcopy(base_weights)
        weights["paris_crack_growth"] = paris_weight
        weights["palmgren_miner"] = miner_weight
        for repeat_index, seed in enumerate(seeds, start=1):
            artifact_dir = calibration_root / profile / f"seed_{repeat_index:02d}"
            started = time.time()
            try:
                model, history, _, metrics = train_one_model(
                    prepared=prepared,
                    model_name=model_name,
                    model_config=model_config,
                    weights=weights,
                    physics=config["physics"],
                    training_config=config["training"],
                    seed=seed,
                    artifact_dir=artifact_dir,
                    evaluation_split="validation",
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                diagnostics = _best_epoch_physics_diagnostics(history, weights)
                (artifact_dir / "candidate_weights.json").write_text(
                    json.dumps(weights, indent=2), encoding="utf-8"
                )
                rows.append(
                    {
                        "dataset": dataset_config["name"],
                        "model": model_name,
                        "weight_profile": profile,
                        "paris_weight": paris_weight,
                        "miner_weight": miner_weight,
                        "seed_repeat": repeat_index,
                        "seed": seed,
                        "status": "ok",
                        "selection_split": "validation",
                        "test_evaluated": False,
                        "training_seconds": time.time() - started,
                        "artifact_directory": artifact_dir.relative_to(
                            output_root
                        ).as_posix(),
                        **{f"validation_{name}": value for name, value in metrics.items()},
                        **diagnostics,
                    }
                )
            except Exception as exc:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                rows.append(
                    {
                        "dataset": dataset_config["name"],
                        "model": model_name,
                        "weight_profile": profile,
                        "paris_weight": paris_weight,
                        "miner_weight": miner_weight,
                        "seed_repeat": repeat_index,
                        "seed": seed,
                        "status": "failed",
                        "selection_split": "validation",
                        "test_evaluated": False,
                        "training_seconds": time.time() - started,
                        "error": str(exc),
                    }
                )

    grid = pd.DataFrame(rows)
    grid.to_csv(calibration_root / "validation_grid_results.csv", index=False)
    successful = grid[grid["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("Every Run 3 Strong-PINN calibration candidate failed.")
    summary = (
        successful.groupby(
            ["weight_profile", "paris_weight", "miner_weight"], as_index=False
        )
        .agg(
            seed_repeats=("seed", "nunique"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
            validation_mae_mean=("validation_mae", "mean"),
            validation_r2_mean=("validation_r2", "mean"),
            weighted_physics_to_data_ratio_mean=(
                "weighted_physics_to_data_ratio",
                "mean",
            ),
            gradient_physics_to_data_ratio_mean=(
                "gradient_physics_to_data_ratio",
                "mean",
            ),
            training_seconds_total=("training_seconds", "sum"),
        )
        .sort_values(["validation_rmse_mean", "validation_rmse_std"])
        .reset_index(drop=True)
    )
    summary["validation_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(calibration_root / "validation_grid_summary.csv", index=False)
    eligible = summary[summary["seed_repeats"] == len(seeds)]
    if eligible.empty:
        raise RuntimeError("No calibration profile completed every configured seed.")
    selected = eligible.iloc[0]
    selected_profile = str(selected["weight_profile"])
    selection = {
        "selected_profile": selected_profile,
        "paris_weight": float(selected["paris_weight"]),
        "miner_weight": float(selected["miner_weight"]),
        "selection_split": "validation",
        "selection_metric": "mean validation RMSE across configured seeds",
        "validation_rmse_mean": float(selected["validation_rmse_mean"]),
        "validation_rmse_std": float(selected["validation_rmse_std"]),
        "test_was_used_for_selection": False,
        "selected_at_utc": _utc_now(),
    }
    (calibration_root / "selected_profile.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_rows: list[dict[str, Any]] = []
    selected_validation_rows = successful[
        successful["weight_profile"] == selected_profile
    ].sort_values("seed_repeat")
    for candidate in selected_validation_rows.itertuples(index=False):
        artifact_dir = output_root / candidate.artifact_directory
        model, checkpoint = _load_checkpoint_model(
            artifact_dir / "checkpoint.pt",
            input_dim=len(prepared.feature_columns),
            model_config=model_config,
            device=device,
        )
        prediction_frame, metrics = evaluate_model(
            model,
            prepared.test,
            int(config["training"]["batch_size"]),
            device,
        )
        metrics.update(
            {
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_mse": float(checkpoint["best_validation_mse"]),
                "evaluation_split": "test",
            }
        )
        prediction_frame.to_csv(artifact_dir / "predictions.csv", index=False)
        (artifact_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        test_rows.append(
            {
                "dataset": dataset_config["name"],
                "model": model_name,
                "weight_profile": selected_profile,
                "paris_weight": float(candidate.paris_weight),
                "miner_weight": float(candidate.miner_weight),
                "seed_repeat": int(candidate.seed_repeat),
                "seed": int(candidate.seed),
                "status": "ok",
                "seconds": float(candidate.training_seconds),
                "selection_split": "validation",
                "test_evaluated": True,
                **metrics,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_test = pd.DataFrame(test_rows)
    selected_test.to_csv(
        calibration_root / "selected_test_results.csv", index=False
    )
    _comparison_summary(selected_test).to_csv(
        calibration_root / "selected_test_summary.csv", index=False
    )
    return grid, selected_test, selection


def run_run3_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    refresh_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(project_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(
            f"Run 3 output directory is not empty: {output_root}. "
            "Use a new directory to avoid overwriting an experiment."
        )
    if config.get("run_label") != "run_03":
        raise ValueError("Run 3 controller requires run_label='run_03'.")
    enabled = enabled_dataset_configs(config)
    if [dataset["name"] for dataset in enabled] != [config["calibration"]["dataset"]]:
        raise ValueError("Run 3 must enable only its calibration dataset.")

    started_utc = _utc_now()
    started = time.time()
    environment = validate_run3_runtime(config)
    clean_config = copy.deepcopy(config)
    clean_config.pop("_config_path", None)
    resolved_config_path = output_root / "resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(clean_config, indent=2), encoding="utf-8"
    )
    environment_path = output_root / "environment.json"
    environment_path.write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (output_root / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    git = _git_state(project_root)
    (output_root / "git_commit.txt").write_text(
        (git["commit"] or "UNAVAILABLE_UPLOAD_SOURCE_MANIFEST_USED") + "\n",
        encoding="utf-8",
    )
    source_tree_hash, source_files = _write_source_manifest(
        project_root, output_root
    )

    dataset_config = enabled[0]
    frame = load_or_extract_dataset(
        dataset_config,
        project_root=project_root,
        cache_dir=cache_dir,
        refresh=refresh_features,
    )
    cache_path = cache_dir / f"{dataset_config['name']}_features.csv"
    prepared = prepare_sequence_dataset(
        frame,
        dataset_config,
        sequence_length=int(config["training"]["sequence_length"]),
    )
    split_payload = {
        dataset_config["name"]: copy.deepcopy(dataset_config["split"])
    }
    (output_root / "data_split.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    dataset_summary = {
        "dataset": dataset_config["name"],
        "feature_cache": str(cache_path),
        "feature_cache_sha256": _sha256(cache_path),
        "feature_rows": len(frame),
        "feature_columns": prepared.feature_columns,
        "sequence_length": int(config["training"]["sequence_length"]),
        "time_scale_seconds": prepared.time_scale_seconds,
        "splits": {
            name: {
                "frame_rows": len(prepared.split_frames[name]),
                "sequences": len(getattr(prepared, name)),
                "run_ids": sorted(
                    prepared.split_frames[name]["run_id"].unique().tolist()
                ),
            }
            for name in ("train", "validation", "test")
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )
    (output_root / "preprocessing.json").write_text(
        json.dumps(
            {
                "fit_split": "train",
                "feature_columns": prepared.feature_columns,
                "scaler_mean": prepared.scaler.mean_.tolist(),
                "scaler_scale": prepared.scaler.scale_.tolist(),
                "sequence_length": int(config["training"]["sequence_length"]),
                "time_scale_seconds": prepared.time_scale_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    dataset_output = output_root / dataset_config["name"]
    dataset_output.mkdir(parents=True, exist_ok=True)
    (dataset_output / "assumptions.json").write_text(
        json.dumps(
            {
                "physics_assumptions": dataset_config.get(
                    "physics_assumptions", []
                ),
                "operating_conditions": dataset_config.get(
                    "operating_conditions", {}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reference_results, _ = run_dataset_experiment(
        prepared, dataset_config, config, output_root
    )
    grid, selected_test, selection = run_strong_pinn_validation_grid(
        prepared, dataset_config, config, output_root
    )
    combined = pd.concat(
        [frame for frame in (reference_results, selected_test) if not frame.empty],
        ignore_index=True,
    )
    combined.to_csv(output_root / "all_model_comparisons.csv", index=False)
    summary = _comparison_summary(combined)
    summary.to_csv(output_root / "all_model_comparisons_summary.csv", index=False)
    combined.to_csv(dataset_output / "model_comparison.csv", index=False)
    summary.to_csv(dataset_output / "model_comparison_summary.csv", index=False)

    failures = [
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("failure.txt")
    ]
    (output_root / "failure_report.json").write_text(
        json.dumps({"failures": failures}, indent=2), encoding="utf-8"
    )
    inventory = _artifact_inventory(output_root)
    inventory.to_csv(output_root / "artifact_inventory.csv", index=False)
    finished_utc = _utc_now()
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "experiment_name": config["experiment"]["name"],
        "run_id": config["run_label"],
        "status": "completed" if not failures else "partial",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": time.time() - started,
        "git": git,
        "source_tree_sha256": source_tree_hash,
        "source_file_count": source_files,
        "resolved_config_sha256": _sha256(resolved_config_path),
        "data_split_sha256": _sha256(output_root / "data_split.json"),
        "dataset_feature_cache_sha256": dataset_summary[
            "feature_cache_sha256"
        ],
        "seeds": _training_seeds(config),
        "requested_reference_models": [
            name
            for name, model in config["models"].items()
            if model.get("enabled", False)
        ],
        "calibration_candidates": len(grid),
        "selected_calibration_profile": selection,
        "completed_test_rows": int((combined["status"] == "ok").sum()),
        "failed_test_rows": int((combined["status"] != "ok").sum()),
        "environment": environment,
        "artifact_count": len(inventory),
        "failure_files": failures,
        "test_access_policy": (
            "Calibration candidates were ranked on validation only. The test split "
            "was evaluated only for the frozen selected Strong-PINN profile and "
            "predeclared reference models."
        ),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return combined, grid
