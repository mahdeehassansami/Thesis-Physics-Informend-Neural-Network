from __future__ import annotations

import copy
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from thesis_work.sequence_models import LSTMRUL


FAMILIES = [
    "linear_increasing",
    "progressively_increasing",
    "step_like",
    "gamma",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_exp7_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"EXP-007 configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def _git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "pyyaml")
        },
    }
    if cuda:
        free, total = torch.cuda.mem_get_info()
        result.update(
            {
                "gpu_memory_free_bytes": int(free),
                "gpu_memory_total_bytes": int(total),
                "selected_device": "cuda:0",
            }
        )
    else:
        result["selected_device"] = "cpu"
    return result


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _load_split(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    split_path = project_root / config["data"]["split_file"]
    document = json.loads(split_path.read_text(encoding="utf-8"))
    key = config["data"]["split_key"]
    if key not in document:
        raise KeyError(f"Split key {key!r} is absent from {split_path}.")
    return copy.deepcopy(document[key])


def validate_exp7_config(
    config: dict[str, Any],
    project_root: str | Path,
    feature_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if config.get("experiment", {}).get("id") != "EXP-007":
        raise ValueError("The active configuration is not EXP-007.")
    if config["experiment"].get("method_scope") != (
        "diagnostic_feasibility_before_high_capacity_pinn_integration"
    ):
        raise ValueError("EXP-007 method scope changed from the locked diagnostic gate.")
    if config["data"].get("target_test_access") != "evaluation_only":
        raise ValueError("Test RUL must remain evaluation-only.")
    if config["credibility"].get("freeze_before_test") is not True:
        raise ValueError("The credibility estimator must be frozen before test evaluation.")
    if len(config["training"].get("seeds", [])) < 5:
        raise ValueError("EXP-007 requires at least five declared seeds.")
    if config["training"].get("oom_policy") != "fail_and_record":
        raise ValueError("EXP-007 allows only the declared fail-and-record OOM policy.")

    split = _load_split(config, root)
    sets = {
        name: set(split[f"{name}_runs"])
        for name in ("train", "validation", "test")
    }
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Controlled trajectory splits overlap.")
    if [len(sets[name]) for name in ("train", "validation", "test")] != [24, 8, 8]:
        raise ValueError("EXP-007 requires the immutable controlled 24/8/8 split.")

    holdout_union: set[str] = set()
    for fold in config["cross_fit"]["folds"]:
        holdout = set(fold["holdout_runs"])
        validation = set(fold["validation_runs"])
        if holdout & validation:
            raise ValueError(f"Cross-fit fold {fold['fold_id']} overlaps holdout and validation.")
        if not (holdout | validation) <= sets["train"]:
            raise ValueError(f"Cross-fit fold {fold['fold_id']} leaves the training population.")
        if holdout_union & holdout:
            raise ValueError("A training trajectory appears in multiple cross-fit holdouts.")
        holdout_union |= holdout
    if holdout_union != sets["train"]:
        raise ValueError("Cross-fit holdouts must cover every controlled training trajectory once.")

    forbidden = set(config["corruptions"]["forbidden_credibility_inputs"])
    evidence = set(config["credibility"]["numeric_evidence"]) | set(
        config["credibility"]["categorical_evidence"]
    )
    overlap = sorted(forbidden & evidence)
    if overlap:
        raise ValueError(f"Forbidden target/truth fields entered credibility evidence: {overlap}")
    if config["prior_templates"].get("interpretation") != (
        "empirical_family_template_not_claimed_governing_equation"
    ):
        raise ValueError("Empirical simulator templates must not be relabeled as governing equations.")
    tolerance_low, tolerance_high = map(
        float, config["corruptions"]["validity_tolerance"]
    )
    for partition, values in config["corruptions"]["specifications"].items():
        uncertain = list(map(float, values["uncertain_scales"]))
        severe = list(map(float, values["severe_scales"]))
        if not all(tolerance_low <= value <= tolerance_high for value in uncertain):
            raise ValueError(f"{partition} uncertain scales leave the valid tolerance.")
        if not all(value < tolerance_low or value > tolerance_high for value in severe):
            raise ValueError(f"{partition} severe scales enter the valid tolerance.")

    path = Path(feature_path) if feature_path is not None else root / config["data"]["feature_cache"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = sha256_file(path)
    expected_sha = config["data"]["expected_feature_cache_sha256"]
    if observed_sha.lower() != str(expected_sha).lower():
        raise ValueError(f"Controlled feature-cache hash mismatch: {observed_sha} != {expected_sha}")

    frame = pd.read_csv(path)
    required = {
        "run_id",
        "official_partition",
        "condition_id",
        "sample_index",
        "elapsed_minutes",
        "rul_norm",
        "degradation_value",
        "degradation_family",
        *config["data"]["feature_columns"],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Controlled cache is missing columns: {missing}")
    if frame[["run_id", "sample_index"]].duplicated().any():
        raise ValueError("Controlled cache contains duplicate run/sample identifiers.")
    if not np.isfinite(frame[list(required - {"run_id", "official_partition", "condition_id", "degradation_family"})].to_numpy(dtype=float)).all():
        raise ValueError("Controlled cache contains non-finite numeric values.")
    if set(frame["degradation_family"].unique()) != set(FAMILIES):
        raise ValueError("Controlled cache progression families changed.")
    expected_membership = {
        run_id: partition for partition, ids in sets.items() for run_id in ids
    }
    observed_membership = dict(
        frame[["run_id", "official_partition"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_membership != expected_membership:
        raise ValueError("Controlled cache membership disagrees with the immutable split.")
    minimum_length = int(frame.groupby("run_id").size().min())
    if minimum_length < int(config["data"]["sequence_length"]):
        raise ValueError("A controlled trajectory is shorter than the declared sequence length.")
    return {
        "feature_path": str(path.resolve()),
        "feature_sha256": observed_sha,
        "rows": int(len(frame)),
        "runs": int(frame["run_id"].nunique()),
        "split_counts": {key: len(value) for key, value in sets.items()},
        "minimum_snapshots_per_run": minimum_length,
        "maximum_snapshots_per_run": int(frame.groupby("run_id").size().max()),
        "families": sorted(frame["degradation_family"].unique()),
    }


def validate_exp7_runtime(
    config: dict[str, Any], project_root: str | Path, feature_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qualification = validate_exp7_config(config, project_root, feature_path)
    environment = _environment()
    git = _git_state(project_root)
    runtime = config["runtime"]
    if runtime.get("require_cuda", True) and not environment["cuda_available"]:
        raise RuntimeError("EXP-007 requires a CUDA GPU; select a Colab T4 runtime.")
    required_name = runtime.get("required_gpu_name_contains")
    if required_name and required_name.lower() not in str(environment["gpu_name"]).lower():
        raise RuntimeError(
            f"EXP-007 requires a {required_name} GPU; assigned device is {environment['gpu_name']}."
        )
    expected_commit = config["repository"].get("expected_commit")
    if not expected_commit or len(str(expected_commit)) != 40:
        raise RuntimeError("EXP-007 requires the exact pushed commit from expected_commit.txt.")
    if git["commit"] != expected_commit:
        raise RuntimeError(f"Git checkout mismatch: {git['commit']} != {expected_commit}")
    if config["repository"].get("require_clean_git", True) and git["dirty"]:
        raise RuntimeError("EXP-007 refuses to train from a dirty Git checkout.")
    return environment, git, qualification


@dataclass
class BackboneFit:
    model: torch.nn.Module
    scaler: StandardScaler
    time_scale_minutes: float
    history: pd.DataFrame
    parameter_count: int
    best_epoch: int
    best_validation_mse: float


def _sequence_arrays(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
    sequence_length: int,
    stride: int,
    time_scale_minutes: float,
) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    times: list[float] = []
    targets: list[float] = []
    run_ids: list[str] = []
    sample_indices: list[int] = []
    elapsed: list[float] = []
    for run_id, run in frame.groupby("run_id", sort=True):
        run = run.sort_values("sample_index")
        values = scaler.transform(run[feature_columns].to_numpy(dtype=float))
        for end in range(sequence_length - 1, len(run), stride):
            row = run.iloc[end]
            xs.append(values[end - sequence_length + 1 : end + 1])
            times.append(float(row["elapsed_minutes"]) / time_scale_minutes)
            targets.append(float(row["rul_norm"]))
            run_ids.append(str(run_id))
            sample_indices.append(int(row["sample_index"]))
            elapsed.append(float(row["elapsed_minutes"]))
    if not xs:
        raise ValueError("No causal sequences were constructed.")
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "time": np.asarray(times, dtype=np.float32).reshape(-1, 1),
        "target": np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        "run_id": run_ids,
        "sample_index": np.asarray(sample_indices, dtype=int),
        "elapsed_minutes": np.asarray(elapsed, dtype=float),
    }


def _predict_backbone(
    fit: BackboneFit,
    frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    arrays = _sequence_arrays(
        frame,
        fit.scaler,
        config["data"]["feature_columns"],
        int(config["data"]["sequence_length"]),
        int(config["data"]["sequence_stride"]),
        fit.time_scale_minutes,
    )
    dataset = TensorDataset(
        torch.from_numpy(arrays["x"]),
        torch.from_numpy(arrays["time"]),
        torch.from_numpy(arrays["target"]),
    )
    loader = DataLoader(dataset, batch_size=int(config["backbone"]["batch_size"]), shuffle=False)
    prediction: list[np.ndarray] = []
    fit.model.eval()
    with torch.no_grad():
        for x, coordinate, _ in loader:
            prediction.append(fit.model(x.to(device), coordinate.to(device)).cpu().numpy())
    return pd.DataFrame(
        {
            "run_id": arrays["run_id"],
            "sample_index": arrays["sample_index"],
            "elapsed_minutes": arrays["elapsed_minutes"],
            "target_rul": arrays["target"].reshape(-1),
            "backbone_rul": np.clip(np.concatenate(prediction).reshape(-1), 0.0, 1.0),
        }
    )


def _train_backbone(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    artifact_dir: Path,
    phase: str,
) -> BackboneFit:
    seed_everything(seed, bool(config["training"].get("deterministic_torch", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    columns = config["data"]["feature_columns"]
    scaler = StandardScaler().fit(train_frame[columns].to_numpy(dtype=float))
    time_scale = max(float(train_frame["elapsed_minutes"].max()), 1.0)
    train_arrays = _sequence_arrays(
        train_frame,
        scaler,
        columns,
        int(config["data"]["sequence_length"]),
        int(config["data"]["sequence_stride"]),
        time_scale,
    )
    validation_arrays = _sequence_arrays(
        validation_frame,
        scaler,
        columns,
        int(config["data"]["sequence_length"]),
        int(config["data"]["sequence_stride"]),
        time_scale,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_arrays["x"]),
        torch.from_numpy(train_arrays["time"]),
        torch.from_numpy(train_arrays["target"]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_arrays["x"]),
        torch.from_numpy(validation_arrays["time"]),
        torch.from_numpy(validation_arrays["target"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["backbone"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["training"].get("dataloader_workers", 0)),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["backbone"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = LSTMRUL(len(columns), int(config["backbone"]["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["backbone"]["learning_rate"]),
        weight_decay=float(config["backbone"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(config["backbone"]["patience"]) // 3),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(config["backbone"]["patience"])
    rows: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, int(config["backbone"]["epochs"]) + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x, coordinate, target in train_loader:
            x, coordinate, target = x.to(device), coordinate.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, coordinate)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite EXP-007 backbone loss in {phase}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["backbone"]["gradient_clip"]))
            optimizer.step()
            train_sum += float(loss.detach()) * len(x)
            train_count += len(x)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for x, coordinate, target in validation_loader:
                prediction = model(x.to(device), coordinate.to(device))
                loss = torch.nn.functional.mse_loss(prediction, target.to(device), reduction="sum")
                validation_sum += float(loss)
                validation_count += len(x)
        validation_mse = validation_sum / max(1, validation_count)
        scheduler.step(validation_mse)
        rows.append(
            {
                "phase": phase,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_sum / max(1, train_count),
                "validation_loss": validation_mse,
                "data_loss": train_sum / max(1, train_count),
                "physics_loss": float("nan"),
                "train_mse": train_sum / max(1, train_count),
                "validation_mse": validation_mse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - training_started,
            }
        )
        if validation_mse < best_loss - float(config["backbone"]["minimum_improvement"]):
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_left = int(config["backbone"]["patience"])
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_state is None:
        raise RuntimeError(f"No valid backbone checkpoint was produced for {phase}.")
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(artifact_dir / "history.csv", index=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.save(
        {
            "experiment_id": "EXP-007",
            "phase": phase,
            "seed": seed,
            "feature_columns": columns,
            "sequence_length": int(config["data"]["sequence_length"]),
            "model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "time_scale_minutes": time_scale,
            "best_epoch": best_epoch,
            "best_validation_mse": best_loss,
            "parameter_count": parameter_count,
        },
        artifact_dir / "checkpoint.pt",
    )
    return BackboneFit(
        model=model,
        scaler=scaler,
        time_scale_minutes=time_scale,
        history=history,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        best_validation_mse=best_loss,
    )


@dataclass
class TemplateLibrary:
    grid: np.ndarray
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]


def fit_template_library(frame: pd.DataFrame, config: dict[str, Any]) -> TemplateLibrary:
    points = int(config["prior_templates"]["lifecycle_grid_points"])
    bandwidth = float(config["prior_templates"]["gaussian_bandwidth"])
    std_floor = float(config["prior_templates"]["minimum_standard_deviation"])
    grid = np.linspace(0.0, 1.0, points)
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    counts = frame.groupby("run_id").size().to_dict()
    for family in FAMILIES:
        subset = frame[frame["degradation_family"] == family].copy()
        if subset["run_id"].nunique() < 2:
            raise ValueError(f"At least two source trajectories are required for {family}.")
        lifecycle = 1.0 - subset["rul_norm"].to_numpy(dtype=float)
        values = subset["degradation_value"].to_numpy(dtype=float)
        run_weight = np.asarray([1.0 / counts[run] for run in subset["run_id"]], dtype=float)
        family_mean: list[float] = []
        family_std: list[float] = []
        for coordinate in grid:
            kernel = np.exp(-0.5 * ((lifecycle - coordinate) / bandwidth) ** 2) * run_weight
            kernel_sum = float(kernel.sum())
            if kernel_sum <= 1e-12:
                index = int(np.argmin(np.abs(lifecycle - coordinate)))
                family_mean.append(float(values[index]))
                family_std.append(std_floor)
                continue
            mean = float(np.sum(kernel * values) / kernel_sum)
            variance = float(np.sum(kernel * (values - mean) ** 2) / kernel_sum)
            family_mean.append(mean)
            family_std.append(max(math.sqrt(max(variance, 0.0)), std_floor))
        projected = np.maximum.accumulate(np.clip(np.asarray(family_mean), 0.0, 1.0))
        projected[-1] = 1.0
        means[family] = projected
        stds[family] = np.asarray(family_std)
    return TemplateLibrary(grid=grid, mean=means, std=stds)


def _fit_degradation_proxy(
    frame: pd.DataFrame, config: dict[str, Any], seed: int
) -> ExtraTreesRegressor:
    proxy = ExtraTreesRegressor(
        n_estimators=int(config["degradation_proxy"]["estimators"]),
        max_depth=int(config["degradation_proxy"]["max_depth"]),
        min_samples_leaf=int(config["degradation_proxy"]["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    )
    proxy.fit(
        frame[config["data"]["feature_columns"]].to_numpy(dtype=float),
        frame["degradation_value"].to_numpy(dtype=float),
    )
    return proxy


def _candidate_specs(true_family: str, partition: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    values = config["corruptions"]["specifications"][partition]
    uncertain_low, uncertain_high = map(float, values["uncertain_scales"])
    severe_low, severe_high = map(float, values["severe_scales"])
    roles = [
        ("exact", 1.0, True, "none"),
        ("uncertain_low", uncertain_low, True, "parameter_uncertainty"),
        ("uncertain_high", uncertain_high, True, "parameter_uncertainty"),
        ("parameter_fast", severe_low, False, "time_scale_fast"),
        ("parameter_slow", severe_high, False, "time_scale_slow"),
    ]
    specifications: list[dict[str, Any]] = []
    for family in FAMILIES:
        for role, factor, within_tolerance, same_family_corruption in roles:
            same_family = family == true_family
            valid = bool(same_family and within_tolerance)
            corruption = same_family_corruption if same_family else "wrong_progression_family"
            specifications.append(
                {
                    "candidate_spec": f"{family}__{role}",
                    "candidate_family": family,
                    "time_scale_factor": factor,
                    "validity_label": int(valid),
                    "corruption_type": corruption,
                }
            )
    return specifications


def build_credibility_evidence(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    fit: BackboneFit,
    proxy: ExtraTreesRegressor,
    templates: TemplateLibrary,
    config: dict[str, Any],
    partition: str,
) -> pd.DataFrame:
    source_columns = [
        "run_id",
        "sample_index",
        "condition_id",
        "degradation_family",
        "degradation_value",
        *config["data"]["feature_columns"],
    ]
    joined = predictions.merge(frame[source_columns], on=["run_id", "sample_index"], how="left", validate="one_to_one")
    if joined["degradation_family"].isna().any():
        raise ValueError("Backbone predictions could not be matched to controlled samples.")
    current = joined[config["data"]["feature_columns"]].to_numpy(dtype=float)
    proxy_trees = np.vstack([tree.predict(current) for tree in proxy.estimators_])
    proxy_mean = np.clip(proxy_trees.mean(axis=0), 0.0, 1.0)
    proxy_std = np.maximum(proxy_trees.std(axis=0), 1e-6)
    standardized = fit.scaler.transform(current)
    covariate_shift = np.sqrt(np.mean(standardized**2, axis=1))
    op_indices = [config["data"]["feature_columns"].index(name) for name in config["data"]["operating_condition_columns"]]
    operation_shift = np.sqrt(np.mean(standardized[:, op_indices] ** 2, axis=1))
    rows: list[dict[str, Any]] = []
    for position, row in joined.reset_index(drop=True).iterrows():
        backbone_damage = float(np.clip(1.0 - row["backbone_rul"], 0.0, 1.0))
        for spec in _candidate_specs(str(row["degradation_family"]), partition, config):
            family = spec["candidate_family"]
            factor = float(spec["time_scale_factor"])
            evaluated_lifecycle = float(np.clip(backbone_damage / factor, 0.0, 1.0))
            prior_degradation = float(np.interp(evaluated_lifecycle, templates.grid, templates.mean[family]))
            template_uncertainty = float(np.interp(evaluated_lifecycle, templates.grid, templates.std[family]))
            absolute_residual = abs(float(proxy_mean[position]) - prior_degradation)
            standardized_residual = absolute_residual / math.sqrt(
                proxy_std[position] ** 2 + template_uncertainty**2 + 1e-8
            )
            inverse_values, inverse_indices = np.unique(
                np.maximum.accumulate(templates.mean[family]), return_index=True
            )
            if len(inverse_values) == 1:
                inverse_lifecycle = float(templates.grid[inverse_indices[0]])
            else:
                inverse_lifecycle = float(
                    np.interp(
                        proxy_mean[position],
                        inverse_values,
                        templates.grid[inverse_indices],
                    )
                )
            prior_damage = float(np.clip(factor * inverse_lifecycle, 0.0, 1.0))
            rows.append(
                {
                    "partition": partition,
                    "run_id": row["run_id"],
                    "sample_index": int(row["sample_index"]),
                    "elapsed_minutes": float(row["elapsed_minutes"]),
                    "condition_id": row["condition_id"],
                    "true_family": row["degradation_family"],
                    "candidate_spec": spec["candidate_spec"],
                    "candidate_family": family,
                    "time_scale_factor": factor,
                    "corruption_type": spec["corruption_type"],
                    "validity_label": int(spec["validity_label"]),
                    "target_rul": float(row["target_rul"]),
                    "backbone_rul": float(row["backbone_rul"]),
                    "backbone_damage": backbone_damage,
                    "proxy_degradation": float(proxy_mean[position]),
                    "proxy_uncertainty": float(proxy_std[position]),
                    "prior_degradation": prior_degradation,
                    "prior_rul": 1.0 - prior_damage,
                    "template_absolute_residual": absolute_residual,
                    "template_standardized_residual": standardized_residual,
                    "lifecycle_disagreement": abs(prior_damage - backbone_damage),
                    "template_uncertainty": template_uncertainty,
                    "covariate_shift_score": float(covariate_shift[position]),
                    "operation_shift_score": float(operation_shift[position]),
                    "causal_prefix_length_log": math.log1p(int(row["sample_index"]) + 1),
                    "true_degradation_evaluation_only": float(row["degradation_value"]),
                }
            )
    evidence = pd.DataFrame(rows).sort_values(["run_id", "candidate_spec", "sample_index"])
    grouping = ["run_id", "candidate_spec"]
    window = int(config["data"]["sequence_length"])
    evidence["rolling_absolute_residual"] = evidence.groupby(grouping)["template_absolute_residual"].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    evidence["rolling_lifecycle_disagreement"] = evidence.groupby(grouping)["lifecycle_disagreement"].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    evidence["_prior_damage"] = 1.0 - evidence["prior_rul"]
    evidence["_prior_recovery"] = evidence.groupby(grouping)["_prior_damage"].diff().fillna(0.0) < -1e-6
    evidence["prior_monotonic_violation_rate"] = evidence.groupby(grouping)["_prior_recovery"].transform(
        lambda values: values.expanding().mean()
    )
    sample_group = ["run_id", "sample_index"]
    best = evidence.groupby(sample_group)["template_standardized_residual"].transform("min")
    evidence["residual_relative_to_best"] = evidence["template_standardized_residual"] - best
    evidence["residual_rank_fraction"] = evidence.groupby(sample_group)["template_standardized_residual"].rank(
        method="average", pct=True
    )
    return evidence.drop(columns=["_prior_damage", "_prior_recovery"]).reset_index(drop=True)


@dataclass
class CredibilityFit:
    feature_names: list[str]
    scaler: StandardScaler
    classifier: LogisticRegression
    calibrator: LogisticRegression
    threshold: float


def _credibility_matrix(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    numeric = list(config["credibility"]["numeric_evidence"])
    values = [frame[numeric].to_numpy(dtype=float)]
    names = list(numeric)
    for family in FAMILIES:
        values.append((frame["candidate_family"].to_numpy() == family).astype(float).reshape(-1, 1))
        names.append(f"candidate_family__{family}")
    matrix = np.concatenate(values, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Credibility evidence contains non-finite values.")
    return matrix, names


def _raw_probability(model: LogisticRegression, scaled: np.ndarray) -> np.ndarray:
    return np.clip(model.predict_proba(scaled)[:, 1], 1e-6, 1.0 - 1e-6)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability)).reshape(-1, 1)


def fit_credibility_estimator(
    train_evidence: pd.DataFrame,
    validation_evidence: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> CredibilityFit:
    x_train, names = _credibility_matrix(train_evidence, config)
    x_validation, validation_names = _credibility_matrix(validation_evidence, config)
    if names != validation_names:
        raise ValueError("Credibility evidence columns changed between train and validation.")
    y_train = train_evidence["validity_label"].to_numpy(dtype=int)
    y_validation = validation_evidence["validity_label"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=float(config["credibility"]["regularization_c"]),
        class_weight=config["credibility"]["class_weight"],
        max_iter=2000,
        random_state=seed,
    ).fit(scaler.transform(x_train), y_train)
    raw_validation = _raw_probability(classifier, scaler.transform(x_validation))
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        _logit(raw_validation), y_validation
    )
    calibrated = calibrator.predict_proba(_logit(raw_validation))[:, 1]
    fpr, tpr, thresholds = roc_curve(y_validation, calibrated)
    finite = np.isfinite(thresholds)
    scores = tpr[finite] - fpr[finite]
    candidate_thresholds = thresholds[finite]
    best_score = float(scores.max())
    tied = candidate_thresholds[np.isclose(scores, best_score)]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return CredibilityFit(names, scaler, classifier, calibrator, threshold)


def apply_credibility(
    fit: CredibilityFit, evidence: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    matrix, names = _credibility_matrix(evidence, config)
    if names != fit.feature_names:
        raise ValueError("Credibility feature order changed at inference.")
    raw = _raw_probability(fit.classifier, fit.scaler.transform(matrix))
    calibrated = fit.calibrator.predict_proba(_logit(raw))[:, 1]
    result = evidence.copy()
    result["credibility_raw"] = raw
    result["credibility"] = np.clip(calibrated, 0.0, 1.0)
    result["credibility_threshold"] = fit.threshold
    result["fallback"] = result["credibility"] < fit.threshold
    return result


def _ece(target: np.ndarray, probability: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return value if total else float("nan")


def aggregate_credibility_units(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["seed", "run_id", "candidate_spec", "true_family", "candidate_family", "corruption_type", "condition_id", "validity_label"]
    return (
        frame.groupby(columns, as_index=False)
        .agg(credibility=("credibility", "mean"), fallback=("fallback", "mean"))
    )


def credibility_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    units = aggregate_credibility_units(frame) if "seed" in frame else frame
    target = units["validity_label"].to_numpy(dtype=int)
    probability = units["credibility"].to_numpy(dtype=float)
    threshold = float(frame["credibility_threshold"].iloc[0]) if "credibility_threshold" in frame else 0.5
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "ece": _ece(target, probability, int(config["evaluation"]["calibration_bins"])),
        "threshold": threshold,
        "all_on_fraction": float(np.mean(probability >= threshold)),
        "all_off_fraction": float(np.mean(probability < threshold)),
        "trajectory_candidate_units": int(len(units)),
    }


def _macro_run_rmse(frame: pd.DataFrame, prediction_column: str) -> float:
    values = []
    for _, run in frame.groupby("run_id"):
        values.append(math.sqrt(mean_squared_error(run["target_rul"], run[prediction_column])))
    return float(np.mean(values))


def _choose_validation_scalar(validation: pd.DataFrame, maximum_blend: float) -> float:
    candidates = np.linspace(0.0, 1.0, 11)
    scores = []
    for scalar in candidates:
        prediction = np.clip(
            validation["backbone_rul"]
            + maximum_blend * scalar * (validation["prior_rul"] - validation["backbone_rul"]),
            0.0,
            1.0,
        )
        work = validation.assign(_prediction=prediction)
        scores.append((_macro_run_rmse(work, "_prediction"), float(scalar)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def make_control_predictions(
    evidence: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    validation_scalar: float,
) -> pd.DataFrame:
    maximum_blend = float(config["credibility"]["maximum_prior_blend"])
    residual_credibility = np.exp(-np.clip(evidence["template_standardized_residual"].to_numpy(dtype=float), 0.0, 50.0))
    rng = np.random.default_rng(seed + 7007)
    controls = {
        "data_only": np.zeros(len(evidence)),
        "priorcred_thresholded": np.where(evidence["fallback"], 0.0, evidence["credibility"]),
        "inverse_residual": residual_credibility,
        "validation_selected_scalar": np.full(len(evidence), validation_scalar),
        "random_credibility": rng.uniform(0.0, 1.0, len(evidence)),
        "all_on": np.ones(len(evidence)),
        "all_off": np.zeros(len(evidence)),
        "oracle": evidence["validity_label"].to_numpy(dtype=float),
        "anti_oracle": 1.0 - evidence["validity_label"].to_numpy(dtype=float),
    }
    rows: list[pd.DataFrame] = []
    for method, credibility in controls.items():
        blend = maximum_blend * np.asarray(credibility, dtype=float)
        prediction = np.clip(
            evidence["backbone_rul"].to_numpy(dtype=float)
            + blend * (
                evidence["prior_rul"].to_numpy(dtype=float)
                - evidence["backbone_rul"].to_numpy(dtype=float)
            ),
            0.0,
            1.0,
        )
        selected = evidence[
            [
                "partition",
                "run_id",
                "sample_index",
                "condition_id",
                "true_family",
                "candidate_spec",
                "candidate_family",
                "corruption_type",
                "validity_label",
                "target_rul",
                "backbone_rul",
                "prior_rul",
            ]
        ].copy()
        selected["method"] = method
        selected["credibility_value"] = credibility
        selected["blend_weight"] = blend
        selected["predicted_rul"] = prediction
        selected["absolute_error"] = np.abs(selected["target_rul"] - prediction)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def summarize_rul_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    for (seed, method), group in frame.groupby(["seed", "method"]):
        target = group["target_rul"].to_numpy(dtype=float)
        prediction = group["predicted_rul"].to_numpy(dtype=float)
        comparison_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "status": "completed",
                "mae": float(mean_absolute_error(target, prediction)),
                "mse": float(mean_squared_error(target, prediction)),
                "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
                "r2": float(r2_score(target, prediction)),
                "macro_run_rmse": _macro_run_rmse(group, "predicted_rul"),
                "samples": int(len(group)),
                "runs": int(group["run_id"].nunique()),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    regret_rows: list[dict[str, Any]] = []
    keys = ["seed", "validity_label", "corruption_type"]
    for key, subset in frame.groupby(keys):
        baseline = subset[subset["method"] == "data_only"]
        baseline_rmse = math.sqrt(mean_squared_error(baseline["target_rul"], baseline["predicted_rul"]))
        for method, group in subset.groupby("method"):
            rmse = math.sqrt(mean_squared_error(group["target_rul"], group["predicted_rul"]))
            regret_rows.append(
                {
                    "seed": int(key[0]),
                    "validity_label": int(key[1]),
                    "corruption_type": key[2],
                    "method": method,
                    "rmse": float(rmse),
                    "data_only_rmse": float(baseline_rmse),
                    "physics_regret": float(rmse - baseline_rmse),
                    "positive_physics_regret": float(max(0.0, rmse - baseline_rmse)),
                }
            )
    return comparison, pd.DataFrame(regret_rows)


def parameter_recovery(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    correct_family = frame[frame["candidate_family"] == frame["true_family"]].copy()
    correct_family["recovery_score"] = (
        correct_family["template_standardized_residual"]
        + correct_family["lifecycle_disagreement"]
    )
    indices = correct_family.groupby(["run_id", "sample_index"])["recovery_score"].idxmin()
    recovered = correct_family.loc[indices].copy()
    recovered["seed"] = seed
    recovered["true_time_scale_factor"] = 1.0
    recovered["absolute_scale_error"] = np.abs(recovered["time_scale_factor"] - 1.0)
    return recovered[
        [
            "seed",
            "run_id",
            "sample_index",
            "true_family",
            "candidate_spec",
            "time_scale_factor",
            "true_time_scale_factor",
            "absolute_scale_error",
            "recovery_score",
        ]
    ]


def _serialize_credibility_fit(fit: CredibilityFit) -> dict[str, Any]:
    return {
        "feature_names": fit.feature_names,
        "scaler_mean": fit.scaler.mean_.tolist(),
        "scaler_scale": fit.scaler.scale_.tolist(),
        "classifier_coefficients": fit.classifier.coef_.reshape(-1).tolist(),
        "classifier_intercept": float(fit.classifier.intercept_[0]),
        "calibrator_coefficient": float(fit.calibrator.coef_.reshape(-1)[0]),
        "calibrator_intercept": float(fit.calibrator.intercept_[0]),
        "threshold": fit.threshold,
    }


def _run_seed(
    seed: int,
    frame: pd.DataFrame,
    split: dict[str, Any],
    config: dict[str, Any],
    seed_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_population = set(split["train_runs"])
    oof_parts: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(config["cross_fit"]["folds"]):
        fold_dir = seed_dir / "crossfit" / fold["fold_id"]
        evidence_path = fold_dir / "evidence.csv"
        history_path = fold_dir / "history.csv"
        if evidence_path.exists() and history_path.exists():
            oof_parts.append(pd.read_csv(evidence_path))
            histories.append(pd.read_csv(history_path))
            continue
        holdout = set(fold["holdout_runs"])
        validation = set(fold["validation_runs"])
        inner_train = train_population - holdout - validation
        train_frame = frame[frame["run_id"].isin(inner_train)].copy()
        validation_frame = frame[frame["run_id"].isin(validation)].copy()
        holdout_frame = frame[frame["run_id"].isin(holdout)].copy()
        fit = _train_backbone(
            train_frame,
            validation_frame,
            config,
            seed + fold_index * 101,
            fold_dir,
            fold["fold_id"],
        )
        proxy = _fit_degradation_proxy(train_frame, config, seed + fold_index * 101)
        templates = fit_template_library(train_frame, config)
        predictions = _predict_backbone(fit, holdout_frame, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        evidence = build_credibility_evidence(predictions, holdout_frame, fit, proxy, templates, config, "train")
        evidence["crossfit_fold"] = fold["fold_id"]
        evidence.to_csv(evidence_path, index=False)
        oof_parts.append(evidence)
        histories.append(fit.history)

    train_evidence = pd.concat(oof_parts, ignore_index=True)
    if set(train_evidence["run_id"].unique()) != train_population:
        raise RuntimeError("Cross-fitting did not produce evidence for every training trajectory.")

    final_dir = seed_dir / "final_backbone"
    validation_evidence_path = final_dir / "validation_evidence.csv"
    test_evidence_path = final_dir / "test_evidence.csv"
    if validation_evidence_path.exists() and test_evidence_path.exists() and (final_dir / "history.csv").exists():
        validation_evidence = pd.read_csv(validation_evidence_path)
        test_evidence = pd.read_csv(test_evidence_path)
        histories.append(pd.read_csv(final_dir / "history.csv"))
        checkpoint = torch.load(final_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        final_parameter_count = int(checkpoint["parameter_count"])
        final_best_epoch = int(checkpoint["best_epoch"])
        final_validation_mse = float(checkpoint["best_validation_mse"])
    else:
        train_frame = frame[frame["run_id"].isin(split["train_runs"])].copy()
        validation_frame = frame[frame["run_id"].isin(split["validation_runs"])].copy()
        test_frame = frame[frame["run_id"].isin(split["test_runs"])].copy()
        final_fit = _train_backbone(train_frame, validation_frame, config, seed, final_dir, "final")
        proxy = _fit_degradation_proxy(train_frame, config, seed)
        templates = fit_template_library(train_frame, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation_predictions = _predict_backbone(final_fit, validation_frame, config, device)
        test_predictions = _predict_backbone(final_fit, test_frame, config, device)
        validation_evidence = build_credibility_evidence(
            validation_predictions, validation_frame, final_fit, proxy, templates, config, "validation"
        )
        test_evidence = build_credibility_evidence(
            test_predictions, test_frame, final_fit, proxy, templates, config, "test"
        )
        validation_evidence.to_csv(validation_evidence_path, index=False)
        test_evidence.to_csv(test_evidence_path, index=False)
        histories.append(final_fit.history)
        final_parameter_count = final_fit.parameter_count
        final_best_epoch = final_fit.best_epoch
        final_validation_mse = final_fit.best_validation_mse

    credibility_fit = fit_credibility_estimator(train_evidence, validation_evidence, config, seed)
    train_scored = apply_credibility(credibility_fit, train_evidence, config)
    validation_scored = apply_credibility(credibility_fit, validation_evidence, config)
    test_scored = apply_credibility(credibility_fit, test_evidence, config)
    for scored in (train_scored, validation_scored, test_scored):
        scored["seed"] = seed
    maximum_blend = float(config["credibility"]["maximum_prior_blend"])
    validation_scalar = _choose_validation_scalar(validation_scored, maximum_blend)
    validation_rul = make_control_predictions(validation_scored, config, seed, validation_scalar)
    test_rul = make_control_predictions(test_scored, config, seed, validation_scalar)
    for scored in (validation_rul, test_rul):
        scored["seed"] = seed
    metrics = credibility_metrics(test_scored, config)
    metrics.update(
        {
            "seed": seed,
            "status": "completed",
            "validation_selected_scalar": validation_scalar,
            "backbone_parameter_count": final_parameter_count,
            "backbone_best_epoch": final_best_epoch,
            "backbone_best_validation_mse": final_validation_mse,
            "seconds": time.perf_counter() - started,
        }
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(histories, ignore_index=True).to_csv(seed_dir / "training_history.csv", index=False)
    train_scored.to_csv(seed_dir / "train_credibility_predictions.csv", index=False)
    validation_scored.to_csv(seed_dir / "validation_credibility_predictions.csv", index=False)
    test_scored.to_csv(seed_dir / "test_credibility_predictions.csv", index=False)
    validation_rul.to_csv(seed_dir / "validation_rul_predictions.csv", index=False)
    test_rul.to_csv(seed_dir / "test_rul_predictions.csv", index=False)
    parameter_recovery(test_scored, seed).to_csv(seed_dir / "parameter_recovery.csv", index=False)
    (seed_dir / "credibility_estimator.json").write_text(
        json.dumps(_serialize_credibility_fit(credibility_fit), indent=2), encoding="utf-8"
    )
    (seed_dir / "job_result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _bootstrap_auroc(units: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    seeds = sorted(units["seed"].unique())
    runs = sorted(units["run_id"].unique())
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    estimates: list[float] = []
    for _ in range(int(config["evaluation"]["bootstrap_replicates"])):
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        parts: list[pd.DataFrame] = []
        for run in sampled_runs:
            for seed in sampled_seeds:
                parts.append(units[(units["run_id"] == run) & (units["seed"] == seed)])
        sample = pd.concat(parts, ignore_index=True)
        if sample["validity_label"].nunique() == 2:
            estimates.append(float(roc_auc_score(sample["validity_label"], sample["credibility"])))
    if not estimates:
        raise RuntimeError("Trajectory/seed bootstrap produced no valid AUROC estimates.")
    return {
        "replicates_requested": int(config["evaluation"]["bootstrap_replicates"]),
        "replicates_valid": len(estimates),
        "auroc_ci_lower_95": float(np.quantile(estimates, 0.025)),
        "auroc_ci_upper_95": float(np.quantile(estimates, 0.975)),
        "auroc_bootstrap_median": float(np.median(estimates)),
    }


def _plots(output_root: Path, credibility: pd.DataFrame, comparison: pd.DataFrame) -> None:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    units = aggregate_credibility_units(credibility)
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(units["validity_label"], units["credibility"])
    ax.plot(fpr, tpr, label=f"AUROC={roc_auc_score(units['validity_label'], units['credibility']):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.5")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="EXP-007 held-out credibility ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "credibility_roc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name in ((0, "corrupt"), (1, "valid")):
        values = units.loc[units["validity_label"] == label, "credibility"]
        ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
    ax.set(xlabel="Credibility", ylabel="Trajectory-candidate count", title="Credibility separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "credibility_distribution.png", dpi=180)
    plt.close(fig)

    summary = comparison.groupby("method", as_index=False)["macro_run_rmse"].mean().sort_values("macro_run_rmse")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["method"], summary["macro_run_rmse"])
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Mean macro trajectory RMSE", title="EXP-007 feasibility controls")
    fig.tight_layout()
    fig.savefig(plot_dir / "rul_control_comparison.png", dpi=180)
    plt.close(fig)


def _copy_completed_state(source: Path, destination: Path | None) -> None:
    if destination is None:
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _write_summary(output_root: Path, gate: dict[str, Any], metrics: pd.DataFrame) -> None:
    rows = [
        "# EXP-007 synthetic credibility feasibility",
        "",
        f"Status: **{gate['status']}**",
        "",
        f"Held-out trajectory-candidate AUROC: `{gate['aggregate_auroc']:.6f}`",
        f"Trajectory/seed bootstrap 95% interval: `[{gate['auroc_ci_lower_95']:.6f}, {gate['auroc_ci_upper_95']:.6f}]`",
        f"Mean fallback fraction: `{gate['mean_fallback_fraction']:.6f}`",
        f"Decision: **{gate['decision']}**",
        "",
        "The RUL blending results are a diagnostic feasibility analysis. They are not the final",
        "physics-loss integration or a real-bearing publication claim.",
        "",
        "## Per-seed credibility metrics",
        "",
        "```text",
        metrics.to_string(index=False),
        "```",
    ]
    (output_root / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def finalize_exp7_artifacts(root: str | Path) -> Path:
    root = Path(root)
    excluded_names = {"run_manifest.json", "artifact_inventory.csv", "codex_results_bundle.zip"}
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded_names:
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(records).to_csv(root / "artifact_inventory.csv", index=False)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_count_excluding_manifest_inventory_bundle"] = len(records)
        manifest["artifact_inventory_sha256"] = sha256_file(root / "artifact_inventory.csv")
        notebook = root / "executed_notebook.ipynb"
        manifest["executed_notebook_sha256"] = sha256_file(notebook) if notebook.exists() else None
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = root / "codex_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    binary_patterns = ["*.pt", "*.pth", "*.pkl", "*.joblib"]
    duplicate_or_training_evidence_patterns = [
        "seeds/*/crossfit/*/evidence.csv",
        "seeds/*/train_credibility_predictions.csv",
        "seeds/*/validation_credibility_predictions.csv",
        "seeds/*/test_credibility_predictions.csv",
        "seeds/*/validation_rul_predictions.csv",
        "seeds/*/test_rul_predictions.csv",
        "predictions/credibility_predictions.csv",
        "predictions/rul_predictions.csv",
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == bundle:
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in binary_patterns):
                continue
            if any(
                fnmatch.fnmatch(relative, pattern)
                for pattern in duplicate_or_training_evidence_patterns
            ):
                continue
            archive.write(path, relative)
    return bundle


def run_exp7_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    feature_path: str | Path,
    output_root: str | Path,
    recovery_root: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    feature_path = Path(feature_path).resolve()
    output_root = Path(output_root).resolve()
    recovery = Path(recovery_root).resolve() if recovery_root is not None else None
    if recovery is not None and recovery.exists() and not output_root.exists():
        shutil.copytree(recovery, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()
    environment, git, qualification = validate_exp7_runtime(config, project_root, feature_path)
    split = _load_split(config, project_root)
    resolved_config = copy.deepcopy(config)
    resolved_config.pop("_config_path", None)
    identity = {
        "experiment_id": "EXP-007",
        "git_commit": git["commit"],
        "config_sha256": _json_sha256(resolved_config),
        "feature_sha256": qualification["feature_sha256"],
        "split_sha256": _json_sha256(split),
        "seeds": list(config["training"]["seeds"]),
    }
    state_path = output_root / "run_state.json"
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing EXP-007 recovery state is incompatible with this run.")
    else:
        state_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    log_path = output_root / "training.log"
    _append_log(
        log_path,
        f"EXP-007 start/resume commit={git['commit']} feature_sha256={qualification['feature_sha256']}",
    )
    Path(output_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    (output_root / "git_commit.txt").write_text(git["commit"] + "\n", encoding="utf-8")
    (output_root / "environment.txt").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (output_root / "dataset_summary.json").write_text(json.dumps(qualification, indent=2), encoding="utf-8")
    (output_root / "data_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    frame = pd.read_csv(feature_path)
    started_utc = _utc_now()
    failures: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]:
        seed = int(seed)
        seed_dir = output_root / "seeds" / f"seed_{seed:05d}"
        job_path = seed_dir / "job_result.json"
        if job_path.exists():
            result = json.loads(job_path.read_text(encoding="utf-8"))
            if result.get("status") == "completed" and int(result.get("seed")) == seed:
                _append_log(log_path, f"seed={seed} resume-skip status=completed")
                continue
        try:
            result = _run_seed(seed, frame, split, config, seed_dir)
            _append_log(
                log_path,
                f"seed={seed} status=completed auroc={result['auroc']:.6f} seconds={result['seconds']:.3f}",
            )
            _copy_completed_state(output_root, recovery)
        except Exception as exc:
            seed_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "seed": seed,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            (seed_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            _append_log(log_path, f"seed={seed} status=failed error={type(exc).__name__}: {exc}")
            _copy_completed_state(output_root, recovery)

    completed_dirs = [
        output_root / "seeds" / f"seed_{int(seed):05d}"
        for seed in config["training"]["seeds"]
        if (output_root / "seeds" / f"seed_{int(seed):05d}" / "job_result.json").exists()
    ]
    if not completed_dirs:
        (output_root / "failure_report.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        _copy_completed_state(output_root, recovery)
        raise RuntimeError("Every EXP-007 seed failed; inspect failure_report.json.")
    credibility_frames = [pd.read_csv(path / "test_credibility_predictions.csv") for path in completed_dirs]
    rul_frames = [pd.read_csv(path / "test_rul_predictions.csv") for path in completed_dirs]
    history_frames = [pd.read_csv(path / "training_history.csv") for path in completed_dirs]
    recovery_frames = [pd.read_csv(path / "parameter_recovery.csv") for path in completed_dirs]
    credibility = pd.concat(credibility_frames, ignore_index=True)
    rul = pd.concat(rul_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    recovered = pd.concat(recovery_frames, ignore_index=True)
    credibility.to_csv(output_root / "credibility_predictions.csv", index=False)
    rul.to_csv(output_root / "rul_predictions.csv", index=False)
    history.to_csv(output_root / "training_history.csv", index=False)
    recovered.to_csv(output_root / "parameter_recovery.csv", index=False)
    metric_rows = []
    for seed, group in credibility.groupby("seed"):
        row = credibility_metrics(group, config)
        row["seed"] = int(seed)
        metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(output_root / "credibility_metrics.csv", index=False)
    comparison, regret = summarize_rul_predictions(rul)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    regret.to_csv(output_root / "physics_regret.csv", index=False)
    units = aggregate_credibility_units(credibility)
    aggregate_metrics = {
        "auroc": float(roc_auc_score(units["validity_label"], units["credibility"])),
        "auprc": float(average_precision_score(units["validity_label"], units["credibility"])),
        "brier": float(brier_score_loss(units["validity_label"], units["credibility"])),
        "ece": _ece(
            units["validity_label"].to_numpy(dtype=int),
            units["credibility"].to_numpy(dtype=float),
            int(config["evaluation"]["calibration_bins"]),
        ),
    }
    bootstrap = _bootstrap_auroc(units, config)
    statistical = {**aggregate_metrics, **bootstrap, "aggregation_unit": "trajectory_candidate_then_seed"}
    (output_root / "statistical_summary.json").write_text(json.dumps(statistical, indent=2), encoding="utf-8")
    metrics_dir = output_root / "metrics"
    predictions_dir = output_root / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "credibility_metrics.json").write_text(
        json.dumps(
            {
                "aggregate": aggregate_metrics,
                "bootstrap": bootstrap,
                "per_seed": metric_table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(output_root / "credibility_predictions.csv", predictions_dir / "credibility_predictions.csv")
    shutil.copy2(output_root / "rul_predictions.csv", predictions_dir / "rul_predictions.csv")
    mean_fallback = float(units["fallback"].mean())
    passes_auc = aggregate_metrics["auroc"] >= float(config["success_criteria"]["minimum_test_auroc"])
    passes_ci = bootstrap["auroc_ci_lower_95"] > float(config["success_criteria"]["minimum_auroc_ci_lower"])
    max_fraction = float(config["success_criteria"]["maximum_all_on_fraction_without_explanation"])
    passes_collapse = mean_fallback <= max_fraction and (1.0 - mean_fallback) <= max_fraction
    decision = "proceed_to_exp008" if passes_auc and passes_ci and passes_collapse and not failures else "stop_and_diagnose_identifiability"
    gate = {
        "experiment_id": "EXP-007",
        "status": "completed" if len(completed_dirs) == len(config["training"]["seeds"]) else "partial",
        "aggregate_auroc": aggregate_metrics["auroc"],
        **bootstrap,
        "mean_fallback_fraction": mean_fallback,
        "mean_all_on_fraction": 1.0 - mean_fallback,
        "passes_minimum_auroc": passes_auc,
        "passes_ci_lower_bound": passes_ci,
        "passes_anti_collapse": passes_collapse,
        "decision": decision,
    }
    (output_root / "gate_decision.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (output_root / "failure_report.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    _append_log(log_path, f"gate decision={decision} aggregate_auroc={aggregate_metrics['auroc']:.6f}")
    _plots(output_root, credibility, comparison)
    _write_summary(output_root, gate, metric_table)
    requested = [int(seed) for seed in config["training"]["seeds"]]
    completed = [int(path.name.split("_")[-1]) for path in completed_dirs]
    manifest = {
        "experiment_id": "EXP-007",
        "run_id": config["experiment"]["run_id"],
        "status": gate["status"],
        "git_commit": git["commit"],
        "config_sha256": identity["config_sha256"],
        "split_sha256": identity["split_sha256"],
        "dataset_id": config["data"]["dataset_id"],
        "dataset_fingerprint": qualification["feature_sha256"],
        "requested_seeds": requested,
        "completed_seeds": completed,
        "failed_seeds": sorted(set(requested) - set(completed)),
        "requested_models": ["causal_lstm_backbone", "extra_trees_degradation_proxy", "logistic_credibility_estimator"],
        "completed_models": ["causal_lstm_backbone", "extra_trees_degradation_proxy", "logistic_credibility_estimator"] if completed else [],
        "skipped_baselines": config["comparators"].get("excluded", {}),
        "environment": environment,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "gate_decision": decision,
        "test_access_policy": "Test RUL and validity labels are used only after backbone, templates, proxy, classifier, calibration, threshold, and scalar baseline are frozen.",
        "resume_policy": config["training"]["recovery_granularity"],
        "oom_policy": config["training"]["oom_policy"],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finalize_exp7_artifacts(output_root)
    _copy_completed_state(output_root, recovery)
    return gate
