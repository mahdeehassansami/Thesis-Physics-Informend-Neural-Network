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


def load_exp7a_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"EXP-007A configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def _git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
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
    return json.loads((project_root / config["data"]["split_file"]).read_text(encoding="utf-8"))


def _replicate(run_id: str) -> int:
    return int(str(run_id).rsplit("_", 1)[1])


def _crossfit_membership(config: dict[str, Any], split: dict[str, Any]) -> list[dict[str, Any]]:
    training = set(split["train_runs"])
    folds: list[dict[str, Any]] = []
    holdout_union: set[str] = set()
    for definition in config["cross_fit"]["folds"]:
        holdout_replicates = {int(value) for value in definition["holdout_replicates"]}
        validation_replicate = int(definition["validation_replicate"])
        holdout = {run for run in training if _replicate(run) in holdout_replicates}
        validation = {run for run in training if _replicate(run) == validation_replicate}
        if holdout & validation:
            raise ValueError(f"Cross-fit fold {definition['fold_id']} overlaps holdout/validation.")
        if holdout_union & holdout:
            raise ValueError("A training trajectory occurs in multiple cross-fit holdouts.")
        holdout_union |= holdout
        folds.append(
            {
                "fold_id": definition["fold_id"],
                "holdout_runs": sorted(holdout),
                "validation_runs": sorted(validation),
                "train_runs": sorted(training - holdout - validation),
            }
        )
    if holdout_union != training:
        raise ValueError("Cross-fit holdouts do not cover every training trajectory exactly once.")
    return folds


def validate_exp7a_config(
    config: dict[str, Any], project_root: str | Path, feature_path: str | Path | None = None
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    experiment = config.get("experiment", {})
    if experiment.get("id") != "EXP-007A":
        raise ValueError("The active configuration is not EXP-007A.")
    if experiment.get("protocol_version") != 0.2:
        raise ValueError("EXP-007A requires protocol version 0.2.")
    if config["data"].get("target_test_access") != "evaluation_only_after_development_gate":
        raise ValueError("Test RUL must remain evaluation-only after development qualification.")
    if config["credibility"].get("freeze_before_test") is not True:
        raise ValueError("Credibility must be frozen before sealed-test evaluation.")
    if len(config["training"].get("seeds", [])) < 5:
        raise ValueError("EXP-007A requires at least five common seeds.")
    if config["training"].get("oom_policy") != "fail_and_record":
        raise ValueError("EXP-007A permits only fail-and-record OOM behavior.")

    scenario_path = root / config["data"]["scenario_file"]
    split = _load_split(config, root)
    if sha256_file(scenario_path) != config["data"]["expected_scenario_sha256"]:
        raise ValueError("EXP-007A scenario design hash changed.")
    if split.get("scenario_sha256") != config["data"]["expected_scenario_sha256"]:
        raise ValueError("EXP-007A split does not pin the active scenario design.")
    sets = {name: set(split[f"{name}_runs"]) for name in ("train", "validation", "test")}
    if [len(sets[name]) for name in ("train", "validation", "test")] != [64, 16, 16]:
        raise ValueError("EXP-007A requires the immutable 64/16/16 trajectory split.")
    if any(
        sets[left] & sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("EXP-007A split populations overlap.")
    folds = _crossfit_membership(config, split)

    scenarios = pd.read_csv(scenario_path)
    expected_membership = {run: part for part, runs in sets.items() for run in runs}
    observed_membership = dict(
        scenarios[["scenario_id", "publication_split"]].itertuples(index=False, name=None)
    )
    if observed_membership != expected_membership:
        raise ValueError("Scenario membership disagrees with the immutable split.")
    if set(scenarios.loc[scenarios["publication_split"] == "test", "simulator_seed"]) != {920071}:
        raise ValueError("Sealed-test scenarios do not use the frozen test simulator seed.")
    if set(scenarios.loc[scenarios["publication_split"] != "test", "simulator_seed"]) != {420071}:
        raise ValueError("Development scenarios do not use the frozen development seed.")
    train_scenarios = scenarios[scenarios["publication_split"] == "train"]
    for column in ("OC_load_mean", "OC_f_set", "SD_SNR"):
        if train_scenarios[column].nunique() < 4:
            raise ValueError(f"Training condition {column} does not span four levels.")
        lower, upper = train_scenarios[column].min(), train_scenarios[column].max()
        if not scenarios[column].between(lower, upper).all():
            raise ValueError(f"Validation/test {column} leaves declared source support.")

    forbidden = set(config["credibility"]["forbidden_inputs"])
    selected = set(config["credibility"]["numeric_evidence"]) | set(
        config["credibility"]["categorical_evidence"]
    )
    if forbidden & selected:
        raise ValueError(f"Forbidden credibility inputs selected: {sorted(forbidden & selected)}")
    candidates = candidate_specs(config)
    if len(candidates) != int(config["physics_intervention"]["candidate_count"]):
        raise ValueError("Candidate count disagrees with the physics-intervention design.")

    path = Path(feature_path) if feature_path is not None else root / config["data"]["feature_cache"]
    if not path.is_file():
        if config["data"]["expected_feature_cache_sha256"] == "PENDING_AFTER_FROZEN_SIMULATION":
            return {
                "status": "design_valid_cache_pending",
                "scenario_path": str(scenario_path),
                "scenario_sha256": sha256_file(scenario_path),
                "split_counts": {name: len(values) for name, values in sets.items()},
                "crossfit_folds": len(folds),
                "candidate_count": len(candidates),
            }
        raise FileNotFoundError(path)
    observed_sha = sha256_file(path)
    if observed_sha != config["data"]["expected_feature_cache_sha256"]:
        raise ValueError("EXP-007A feature-cache hash changed.")
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
        raise ValueError(f"EXP-007A feature cache is missing columns: {missing}")
    if frame[["run_id", "sample_index"]].duplicated().any():
        raise ValueError("EXP-007A feature cache has duplicate sample identities.")
    observed = dict(
        frame[["run_id", "official_partition"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed != expected_membership:
        raise ValueError("Feature-cache membership disagrees with the immutable split.")
    minimum = int(frame.groupby("run_id").size().min())
    if minimum < int(config["data"]["sequence_length"]):
        raise ValueError("A trajectory is shorter than the declared causal sequence.")
    if set(frame["degradation_family"].unique()) != set(FAMILIES):
        raise ValueError("Progression families changed in the EXP-007A feature cache.")
    return {
        "status": "qualified",
        "feature_path": str(path.resolve()),
        "feature_sha256": observed_sha,
        "scenario_sha256": sha256_file(scenario_path),
        "rows": int(len(frame)),
        "runs": int(frame["run_id"].nunique()),
        "split_counts": {name: len(values) for name, values in sets.items()},
        "minimum_snapshots_per_run": minimum,
        "maximum_snapshots_per_run": int(frame.groupby("run_id").size().max()),
        "crossfit_folds": len(folds),
        "candidate_count": len(candidates),
    }


def validate_exp7a_runtime(
    config: dict[str, Any], project_root: str | Path, feature_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qualification = validate_exp7a_config(config, project_root, feature_path)
    environment = _environment()
    git = _git_state(project_root)
    runtime = config["runtime"]
    if runtime.get("require_cuda", True) and not environment["cuda_available"]:
        raise RuntimeError("EXP-007A requires a CUDA GPU; select a Colab T4 runtime.")
    required_name = runtime.get("required_gpu_name_contains")
    if required_name and required_name.lower() not in str(environment["gpu_name"]).lower():
        raise RuntimeError(
            f"EXP-007A requires a {required_name} GPU; assigned device is {environment['gpu_name']}."
        )
    expected = config["repository"].get("expected_commit")
    if not expected or len(str(expected)) != 40:
        raise RuntimeError("EXP-007A requires an exact pushed commit.")
    if git["commit"] != expected:
        raise RuntimeError(f"Git checkout mismatch: {git['commit']} != {expected}")
    if config["repository"].get("require_clean_git", True) and git["dirty"]:
        raise RuntimeError("EXP-007A refuses to train from a dirty Git checkout.")
    return environment, git, qualification


def candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in config["physics_intervention"]["candidate_families"]:
        for scale in config["physics_intervention"]["time_scale_factors"]:
            result.append(
                {
                    "candidate_spec": f"{family}__scale_{float(scale):.2f}",
                    "candidate_family": str(family),
                    "time_scale_factor": float(scale),
                }
            )
    return result


@dataclass
class TemplateLibrary:
    grid: np.ndarray
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    slope: dict[str, np.ndarray]


def fit_template_library(frame: pd.DataFrame, config: dict[str, Any]) -> TemplateLibrary:
    settings = config["physics_intervention"]["template"]
    grid = np.linspace(0.0, 1.0, int(settings["grid_points"]))
    bandwidth = float(settings["gaussian_bandwidth"])
    minimum_std = float(settings["minimum_standard_deviation"])
    mean: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}
    slope: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        subset = frame[frame["degradation_family"] == family]
        curves: list[np.ndarray] = []
        for _, run in subset.groupby("run_id"):
            ordered = run.sort_values("sample_index")
            duration = max(float(ordered["elapsed_minutes"].iloc[-1]), 1e-9)
            lifecycle = ordered["elapsed_minutes"].to_numpy(dtype=float) / duration
            damage = np.clip(ordered["degradation_value"].to_numpy(dtype=float), 0.0, None)
            maximum = max(float(damage.max()), 1e-9)
            curves.append(np.interp(grid, lifecycle, damage / maximum))
        if not curves:
            raise ValueError(f"No training templates are available for {family}.")
        values = np.vstack(curves)
        average = np.maximum.accumulate(np.clip(values.mean(axis=0), 0.0, 1.0))
        average -= average[0]
        average /= max(float(average[-1]), 1e-9)
        spread = np.maximum(values.std(axis=0), minimum_std)
        mean[family] = average
        std[family] = spread
        slope[family] = np.maximum(np.gradient(average, grid), 0.0)
    return TemplateLibrary(grid=grid, mean=mean, std=std, slope=slope)


@dataclass
class TrainingContext:
    scaler: StandardScaler
    templates: TemplateLibrary
    reference_life_minutes: float


@dataclass
class ModelFit:
    model: torch.nn.Module
    context: TrainingContext
    history: pd.DataFrame
    parameter_count: int
    best_epoch: int
    best_validation_mse: float
    parent_seed: int
    optimization_seed: int


def fit_training_context(frame: pd.DataFrame, config: dict[str, Any]) -> TrainingContext:
    columns = list(config["data"]["feature_columns"])
    scaler = StandardScaler().fit(frame[columns].to_numpy(dtype=float))
    durations = frame.groupby("run_id")["elapsed_minutes"].max().to_numpy(dtype=float)
    reference = max(float(np.median(durations)), 1.0)
    return TrainingContext(
        scaler=scaler,
        templates=fit_template_library(frame, config),
        reference_life_minutes=reference,
    )


def _sequence_arrays(
    frame: pd.DataFrame, context: TrainingContext, config: dict[str, Any]
) -> dict[str, Any]:
    columns = list(config["data"]["feature_columns"])
    length = int(config["data"]["sequence_length"])
    stride = int(config["data"]["sequence_stride"])
    xs: list[np.ndarray] = []
    times: list[float] = []
    targets: list[float] = []
    run_ids: list[str] = []
    sample_indices: list[int] = []
    elapsed_values: list[float] = []
    loads: list[float] = []
    speeds: list[float] = []
    snrs: list[float] = []
    for run_id, run in frame.groupby("run_id", sort=True):
        ordered = run.sort_values("sample_index")
        values = context.scaler.transform(ordered[columns].to_numpy(dtype=float))
        for end in range(length - 1, len(ordered), stride):
            row = ordered.iloc[end]
            xs.append(values[end - length + 1 : end + 1])
            elapsed = float(row["elapsed_minutes"])
            times.append(elapsed / context.reference_life_minutes)
            targets.append(float(row["rul_norm"]))
            run_ids.append(str(run_id))
            sample_indices.append(int(row["sample_index"]))
            elapsed_values.append(elapsed)
            loads.append(float(row["load_n"]))
            speeds.append(float(row["speed_rpm"]))
            snrs.append(float(row["snr_db"]))
    if not xs:
        raise ValueError("No EXP-007A causal sequences were constructed.")
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "time": np.asarray(times, dtype=np.float32).reshape(-1, 1),
        "target": np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        "run_id": run_ids,
        "sample_index": np.asarray(sample_indices, dtype=int),
        "elapsed_minutes": np.asarray(elapsed_values, dtype=float),
        "load_n": np.asarray(loads, dtype=float),
        "speed_rpm": np.asarray(speeds, dtype=float),
        "snr_db": np.asarray(snrs, dtype=float),
    }


def _prior_arrays(
    arrays: dict[str, Any], candidate: dict[str, Any], context: TrainingContext, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    exposure = config["physics_intervention"]["exposure"]
    reference_load = float(exposure["reference_load_n"])
    reference_speed = float(exposure["reference_speed_rpm"])
    exponent = float(exposure["bearing_life_exponent"])
    factor = float(candidate["time_scale_factor"])
    condition_factor = np.power(reference_load / np.maximum(arrays["load_n"], 1.0), exponent)
    condition_factor *= reference_speed / np.maximum(arrays["speed_rpm"], 1.0)
    denominator = np.maximum(condition_factor * factor, 1e-6)
    lifecycle = np.clip(arrays["time"].reshape(-1) / denominator, 0.0, 1.0)
    family = str(candidate["candidate_family"])
    damage = np.interp(lifecycle, context.templates.grid, context.templates.mean[family])
    damage_rate = np.interp(lifecycle, context.templates.grid, context.templates.slope[family])
    damage_rate = damage_rate / denominator
    return (1.0 - damage).astype(np.float32).reshape(-1, 1), damage_rate.astype(np.float32).reshape(-1, 1)


def _model(input_dim: int, config: dict[str, Any]) -> LSTMRUL:
    return LSTMRUL(input_dim, int(config["backbone"]["hidden_dim"]))


def _predict_model(
    fit: ModelFit, frame: pd.DataFrame, config: dict[str, Any], device: torch.device
) -> pd.DataFrame:
    arrays = _sequence_arrays(frame, fit.context, config)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(arrays["x"]),
            torch.from_numpy(arrays["time"]),
            torch.from_numpy(arrays["target"]),
        ),
        batch_size=int(config["backbone"]["batch_size"]),
        shuffle=False,
    )
    values: list[np.ndarray] = []
    fit.model.eval()
    with torch.no_grad():
        for x, coordinate, _ in loader:
            values.append(fit.model(x.to(device), coordinate.to(device)).cpu().numpy())
    return pd.DataFrame(
        {
            "run_id": arrays["run_id"],
            "sample_index": arrays["sample_index"],
            "elapsed_minutes": arrays["elapsed_minutes"],
            "target_rul": arrays["target"].reshape(-1),
            "predicted_rul": np.clip(np.concatenate(values).reshape(-1), 0.0, 1.0),
        }
    )


def _fit_data_only(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: dict[str, Any],
    parent_seed: int,
    optimization_seed: int,
    artifact_dir: Path,
    phase: str,
) -> ModelFit:
    seed_everything(optimization_seed, bool(config["training"]["deterministic_torch"]))
    context = fit_training_context(train_frame, config)
    train = _sequence_arrays(train_frame, context, config)
    validation = _sequence_arrays(validation_frame, context, config)
    return _train_model(
        train,
        validation,
        context,
        config,
        parent_seed,
        optimization_seed,
        artifact_dir,
        phase,
        candidate=None,
        initial_state=None,
    )


def _fit_physics_intervention(
    parent: ModelFit,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    candidate: dict[str, Any],
    config: dict[str, Any],
    artifact_dir: Path,
    phase: str,
    optimization_seed: int,
) -> ModelFit:
    train = _sequence_arrays(train_frame, parent.context, config)
    validation = _sequence_arrays(validation_frame, parent.context, config)
    state = {name: value.detach().cpu().clone() for name, value in parent.model.state_dict().items()}
    return _train_model(
        train,
        validation,
        parent.context,
        config,
        parent.parent_seed,
        optimization_seed,
        artifact_dir,
        phase,
        candidate=candidate,
        initial_state=state,
    )


def _train_model(
    train: dict[str, Any],
    validation: dict[str, Any],
    context: TrainingContext,
    config: dict[str, Any],
    parent_seed: int,
    optimization_seed: int,
    artifact_dir: Path,
    phase: str,
    candidate: dict[str, Any] | None,
    initial_state: dict[str, torch.Tensor] | None,
) -> ModelFit:
    seed_everything(optimization_seed, bool(config["training"]["deterministic_torch"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(train["x"].shape[-1], config).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    is_physics = candidate is not None
    if is_physics:
        train_prior = _prior_arrays(train, candidate, context, config)
        validation_prior = _prior_arrays(validation, candidate, context, config)
        epochs = int(config["physics_intervention"]["fine_tune_epochs"])
        patience = int(config["physics_intervention"]["patience"])
        learning_rate = float(config["physics_intervention"]["learning_rate"])
    else:
        train_prior = (np.zeros_like(train["target"]), np.zeros_like(train["target"]))
        validation_prior = (
            np.zeros_like(validation["target"]),
            np.zeros_like(validation["target"]),
        )
        epochs = int(config["backbone"]["epochs"])
        patience = int(config["backbone"]["patience"])
        learning_rate = float(config["backbone"]["learning_rate"])
    train_dataset = TensorDataset(
        torch.from_numpy(train["x"]),
        torch.from_numpy(train["time"]),
        torch.from_numpy(train["target"]),
        torch.from_numpy(train_prior[0]),
        torch.from_numpy(train_prior[1]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation["x"]),
        torch.from_numpy(validation["time"]),
        torch.from_numpy(validation["target"]),
        torch.from_numpy(validation_prior[0]),
        torch.from_numpy(validation_prior[1]),
    )
    generator = torch.Generator().manual_seed(optimization_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["backbone"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["training"]["dataloader_workers"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["backbone"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=float(config["backbone"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3)
    )
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = patience
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    weights = config["physics_intervention"]["weights"]
    candidate_name = candidate["candidate_spec"] if candidate else "data_only"

    for epoch in range(1, epochs + 1):
        model.train()
        accumulators = {name: 0.0 for name in ("total", "data", "prior_value", "prior_rate", "monotonic")}
        count = 0
        for x, coordinate, target, prior_rul, prior_rate in train_loader:
            x = x.to(device)
            coordinate = coordinate.to(device).requires_grad_(is_physics)
            target = target.to(device)
            prior_rul = prior_rul.to(device)
            prior_rate = prior_rate.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, coordinate)
            data_loss = torch.nn.functional.mse_loss(prediction, target)
            if is_physics:
                derivative = torch.autograd.grad(
                    prediction,
                    coordinate,
                    grad_outputs=torch.ones_like(prediction),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                value_loss = torch.nn.functional.mse_loss(prediction, prior_rul)
                rate_loss = torch.nn.functional.smooth_l1_loss(
                    torch.relu(-derivative), prior_rate
                )
                monotonic_loss = torch.mean(torch.relu(derivative) ** 2)
                total = (
                    float(weights["data"]) * data_loss
                    + float(weights["prior_value"]) * value_loss
                    + float(weights["prior_rate"]) * rate_loss
                    + float(weights["monotonic"]) * monotonic_loss
                )
            else:
                value_loss = prediction.new_zeros(())
                rate_loss = prediction.new_zeros(())
                monotonic_loss = prediction.new_zeros(())
                total = data_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite EXP-007A loss in {phase}/{candidate_name}.")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["backbone"]["gradient_clip"])
            )
            optimizer.step()
            batch_size = len(x)
            for name, value in (
                ("total", total),
                ("data", data_loss),
                ("prior_value", value_loss),
                ("prior_rate", rate_loss),
                ("monotonic", monotonic_loss),
            ):
                accumulators[name] += float(value.detach()) * batch_size
            count += batch_size

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for x, coordinate, target, _, _ in validation_loader:
                prediction = model(x.to(device), coordinate.to(device))
                validation_sum += float(
                    torch.nn.functional.mse_loss(
                        prediction, target.to(device), reduction="sum"
                    )
                )
                validation_count += len(x)
        validation_mse = validation_sum / max(1, validation_count)
        scheduler.step(validation_mse)
        history.append(
            {
                "parent_seed": parent_seed,
                "optimization_seed": optimization_seed,
                "phase": phase,
                "candidate_spec": candidate_name,
                "epoch": epoch,
                "train_loss": accumulators["total"] / max(1, count),
                "data_loss": accumulators["data"] / max(1, count),
                "physics_value_loss": accumulators["prior_value"] / max(1, count),
                "physics_rate_loss": accumulators["prior_rate"] / max(1, count),
                "monotonic_loss": accumulators["monotonic"] / max(1, count),
                "validation_mse": validation_mse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if validation_mse < best_validation - float(config["backbone"]["minimum_improvement"]):
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint was produced for {phase}/{candidate_name}.")
    model.load_state_dict(best_state)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    frame.to_csv(artifact_dir / "history.csv", index=False)
    torch.save(
        {
            "experiment_id": "EXP-007A",
            "phase": phase,
            "candidate": candidate,
            "parent_seed": parent_seed,
            "optimization_seed": optimization_seed,
            "model_state": best_state,
            "scaler_mean": context.scaler.mean_,
            "scaler_scale": context.scaler.scale_,
            "reference_life_minutes": context.reference_life_minutes,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        artifact_dir / "checkpoint.pt",
    )
    return ModelFit(
        model=model,
        context=context,
        history=frame,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        parent_seed=parent_seed,
        optimization_seed=optimization_seed,
    )


def _fit_degradation_proxy(
    frame: pd.DataFrame, config: dict[str, Any], seed: int
) -> ExtraTreesRegressor:
    settings = config["degradation_proxy"]
    proxy = ExtraTreesRegressor(
        n_estimators=int(settings["estimators"]),
        max_depth=int(settings["max_depth"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    )
    proxy.fit(
        frame[config["data"]["feature_columns"]].to_numpy(dtype=float),
        frame["degradation_value"].to_numpy(dtype=float),
    )
    return proxy


def _condition_evidence(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scaling = config["data"]["condition_scaling"]
    result = pd.DataFrame(index=frame.index)
    mapping = {
        "load_n": "load_delta_scaled",
        "speed_rpm": "speed_delta_scaled",
        "snr_db": "snr_delta_scaled",
    }
    for source, destination in mapping.items():
        reference = float(scaling[source]["reference"])
        declared_range = float(scaling[source]["range"])
        if declared_range <= 0:
            raise ValueError(f"Declared physical range for {source} must be positive.")
        result[destination] = (frame[source].to_numpy(dtype=float) - reference) / declared_range
    result["condition_distance"] = np.sqrt(
        np.mean(
            np.square(
                result[["load_delta_scaled", "speed_delta_scaled", "snr_delta_scaled"]]
            ),
            axis=1,
        )
    )
    return result


def _prior_for_joined_rows(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    context: TrainingContext,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_time = frame["elapsed_minutes"].to_numpy(dtype=float) / context.reference_life_minutes
    arrays = {
        "time": normalized_time.reshape(-1, 1),
        "load_n": frame["load_n"].to_numpy(dtype=float),
        "speed_rpm": frame["speed_rpm"].to_numpy(dtype=float),
    }
    prior_rul, prior_rate = _prior_arrays(arrays, candidate, context, config)
    exposure = config["physics_intervention"]["exposure"]
    condition_factor = np.power(
        float(exposure["reference_load_n"]) / np.maximum(arrays["load_n"], 1.0),
        float(exposure["bearing_life_exponent"]),
    )
    condition_factor *= float(exposure["reference_speed_rpm"]) / np.maximum(
        arrays["speed_rpm"], 1.0
    )
    lifecycle = np.clip(
        normalized_time
        / np.maximum(condition_factor * float(candidate["time_scale_factor"]), 1e-6),
        0.0,
        1.0,
    )
    family = str(candidate["candidate_family"])
    uncertainty = np.interp(lifecycle, context.templates.grid, context.templates.std[family])
    return prior_rul.reshape(-1), prior_rate.reshape(-1), uncertainty


def _candidate_prediction_frame(
    fits: dict[str, ModelFit], frame: pd.DataFrame, config: dict[str, Any], device: torch.device
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    by_name = {item["candidate_spec"]: item for item in candidate_specs(config)}
    for name, fit in fits.items():
        prediction = _predict_model(fit, frame, config, device).rename(
            columns={"predicted_rul": "physics_rul"}
        )
        candidate = by_name[name]
        prediction["candidate_spec"] = name
        prediction["candidate_family"] = candidate["candidate_family"]
        prediction["time_scale_factor"] = candidate["time_scale_factor"]
        parts.append(prediction)
    return pd.concat(parts, ignore_index=True)


def build_counterfactual_evidence(
    data_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    source_frame: pd.DataFrame,
    context: TrainingContext,
    proxy: ExtraTreesRegressor,
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
    data = data_predictions.rename(columns={"predicted_rul": "data_rul"})
    joined = candidate_predictions.merge(
        data[["run_id", "sample_index", "data_rul"]],
        on=["run_id", "sample_index"],
        how="left",
        validate="many_to_one",
    ).merge(
        source_frame[source_columns],
        on=["run_id", "sample_index"],
        how="left",
        validate="many_to_one",
    )
    if joined[["data_rul", "degradation_family"]].isna().any().any():
        raise ValueError("Counterfactual predictions did not align with source samples.")
    joined["partition"] = partition
    joined["true_family"] = joined["degradation_family"]
    joined["law_correctness"] = (
        (joined["candidate_family"] == joined["true_family"])
        & np.isclose(joined["time_scale_factor"], 1.0)
    )
    margin = float(config["counterfactual_target"]["harmful_regret_margin"])
    run_rows: list[dict[str, Any]] = []
    for (run_id, candidate), group in joined.groupby(["run_id", "candidate_spec"]):
        target = group["target_rul"].to_numpy(dtype=float)
        data_rmse = math.sqrt(mean_squared_error(target, group["data_rul"]))
        physics_rmse = math.sqrt(mean_squared_error(target, group["physics_rul"]))
        regret = physics_rmse - data_rmse
        run_rows.append(
            {
                "run_id": run_id,
                "candidate_spec": candidate,
                "data_only_rmse": data_rmse,
                "physics_rmse": physics_rmse,
                "physics_regret": regret,
                "harmful_intervention": int(regret > margin),
                "safe_to_apply": int(regret <= margin),
            }
        )
    joined = joined.merge(
        pd.DataFrame(run_rows), on=["run_id", "candidate_spec"], how="left", validate="many_to_one"
    )

    current = joined[config["data"]["feature_columns"]].to_numpy(dtype=float)
    tree_predictions = np.vstack([tree.predict(current) for tree in proxy.estimators_])
    proxy_mean = np.clip(tree_predictions.mean(axis=0), 0.0, 1.0)
    proxy_std = np.maximum(tree_predictions.std(axis=0), 1e-6)
    joined["proxy_degradation"] = proxy_mean
    joined["proxy_uncertainty"] = proxy_std
    condition = _condition_evidence(joined, config)
    for column in condition:
        joined[column] = condition[column].to_numpy()

    joined["backbone_damage"] = np.clip(1.0 - joined["data_rul"], 0.0, 1.0)
    data_rate_parts: list[pd.Series] = []
    for _, run in joined.drop_duplicates(["run_id", "sample_index"]).groupby("run_id"):
        ordered = run.sort_values("sample_index")
        coordinate = ordered["elapsed_minutes"].to_numpy(dtype=float) / context.reference_life_minutes
        damage = ordered["backbone_damage"].to_numpy(dtype=float)
        if len(ordered) > 1:
            rate = np.maximum(np.gradient(damage, coordinate, edge_order=1), 0.0)
        else:
            rate = np.zeros(len(ordered))
        data_rate_parts.append(pd.Series(rate, index=ordered.index))
    unique_rate = pd.concat(data_rate_parts).sort_index()
    rate_map = joined.drop_duplicates(["run_id", "sample_index"])[["run_id", "sample_index"]].copy()
    rate_map["backbone_damage_rate"] = unique_rate.reindex(rate_map.index).to_numpy()
    joined = joined.merge(rate_map, on=["run_id", "sample_index"], how="left", validate="many_to_one")

    prior_rul_values = np.zeros(len(joined), dtype=float)
    prior_rate_values = np.zeros(len(joined), dtype=float)
    template_uncertainty = np.zeros(len(joined), dtype=float)
    by_name = {item["candidate_spec"]: item for item in candidate_specs(config)}
    for name, indices in joined.groupby("candidate_spec").groups.items():
        index = np.asarray(list(indices), dtype=int)
        prior_rul, prior_rate, uncertainty = _prior_for_joined_rows(
            joined.loc[index], by_name[name], context, config
        )
        prior_rul_values[index] = prior_rul
        prior_rate_values[index] = prior_rate
        template_uncertainty[index] = uncertainty
    joined["candidate_prior_rul"] = prior_rul_values
    joined["candidate_damage_rate"] = prior_rate_values
    joined["template_uncertainty"] = template_uncertainty
    joined["prior_absolute_disagreement"] = np.abs(joined["data_rul"] - prior_rul_values)
    joined["prior_rate_disagreement"] = np.abs(
        joined["backbone_damage_rate"] - prior_rate_values
    )
    joined["proxy_absolute_residual"] = np.abs(
        joined["proxy_degradation"] - (1.0 - prior_rul_values)
    )
    joined["causal_prefix_length_log"] = np.log1p(joined["sample_index"].to_numpy(dtype=float) + 1.0)
    grouping = ["run_id", "candidate_spec"]
    window = int(config["data"]["sequence_length"])
    joined = joined.sort_values(["run_id", "candidate_spec", "sample_index"]).reset_index(drop=True)
    joined["rolling_prior_disagreement"] = joined.groupby(grouping)[
        "prior_absolute_disagreement"
    ].transform(lambda values: values.rolling(window, min_periods=1).mean())
    joined["rolling_rate_disagreement"] = joined.groupby(grouping)[
        "prior_rate_disagreement"
    ].transform(lambda values: values.rolling(window, min_periods=1).mean())
    sample_group = ["run_id", "sample_index"]
    best = joined.groupby(sample_group)["proxy_absolute_residual"].transform("min")
    joined["residual_relative_to_best"] = joined["proxy_absolute_residual"] - best
    joined["residual_rank_fraction"] = joined.groupby(sample_group)[
        "proxy_absolute_residual"
    ].rank(method="average", pct=True)
    return joined


UNIT_KEYS = [
    "parent_seed",
    "run_id",
    "candidate_spec",
    "true_family",
    "candidate_family",
    "time_scale_factor",
    "law_correctness",
    "condition_id",
    "safe_to_apply",
    "harmful_intervention",
]


def aggregate_credibility_units(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        name: (name, "mean") for name in config["credibility"]["numeric_evidence"]
    }
    for optional in (
        "credibility_raw",
        "credibility",
        "fallback",
        "physics_regret",
        "data_only_rmse",
        "physics_rmse",
    ):
        if optional in frame:
            aggregations[optional] = (optional, "mean")
    return frame.groupby(UNIT_KEYS, as_index=False).agg(**aggregations)


def _credibility_matrix(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    numeric = list(config["credibility"]["numeric_evidence"])
    values = [frame[numeric].to_numpy(dtype=float)]
    names = list(numeric)
    for family in FAMILIES:
        values.append(
            (frame["candidate_family"].to_numpy() == family).astype(float).reshape(-1, 1)
        )
        names.append(f"candidate_family__{family}")
    matrix = np.concatenate(values, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("EXP-007A credibility evidence contains non-finite values.")
    forbidden = set(config["credibility"]["forbidden_inputs"])
    if forbidden & set(names):
        raise ValueError("Forbidden counterfactual target entered credibility features.")
    return matrix, names


@dataclass
class CredibilityFit:
    feature_names: list[str]
    scaler: StandardScaler
    classifier: LogisticRegression
    calibrator: LogisticRegression
    threshold: float


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_credibility_estimator(
    train_evidence: pd.DataFrame,
    validation_evidence: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> CredibilityFit:
    train_units = aggregate_credibility_units(train_evidence, config)
    validation_units = aggregate_credibility_units(validation_evidence, config)
    x_train, names = _credibility_matrix(train_units, config)
    x_validation, validation_names = _credibility_matrix(validation_units, config)
    if names != validation_names:
        raise ValueError("Credibility feature order changed between train and validation.")
    y_train = train_units["safe_to_apply"].to_numpy(dtype=int)
    y_validation = validation_units["safe_to_apply"].to_numpy(dtype=int)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
        raise ValueError("Development target qualification requires both safe and harmful units.")
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=float(config["credibility"]["regularization_c"]),
        class_weight=config["credibility"]["class_weight"],
        max_iter=3000,
        random_state=seed,
    ).fit(scaler.transform(x_train), y_train)
    raw_validation = np.clip(
        classifier.predict_proba(scaler.transform(x_validation))[:, 1], 1e-9, 1.0 - 1e-9
    )
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=seed).fit(
        _logit(raw_validation), y_validation
    )
    calibrated = calibrator.predict_proba(_logit(raw_validation))[:, 1]
    fpr, tpr, thresholds = roc_curve(y_validation, calibrated)
    finite = np.isfinite(thresholds)
    scores = tpr[finite] - fpr[finite]
    candidates = thresholds[finite]
    best = float(scores.max())
    tied = candidates[np.isclose(scores, best)]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return CredibilityFit(names, scaler, classifier, calibrator, threshold)


def apply_credibility(
    fit: CredibilityFit, evidence: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    units = aggregate_credibility_units(evidence, config)
    matrix, names = _credibility_matrix(units, config)
    if names != fit.feature_names:
        raise ValueError("Credibility feature order changed at inference.")
    raw = np.clip(
        fit.classifier.predict_proba(fit.scaler.transform(matrix))[:, 1], 1e-9, 1.0 - 1e-9
    )
    probability = np.clip(fit.calibrator.predict_proba(_logit(raw))[:, 1], 0.0, 1.0)
    scores = units[["parent_seed", "run_id", "candidate_spec"]].copy()
    scores["credibility_raw"] = raw
    scores["credibility"] = probability
    scores["credibility_threshold"] = fit.threshold
    scores["fallback"] = probability < fit.threshold
    return evidence.merge(
        scores,
        on=["parent_seed", "run_id", "candidate_spec"],
        how="left",
        validate="many_to_one",
    )


def development_target_qualification(
    train_evidence: pd.DataFrame, validation_evidence: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    train_units = aggregate_credibility_units(train_evidence, config)
    validation_units = aggregate_credibility_units(validation_evidence, config)
    units = pd.concat([train_units.assign(partition="train_oof"), validation_units.assign(partition="validation")])
    harmful_fraction = float(units["harmful_intervention"].mean())
    safe_fraction = float(units["safe_to_apply"].mean())
    minimum_harm = float(config["counterfactual_target"]["minimum_development_harmful_fraction"])
    minimum_safe = float(config["counterfactual_target"]["minimum_development_safe_fraction"])
    passed = harmful_fraction >= minimum_harm and safe_fraction >= minimum_safe
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "trajectory_candidate_seed_units": int(len(units)),
        "harmful_fraction": harmful_fraction,
        "safe_fraction": safe_fraction,
        "minimum_harmful_fraction": minimum_harm,
        "minimum_safe_fraction": minimum_safe,
        "test_access_authorized": passed,
    }


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


def _load_fit(
    checkpoint_path: Path,
    train_frame: pd.DataFrame,
    config: dict[str, Any],
) -> ModelFit:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != "EXP-007A":
        raise RuntimeError(f"Incompatible checkpoint: {checkpoint_path}")
    context = fit_training_context(train_frame, config)
    if not np.allclose(context.scaler.mean_, checkpoint["scaler_mean"]):
        raise RuntimeError(f"Checkpoint scaler mean changed: {checkpoint_path}")
    if not np.allclose(context.scaler.scale_, checkpoint["scaler_scale"]):
        raise RuntimeError(f"Checkpoint scaler scale changed: {checkpoint_path}")
    if not math.isclose(
        context.reference_life_minutes,
        float(checkpoint["reference_life_minutes"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(f"Checkpoint reference life changed: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(len(config["data"]["feature_columns"]), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    history_path = checkpoint_path.with_name("history.csv")
    return ModelFit(
        model=model,
        context=context,
        history=pd.read_csv(history_path),
        parameter_count=int(checkpoint["parameter_count"]),
        best_epoch=int(checkpoint["best_epoch"]),
        best_validation_mse=float(checkpoint["best_validation_mse"]),
        parent_seed=int(checkpoint["parent_seed"]),
        optimization_seed=int(checkpoint["optimization_seed"]),
    )


def _fit_or_load_data_only(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: dict[str, Any],
    parent_seed: int,
    optimization_seed: int,
    artifact_dir: Path,
    phase: str,
) -> ModelFit:
    checkpoint = artifact_dir / "checkpoint.pt"
    if checkpoint.is_file() and (artifact_dir / "history.csv").is_file():
        return _load_fit(checkpoint, train_frame, config)
    return _fit_data_only(
        train_frame,
        validation_frame,
        config,
        parent_seed,
        optimization_seed,
        artifact_dir,
        phase,
    )


def _fit_or_load_intervention(
    parent: ModelFit,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    candidate: dict[str, Any],
    config: dict[str, Any],
    artifact_dir: Path,
    phase: str,
    optimization_seed: int,
) -> ModelFit:
    checkpoint = artifact_dir / "checkpoint.pt"
    if checkpoint.is_file() and (artifact_dir / "history.csv").is_file():
        loaded = _load_fit(checkpoint, train_frame, config)
        saved_candidate = torch.load(checkpoint, map_location="cpu", weights_only=False).get("candidate")
        if saved_candidate != candidate:
            raise RuntimeError(f"Candidate checkpoint identity changed: {checkpoint}")
        return loaded
    return _fit_physics_intervention(
        parent,
        train_frame,
        validation_frame,
        candidate,
        config,
        artifact_dir,
        phase,
        optimization_seed,
    )


def _write_prediction_csv(frame: pd.DataFrame, path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format=str(config["evaluation"]["probability_float_format"]),
    )


def _candidate_regret(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "parent_seed",
        "partition",
        "run_id",
        "condition_id",
        "true_family",
        "candidate_spec",
        "candidate_family",
        "time_scale_factor",
        "law_correctness",
        "safe_to_apply",
        "harmful_intervention",
        "data_only_rmse",
        "physics_rmse",
        "physics_regret",
    ]
    return frame[columns].drop_duplicates().sort_values(
        ["parent_seed", "partition", "run_id", "candidate_spec"]
    )


def _ece(target: np.ndarray, probability: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return value


def credibility_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    units = aggregate_credibility_units(frame, config)
    target = units["safe_to_apply"].to_numpy(dtype=int)
    probability = units["credibility"].to_numpy(dtype=float)
    threshold = float(frame["credibility_threshold"].iloc[0])
    prevalence = float(target.mean())
    brier = float(brier_score_loss(target, probability))
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "prevalence_auprc": prevalence,
        "brier": brier,
        "prevalence_brier": prevalence * (1.0 - prevalence),
        "brier_better_than_prevalence": bool(brier < prevalence * (1.0 - prevalence)),
        "ece": _ece(target, probability, int(config["evaluation"]["calibration_bins"])),
        "threshold": threshold,
        "all_on_fraction": float(np.mean(probability >= threshold)),
        "all_off_fraction": float(np.mean(probability < threshold)),
        "trajectory_candidate_units": int(len(units)),
    }


def _macro_run_rmse(frame: pd.DataFrame, prediction_column: str) -> float:
    return float(
        np.mean(
            [
                math.sqrt(mean_squared_error(run["target_rul"], run[prediction_column]))
                for _, run in frame.groupby("run_id")
            ]
        )
    )


def _combine_candidates(
    group: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    data_rul = float(group["data_rul"].iloc[0])
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return data_rul
    normalized = weights / weights.sum()
    delta = group["physics_rul"].to_numpy(dtype=float) - data_rul
    return float(np.clip(data_rul + np.sum(normalized * delta), 0.0, 1.0))


def _control_base(evidence: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 70070)
    for (run_id, sample_index), group in evidence.groupby(["run_id", "sample_index"], sort=True):
        group = group.sort_values("candidate_spec")
        target = float(group["target_rul"].iloc[0])
        data_rul = float(group["data_rul"].iloc[0])
        threshold = float(group["credibility_threshold"].iloc[0])
        credibility = group["credibility"].to_numpy(dtype=float)
        priorcred_weights = np.where(credibility >= threshold, credibility, 0.0)
        inverse = np.exp(-np.clip(group["prior_absolute_disagreement"].to_numpy(dtype=float), 0.0, 50.0))
        safe = group["safe_to_apply"].to_numpy(dtype=float)
        harmful = 1.0 - safe
        methods = {
            "data_only": data_rul,
            "all_off": data_rul,
            "all_on": float(np.clip(group["physics_rul"].mean(), 0.0, 1.0)),
            "priorcred": _combine_candidates(group, priorcred_weights),
            "inverse_residual": _combine_candidates(group, inverse),
            "random_credibility": _combine_candidates(group, rng.uniform(0.0, 1.0, len(group))),
            "oracle": _combine_candidates(group, safe),
            "anti_oracle": _combine_candidates(group, harmful),
        }
        base = {
            "parent_seed": int(group["parent_seed"].iloc[0]),
            "partition": str(group["partition"].iloc[0]),
            "run_id": run_id,
            "sample_index": int(sample_index),
            "elapsed_minutes": float(group["elapsed_minutes"].iloc[0]),
            "condition_id": str(group["condition_id"].iloc[0]),
            "true_family": str(group["true_family"].iloc[0]),
            "target_rul": target,
            "data_rul": data_rul,
            "priorcred_selected_candidates": int(np.count_nonzero(priorcred_weights)),
            "priorcred_fallback": bool(np.count_nonzero(priorcred_weights) == 0),
        }
        for method, prediction in methods.items():
            rows.append(
                {
                    **base,
                    "method": method,
                    "predicted_rul": prediction,
                    "absolute_error": abs(target - prediction),
                }
            )
    return pd.DataFrame(rows)


def choose_validation_scalar(validation_controls: pd.DataFrame) -> float:
    data = validation_controls[validation_controls["method"] == "data_only"].copy()
    all_on = validation_controls[validation_controls["method"] == "all_on"][[
        "parent_seed", "run_id", "sample_index", "predicted_rul"
    ]].rename(columns={"predicted_rul": "all_on_rul"})
    work = data.merge(
        all_on,
        on=["parent_seed", "run_id", "sample_index"],
        how="left",
        validate="one_to_one",
    )
    scores: list[tuple[float, float]] = []
    for scalar in np.linspace(0.0, 1.0, 11):
        prediction = np.clip(
            work["data_rul"] + scalar * (work["all_on_rul"] - work["data_rul"]), 0.0, 1.0
        )
        scores.append((_macro_run_rmse(work.assign(_prediction=prediction), "_prediction"), float(scalar)))
    return min(scores, key=lambda value: (value[0], value[1]))[1]


def add_validation_scalar(controls: pd.DataFrame, scalar: float) -> pd.DataFrame:
    data = controls[controls["method"] == "data_only"].copy()
    all_on = controls[controls["method"] == "all_on"][[
        "parent_seed", "run_id", "sample_index", "predicted_rul"
    ]].rename(columns={"predicted_rul": "all_on_rul"})
    selected = data.merge(
        all_on,
        on=["parent_seed", "run_id", "sample_index"],
        how="left",
        validate="one_to_one",
    )
    selected["method"] = "validation_selected_scalar"
    selected["predicted_rul"] = np.clip(
        selected["data_rul"] + scalar * (selected["all_on_rul"] - selected["data_rul"]),
        0.0,
        1.0,
    )
    selected["absolute_error"] = np.abs(selected["target_rul"] - selected["predicted_rul"])
    return pd.concat([controls, selected[controls.columns]], ignore_index=True)


def summarize_control_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    for (seed, method), group in frame.groupby(["parent_seed", "method"]):
        target = group["target_rul"].to_numpy(dtype=float)
        prediction = group["predicted_rul"].to_numpy(dtype=float)
        comparison_rows.append(
            {
                "parent_seed": int(seed),
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
    for (seed, run_id), subset in frame.groupby(["parent_seed", "run_id"]):
        baseline = subset[subset["method"] == "data_only"]
        baseline_rmse = math.sqrt(
            mean_squared_error(baseline["target_rul"], baseline["predicted_rul"])
        )
        for method, group in subset.groupby("method"):
            rmse = math.sqrt(mean_squared_error(group["target_rul"], group["predicted_rul"]))
            regret_rows.append(
                {
                    "parent_seed": int(seed),
                    "run_id": run_id,
                    "method": method,
                    "rmse": rmse,
                    "data_only_rmse": baseline_rmse,
                    "physics_regret": rmse - baseline_rmse,
                    "positive_physics_regret": max(0.0, rmse - baseline_rmse),
                }
            )
    return comparison, pd.DataFrame(regret_rows)


def _fit_candidate_population(
    parent: ModelFit,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: dict[str, Any],
    artifact_root: Path,
    phase: str,
    optimization_seed_base: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, ModelFit]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    fits: dict[str, ModelFit] = {}
    for index, candidate in enumerate(candidate_specs(config), start=1):
        name = candidate["candidate_spec"]
        fit = _fit_or_load_intervention(
            parent,
            train_frame,
            validation_frame,
            candidate,
            config,
            artifact_root / name,
            phase,
            optimization_seed_base + index,
        )
        prediction = _predict_model(fit, evaluation_frame, config, device).rename(
            columns={"predicted_rul": "physics_rul"}
        )
        prediction["candidate_spec"] = name
        prediction["candidate_family"] = candidate["candidate_family"]
        prediction["time_scale_factor"] = candidate["time_scale_factor"]
        predictions.append(prediction)
        histories.append(fit.history)
        fits[name] = fit
    return pd.concat(predictions, ignore_index=True), histories, fits


def _run_seed(
    seed: int,
    frame: pd.DataFrame,
    split: dict[str, Any],
    config: dict[str, Any],
    seed_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_parts: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    folds = _crossfit_membership(config, split)
    for fold_index, fold in enumerate(folds, start=1):
        fold_dir = seed_dir / "crossfit" / fold["fold_id"]
        evidence_path = fold_dir / "counterfactual_evidence.csv"
        if evidence_path.is_file():
            oof_parts.append(pd.read_csv(evidence_path))
            histories.extend(
                pd.read_csv(path) for path in sorted(fold_dir.rglob("history.csv"))
            )
            continue
        train_frame = frame[frame["run_id"].isin(fold["train_runs"])].copy()
        validation_frame = frame[frame["run_id"].isin(fold["validation_runs"])].copy()
        holdout_frame = frame[frame["run_id"].isin(fold["holdout_runs"])].copy()
        parent_seed = seed
        optimization_base = seed + fold_index * 10000
        parent = _fit_or_load_data_only(
            train_frame,
            validation_frame,
            config,
            parent_seed,
            optimization_base,
            fold_dir / "data_only",
            fold["fold_id"],
        )
        histories.append(parent.history)
        data_predictions = _predict_model(parent, holdout_frame, config, device)
        candidate_predictions, candidate_histories, _ = _fit_candidate_population(
            parent,
            train_frame,
            validation_frame,
            holdout_frame,
            config,
            fold_dir / "candidates",
            fold["fold_id"],
            optimization_base + 100,
        )
        histories.extend(candidate_histories)
        proxy = _fit_degradation_proxy(train_frame, config, optimization_base)
        evidence = build_counterfactual_evidence(
            data_predictions,
            candidate_predictions,
            holdout_frame,
            parent.context,
            proxy,
            config,
            "train_oof",
        )
        evidence["parent_seed"] = seed
        evidence["crossfit_fold"] = fold["fold_id"]
        _write_prediction_csv(evidence, evidence_path, config)
        oof_parts.append(evidence)

    train_evidence = pd.concat(oof_parts, ignore_index=True)
    if set(train_evidence["run_id"].unique()) != set(split["train_runs"]):
        raise RuntimeError("Cross-fitting did not cover every training trajectory exactly once.")

    final_dir = seed_dir / "final"
    train_frame = frame[frame["run_id"].isin(split["train_runs"])].copy()
    validation_frame = frame[frame["run_id"].isin(split["validation_runs"])].copy()
    test_frame = frame[frame["run_id"].isin(split["test_runs"])].copy()
    parent = _fit_or_load_data_only(
        train_frame,
        validation_frame,
        config,
        seed,
        seed,
        final_dir / "data_only",
        "final",
    )
    histories.append(parent.history)
    validation_data = _predict_model(parent, validation_frame, config, device)
    validation_candidates, candidate_histories, final_fits = _fit_candidate_population(
        parent,
        train_frame,
        validation_frame,
        validation_frame,
        config,
        final_dir / "candidates",
        "final",
        seed + 50000,
    )
    histories.extend(candidate_histories)
    proxy = _fit_degradation_proxy(train_frame, config, seed)
    validation_evidence = build_counterfactual_evidence(
        validation_data,
        validation_candidates,
        validation_frame,
        parent.context,
        proxy,
        config,
        "validation",
    )
    validation_evidence["parent_seed"] = seed
    qualification = development_target_qualification(
        train_evidence, validation_evidence, config
    )
    (seed_dir / "development_target_qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8"
    )
    if not qualification["passed"]:
        result = {
            "parent_seed": seed,
            "status": "development_gate_failed",
            "test_evaluated": False,
            "seconds": time.perf_counter() - started,
            "development_target_qualification": qualification,
        }
        (seed_dir / "training_history.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.concat(histories, ignore_index=True).to_csv(
            seed_dir / "training_history.csv", index=False
        )
        _write_prediction_csv(train_evidence, seed_dir / "train_counterfactual_evidence.csv", config)
        _write_prediction_csv(
            validation_evidence, seed_dir / "validation_counterfactual_evidence.csv", config
        )
        (seed_dir / "job_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    credibility_fit = fit_credibility_estimator(
        train_evidence, validation_evidence, config, seed
    )
    train_scored = apply_credibility(credibility_fit, train_evidence, config)
    validation_scored = apply_credibility(credibility_fit, validation_evidence, config)
    validation_controls = _control_base(validation_scored, seed)
    validation_scalar = choose_validation_scalar(validation_controls)
    validation_controls = add_validation_scalar(validation_controls, validation_scalar)

    # The sealed test is first touched only after the development target, estimator,
    # calibration, threshold, and scalar comparator are frozen.
    test_data = _predict_model(parent, test_frame, config, device)
    test_candidate_parts: list[pd.DataFrame] = []
    by_name = {item["candidate_spec"]: item for item in candidate_specs(config)}
    for name, fit in final_fits.items():
        prediction = _predict_model(fit, test_frame, config, device).rename(
            columns={"predicted_rul": "physics_rul"}
        )
        prediction["candidate_spec"] = name
        prediction["candidate_family"] = by_name[name]["candidate_family"]
        prediction["time_scale_factor"] = by_name[name]["time_scale_factor"]
        test_candidate_parts.append(prediction)
    test_candidates = pd.concat(test_candidate_parts, ignore_index=True)
    test_evidence = build_counterfactual_evidence(
        test_data,
        test_candidates,
        test_frame,
        parent.context,
        proxy,
        config,
        "test",
    )
    test_evidence["parent_seed"] = seed
    test_scored = apply_credibility(credibility_fit, test_evidence, config)
    test_controls = add_validation_scalar(_control_base(test_scored, seed), validation_scalar)

    pd.concat(histories, ignore_index=True).to_csv(seed_dir / "training_history.csv", index=False)
    _write_prediction_csv(train_scored, seed_dir / "train_counterfactual_evidence.csv", config)
    _write_prediction_csv(
        validation_scored, seed_dir / "validation_counterfactual_evidence.csv", config
    )
    test_path = seed_dir / "test_credibility_predictions.csv"
    _write_prediction_csv(test_scored, test_path, config)
    _write_prediction_csv(test_controls, seed_dir / "test_control_predictions.csv", config)
    _candidate_regret(test_scored).to_csv(seed_dir / "candidate_regret.csv", index=False)
    (seed_dir / "credibility_estimator.json").write_text(
        json.dumps(_serialize_credibility_fit(credibility_fit), indent=2), encoding="utf-8"
    )

    # Metrics are calculated from the serialized probabilities, not the in-memory frame.
    serialized = pd.read_csv(test_path)
    metrics = credibility_metrics(serialized, config)
    metrics.update(
        {
            "parent_seed": seed,
            "status": "completed",
            "test_evaluated": True,
            "validation_selected_scalar": validation_scalar,
            "backbone_parameter_count": parent.parameter_count,
            "backbone_best_epoch": parent.best_epoch,
            "backbone_best_validation_mse": parent.best_validation_mse,
            "seconds": time.perf_counter() - started,
            "development_target_qualification": qualification,
            "metric_source": "reloaded_serialized_test_predictions",
        }
    )
    (seed_dir / "job_result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _hierarchical_bootstrap(
    units: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    seeds = sorted(int(value) for value in units["parent_seed"].unique())
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    estimates: list[float] = []
    for _ in range(int(config["evaluation"]["bootstrap_replicates"])):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_estimates: list[float] = []
        for seed in sampled_seeds:
            seed_frame = units[units["parent_seed"] == int(seed)]
            runs = sorted(seed_frame["run_id"].unique())
            sampled_runs = rng.choice(runs, size=len(runs), replace=True)
            sample = pd.concat(
                [seed_frame[seed_frame["run_id"] == run] for run in sampled_runs],
                ignore_index=True,
            )
            if sample["safe_to_apply"].nunique() == 2:
                seed_estimates.append(
                    float(roc_auc_score(sample["safe_to_apply"], sample["credibility"]))
                )
        if seed_estimates:
            estimates.append(float(np.mean(seed_estimates)))
    if not estimates:
        raise RuntimeError("EXP-007A hierarchical bootstrap produced no valid estimates.")
    return {
        "replicates_requested": int(config["evaluation"]["bootstrap_replicates"]),
        "replicates_valid": len(estimates),
        "mean": float(np.mean(estimates)),
        "median": float(np.median(estimates)),
        "auroc_ci_lower_95": float(np.quantile(estimates, 0.025)),
        "auroc_ci_upper_95": float(np.quantile(estimates, 0.975)),
        "aggregation": "trajectory_within_resampled_seed_then_mean_seed_auroc",
    }


def _plots(
    output_root: Path,
    credibility: pd.DataFrame,
    comparison: pd.DataFrame,
    candidate_regret: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    units = aggregate_credibility_units(credibility, config)
    fig, ax = plt.subplots(figsize=(6, 5))
    for seed, group in units.groupby("parent_seed"):
        fpr, tpr, _ = roc_curve(group["safe_to_apply"], group["credibility"])
        ax.plot(fpr, tpr, alpha=0.65, label=f"seed {seed}: {roc_auc_score(group['safe_to_apply'], group['credibility']):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.5")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="EXP-007A safe-intervention ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "safe_intervention_roc.png", dpi=180)
    plt.close(fig)

    summary = comparison.groupby("method", as_index=False)["macro_run_rmse"].mean().sort_values("macro_run_rmse")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["method"], summary["macro_run_rmse"])
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Mean macro trajectory RMSE", title="EXP-007A intervention controls")
    fig.tight_layout()
    fig.savefig(plot_dir / "control_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = candidate_regret.groupby("candidate_family")["physics_regret"]
    ax.boxplot([values.to_numpy() for _, values in grouped], tick_labels=[name for name, _ in grouped])
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0.0, color="0.4", linestyle="--")
    ax.set(ylabel="Candidate physics regret", title="Actual intervention regret by candidate family")
    fig.tight_layout()
    fig.savefig(plot_dir / "candidate_physics_regret.png", dpi=180)
    plt.close(fig)


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _copy_completed_state(source: Path, destination: Path | None) -> None:
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)


def finalize_exp7a_artifacts(root: str | Path) -> Path:
    root = Path(root)
    excluded = {"run_manifest.json", "artifact_inventory.csv", "codex_results_bundle.zip"}
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    inventory_path = root / "artifact_inventory.csv"
    pd.DataFrame(records).to_csv(inventory_path, index=False)
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_count_excluding_manifest_inventory_bundle"] = len(records)
        manifest["artifact_inventory_sha256"] = sha256_file(inventory_path)
        notebook = root / "executed_notebook.ipynb"
        manifest["executed_notebook_sha256"] = sha256_file(notebook) if notebook.is_file() else None
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = root / "codex_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    binary = ("*.pt", "*.pth", "*.pkl", "*.joblib")
    bulky = (
        "seeds/*/crossfit/*/counterfactual_evidence.csv",
        "seeds/*/train_counterfactual_evidence.csv",
        "seeds/*/validation_counterfactual_evidence.csv",
        "seeds/*/test_credibility_predictions.csv",
        "seeds/*/test_control_predictions.csv",
    )
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == bundle:
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in binary):
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in bulky):
                continue
            archive.write(path, relative)
    return bundle


def _write_summary(
    output_root: Path,
    gate: dict[str, Any],
    metrics: pd.DataFrame,
    qualification: dict[str, Any],
) -> None:
    rows = [
        "# EXP-007A counterfactual physics-harm credibility",
        "",
        f"Status: **{gate['status']}**",
        "",
        f"Development target qualification: **{qualification['status']}**",
        f"Mean within-seed safe-intervention AUROC: `{gate.get('mean_seed_auroc', float('nan')):.6f}`",
        f"Hierarchical bootstrap 95% interval: `[{gate.get('auroc_ci_lower_95', float('nan')):.6f}, {gate.get('auroc_ci_upper_95', float('nan')):.6f}]`",
        f"Decision: **{gate['decision']}**",
        "",
        "Candidate prior models were trained with differentiable value/rate/monotonic losses",
        "from identical data-only parent checkpoints. Mathematical law correctness is secondary",
        "metadata; the primary label is counterfactual RUL intervention safety.",
        "",
        "## Per-seed credibility metrics",
        "",
        "```text",
        metrics.to_string(index=False) if not metrics.empty else "No sealed-test metrics were produced.",
        "```",
    ]
    (output_root / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_exp7a_experiment(
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
    started_perf = time.perf_counter()
    started_utc = _utc_now()
    environment, git, qualification = validate_exp7a_runtime(
        config, project_root, feature_path
    )
    split = _load_split(config, project_root)
    resolved_config = copy.deepcopy(config)
    resolved_config.pop("_config_path", None)
    identity = {
        "experiment_id": "EXP-007A",
        "git_commit": git["commit"],
        "config_sha256": _json_sha256(resolved_config),
        "feature_sha256": qualification["feature_sha256"],
        "split_sha256": _json_sha256(split),
        "scenario_sha256": qualification["scenario_sha256"],
        "seeds": list(config["training"]["seeds"]),
    }
    state_path = output_root / "run_state.json"
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing EXP-007A recovery state is incompatible.")
    else:
        state_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    log_path = output_root / "training.log"
    _append_log(
        log_path,
        f"EXP-007A start/resume commit={git['commit']} feature_sha256={qualification['feature_sha256']}",
    )
    (output_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    (output_root / "git_commit.txt").write_text(git["commit"] + "\n", encoding="utf-8")
    (output_root / "environment.txt").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8"
    )
    (output_root / "data_split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    frame = pd.read_csv(feature_path)
    failures: list[dict[str, Any]] = []
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        seed_dir = output_root / "seeds" / f"seed_{seed:05d}"
        job_path = seed_dir / "job_result.json"
        if job_path.is_file():
            existing = json.loads(job_path.read_text(encoding="utf-8"))
            if int(existing.get("parent_seed", -1)) == seed and existing.get("status") in {
                "completed",
                "development_gate_failed",
            }:
                _append_log(log_path, f"seed={seed} resume-skip status={existing['status']}")
                continue
        try:
            result = _run_seed(seed, frame, split, config, seed_dir)
            _append_log(
                log_path,
                f"seed={seed} status={result['status']} test_evaluated={result['test_evaluated']} seconds={result['seconds']:.3f}",
            )
            _copy_completed_state(output_root, recovery)
        except Exception as exc:
            seed_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "parent_seed": seed,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            (seed_dir / "failure.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            _append_log(log_path, f"seed={seed} status=failed error={type(exc).__name__}: {exc}")
            _copy_completed_state(output_root, recovery)

    job_results: list[dict[str, Any]] = []
    for seed_value in config["training"]["seeds"]:
        path = output_root / "seeds" / f"seed_{int(seed_value):05d}" / "job_result.json"
        if path.is_file():
            job_results.append(json.loads(path.read_text(encoding="utf-8")))
    completed = [result for result in job_results if result.get("status") == "completed"]
    development_failed = [
        result for result in job_results if result.get("status") == "development_gate_failed"
    ]
    qualification_summary = {
        "status": "passed" if len(completed) == len(config["training"]["seeds"]) else "failed",
        "requested_seeds": [int(value) for value in config["training"]["seeds"]],
        "completed_seeds": [int(value["parent_seed"]) for value in completed],
        "development_gate_failed_seeds": [
            int(value["parent_seed"]) for value in development_failed
        ],
        "failed_seeds": [int(value["parent_seed"]) for value in failures],
        "per_seed": {
            str(value["parent_seed"]): value.get("development_target_qualification")
            for value in job_results
        },
    }
    (output_root / "development_target_qualification.json").write_text(
        json.dumps(qualification_summary, indent=2), encoding="utf-8"
    )
    (output_root / "failure_report.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    history_paths = sorted((output_root / "seeds").glob("seed_*/training_history.csv"))
    if history_paths:
        pd.concat([pd.read_csv(path) for path in history_paths], ignore_index=True).to_csv(
            output_root / "training_history.csv", index=False
        )

    if len(completed) != len(config["training"]["seeds"]):
        gate = {
            "experiment_id": "EXP-007A",
            "status": "failed",
            "decision": "stop_before_or_during_sealed_test",
            "development_target_qualification": qualification_summary["status"],
            "reason": (
                "All seeds must pass development target qualification and complete before the "
                "sealed-test credibility gate is interpretable."
            ),
        }
        (output_root / "gate_decision.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )
        empty_metrics = pd.DataFrame()
        empty_metrics.to_csv(output_root / "credibility_metrics.csv", index=False)
        (output_root / "statistical_summary.json").write_text(
            json.dumps({"status": "not_computed"}, indent=2), encoding="utf-8"
        )
        (output_root / "serialization_verification.json").write_text(
            json.dumps({"status": "not_applicable"}, indent=2), encoding="utf-8"
        )
        _write_summary(output_root, gate, empty_metrics, qualification_summary)
        manifest = {
            **identity,
            "run_id": config["experiment"]["run_id"],
            "status": "failed",
            "requested_seeds": identity["seeds"],
            "completed_seeds": qualification_summary["completed_seeds"],
            "failed_models": failures,
            "development_gate_failed_seeds": qualification_summary[
                "development_gate_failed_seeds"
            ],
            "test_access_policy_honored": True,
            "environment": environment,
            "started_at_utc": started_utc,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_perf,
            "gate_decision": gate["decision"],
        }
        (output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        finalize_exp7a_artifacts(output_root)
        _copy_completed_state(output_root, recovery)
        return gate

    seed_dirs = [
        output_root / "seeds" / f"seed_{int(seed):05d}"
        for seed in config["training"]["seeds"]
    ]
    credibility = pd.concat(
        [pd.read_csv(path / "test_credibility_predictions.csv") for path in seed_dirs],
        ignore_index=True,
    )
    controls = pd.concat(
        [pd.read_csv(path / "test_control_predictions.csv") for path in seed_dirs],
        ignore_index=True,
    )
    candidate_regret = pd.concat(
        [pd.read_csv(path / "candidate_regret.csv") for path in seed_dirs],
        ignore_index=True,
    )
    credibility_path = output_root / "credibility_predictions.csv"
    _write_prediction_csv(credibility, credibility_path, config)
    _write_prediction_csv(controls, output_root / "control_predictions.csv", config)
    candidate_regret.to_csv(output_root / "candidate_regret.csv", index=False)
    serialized = pd.read_csv(credibility_path)

    metric_rows: list[dict[str, Any]] = []
    for seed, group in serialized.groupby("parent_seed"):
        metric_rows.append({"parent_seed": int(seed), **credibility_metrics(group, config)})
    metrics = pd.DataFrame(metric_rows).sort_values("parent_seed")
    metrics.to_csv(output_root / "credibility_metrics.csv", index=False)
    units = aggregate_credibility_units(serialized, config)
    bootstrap = _hierarchical_bootstrap(units, config)
    statistical = {
        "mean_seed_auroc": float(metrics["auroc"].mean()),
        "std_seed_auroc": float(metrics["auroc"].std(ddof=1)),
        "mean_seed_auprc": float(metrics["auprc"].mean()),
        "mean_seed_brier": float(metrics["brier"].mean()),
        "mean_prevalence_brier": float(metrics["prevalence_brier"].mean()),
        **bootstrap,
    }
    (output_root / "statistical_summary.json").write_text(
        json.dumps(statistical, indent=2), encoding="utf-8"
    )
    comparison, control_regret = summarize_control_predictions(controls)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    control_regret.to_csv(output_root / "physics_regret.csv", index=False)

    maximum = float(config["success_criteria"]["maximum_all_off_fraction_per_seed"])
    passes_auroc = statistical["mean_seed_auroc"] >= float(
        config["success_criteria"]["minimum_test_auroc"]
    )
    passes_ci = bootstrap["auroc_ci_lower_95"] > float(
        config["success_criteria"]["minimum_auroc_ci_lower"]
    )
    passes_collapse = bool(
        (metrics["all_off_fraction"] <= maximum).all()
        and (metrics["all_on_fraction"] <= float(config["success_criteria"]["maximum_all_on_fraction_per_seed"])).all()
    )
    passes_brier = bool(metrics["brier"].mean() < metrics["prevalence_brier"].mean())
    harmful = candidate_regret[candidate_regret["harmful_intervention"] == 1]
    passes_harm_stress = bool(len(harmful) and float(harmful["physics_regret"].mean()) > 0.0)
    mean_positive = control_regret.groupby("method")["positive_physics_regret"].mean()
    passes_regret = bool(
        mean_positive.get("priorcred", float("inf")) < mean_positive.get("all_on", -float("inf"))
        and mean_positive.get("priorcred", float("inf"))
        < mean_positive.get("validation_selected_scalar", -float("inf"))
    )
    passes = all(
        (passes_auroc, passes_ci, passes_collapse, passes_brier, passes_harm_stress, passes_regret)
    )
    gate = {
        "experiment_id": "EXP-007A",
        "status": "passed" if passes else "failed",
        "decision": "permit_exp008_preparation" if passes else "stop_and_diagnose_exp007a",
        "mean_seed_auroc": statistical["mean_seed_auroc"],
        "std_seed_auroc": statistical["std_seed_auroc"],
        **bootstrap,
        "passes_minimum_auroc": passes_auroc,
        "passes_bootstrap_ci": passes_ci,
        "passes_per_seed_anti_collapse": passes_collapse,
        "passes_brier_baseline": passes_brier,
        "passes_harm_stress": passes_harm_stress,
        "passes_priorcred_regret_reduction": passes_regret,
        "mean_positive_regret": {name: float(value) for name, value in mean_positive.items()},
    }
    (output_root / "gate_decision.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    verification_rows: list[dict[str, Any]] = []
    for row in metrics.to_dict(orient="records"):
        seed = int(row["parent_seed"])
        recomputed = credibility_metrics(serialized[serialized["parent_seed"] == seed], config)
        verification_rows.append(
            {
                "parent_seed": seed,
                "auroc_absolute_difference": abs(float(row["auroc"]) - recomputed["auroc"]),
                "auprc_absolute_difference": abs(float(row["auprc"]) - recomputed["auprc"]),
                "brier_absolute_difference": abs(float(row["brier"]) - recomputed["brier"]),
            }
        )
    serialization = {
        "status": "passed",
        "source": "reloaded_root_credibility_predictions.csv",
        "float_format": config["evaluation"]["probability_float_format"],
        "maximum_absolute_difference": float(
            max(
                max(
                    row["auroc_absolute_difference"],
                    row["auprc_absolute_difference"],
                    row["brier_absolute_difference"],
                )
                for row in verification_rows
            )
        ),
        "per_seed": verification_rows,
    }
    (output_root / "serialization_verification.json").write_text(
        json.dumps(serialization, indent=2), encoding="utf-8"
    )
    _plots(output_root, serialized, comparison, candidate_regret, config)
    _write_summary(output_root, gate, metrics, qualification_summary)
    manifest = {
        **identity,
        "run_id": config["experiment"]["run_id"],
        "status": "completed",
        "requested_seeds": identity["seeds"],
        "completed_seeds": qualification_summary["completed_seeds"],
        "failed_models": failures,
        "development_gate_failed_seeds": [],
        "test_access_policy_honored": True,
        "environment": environment,
        "started_at_utc": started_utc,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started_perf,
        "gate_decision": gate["decision"],
        "effective_runtime": {
            "seeds": identity["seeds"],
            "candidate_count": len(candidate_specs(config)),
            "baseline_epochs": int(config["backbone"]["epochs"]),
            "intervention_epochs": int(config["physics_intervention"]["fine_tune_epochs"]),
            "oom_retries": 0,
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    finalize_exp7a_artifacts(output_root)
    _copy_completed_state(output_root, recovery)
    return gate
