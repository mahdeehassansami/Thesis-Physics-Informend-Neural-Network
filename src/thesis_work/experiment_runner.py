from __future__ import annotations

import copy
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from thesis_work.multi_dataset import (
    MODEL_FEATURES,
    PreparedDataset,
    enabled_dataset_configs,
    load_or_extract_dataset,
    prepare_sequence_dataset,
)
from thesis_work.sequence_models import build_model, calculate_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def predict_model(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    loader = _loader(dataset, batch_size, False, seed=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    run_ids: list[str] = []
    sample_indices: list[np.ndarray] = []
    rul_scales: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            time_coordinate = batch["time"].to(device)
            prediction = model(x, time_coordinate)
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target"].numpy())
            run_ids.extend(list(batch["run_id"]))
            sample_indices.append(batch["sample_index"].numpy())
            rul_scales.append(batch["rul_scale_seconds"].numpy())
    return (
        np.concatenate(targets).reshape(-1),
        np.clip(np.concatenate(predictions).reshape(-1), 0, 1),
        run_ids,
        np.concatenate(sample_indices).reshape(-1),
        np.concatenate(rul_scales).reshape(-1),
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "mse": float(mean_squared_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    started = time.perf_counter()
    target, prediction, run_ids, sample_indices, rul_scales = predict_model(
        model, dataset, batch_size, device
    )
    inference_seconds = time.perf_counter() - started
    target_seconds = target * rul_scales
    prediction_seconds = prediction * rul_scales
    frame = pd.DataFrame(
        {
            "run_id": run_ids,
            "sample_index": sample_indices,
            "target_rul": target,
            "predicted_rul": prediction,
            "rul_scale_seconds": rul_scales,
            "target_rul_seconds": target_seconds,
            "predicted_rul_seconds": prediction_seconds,
        }
    )
    metrics = regression_metrics(target, prediction)
    original_metrics = regression_metrics(target_seconds, prediction_seconds)
    metrics.update(
        {
            "mae_seconds": original_metrics["mae"],
            "mse_seconds2": original_metrics["mse"],
            "rmse_seconds": original_metrics["rmse"],
            "r2_seconds": original_metrics["r2"],
            "inference_seconds": inference_seconds,
            "inference_seconds_per_sample": inference_seconds / max(1, len(frame)),
        }
    )
    return frame, metrics


def _loss_gradient_norm(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + torch.sum(gradient.detach() ** 2)
    return float(torch.sqrt(squared).detach())


def train_one_model(
    prepared: PreparedDataset,
    model_name: str,
    model_config: dict[str, Any],
    weights: dict[str, float],
    physics: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    artifact_dir: str | Path,
    evaluation_split: str = "test",
    save_final_evaluation: bool = False,
) -> tuple[torch.nn.Module, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'.")
    seed_everything(seed)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name,
        input_dim=len(prepared.feature_columns),
        model_config=model_config,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(training_config["patience"]) // 3),
    )
    batch_size = int(training_config["batch_size"])
    train_loader = _loader(prepared.train, batch_size, True, seed)
    feature_indices = {
        name: index for index, name in enumerate(prepared.feature_columns)
    }

    history_rows: list[dict[str, float]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(training_config["patience"])
    epochs = int(training_config["epochs"])
    gradient_interval = max(
        1, int(training_config.get("gradient_diagnostics_interval", 10))
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, float] = {}
        gradient_diagnostics = {
            "gradient_data_weighted": float("nan"),
            "gradient_physics_weighted": float("nan"),
            "gradient_total": float("nan"),
        }
        batches = 0
        for raw_batch in train_loader:
            batch = _move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, components, _ = calculate_loss(
                model,
                batch,
                weights=weights,
                feature_indices=feature_indices,
                physics=physics,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite loss in {model_name}; components="
                    f"{ {name: float(value.detach()) for name, value in components.items()} }"
                )
            if batches == 0 and (epoch == 1 or epoch % gradient_interval == 0):
                data_loss = float(weights.get("data", 1.0)) * components["data"]
                physics_terms = [
                    float(weights.get(name, 0.0)) * value
                    for name, value in components.items()
                    if name != "data" and float(weights.get(name, 0.0)) != 0.0
                ]
                gradient_diagnostics["gradient_data_weighted"] = _loss_gradient_norm(
                    data_loss, trainable_parameters
                )
                if physics_terms:
                    physics_loss = torch.stack(physics_terms).sum()
                    gradient_diagnostics[
                        "gradient_physics_weighted"
                    ] = _loss_gradient_norm(physics_loss, trainable_parameters)
                else:
                    gradient_diagnostics["gradient_physics_weighted"] = 0.0
                gradient_diagnostics["gradient_total"] = _loss_gradient_norm(
                    total, trainable_parameters
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config.get("gradient_clip", 5.0))
            )
            optimizer.step()

            batches += 1
            running["total"] = running.get("total", 0.0) + float(total.detach())
            for name, value in components.items():
                running[name] = running.get(name, 0.0) + float(value.detach())

        validation_target, validation_prediction, _, _, _ = predict_model(
            model, prepared.validation, batch_size, device
        )
        validation_mse = float(
            mean_squared_error(validation_target, validation_prediction)
        )
        scheduler.step(validation_mse)
        averages = {
            name: value / max(1, batches) for name, value in running.items()
        }
        row = {
            "epoch": epoch,
            "validation_mse": validation_mse,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **averages,
            **gradient_diagnostics,
        }
        for name, value in averages.items():
            if name != "total":
                row[f"weighted_{name}"] = value * float(weights.get(name, 0.0))
        history_rows.append(row)

        if validation_mse < best_validation - float(
            training_config.get("minimum_improvement", 1e-6)
        ):
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_left = int(training_config["patience"])
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} never produced a valid validation checkpoint.")
    history = pd.DataFrame(history_rows)
    history.to_csv(artifact_dir / "history.csv", index=False)
    final_epoch = int(history.iloc[-1]["epoch"])
    final_validation_mse = float(history.iloc[-1]["validation_mse"])
    final_test_metrics: dict[str, Any] = {}
    if save_final_evaluation:
        final_validation_frame, final_validation_metrics = evaluate_model(
            model, prepared.validation, batch_size, device
        )
        final_validation_metrics.update(
            {
                "parameter_count": parameter_count,
                "epoch": final_epoch,
                "evaluation_split": "validation",
                "checkpoint_role": "final_epoch",
            }
        )
        final_validation_frame.to_csv(
            artifact_dir / "final_validation_predictions.csv", index=False
        )
        (artifact_dir / "final_validation_metrics.json").write_text(
            json.dumps(final_validation_metrics, indent=2), encoding="utf-8"
        )
        if evaluation_split == "test":
            final_prediction_frame, final_test_metrics = evaluate_model(
                model, prepared.test, batch_size, device
            )
            final_test_metrics.update(
                {
                    "parameter_count": parameter_count,
                    "epoch": final_epoch,
                    "evaluation_split": "test",
                    "checkpoint_role": "final_epoch",
                }
            )
            final_prediction_frame.to_csv(
                artifact_dir / "final_predictions.csv", index=False
            )
            (artifact_dir / "final_test_metrics.json").write_text(
                json.dumps(final_test_metrics, indent=2), encoding="utf-8"
            )

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_name": model_name,
            "model_config": model_config,
            "feature_columns": prepared.feature_columns,
            "state_dict": best_state,
            "weights": weights,
            "physics": physics,
            "training_config": training_config,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "final_epoch": final_epoch,
            "final_validation_mse": final_validation_mse,
        },
        artifact_dir / "checkpoint.pt",
    )

    common_metrics: dict[str, Any] = {
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_mse": best_validation,
        "final_epoch": final_epoch,
        "final_validation_mse": final_validation_mse,
        "checkpoint_role": "best_validation",
    }
    for name, value in final_test_metrics.items():
        if isinstance(value, (int, float)):
            common_metrics[f"final_test_{name}"] = value
    validation_frame, validation_metrics = evaluate_model(
        model, prepared.validation, batch_size, device
    )
    validation_metrics.update(common_metrics)
    validation_metrics["evaluation_split"] = "validation"
    validation_frame.to_csv(
        artifact_dir / "validation_predictions.csv", index=False
    )
    (artifact_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )

    if evaluation_split == "validation":
        return model, history, validation_frame, validation_metrics

    prediction_frame, test_metrics = evaluate_model(
        model, prepared.test, batch_size, device
    )
    test_metrics.update(common_metrics)
    test_metrics["evaluation_split"] = "test"
    prediction_frame.to_csv(artifact_dir / "predictions.csv", index=False)
    (artifact_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    return model, history, prediction_frame, test_metrics

def _weight_profiles_for_model(
    model_name: str,
    model_config: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    profiles = model_config.get("profiles", ["data_only"])
    return [
        (profile, copy.deepcopy(config["weight_profiles"][profile]))
        for profile in profiles
    ]


def _comparison_summary(result: pd.DataFrame) -> pd.DataFrame:
    successful = result[result["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "seed_repeats": ("seed", "nunique"),
        "seconds_mean": ("seconds", "mean"),
        "seconds_total": ("seconds", "sum"),
        "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"),
        "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"),
        "r2_mean": ("r2", "mean"),
        "r2_std": ("r2", "std"),
    }
    optional = {
        "mae_seconds_mean": ("mae_seconds", "mean"),
        "mae_seconds_std": ("mae_seconds", "std"),
        "rmse_seconds_mean": ("rmse_seconds", "mean"),
        "rmse_seconds_std": ("rmse_seconds", "std"),
        "r2_seconds_mean": ("r2_seconds", "mean"),
        "inference_seconds_mean": ("inference_seconds", "mean"),
        "parameter_count": ("parameter_count", "first"),
        "best_epoch_mean": ("best_epoch", "mean"),
    }
    for output_name, specification in optional.items():
        if specification[0] in successful.columns:
            aggregations[output_name] = specification
    return (
        successful.groupby(["dataset", "model", "weight_profile"], as_index=False)
        .agg(**aggregations)
        .sort_values(["dataset", "rmse_mean"])
        .reset_index(drop=True)
    )

def run_dataset_experiment(
    prepared: PreparedDataset,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    output_root: str | Path,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.nn.Module]]:
    output_root = Path(output_root)
    dataset_output = output_root / dataset_config["name"]
    dataset_output.mkdir(parents=True, exist_ok=True)
    training = copy.deepcopy(config["training"])
    if training_overrides:
        training.update(training_overrides)

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, torch.nn.Module] = {}
    seed_base = int(config.get("seed", 42))
    explicit_seeds = [int(value) for value in training.get("seeds", [])]
    repeats = len(explicit_seeds) if explicit_seeds else int(
        training.get("seed_repeats", 1)
    )
    seed_stride = int(training.get("seed_stride", 1000))
    if repeats < 1:
        raise ValueError("At least one training seed is required.")
    if explicit_seeds and int(training.get("seed_repeats", repeats)) != repeats:
        raise ValueError("seed_repeats must match the length of training.seeds.")

    for model_index, (model_name, model_config) in enumerate(
        config["models"].items(), start=1
    ):
        if not model_config.get("enabled", True):
            continue
        for profile_name, weights in _weight_profiles_for_model(
            model_name, model_config, config
        ):
            for repeat_index in range(repeats):
                repeat_number = repeat_index + 1
                seed = (
                    explicit_seeds[repeat_index]
                    if explicit_seeds
                    else seed_base + model_index + repeat_index * seed_stride
                )
                label = (
                    f"{model_name}__{profile_name}__seed_{repeat_number:02d}"
                )
                artifact_dir = dataset_output / label
                started = time.time()
                try:
                    model, _, _, metrics = train_one_model(
                        prepared=prepared,
                        model_name=model_name,
                        model_config=model_config,
                        weights=weights,
                        physics=config["physics"],
                        training_config=training,
                        seed=seed,
                        artifact_dir=artifact_dir,
                    )
                    trained_models[label] = model
                    rows.append(
                        {
                            "dataset": dataset_config["name"],
                            "model": model_name,
                            "weight_profile": profile_name,
                            "seed_repeat": repeat_number,
                            "seed": seed,
                            "status": "ok",
                            "seconds": time.time() - started,
                            **metrics,
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
                            "weight_profile": profile_name,
                            "seed_repeat": repeat_number,
                            "seed": seed,
                            "status": "failed",
                            "seconds": time.time() - started,
                            "error": str(exc),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(dataset_output / "model_comparison.csv", index=False)
    _comparison_summary(result).to_csv(
        dataset_output / "model_comparison_summary.csv", index=False
    )
    return result, trained_models

def _set_nested(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def run_sensitivity_analysis(
    prepared: PreparedDataset,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    output_root: str | Path,
) -> pd.DataFrame:
    sensitivity_config = config["sensitivity"]
    if not sensitivity_config.get("enabled", True):
        return pd.DataFrame()
    output_dir = Path(output_root) / dataset_config["name"] / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = sensitivity_config.get("model", "strong_pinn")
    model_config = config["models"][model_name]
    base_profile = sensitivity_config["base_weight_profile"]
    base_weights = copy.deepcopy(config["weight_profiles"][base_profile])
    training_overrides = {
        "epochs": int(sensitivity_config["epochs"]),
        "patience": int(sensitivity_config["patience"]),
    }

    rows = []
    trial = 0
    for parameter in sensitivity_config["parameters"]:
        path = parameter["path"]
        for value in parameter["values"]:
            trial += 1
            trial_config = copy.deepcopy(config)
            _set_nested(trial_config, path, value)
            weights = (
                trial_config["weight_profiles"][base_profile]
                if path.startswith("weight_profiles.")
                else base_weights
            )
            trial_dir = output_dir / f"trial_{trial:03d}"
            try:
                _, _, _, metrics = train_one_model(
                    prepared=prepared,
                    model_name=model_name,
                    model_config=model_config,
                    weights=weights,
                    physics=trial_config["physics"],
                    training_config={**config["training"], **training_overrides},
                    seed=int(config.get("seed", 42)) + 10_000,
                    artifact_dir=trial_dir,
                    evaluation_split="validation",
                )
                rows.append(
                    {
                        "dataset": dataset_config["name"],
                        "model": model_name,
                        "parameter": path,
                        "value": value,
                        "selection_split": "validation",
                        "test_evaluated": False,
                        "status": "ok",
                        **metrics,
                    }
                )
            except Exception as exc:
                (trial_dir / "failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                rows.append(
                    {
                        "dataset": dataset_config["name"],
                        "model": model_name,
                        "parameter": path,
                        "value": value,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(output_dir / "sensitivity_results.csv", index=False)
    if not sensitivity.empty and (sensitivity["status"] == "ok").any():
        successful = sensitivity[sensitivity["status"] == "ok"].copy()
        summary = (
            successful.groupby("parameter")
            .agg(
                rmse_min=("rmse", "min"),
                rmse_max=("rmse", "max"),
                rmse_range=("rmse", lambda values: float(values.max() - values.min())),
                trials=("rmse", "size"),
            )
            .sort_values("rmse_range", ascending=False)
            .reset_index()
        )
        summary.to_csv(output_dir / "sensitivity_ranking.csv", index=False)
    return sensitivity


def run_all_experiments(
    config: dict[str, Any],
    project_root: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    refresh_features: bool = False,
    run_sensitivity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    comparison_frames = []
    sensitivity_frames = []
    for dataset_config in enabled_dataset_configs(config):
        frame = load_or_extract_dataset(
            dataset_config,
            project_root=project_root,
            cache_dir=cache_dir,
            refresh=refresh_features,
        )
        prepared = prepare_sequence_dataset(
            frame,
            dataset_config,
            sequence_length=int(config["training"]["sequence_length"]),
        )
        comparison, _ = run_dataset_experiment(
            prepared,
            dataset_config,
            config,
            output_root,
        )
        comparison_frames.append(comparison)
        assumptions_path = output_root / dataset_config["name"] / "assumptions.json"
        assumptions_path.write_text(
            json.dumps(
                {
                    "physics_assumptions": dataset_config.get(
                        "physics_assumptions", []
                    ),
                    "operating_conditions": dataset_config.get(
                        "operating_conditions", {}
                    ),
                    "important": (
                        "The aSKF expression is a differentiable calibrated surrogate "
                        "for catalog curves; it is not claimed to reproduce a full "
                        "manufacturer/ISO lookup without calibration."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        sensitivity_datasets = config['sensitivity'].get('datasets')
        sensitivity_selected = (
            not sensitivity_datasets
            or dataset_config['name'] in sensitivity_datasets
        )
        if run_sensitivity and sensitivity_selected:
            sensitivity = run_sensitivity_analysis(
                prepared, dataset_config, config, output_root
            )
            if not sensitivity.empty:
                sensitivity_frames.append(sensitivity)

    all_results = (
        pd.concat(comparison_frames, ignore_index=True)
        if comparison_frames
        else pd.DataFrame()
    )
    all_sensitivity = (
        pd.concat(sensitivity_frames, ignore_index=True)
        if sensitivity_frames
        else pd.DataFrame()
    )
    all_results.to_csv(output_root / "all_model_comparisons.csv", index=False)
    _comparison_summary(all_results).to_csv(
        output_root / "all_model_comparisons_summary.csv", index=False
    )
    all_sensitivity.to_csv(output_root / "all_sensitivity_results.csv", index=False)
    return all_results, all_sensitivity
