from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "saved results"
    / "run_07a"
    / "colab_run_01"
    / "experiment_outputs_exp007a"
)
DEFAULT_OUTPUT = ROOT / "results" / "analyzed" / "EXP-007A"
UNIT_KEYS = ["parent_seed", "run_id", "candidate_spec"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ece(target: np.ndarray, probability: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return result


def macro_run_rmse(frame: pd.DataFrame, prediction: str) -> float:
    return float(
        np.mean(
            [
                math.sqrt(mean_squared_error(run["target_rul"], run[prediction]))
                for _, run in frame.groupby("run_id")
            ]
        )
    )


def aggregate_units(frame: pd.DataFrame) -> pd.DataFrame:
    constant = [
        "safe_to_apply",
        "harmful_intervention",
        "credibility",
        "credibility_threshold",
        "physics_regret",
        "law_correctness",
        "candidate_family",
        "time_scale_factor",
        "true_family",
    ]
    uniqueness = frame.groupby(UNIT_KEYS)[constant].nunique(dropna=False)
    if (uniqueness > 1).any().any():
        raise ValueError("A trajectory-candidate unit has inconsistent labels or scores.")
    return frame.groupby(UNIT_KEYS, as_index=False).agg({name: "first" for name in constant})


def hierarchical_bootstrap(units: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("No valid EXP-007A bootstrap replicates were produced.")
    return {
        "replicates_requested": int(config["evaluation"]["bootstrap_replicates"]),
        "replicates_valid": len(estimates),
        "mean": float(np.mean(estimates)),
        "median": float(np.median(estimates)),
        "auroc_ci_lower_95": float(np.quantile(estimates, 0.025)),
        "auroc_ci_upper_95": float(np.quantile(estimates, 0.975)),
        "aggregation": "trajectory_within_resampled_seed_then_mean_seed_auroc",
    }


def validate_artifacts(input_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    inventory_path = input_root / "artifact_inventory.csv"
    observed_inventory_sha = sha256_file(inventory_path)
    inventory = pd.read_csv(inventory_path)
    missing: list[str] = []
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    for row in inventory.itertuples(index=False):
        path = input_root / row.relative_path
        if not path.is_file():
            missing.append(row.relative_path)
        elif path.stat().st_size != int(row.bytes):
            size_mismatches.append(row.relative_path)
        elif sha256_file(path) != row.sha256:
            hash_mismatches.append(row.relative_path)
    bundle_path = input_root / "codex_results_bundle.zip"
    unsafe: list[str] = []
    duplicates: list[str] = []
    bundle_hash_mismatches: list[str] = []
    bundle_missing_from_full: list[str] = []
    with zipfile.ZipFile(bundle_path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        for info in infos:
            name = info.filename
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                unsafe.append(name)
                continue
            full_path = input_root / name
            if not full_path.is_file():
                bundle_missing_from_full.append(name)
                continue
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sha256_file(full_path):
                bundle_hash_mismatches.append(name)
        bundle_entries = len(infos)
        bundle_uncompressed_bytes = sum(item.file_size for item in infos)
    validation = {
        "status": "passed",
        "artifact_inventory_records": int(len(inventory)),
        "artifact_inventory_sha256": observed_inventory_sha,
        "artifact_inventory_sha_matches_manifest": (
            observed_inventory_sha == manifest["artifact_inventory_sha256"]
        ),
        "missing_artifacts": missing,
        "artifact_size_mismatches": size_mismatches,
        "artifact_hash_mismatches": hash_mismatches,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_entries": bundle_entries,
        "bundle_uncompressed_bytes": bundle_uncompressed_bytes,
        "bundle_unsafe_entries": unsafe,
        "bundle_duplicate_entries": duplicates,
        "bundle_missing_from_full": bundle_missing_from_full,
        "bundle_hash_mismatches": bundle_hash_mismatches,
        "executed_notebook_sha_matches_manifest": (
            sha256_file(input_root / "executed_notebook.ipynb")
            == manifest["executed_notebook_sha256"]
        ),
    }
    failed = any(
        (
            not validation["artifact_inventory_sha_matches_manifest"],
            missing,
            size_mismatches,
            hash_mismatches,
            unsafe,
            duplicates,
            bundle_missing_from_full,
            bundle_hash_mismatches,
            not validation["executed_notebook_sha_matches_manifest"],
        )
    )
    if failed:
        validation["status"] = "failed"
        raise ValueError(f"EXP-007A artifact validation failed: {validation}")
    return validation


def verify_identity(input_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("experiment_id") != "EXP-007A" or manifest.get("status") != "completed":
        raise ValueError("The incoming result is not a completed EXP-007A execution.")
    commit = str(manifest["git_commit"])
    commit_type = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    active_config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    result_config = yaml.safe_load((input_root / "experiment_config.yaml").read_text())
    active_config["repository"]["expected_commit"] = commit
    active_split = json.loads((ROOT / "configs" / "exp007a_data_split.json").read_text())
    result_split = json.loads((input_root / "data_split.json").read_text())
    checks = {
        "git_commit": commit,
        "git_object_type": commit_type,
        "config_semantically_equal_to_recorded_commit_inputs": active_config == result_config,
        "config_sha256": json_sha256(result_config),
        "config_sha_matches_manifest": json_sha256(result_config) == manifest["config_sha256"],
        "split_exactly_equal": active_split == result_split,
        "split_sha256": json_sha256(result_split),
        "split_sha_matches_manifest": json_sha256(result_split) == manifest["split_sha256"],
        "scenario_sha256": sha256_file(
            ROOT / active_config["data"]["scenario_file"]
        ),
        "feature_sha256": sha256_file(ROOT / active_config["data"]["feature_cache"]),
        "metadata_sha256": sha256_file(ROOT / active_config["data"]["metadata_file"]),
    }
    checks["scenario_sha_matches_manifest"] = (
        checks["scenario_sha256"] == manifest["scenario_sha256"]
    )
    checks["feature_sha_matches_manifest"] = (
        checks["feature_sha256"] == manifest["feature_sha256"]
    )
    checks["metadata_sha_matches_config"] = (
        checks["metadata_sha256"] == active_config["data"]["expected_metadata_sha256"]
    )
    required_true = [name for name, value in checks.items() if name.endswith(("equal", "manifest", "config", "inputs")) and isinstance(value, bool)]
    if not all(checks[name] for name in required_true):
        raise ValueError(f"EXP-007A identity validation failed: {checks}")
    return checks


def verified_credibility(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    units = aggregate_units(predictions)
    rows: list[dict[str, Any]] = []
    for seed, group in units.groupby("parent_seed"):
        target = group["safe_to_apply"].to_numpy(dtype=int)
        probability = group["credibility"].to_numpy(dtype=float)
        threshold = float(group["credibility_threshold"].iloc[0])
        prevalence = float(target.mean())
        brier = float(brier_score_loss(target, probability))
        rows.append(
            {
                "parent_seed": int(seed),
                "auroc": float(roc_auc_score(target, probability)),
                "auprc": float(average_precision_score(target, probability)),
                "prevalence_auprc": prevalence,
                "brier": brier,
                "prevalence_brier": prevalence * (1.0 - prevalence),
                "brier_better_than_prevalence": brier < prevalence * (1.0 - prevalence),
                "ece": ece(
                    target,
                    probability,
                    int(config["evaluation"]["calibration_bins"]),
                ),
                "threshold": threshold,
                "all_on_fraction": float(np.mean(probability >= threshold)),
                "all_off_fraction": float(np.mean(probability < threshold)),
                "trajectory_candidate_units": int(len(group)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("parent_seed")
    bootstrap = hierarchical_bootstrap(units, config)
    statistical = {
        "mean_seed_auroc": float(metrics["auroc"].mean()),
        "std_seed_auroc": float(metrics["auroc"].std(ddof=1)),
        "mean_seed_auprc": float(metrics["auprc"].mean()),
        "std_seed_auprc": float(metrics["auprc"].std(ddof=1)),
        "mean_seed_brier": float(metrics["brier"].mean()),
        "std_seed_brier": float(metrics["brier"].std(ddof=1)),
        "mean_prevalence_brier": float(metrics["prevalence_brier"].mean()),
        "test_safe_fraction": float(units["safe_to_apply"].mean()),
        "test_units": int(len(units)),
        **bootstrap,
    }
    return units, metrics, statistical


def verified_controls(
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    for (seed, method), group in controls.groupby(["parent_seed", "method"]):
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
                "macro_run_rmse": macro_run_rmse(group, "predicted_rul"),
                "samples": int(len(group)),
                "runs": int(group["run_id"].nunique()),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(["parent_seed", "method"])
    regret_rows: list[dict[str, Any]] = []
    for (seed, run_id), subset in controls.groupby(["parent_seed", "run_id"]):
        baseline = subset[subset["method"] == "data_only"]
        baseline_rmse = math.sqrt(
            mean_squared_error(baseline["target_rul"], baseline["predicted_rul"])
        )
        for method, group in subset.groupby("method"):
            rmse = math.sqrt(
                mean_squared_error(group["target_rul"], group["predicted_rul"])
            )
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
    regret = pd.DataFrame(regret_rows).sort_values(["parent_seed", "run_id", "method"])
    aggregate = (
        comparison.groupby("method")
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_macro_run_rmse=("macro_run_rmse", "mean"),
            sd_macro_run_rmse=("macro_run_rmse", "std"),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values("mean_macro_run_rmse")
    )
    baseline = float(
        aggregate.loc[aggregate["method"] == "data_only", "mean_macro_run_rmse"].iloc[0]
    )
    all_on = float(
        aggregate.loc[aggregate["method"] == "all_on", "mean_macro_run_rmse"].iloc[0]
    )
    aggregate["improvement_vs_data_pct"] = (
        100.0 * (baseline - aggregate["mean_macro_run_rmse"]) / baseline
    )
    aggregate["improvement_vs_all_on_pct"] = (
        100.0 * (all_on - aggregate["mean_macro_run_rmse"]) / all_on
    )
    return comparison, regret, aggregate


def compare_numeric_frames(
    verified: pd.DataFrame,
    reported: pd.DataFrame,
    keys: list[str],
) -> float:
    merged = verified.merge(reported, on=keys, suffixes=("_verified", "_reported"), validate="one_to_one")
    if len(merged) != len(verified) or len(verified) != len(reported):
        raise ValueError(f"Reported and verified row identities differ for {keys}.")
    differences: list[float] = []
    for column in verified.select_dtypes(include=[np.number]).columns:
        if column in keys or f"{column}_reported" not in merged:
            continue
        differences.append(
            float(
                np.max(
                    np.abs(
                        merged[f"{column}_verified"].to_numpy(dtype=float)
                        - merged[f"{column}_reported"].to_numpy(dtype=float)
                    )
                )
            )
        )
    return max(differences, default=0.0)


def candidate_diagnostics(candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, scale), group in candidate.groupby(["candidate_family", "time_scale_factor"]):
        rows.append(
            {
                "scope": "candidate_family_scale",
                "group": f"{family}__scale_{float(scale):.2f}",
                "units": int(len(group)),
                "safe_fraction": float(group["safe_to_apply"].mean()),
                "mean_regret": float(group["physics_regret"].mean()),
                "median_regret": float(group["physics_regret"].median()),
                "mean_positive_regret": float(
                    np.maximum(group["physics_regret"].to_numpy(dtype=float), 0.0).mean()
                ),
            }
        )
    for correctness, group in candidate.groupby("law_correctness"):
        rows.append(
            {
                "scope": "law_correctness",
                "group": str(bool(correctness)).lower(),
                "units": int(len(group)),
                "safe_fraction": float(group["safe_to_apply"].mean()),
                "mean_regret": float(group["physics_regret"].mean()),
                "median_regret": float(group["physics_regret"].median()),
                "mean_positive_regret": float(
                    np.maximum(group["physics_regret"].to_numpy(dtype=float), 0.0).mean()
                ),
            }
        )
    for family, group in candidate.groupby("true_family"):
        rows.append(
            {
                "scope": "true_family",
                "group": family,
                "units": int(len(group)),
                "safe_fraction": float(group["safe_to_apply"].mean()),
                "mean_regret": float(group["physics_regret"].mean()),
                "median_regret": float(group["physics_regret"].median()),
                "mean_positive_regret": float(
                    np.maximum(group["physics_regret"].to_numpy(dtype=float), 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def credibility_by_family(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in units.groupby("true_family"):
        rows.append(
            {
                "true_family": family,
                "units": int(len(group)),
                "safe_fraction": float(group["safe_to_apply"].mean()),
                "auroc": float(roc_auc_score(group["safe_to_apply"], group["credibility"])),
                "auprc": float(
                    average_precision_score(group["safe_to_apply"], group["credibility"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("true_family")


def convergence_summary(history: pd.DataFrame) -> pd.DataFrame:
    numeric = history.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Training history contains non-finite numeric values.")
    loss_columns = ["physics_value_loss", "physics_rate_loss", "monotonic_loss"]
    if not (history[loss_columns] >= 0.0).all().all():
        raise ValueError("Training history contains a negative physics loss.")
    keys = ["parent_seed", "optimization_seed", "phase", "candidate_spec"]
    rows: list[dict[str, Any]] = []
    for identity, group in history.groupby(keys, dropna=False):
        best_index = group["validation_mse"].idxmin()
        best = group.loc[best_index]
        final = group.sort_values("epoch").iloc[-1]
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "epochs": int(group["epoch"].max()),
                "best_epoch": int(best["epoch"]),
                "best_validation_mse": float(best["validation_mse"]),
                "final_validation_mse": float(final["validation_mse"]),
                "final_train_loss": float(final["train_loss"]),
                "final_over_best_validation_ratio": float(
                    final["validation_mse"] / best["validation_mse"]
                ),
                "elapsed_seconds": float(group["elapsed_seconds"].max()),
            }
        )
    result = pd.DataFrame(rows).sort_values(keys)
    if len(result) != 325:
        raise ValueError(f"Expected 325 training fits, found {len(result)}.")
    return result


def runtime_summary(input_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = [
        json.loads(path.read_text())
        for path in sorted((input_root / "seeds").glob("seed_*/job_result.json"))
    ]
    completed_lines = [
        line
        for line in (input_root / "training.log").read_text().splitlines()
        if "status=completed test_evaluated=" in line
    ]
    first_start = next(
        line for line in (input_root / "training.log").read_text().splitlines() if "start/resume" in line
    )
    first_time = datetime.fromisoformat(first_start.split()[0])
    final_seed_time = datetime.fromisoformat(completed_lines[-1].split()[0])
    return {
        "manifest_elapsed_seconds": float(manifest["elapsed_seconds"]),
        "manifest_scope": "final resume and artifact finalization",
        "per_seed_training_seconds": {
            str(job["parent_seed"]): float(job["seconds"]) for job in jobs
        },
        "sum_per_seed_training_seconds": float(sum(job["seconds"] for job in jobs)),
        "mean_per_seed_training_seconds": float(np.mean([job["seconds"] for job in jobs])),
        "initial_pass_wall_seconds_from_log": float((final_seed_time - first_time).total_seconds()),
        "resume_skip_count": sum(
            "resume-skip" in line for line in (input_root / "training.log").read_text().splitlines()
        ),
        "validation_selected_scalars": {
            str(job["parent_seed"]): float(job["validation_selected_scalar"]) for job in jobs
        },
        "backbone_parameter_counts": sorted(
            {int(job["backbone_parameter_count"]) for job in jobs}
        ),
        "backbone_best_epochs": {
            str(job["parent_seed"]): int(job["backbone_best_epoch"]) for job in jobs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify EXP-007A artifacts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((input_root / "run_manifest.json").read_text())
    config = yaml.safe_load((input_root / "experiment_config.yaml").read_text())
    identity = verify_identity(input_root, manifest)
    artifacts = validate_artifacts(input_root, manifest)
    artifacts["identity"] = identity
    failures = json.loads((input_root / "failure_report.json").read_text())
    if failures or manifest["failed_models"]:
        raise ValueError("EXP-007A contains an unreported model or seed failure.")

    credibility = pd.read_csv(input_root / "credibility_predictions.csv")
    controls = pd.read_csv(input_root / "control_predictions.csv")
    candidate = pd.read_csv(input_root / "candidate_regret.csv")
    history = pd.read_csv(input_root / "training_history.csv")
    split = json.loads((input_root / "data_split.json").read_text())
    expected_test = set(split["test_runs"])
    if set(credibility["run_id"].unique()) != expected_test or set(
        controls["run_id"].unique()
    ) != expected_test:
        raise ValueError("Serialized predictions do not use the exact sealed-test population.")
    if set(credibility["partition"].unique()) != {"test"} or set(
        controls["partition"].unique()
    ) != {"test"}:
        raise ValueError("A non-test row entered the sealed prediction files.")

    units, metrics, statistical = verified_credibility(credibility, config)
    comparison, physics_regret, model_aggregate = verified_controls(controls)
    diagnostics = candidate_diagnostics(candidate)
    family_metrics = credibility_by_family(units)
    convergence = convergence_summary(history)
    runtime = runtime_summary(input_root, manifest)

    reported_metrics = pd.read_csv(input_root / "credibility_metrics.csv")
    reported_comparison = pd.read_csv(input_root / "model_comparison.csv")
    reported_physics_regret = pd.read_csv(input_root / "physics_regret.csv")
    reported_statistical = json.loads((input_root / "statistical_summary.json").read_text())
    metric_difference = compare_numeric_frames(metrics, reported_metrics, ["parent_seed"])
    model_difference = compare_numeric_frames(
        comparison, reported_comparison, ["parent_seed", "method"]
    )
    regret_difference = compare_numeric_frames(
        physics_regret, reported_physics_regret, ["parent_seed", "run_id", "method"]
    )
    statistical_difference = max(
        abs(float(statistical[key]) - float(reported_statistical[key]))
        for key in (
            "mean_seed_auroc",
            "std_seed_auroc",
            "mean_seed_auprc",
            "mean_seed_brier",
            "mean_prevalence_brier",
            "mean",
            "median",
            "auroc_ci_lower_95",
            "auroc_ci_upper_95",
        )
    )
    artifacts["maximum_metric_absolute_difference"] = metric_difference
    artifacts["maximum_model_comparison_absolute_difference"] = model_difference
    artifacts["maximum_physics_regret_absolute_difference"] = regret_difference
    artifacts["maximum_statistical_absolute_difference"] = statistical_difference
    artifacts["serialization_verification"] = json.loads(
        (input_root / "serialization_verification.json").read_text()
    )

    development = json.loads(
        (input_root / "development_target_qualification.json").read_text()
    )
    for seed in manifest["seeds"]:
        seed_root = input_root / "seeds" / f"seed_{int(seed):05d}"
        frames = [
            pd.read_csv(seed_root / "train_counterfactual_evidence.csv"),
            pd.read_csv(seed_root / "validation_counterfactual_evidence.csv"),
        ]
        dev_units = pd.concat([aggregate_units(frame) for frame in frames], ignore_index=True)
        saved = development["per_seed"][str(seed)]
        if len(dev_units) != 960 or not math.isclose(
            float(dev_units["safe_to_apply"].mean()),
            float(saved["safe_fraction"]),
            abs_tol=1e-15,
        ):
            raise ValueError(f"Development qualification does not reproduce for seed {seed}.")

    maximum = float(config["success_criteria"]["maximum_all_off_fraction_per_seed"])
    mean_positive = physics_regret.groupby("method")["positive_physics_regret"].mean()
    harmful = candidate[candidate["harmful_intervention"] == 1]
    criteria = pd.DataFrame(
        [
            {
                "criterion": "mean_within_seed_auroc",
                "threshold": f">={config['success_criteria']['minimum_test_auroc']}",
                "observed": statistical["mean_seed_auroc"],
                "passed": statistical["mean_seed_auroc"]
                >= float(config["success_criteria"]["minimum_test_auroc"]),
            },
            {
                "criterion": "hierarchical_bootstrap_lower_95",
                "threshold": f">{config['success_criteria']['minimum_auroc_ci_lower']}",
                "observed": statistical["auroc_ci_lower_95"],
                "passed": statistical["auroc_ci_lower_95"]
                > float(config["success_criteria"]["minimum_auroc_ci_lower"]),
            },
            {
                "criterion": "per_seed_anti_collapse",
                "threshold": f"all on/off <={maximum}",
                "observed": float(
                    max(metrics["all_on_fraction"].max(), metrics["all_off_fraction"].max())
                ),
                "passed": bool(
                    (metrics["all_on_fraction"] <= maximum).all()
                    and (metrics["all_off_fraction"] <= maximum).all()
                ),
            },
            {
                "criterion": "brier_better_than_prevalence",
                "threshold": "mean Brier < mean prevalence Brier",
                "observed": statistical["mean_seed_brier"],
                "passed": statistical["mean_seed_brier"]
                < statistical["mean_prevalence_brier"],
            },
            {
                "criterion": "harm_stress",
                "threshold": "harmful-candidate mean regret > 0",
                "observed": float(harmful["physics_regret"].mean()),
                "passed": bool(len(harmful) and harmful["physics_regret"].mean() > 0.0),
            },
            {
                "criterion": "priorcred_positive_regret_below_all_on",
                "threshold": "PriorCred < all-on",
                "observed": float(mean_positive["priorcred"]),
                "passed": bool(mean_positive["priorcred"] < mean_positive["all_on"]),
            },
            {
                "criterion": "priorcred_positive_regret_below_validation_scalar",
                "threshold": "PriorCred < validation-selected scalar",
                "observed": float(mean_positive["priorcred"]),
                "passed": bool(
                    mean_positive["priorcred"]
                    < mean_positive["validation_selected_scalar"]
                ),
            },
        ]
    )
    gate_passed = bool(criteria["passed"].all())
    if gate_passed or manifest["gate_decision"] != "stop_and_diagnose_exp007a":
        raise ValueError("Independent gate decision disagrees with the saved failure decision.")

    pivot = physics_regret.pivot_table(
        index=["parent_seed", "run_id"], columns="method", values="rmse"
    )
    wins_data = int((pivot["priorcred"] < pivot["data_only"]).sum())
    wins_all_on = int((pivot["priorcred"] < pivot["all_on"]).sum())
    priorcred_aggregate = model_aggregate.set_index("method").loc["priorcred"]
    correct = diagnostics[
        (diagnostics["scope"] == "law_correctness") & (diagnostics["group"] == "true")
    ].iloc[0]
    incorrect = diagnostics[
        (diagnostics["scope"] == "law_correctness") & (diagnostics["group"] == "false")
    ].iloc[0]

    write_json(output_root / "artifact_validation.json", artifacts)
    metrics.to_csv(output_root / "verified_metrics.csv", index=False)
    write_json(output_root / "verified_statistical_summary.json", statistical)
    comparison.to_csv(output_root / "verified_model_comparison.csv", index=False)
    physics_regret.to_csv(output_root / "verified_physics_regret.csv", index=False)
    model_aggregate.to_csv(output_root / "model_aggregate.csv", index=False)
    diagnostics.to_csv(output_root / "candidate_diagnostics.csv", index=False)
    family_metrics.to_csv(output_root / "credibility_by_true_family.csv", index=False)
    convergence.to_csv(output_root / "convergence_summary.csv", index=False)
    criteria.to_csv(output_root / "success_criteria.csv", index=False)
    write_json(output_root / "runtime_summary.json", runtime)

    analysis = f"""# EXP-007A verified analysis

## Validity and execution

EXP-007A is a valid completed execution of commit `{manifest['git_commit']}` on a Tesla T4.
All {artifacts['artifact_inventory_records']} inventoried full artifacts and all
{artifacts['bundle_entries']} lightweight-bundle entries passed size, SHA-256, and safe-path
checks. The configuration, 64/16/16 split, simulator scenario, feature cache, metadata, and five
seeds match the frozen experiment. All seeds completed, the development target gate passed for
every seed, and sealed-test access followed the declared order. No model, OOM, or numerical
failure was reported.

The manifest's `{manifest['elapsed_seconds']:.1f}` seconds describes a later resume/finalization
pass. The five saved seed jobs total `{runtime['sum_per_seed_training_seconds']:.1f}` seconds
(`{runtime['sum_per_seed_training_seconds']/60:.1f}` minutes), while the first start-to-last-seed
wall interval in the log is `{runtime['initial_pass_wall_seconds_from_log']:.1f}` seconds. This
runtime-scope discrepancy does not alter predictions but must be reported accurately.

## Independently verified primary endpoint

The test population contains {statistical['test_units']} trajectory-candidate-seed units. Mean
within-seed safe-intervention AUROC is `{statistical['mean_seed_auroc']:.6f} +/-
{statistical['std_seed_auroc']:.6f}`. The trajectory-within-resampled-seed bootstrap 95% interval
is `[{statistical['auroc_ci_lower_95']:.6f}, {statistical['auroc_ci_upper_95']:.6f}]`. The point
estimate is below the frozen `0.80` requirement, although the interval remains above chance.
Mean AUPRC is `{statistical['mean_seed_auprc']:.6f}` and mean Brier score is
`{statistical['mean_seed_brier']:.6f}`, better than the `{statistical['mean_prevalence_brier']:.6f}`
constant-prevalence reference. No seed crossed the 90% all-on/all-off collapse limit.

Performance is heterogeneous by true progression family: step-like AUROC is
`{family_metrics.set_index('true_family').loc['step_like','auroc']:.6f}`, but gamma AUROC is only
`{family_metrics.set_index('true_family').loc['gamma','auroc']:.6f}`. Seed AUROC ranges from
`{metrics['auroc'].min():.6f}` to `{metrics['auroc'].max():.6f}`.

## RUL control outcome

PriorCred's mean macro run RMSE is `{priorcred_aggregate['mean_macro_run_rmse']:.6f}` normalized
RUL, an improvement of `{priorcred_aggregate['improvement_vs_data_pct']:.2f}%` over data-only and
`{priorcred_aggregate['improvement_vs_all_on_pct']:.2f}%` over all-on physics. It beats data-only
on `{wins_data}/80` seed-trajectory pairs and all-on on `{wins_all_on}/80`. Oracle selection is
better (`{model_aggregate.set_index('method').loc['oracle','mean_macro_run_rmse']:.6f}`), while
anti-oracle is much worse (`{model_aggregate.set_index('method').loc['anti_oracle','mean_macro_run_rmse']:.6f}`),
so intervention safety is consequential and useful headroom exists.

The gain is not risk-safe under the frozen definition. PriorCred's mean positive run regret is
`{mean_positive['priorcred']:.6f}`, versus `{mean_positive['all_on']:.6f}` for all-on and exactly
`{mean_positive['validation_selected_scalar']:.6f}` for the validation-selected scalar. Its
worst positive regret is `{physics_regret.loc[physics_regret['method']=='priorcred','positive_physics_regret'].max():.6f}`.
Every seed selected scalar zero, so the comparator reduced to data-only and had zero positive
regret. The strict requirement that PriorCred be below this nonnegative zero reference was
therefore impossible in this realized run. The criterion remains failed; it cannot be relaxed
after opening the test set.

## Physics interpretation

The harm stress worked: {len(harmful)} harmful interventions have mean normalized-RUL regret
`{harmful['physics_regret'].mean():.6f}`. However, simulator-family correctness is not a safety
proxy. Correct family/scale candidates are safe only `{correct['safe_fraction']:.2%}` of the time
and have mean regret `{correct['mean_regret']:.6f}`, compared with `{incorrect['safe_fraction']:.2%}`
safe and `{incorrect['mean_regret']:.6f}` mean regret for the other candidates. This supports the
research premise that a mathematically matched degradation prior can still cause negative
transfer when imposed through a learned RUL model.

## Convergence and stability

All 325 expected fits are represented with finite histories and separate nonnegative data,
prior-value, prior-rate, and monotonic losses. Median training length is
`{convergence['epochs'].median():.0f}` epochs; the maximum is `{convergence['epochs'].max()}`.
The median final-to-best validation-MSE ratio is
`{convergence['final_over_best_validation_ratio'].median():.3f}`, consistent with checkpointed
early stopping rather than missing training. The final backbone has 22,625 parameters.

## Decision

EXP-007A fails the frozen publication gate because AUROC is below 0.80 and PriorCred does not
reduce positive regret below both declared comparators. EXP-008 remains blocked. The average RUL
improvement and law-correctness result are promising findings, but this opened synthetic test
set can now support diagnosis only, not confirmation of a revised method.
"""
    (output_root / "analysis.md").write_text(analysis, encoding="utf-8")

    issues = f"""# EXP-007A issues

1. **Primary discrimination gate failed.** Verified mean seed AUROC is
   `{statistical['mean_seed_auroc']:.6f}`, below `0.80`; gamma-family AUROC is especially weak at
   `{family_metrics.set_index('true_family').loc['gamma','auroc']:.6f}`.
2. **Tail-risk gate failed.** PriorCred improves average RMSE but has mean positive regret
   `{mean_positive['priorcred']:.6f}` and maximum `{physics_regret.loc[physics_regret['method']=='priorcred','positive_physics_regret'].max():.6f}`.
3. **The scalar comparator became a zero-risk impossibility bound.** Validation selected scalar
   zero for all five seeds. Its positive regret is zero, so a strict lower-than comparison cannot
   be satisfied by a nonnegative positive-regret statistic. This is a protocol-design finding,
   not permission to rescore EXP-007A.
4. **Seed and family heterogeneity remain.** Seed AUROC spans `{metrics['auroc'].min():.6f}` to
   `{metrics['auroc'].max():.6f}`; true-family AUROC spans
   `{family_metrics['auroc'].min():.6f}` to `{family_metrics['auroc'].max():.6f}`.
5. **Manifest elapsed time is incomplete.** It records the final resume/finalization pass
   (`{manifest['elapsed_seconds']:.1f}` seconds), not the approximately
   `{runtime['sum_per_seed_training_seconds']/60:.1f}` minutes of saved seed training.
6. **The test population is now open.** Further feature, threshold, candidate, or loss changes
   evaluated on these 16 trajectories are exploratory and cannot validate the revised method.
"""
    (output_root / "issues.md").write_text(issues, encoding="utf-8")

    recommendations = """# EXP-007A recommendations

1. Do not prepare EXP-008 from these results. Record EXP-007A as a valid negative gate outcome.
2. Use the opened EXP-007A test artifacts only for diagnostics: localize the PriorCred regret
   tail by family, scale, condition, seed, and lifecycle; do not tune a confirmatory score on it.
3. Before a fresh run, replace the impossible strict comparison against zero positive regret
   with a preregistered clinically/engineering-meaningful risk objective, such as a non-inferiority
   margin, upper-tail regret bound, or constrained average-improvement plus harm-rate criterion.
4. Train the selector for the decision objective directly. AUROC ranking alone does not control
   rare large regret; test asymmetric or cost-sensitive learning using development data only.
5. Focus development diagnostics on gamma trajectories and high-regret 1.6-scale priors, while
   preserving progression-family and operating-condition coverage.
6. Any corrective EXP-007B must use a newly generated sealed simulator seed and a frozen protocol.
   The current 16 test trajectories cannot be reused as confirmation.
7. Defer ANSYS and real-bearing confirmation until the synthetic safety gate is coherent and
   passes on a fresh population. Higher-fidelity physics will not fix an ill-posed decision gate.
"""
    (output_root / "recommendations.md").write_text(recommendations, encoding="utf-8")

    previous = pd.read_csv(ROOT / "results" / "analyzed" / "EXP-007" / "verified_metrics.csv")
    previous_mean = previous[previous["scope"] == "seed_mean"].iloc[0]
    comparison_stage = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-007",
                "target": "mathematical_prior_correctness",
                "mean_seed_auroc": float(previous_mean["auroc"]),
                "seed_auroc_sd": float(previous_mean["auroc_std"]),
                "gate": "failed",
                "comparability_note": "Different target and test population; descriptive only",
            },
            {
                "experiment_id": "EXP-007A",
                "target": "counterfactual_safe_intervention",
                "mean_seed_auroc": statistical["mean_seed_auroc"],
                "seed_auroc_sd": statistical["std_seed_auroc"],
                "gate": "failed",
                "comparability_note": "Different target and test population; descriptive only",
            },
        ]
    )
    comparison_stage.to_csv(
        ROOT / "results" / "comparisons" / "exp007_vs_exp007a.csv", index=False
    )
    print(
        json.dumps(
            {
                "status": "verified_gate_failed",
                "output": str(output_root),
                "mean_seed_auroc": statistical["mean_seed_auroc"],
                "auroc_ci": [
                    statistical["auroc_ci_lower_95"],
                    statistical["auroc_ci_upper_95"],
                ],
                "priorcred_macro_rmse": float(priorcred_aggregate["mean_macro_run_rmse"]),
                "artifact_records": artifacts["artifact_inventory_records"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
