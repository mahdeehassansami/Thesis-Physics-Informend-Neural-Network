from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)

from analyze_exp007a_results import (
    candidate_diagnostics,
    compare_numeric_frames,
    convergence_summary,
    credibility_by_family,
    ece,
    json_sha256,
    sha256_file,
    validate_artifacts,
    verified_controls,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "saved results"
    / "run_07b"
    / "colab_run_01"
    / "experiment_outputs_exp007b"
)
DEFAULT_OUTPUT = ROOT / "results" / "analyzed" / "EXP-007B"
CONTROL_KEYS = ["parent_seed", "run_id", "sample_index"]


def aggregate_causal_credibility_units(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    keys = [
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
    columns = [
        *config["credibility"]["numeric_evidence"],
        "credibility_raw",
        "credibility",
        "fallback",
        "physics_regret",
        "data_only_rmse",
        "physics_rmse",
    ]
    aggregations = {
        column: (column, "mean") for column in columns if column in frame.columns
    }
    return frame.groupby(keys, as_index=False).agg(**aggregations)


def verified_causal_credibility(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    units = aggregate_causal_credibility_units(predictions, config)
    rows = []
    for seed, group in units.groupby("parent_seed"):
        target = group["safe_to_apply"].to_numpy(int)
        probability = group["credibility"].to_numpy(float)
        source = predictions[predictions["parent_seed"] == seed]
        threshold = float(source["credibility_threshold"].iloc[0])
        sample_best = source.groupby(CONTROL_KEYS)["credibility"].max().to_numpy(float)
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
                "brier_better_than_prevalence": brier
                < prevalence * (1.0 - prevalence),
                "ece": ece(
                    target,
                    probability,
                    int(config["evaluation"]["calibration_bins"]),
                ),
                "threshold": threshold,
                "all_on_fraction": float(np.mean(sample_best >= threshold)),
                "all_off_fraction": float(np.mean(sample_best < threshold)),
                "trajectory_candidate_units": int(len(group)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("parent_seed")
    bootstrap = fast_hierarchical_bootstrap(units, config)
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


def fast_hierarchical_bootstrap(
    units: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in units["parent_seed"].unique())
    by_seed: dict[int, tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    for seed in seeds:
        seed_frame = units[units["parent_seed"] == seed]
        runs = sorted(str(run) for run in seed_frame["run_id"].unique())
        by_seed[seed] = (
            runs,
            {
                run: (
                    seed_frame.loc[
                        seed_frame["run_id"] == run, "safe_to_apply"
                    ].to_numpy(int),
                    seed_frame.loc[
                        seed_frame["run_id"] == run, "credibility"
                    ].to_numpy(float),
                )
                for run in runs
            },
        )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    estimates = []
    for _ in range(int(config["evaluation"]["bootstrap_replicates"])):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_estimates = []
        for sampled_seed in sampled_seeds:
            runs, arrays = by_seed[int(sampled_seed)]
            sampled_runs = rng.choice(runs, size=len(runs), replace=True)
            target = np.concatenate([arrays[str(run)][0] for run in sampled_runs])
            probability = np.concatenate([arrays[str(run)][1] for run in sampled_runs])
            if np.unique(target).size == 2:
                seed_estimates.append(float(roc_auc_score(target, probability)))
        if seed_estimates:
            estimates.append(float(np.mean(seed_estimates)))
    if not estimates:
        raise RuntimeError("EXP-007B hierarchical bootstrap produced no valid estimates.")
    return {
        "replicates_requested": int(config["evaluation"]["bootstrap_replicates"]),
        "replicates_valid": len(estimates),
        "mean": float(np.mean(estimates)),
        "median": float(np.median(estimates)),
        "auroc_ci_lower_95": float(np.quantile(estimates, 0.025)),
        "auroc_ci_upper_95": float(np.quantile(estimates, 0.975)),
        "aggregation": "trajectory_within_resampled_seed_then_mean_seed_auroc",
    }


def verify_identity(input_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("experiment_id") != "EXP-007B" or manifest.get("status") != "completed":
        raise ValueError("The incoming result is not a completed EXP-007B execution.")
    commit = str(manifest["git_commit"])
    commit_type = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    committed_config = yaml.safe_load(
        subprocess.run(
            ["git", "show", f"{commit}:configs/experiment.yaml"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    active_config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    result_config = yaml.safe_load((input_root / "experiment_config.yaml").read_text())
    committed_config["repository"]["expected_commit"] = commit
    active_config["repository"]["expected_commit"] = commit
    committed_split = json.loads(
        subprocess.run(
            ["git", "show", f"{commit}:configs/exp007b_data_split.json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    active_split = json.loads((ROOT / "configs" / "exp007b_data_split.json").read_text())
    result_split = json.loads((input_root / "data_split.json").read_text())
    checks: dict[str, Any] = {
        "git_commit": commit,
        "git_object_type": commit_type,
        "git_commit_file_matches_manifest": (
            (input_root / "git_commit.txt").read_text().strip() == commit
        ),
        "committed_config_matches_result": committed_config == result_config,
        "active_config_matches_result": active_config == result_config,
        "config_sha256": json_sha256(result_config),
        "config_sha_matches_manifest": json_sha256(result_config)
        == manifest["config_sha256"],
        "committed_split_matches_result": committed_split == result_split,
        "active_split_matches_result": active_split == result_split,
        "split_sha256": json_sha256(result_split),
        "split_sha_matches_manifest": json_sha256(result_split)
        == manifest["split_sha256"],
        "feature_sha256": sha256_file(ROOT / active_config["data"]["feature_cache"]),
        "metadata_sha256": sha256_file(ROOT / active_config["data"]["metadata_file"]),
        "scenario_sha256": sha256_file(ROOT / active_config["data"]["scenario_file"]),
        "fresh_test_seed": int(result_split["sealed_test_simulator_seed"]),
        "opened_exp007a_test_seed_excluded": int(
            result_split["exp007a_opened_test_seed_excluded"]
        ),
    }
    checks.update(
        {
            "feature_sha_matches_manifest": checks["feature_sha256"]
            == manifest["feature_sha256"],
            "metadata_sha_matches_config": checks["metadata_sha256"]
            == result_config["data"]["expected_metadata_sha256"],
            "scenario_sha_matches_manifest": checks["scenario_sha256"]
            == manifest["scenario_sha256"],
            "fresh_test_seed_differs_from_opened_exp007a_seed": checks["fresh_test_seed"]
            != checks["opened_exp007a_test_seed_excluded"],
        }
    )
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if commit_type != "commit" or not all(boolean_checks):
        raise ValueError(f"EXP-007B identity validation failed: {checks}")
    return checks


def verify_prediction_population(
    credibility: pd.DataFrame,
    controls: pd.DataFrame,
    split: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_runs = set(split["test_runs"])
    expected_seeds = set(int(seed) for seed in config["training"]["seeds"])
    expected_methods = {
        "all_off",
        "all_on",
        "anti_oracle",
        "data_only",
        "inverse_residual",
        "oracle",
        "priorcred",
        "random_credibility",
        "validation_selected_scalar",
    }
    checks = {
        "credibility_test_runs_exact": set(credibility["run_id"].unique())
        == expected_runs,
        "control_test_runs_exact": set(controls["run_id"].unique()) == expected_runs,
        "credibility_partition_test_only": set(credibility["partition"].unique())
        == {"test"},
        "control_partition_test_only": set(controls["partition"].unique()) == {"test"},
        "credibility_seeds_exact": set(credibility["parent_seed"].astype(int).unique())
        == expected_seeds,
        "control_seeds_exact": set(controls["parent_seed"].astype(int).unique())
        == expected_seeds,
        "control_methods_exact": set(controls["method"].unique()) == expected_methods,
        "control_duplicate_rows": int(
            controls.duplicated([*CONTROL_KEYS, "method"]).sum()
        ),
        "credibility_duplicate_rows": int(
            credibility.duplicated([*CONTROL_KEYS, "candidate_spec"]).sum()
        ),
        "control_rows": int(len(controls)),
        "credibility_rows": int(len(credibility)),
        "test_samples_per_seed": int(
            controls[controls["method"] == "data_only"]
            .groupby("parent_seed")
            .size()
            .iloc[0]
        ),
    }
    candidate_counts = credibility.groupby(CONTROL_KEYS).size()
    method_counts = controls.groupby(CONTROL_KEYS).size()
    checks["twelve_candidates_per_sample"] = bool(
        (candidate_counts == int(config["physics_intervention"]["candidate_count"])).all()
    )
    checks["nine_controls_per_sample"] = bool((method_counts == len(expected_methods)).all())
    target_consistency = controls.groupby(CONTROL_KEYS)["target_rul"].nunique(dropna=False)
    checks["target_consistent_across_methods"] = bool((target_consistency == 1).all())
    per_seed_population = []
    baseline = controls[controls["method"] == "data_only"]
    for seed, group in baseline.groupby("parent_seed"):
        per_seed_population.append(
            {
                "parent_seed": int(seed),
                "samples": int(len(group)),
                "runs": int(group["run_id"].nunique()),
            }
        )
    checks["per_seed_population"] = per_seed_population
    failed = [
        name
        for name, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    if checks["control_duplicate_rows"] or checks["credibility_duplicate_rows"] or failed:
        raise ValueError(f"EXP-007B prediction population failed validation: {checks}")
    return checks


def verify_controller(
    credibility: pd.DataFrame,
    controls: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    ordered = credibility.sort_values(
        [*CONTROL_KEYS, "credibility", "candidate_spec"],
        ascending=[True, True, True, False, True],
    )
    best = ordered.groupby(CONTROL_KEYS, as_index=False, sort=False).head(1).copy()
    selected = best["credibility"].to_numpy(float) >= best["credibility_threshold"].to_numpy(float)
    blend = float(config["credibility"]["maximum_physics_blend"])
    blended = np.clip(
        best["data_rul"].to_numpy(float)
        + blend * (best["physics_rul"].to_numpy(float) - best["data_rul"].to_numpy(float)),
        0.0,
        1.0,
    )
    best["expected_prediction"] = np.where(selected, blended, best["data_rul"])
    best["expected_selected"] = selected.astype(int)
    prior = controls[controls["method"] == "priorcred"].copy()
    merged = prior.merge(
        best[
            [
                *CONTROL_KEYS,
                "candidate_spec",
                "candidate_family",
                "time_scale_factor",
                "expected_prediction",
                "expected_selected",
            ]
        ],
        on=CONTROL_KEYS,
        how="inner",
        validate="one_to_one",
    )
    selected_rows = merged["expected_selected"] == 1
    prediction_difference = float(
        np.max(np.abs(merged["predicted_rul"] - merged["expected_prediction"]))
    )
    absolute_error_difference = float(
        np.max(
            np.abs(
                merged["absolute_error"]
                - np.abs(merged["target_rul"] - merged["predicted_rul"])
            )
        )
    )
    selected_spec_matches = bool(
        (
            merged.loc[selected_rows, "priorcred_selected_candidate_spec"].fillna("")
            == merged.loc[selected_rows, "candidate_spec"].fillna("")
        ).all()
    )
    selected_family_matches = bool(
        (
            merged.loc[selected_rows, "priorcred_selected_candidate_family"].fillna("")
            == merged.loc[selected_rows, "candidate_family"].fillna("")
        ).all()
    )
    selected_scale_difference = float(
        np.max(
            np.abs(
                merged.loc[selected_rows, "priorcred_selected_time_scale_factor"]
                - merged.loc[selected_rows, "time_scale_factor"]
            )
        )
    )
    baseline = controls[controls["method"] == "data_only"].set_index(CONTROL_KEYS)
    all_off = controls[controls["method"] == "all_off"].set_index(CONTROL_KEYS)
    scalar = controls[controls["method"] == "validation_selected_scalar"].set_index(
        CONTROL_KEYS
    )
    checks = {
        "status": "passed",
        "decision_rule": "single_highest_causal_probability_then_fixed_blend_or_abstain",
        "maximum_physics_blend": blend,
        "rows_verified": int(len(merged)),
        "selected_rows": int(selected_rows.sum()),
        "pooled_intervention_coverage": float(selected_rows.mean()),
        "maximum_prediction_absolute_difference": prediction_difference,
        "maximum_absolute_error_difference": absolute_error_difference,
        "selected_count_matches": bool(
            (
                merged["priorcred_selected_candidates"].astype(int)
                == merged["expected_selected"].astype(int)
            ).all()
        ),
        "fallback_matches": bool(
            (
                merged["priorcred_fallback"].astype(bool)
                == (merged["expected_selected"] == 0)
            ).all()
        ),
        "selected_spec_matches": selected_spec_matches,
        "selected_family_matches": selected_family_matches,
        "maximum_selected_scale_absolute_difference": selected_scale_difference,
        "all_off_equals_data_only": bool(
            np.allclose(all_off["predicted_rul"], baseline["predicted_rul"], atol=0.0)
        ),
        "validation_scalar_equals_data_only": bool(
            np.allclose(scalar["predicted_rul"], baseline["predicted_rul"], atol=0.0)
        ),
    }
    if prediction_difference > 1e-14 or absolute_error_difference > 1e-14 or not all(
        value for value in checks.values() if isinstance(value, bool)
    ):
        checks["status"] = "failed"
        raise ValueError(f"Serialized causal controller failed verification: {checks}")
    return checks


def causal_threshold_metrics(
    scored: pd.DataFrame, threshold: float, config: dict[str, Any]
) -> dict[str, float]:
    best = (
        scored.sort_values(
            [*CONTROL_KEYS, "credibility", "candidate_spec"],
            ascending=[True, True, True, False, True],
        )
        .groupby(CONTROL_KEYS, as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected = best["credibility"].to_numpy(float) >= float(threshold)
    blend = float(config["credibility"]["maximum_physics_blend"])
    blended = np.clip(
        best["data_rul"].to_numpy(float)
        + blend * (best["physics_rul"].to_numpy(float) - best["data_rul"].to_numpy(float)),
        0.0,
        1.0,
    )
    best["controlled_rul"] = np.where(selected, blended, best["data_rul"])
    data_rmse: list[float] = []
    controller_rmse: list[float] = []
    for _, run in best.groupby(["parent_seed", "run_id"], sort=True):
        target = run["target_rul"].to_numpy(float)
        data_rmse.append(math.sqrt(mean_squared_error(target, run["data_rul"])))
        controller_rmse.append(
            math.sqrt(mean_squared_error(target, run["controlled_rul"]))
        )
    regret = np.asarray(controller_rmse) - np.asarray(data_rmse)
    harm_margin = float(config["credibility"]["risk_constraints"]["harm_regret_margin"])
    return {
        "threshold": float(threshold),
        "macro_run_rmse": float(np.mean(controller_rmse)),
        "data_only_macro_run_rmse": float(np.mean(data_rmse)),
        "mean_control_regret": float(regret.mean()),
        "mean_positive_control_regret": float(np.maximum(regret, 0.0).mean()),
        "harmful_run_fraction": float(np.mean(regret > harm_margin)),
        "maximum_control_regret": float(regret.max()),
        "intervention_coverage": float(selected.mean()),
        "trajectory_count": int(len(regret)),
    }


def choose_threshold(
    validation: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    scores = validation.groupby(CONTROL_KEYS)["credibility"].max().to_numpy(float)
    thresholds = np.unique(
        np.concatenate(
            (
                [np.nextafter(float(scores.max()), np.inf)],
                np.quantile(scores, np.linspace(0.0, 1.0, 101)),
            )
        )
    )
    constraints = config["credibility"]["risk_constraints"]
    feasible = []
    for threshold in thresholds:
        metric = causal_threshold_metrics(validation, float(threshold), config)
        if (
            metric["mean_positive_control_regret"]
            <= float(constraints["maximum_mean_positive_regret"])
            and metric["harmful_run_fraction"]
            <= float(constraints["maximum_harmful_run_fraction"])
            and metric["intervention_coverage"]
            >= float(constraints["minimum_intervention_coverage"])
            and metric["intervention_coverage"]
            <= float(constraints["maximum_intervention_coverage"])
        ):
            feasible.append(metric)
    if not feasible:
        raise ValueError("A saved passing seed has no independently feasible threshold.")
    selected = min(
        feasible,
        key=lambda item: (
            item["macro_run_rmse"],
            item["mean_positive_control_regret"],
            -item["intervention_coverage"],
            item["threshold"],
        ),
    )
    return selected, len(feasible)


def verify_development(
    input_root: Path, manifest: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    reported = json.loads(
        (input_root / "development_target_qualification.json").read_text()
    )
    rows = []
    for seed in manifest["seeds"]:
        seed_root = input_root / "seeds" / f"seed_{int(seed):05d}"
        train = pd.read_csv(seed_root / "train_counterfactual_evidence.csv")
        validation = pd.read_csv(seed_root / "validation_counterfactual_evidence.csv")
        unit_keys = ["parent_seed", "run_id", "candidate_spec"]
        constant_columns = [
            "safe_to_apply",
            "harmful_intervention",
            "physics_regret",
            "law_correctness",
            "candidate_family",
            "time_scale_factor",
            "true_family",
        ]
        development_parts = []
        for frame in (train, validation):
            uniqueness = frame.groupby(unit_keys)[constant_columns].nunique(dropna=False)
            if (uniqueness > 1).any().any():
                raise ValueError(
                    f"Development trajectory labels are inconsistent for seed {seed}."
                )
            development_parts.append(
                frame.groupby(unit_keys, as_index=False).agg(
                    {column: "first" for column in constant_columns}
                )
            )
        dev_units = pd.concat(development_parts, ignore_index=True)
        saved = reported["per_seed"][str(seed)]
        selected, feasible_count = choose_threshold(validation, config)
        saved_threshold = saved["selector_threshold_qualification"]
        differences = [
            abs(float(selected[key]) - float(saved_threshold[key]))
            for key in (
                "threshold",
                "macro_run_rmse",
                "data_only_macro_run_rmse",
                "mean_control_regret",
                "mean_positive_control_regret",
                "harmful_run_fraction",
                "maximum_control_regret",
                "intervention_coverage",
            )
        ]
        rows.append(
            {
                "parent_seed": int(seed),
                "development_units": int(len(dev_units)),
                "safe_fraction": float(dev_units["safe_to_apply"].mean()),
                "reported_safe_fraction": float(saved["safe_fraction"]),
                "selected_threshold": float(selected["threshold"]),
                "reported_threshold": float(saved_threshold["threshold"]),
                "feasible_thresholds": int(feasible_count),
                "reported_feasible_thresholds": int(saved_threshold["feasible_thresholds"]),
                "maximum_threshold_metric_absolute_difference": max(differences),
                "test_access_authorized": bool(saved["test_access_authorized"]),
                "passed": bool(saved["passed"] and saved_threshold["passed"]),
            }
        )
    result = pd.DataFrame(rows)
    if (
        reported["status"] != "passed"
        or reported["development_gate_failed_seeds"]
        or not (result["development_units"] == 960).all()
        or not np.allclose(
            result["safe_fraction"], result["reported_safe_fraction"], atol=1e-15
        )
        or not (result["feasible_thresholds"] == result["reported_feasible_thresholds"]).all()
        or result["maximum_threshold_metric_absolute_difference"].max() > 1e-14
        or not result["test_access_authorized"].all()
        or not result["passed"].all()
    ):
        raise ValueError(f"Development qualification failed verification:\n{result}")
    return result


def regret_bootstrap(prior_runs: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in prior_runs["parent_seed"].unique())
    by_seed = {
        seed: prior_runs[prior_runs["parent_seed"] == seed]["physics_regret"].to_numpy(
            float
        )
        for seed in seeds
    }
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]) + 1)
    signed: list[float] = []
    positive: list[float] = []
    for _ in range(int(config["evaluation"]["bootstrap_replicates"])):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        parts = []
        for seed in sampled_seeds:
            values = by_seed[int(seed)]
            sampled_indices = rng.choice(len(values), size=len(values), replace=True)
            parts.append(values[sampled_indices])
        regret = np.concatenate(parts)
        signed.append(float(regret.mean()))
        positive.append(float(np.maximum(regret, 0.0).mean()))
    return {
        "replicates": len(signed),
        "aggregation": "trajectory_within_resampled_seed_then_pooled_mean",
        "mean_regret_ci_lower_95": float(np.quantile(signed, 0.025)),
        "mean_regret_ci_upper_95": float(np.quantile(signed, 0.975)),
        "mean_positive_regret_ci_lower_95": float(np.quantile(positive, 0.025)),
        "mean_positive_regret_ci_upper_95": float(np.quantile(positive, 0.975)),
    }


def risk_slices(
    controls: pd.DataFrame, physics_regret: pd.DataFrame, harm_margin: float
) -> pd.DataFrame:
    prior = controls[controls["method"] == "priorcred"].copy()
    metadata = prior[
        ["parent_seed", "run_id", "true_family", "condition_id"]
    ].drop_duplicates()
    full = physics_regret[physics_regret["method"] == "priorcred"].merge(
        metadata, on=["parent_seed", "run_id"], validate="one_to_one"
    )
    coverage = prior.groupby(["parent_seed", "run_id"], as_index=False).agg(
        intervention_coverage=(
            "priorcred_selected_candidates", lambda values: float(np.mean(values > 0))
        )
    )
    full = full.merge(coverage, on=["parent_seed", "run_id"], validate="one_to_one")
    lifecycle = controls[controls["method"].isin(["data_only", "priorcred"])].copy()
    lifecycle["lifecycle_region"] = np.select(
        [lifecycle["target_rul"] > 2 / 3, lifecycle["target_rul"] > 1 / 3],
        ["early", "middle"],
        default="late",
    )
    lifecycle_rows = []
    keys = ["parent_seed", "run_id", "true_family", "condition_id", "lifecycle_region"]
    for identity, group in lifecycle.groupby(keys, sort=True):
        rmse = {
            method: math.sqrt(
                mean_squared_error(part["target_rul"], part["predicted_rul"])
            )
            for method, part in group.groupby("method")
        }
        prior_part = group[group["method"] == "priorcred"]
        regret = rmse["priorcred"] - rmse["data_only"]
        lifecycle_rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "physics_regret": regret,
                "positive_physics_regret": max(regret, 0.0),
                "intervention_coverage": float(
                    (prior_part["priorcred_selected_candidates"] > 0).mean()
                ),
            }
        )
    lifecycle_frame = pd.DataFrame(lifecycle_rows)
    rows = []
    for dimension, source, column in (
        ("parent_seed", full, "parent_seed"),
        ("true_family", full, "true_family"),
        ("condition_id", full, "condition_id"),
        ("lifecycle_region", lifecycle_frame, "lifecycle_region"),
    ):
        for value, group in source.groupby(column, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "trajectory_seed_units": int(len(group)),
                    "mean_regret": float(group["physics_regret"].mean()),
                    "mean_positive_regret": float(
                        group["positive_physics_regret"].mean()
                    ),
                    "harmful_fraction": float(
                        (group["physics_regret"] > harm_margin).mean()
                    ),
                    "maximum_regret": float(group["physics_regret"].max()),
                    "mean_intervention_coverage": float(
                        group["intervention_coverage"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def selected_frequency(controls: pd.DataFrame) -> pd.DataFrame:
    prior = controls[controls["method"] == "priorcred"].copy()
    prior["lifecycle_region"] = np.select(
        [prior["target_rul"] > 2 / 3, prior["target_rul"] > 1 / 3],
        ["early", "middle"],
        default="late",
    )
    totals = prior.groupby("parent_seed").size().to_dict()
    selected = prior[prior["priorcred_selected_candidates"] > 0]
    result = selected.groupby(
        [
            "parent_seed",
            "priorcred_selected_candidate_family",
            "priorcred_selected_time_scale_factor",
            "lifecycle_region",
        ],
        as_index=False,
    ).agg(selected_samples=("sample_index", "size"))
    result["fraction_of_seed_samples"] = result.apply(
        lambda row: float(row["selected_samples"] / totals[int(row["parent_seed"])]),
        axis=1,
    )
    return result


def runtime_summary(input_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = [
        json.loads(path.read_text())
        for path in sorted((input_root / "seeds").glob("seed_*/job_result.json"))
    ]
    lines = (input_root / "training.log").read_text().splitlines()
    start = datetime.fromisoformat(next(line for line in lines if "start/resume" in line).split()[0])
    completed = [line for line in lines if "status=completed test_evaluated=True" in line]
    last_seed = datetime.fromisoformat(completed[-1].split()[0])
    elapsed = float(manifest["elapsed_seconds"])
    seed_sum = float(sum(job["seconds"] for job in jobs))
    return {
        "manifest_elapsed_seconds": elapsed,
        "manifest_elapsed_minutes": elapsed / 60.0,
        "sum_per_seed_training_seconds": seed_sum,
        "sum_per_seed_training_minutes": seed_sum / 60.0,
        "initial_start_to_last_seed_seconds": float((last_seed - start).total_seconds()),
        "artifact_finalization_seconds_after_last_seed": float(
            datetime.fromisoformat(manifest["finished_at_utc"]).timestamp()
            - last_seed.timestamp()
        ),
        "per_seed_training_seconds": {
            str(job["parent_seed"]): float(job["seconds"]) for job in jobs
        },
        "validation_selected_scalars": {
            str(job["parent_seed"]): float(job["validation_selected_scalar"])
            for job in jobs
        },
        "backbone_parameter_counts": sorted(
            {int(job["backbone_parameter_count"]) for job in jobs}
        ),
        "backbone_best_epochs": {
            str(job["parent_seed"]): int(job["backbone_best_epoch"]) for job in jobs
        },
        "oom_retries": int(manifest["effective_runtime"]["oom_retries"]),
    }


def numeric_json_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    differences = []
    for key, value in a.items():
        if isinstance(value, bool) and key in b:
            if value != b[key]:
                raise ValueError(f"Boolean gate mismatch for {key}: {value} != {b[key]}")
        elif isinstance(value, (int, float)) and key in b and isinstance(b[key], (int, float)):
            differences.append(abs(float(value) - float(b[key])))
        elif isinstance(value, dict) and isinstance(b.get(key), dict):
            differences.append(numeric_json_difference(value, b[key]))
    return max(differences, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify EXP-007B artifacts.")
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
    if failures or manifest["failed_models"] or manifest["completed_seeds"] != manifest["requested_seeds"]:
        raise ValueError("EXP-007B contains a failed or missing seed/model.")

    credibility = pd.read_csv(input_root / "credibility_predictions.csv")
    controls = pd.read_csv(input_root / "control_predictions.csv")
    candidate = pd.read_csv(input_root / "candidate_regret.csv")
    history = pd.read_csv(input_root / "training_history.csv")
    split = json.loads((input_root / "data_split.json").read_text())
    population = verify_prediction_population(credibility, controls, split, config)
    controller = verify_controller(credibility, controls, config)
    development = verify_development(input_root, manifest, config)

    units, metrics, statistical = verified_causal_credibility(credibility, config)
    comparison, physics_regret, model_aggregate = verified_controls(controls)
    prior_runs = physics_regret[physics_regret["method"] == "priorcred"].copy()
    regret_ci = regret_bootstrap(prior_runs, config)
    statistical["causal_controller_regret_bootstrap"] = regret_ci
    diagnostics = candidate_diagnostics(candidate)
    family_metrics = credibility_by_family(units)
    convergence = convergence_summary(history)
    runtime = runtime_summary(input_root, manifest)
    slices = risk_slices(
        controls, physics_regret, float(config["success_criteria"]["harm_regret_margin"])
    )
    frequency = selected_frequency(controls)

    reported_metrics = pd.read_csv(input_root / "credibility_metrics.csv")
    reported_comparison = pd.read_csv(input_root / "model_comparison.csv")
    reported_regret = pd.read_csv(input_root / "physics_regret.csv")
    reported_slices = pd.read_csv(input_root / "risk_slice_metrics.csv", dtype={"value": str})
    slices_for_compare = slices.copy()
    slices_for_compare["value"] = slices_for_compare["value"].astype(str)
    reported_frequency = pd.read_csv(input_root / "selected_candidate_frequency.csv")
    frequency_for_compare = frequency.copy()
    reported_frequency_for_compare = reported_frequency.copy()
    scale_column = "priorcred_selected_time_scale_factor"
    frequency_for_compare[scale_column] = frequency_for_compare[scale_column].round(12)
    reported_frequency_for_compare[scale_column] = reported_frequency_for_compare[
        scale_column
    ].round(12)
    artifacts["maximum_metric_absolute_difference"] = compare_numeric_frames(
        metrics, reported_metrics, ["parent_seed"]
    )
    artifacts["maximum_model_comparison_absolute_difference"] = compare_numeric_frames(
        comparison, reported_comparison, ["parent_seed", "method"]
    )
    artifacts["maximum_physics_regret_absolute_difference"] = compare_numeric_frames(
        physics_regret, reported_regret, ["parent_seed", "run_id", "method"]
    )
    artifacts["maximum_risk_slice_absolute_difference"] = compare_numeric_frames(
        slices_for_compare, reported_slices, ["dimension", "value"]
    )
    artifacts["maximum_selected_frequency_absolute_difference"] = compare_numeric_frames(
        frequency_for_compare,
        reported_frequency_for_compare,
        [
            "parent_seed",
            "priorcred_selected_candidate_family",
            "priorcred_selected_time_scale_factor",
            "lifecycle_region",
        ],
    )

    criteria = config["success_criteria"]
    prior_macro = float(
        comparison.loc[comparison["method"] == "priorcred", "macro_run_rmse"].mean()
    )
    data_macro = float(
        comparison.loc[comparison["method"] == "data_only", "macro_run_rmse"].mean()
    )
    relative_improvement = 100.0 * (data_macro - prior_macro) / data_macro
    mean_regret = float(prior_runs["physics_regret"].mean())
    mean_positive = float(prior_runs["positive_physics_regret"].mean())
    harm_margin = float(criteria["harm_regret_margin"])
    harmful_fraction = float((prior_runs["physics_regret"] > harm_margin).mean())
    maximum_regret = float(prior_runs["physics_regret"].max())
    coverage = float(
        (
            controls.loc[
                controls["method"] == "priorcred", "priorcred_selected_candidates"
            ]
            > 0
        ).mean()
    )
    gate_checks = {
        "passes_minimum_relative_improvement": relative_improvement
        >= float(criteria["minimum_relative_macro_rmse_improvement_percent"]),
        "passes_maximum_mean_positive_regret": mean_positive
        <= float(criteria["maximum_mean_positive_regret"]),
        "passes_maximum_harmful_run_fraction": harmful_fraction
        <= float(criteria["maximum_harmful_run_fraction"]),
        "passes_maximum_observed_run_regret": maximum_regret
        <= float(criteria["maximum_observed_run_regret"]),
        "passes_pooled_intervention_coverage": coverage
        >= float(criteria["minimum_pooled_intervention_coverage"])
        and coverage <= float(criteria["maximum_pooled_intervention_coverage"]),
    }
    gate_passed = all(gate_checks.values())
    gate = {
        "experiment_id": "EXP-007B",
        "status": "passed" if gate_passed else "failed",
        "decision": (
            "permit_higher_fidelity_or_real_bearing_confirmation"
            if gate_passed
            else "stop_and_preserve_exp007b_negative_result"
        ),
        "primary_unit": "complete_test_trajectory_within_neural_seed",
        "trajectory_seed_units": int(len(prior_runs)),
        "data_only_mean_macro_run_rmse": data_macro,
        "causal_controller_mean_macro_run_rmse": prior_macro,
        "relative_macro_rmse_improvement_percent": relative_improvement,
        "mean_control_regret": mean_regret,
        "mean_positive_control_regret": mean_positive,
        "harm_regret_margin": harm_margin,
        "harmful_run_fraction": harmful_fraction,
        "maximum_observed_run_regret": maximum_regret,
        "pooled_intervention_coverage": coverage,
        **gate_checks,
        "safe_intervention_auroc_secondary": statistical["mean_seed_auroc"],
        "auroc_bootstrap_secondary": {
            key: statistical[key]
            for key in (
                "replicates_requested",
                "replicates_valid",
                "mean",
                "median",
                "auroc_ci_lower_95",
                "auroc_ci_upper_95",
                "aggregation",
            )
        },
        "regret_bootstrap_secondary": regret_ci,
    }
    reported_gate = json.loads((input_root / "gate_decision.json").read_text())
    gate_difference = numeric_json_difference(gate, reported_gate)
    if gate["status"] != reported_gate["status"] or gate["decision"] != reported_gate["decision"]:
        raise ValueError("Independent gate status disagrees with the serialized decision.")
    artifacts["maximum_gate_numeric_absolute_difference"] = gate_difference

    reported_statistical = json.loads((input_root / "statistical_summary.json").read_text())
    artifacts["maximum_statistical_numeric_absolute_difference"] = numeric_json_difference(
        statistical, reported_statistical
    )
    artifacts["prediction_population"] = population
    artifacts["controller_verification"] = controller
    artifacts["serialization_verification"] = json.loads(
        (input_root / "serialization_verification.json").read_text()
    )

    success = pd.DataFrame(
        [
            {
                "criterion": "relative_macro_run_rmse_improvement_percent",
                "threshold": f">={criteria['minimum_relative_macro_rmse_improvement_percent']}",
                "observed": relative_improvement,
                "passed": gate_checks["passes_minimum_relative_improvement"],
            },
            {
                "criterion": "mean_positive_control_regret",
                "threshold": f"<={criteria['maximum_mean_positive_regret']}",
                "observed": mean_positive,
                "passed": gate_checks["passes_maximum_mean_positive_regret"],
            },
            {
                "criterion": "harmful_run_fraction",
                "threshold": f"<={criteria['maximum_harmful_run_fraction']}",
                "observed": harmful_fraction,
                "passed": gate_checks["passes_maximum_harmful_run_fraction"],
            },
            {
                "criterion": "maximum_observed_run_regret",
                "threshold": f"<={criteria['maximum_observed_run_regret']}",
                "observed": maximum_regret,
                "passed": gate_checks["passes_maximum_observed_run_regret"],
            },
            {
                "criterion": "pooled_intervention_coverage",
                "threshold": (
                    f"{criteria['minimum_pooled_intervention_coverage']}.."
                    f"{criteria['maximum_pooled_intervention_coverage']}"
                ),
                "observed": coverage,
                "passed": gate_checks["passes_pooled_intervention_coverage"],
            },
        ]
    )

    baseline = model_aggregate.set_index("method").loc["data_only", "mean_macro_run_rmse"]
    all_on = model_aggregate.set_index("method").loc["all_on", "mean_macro_run_rmse"]
    model_aggregate["improvement_vs_data_pct"] = (
        100.0 * (baseline - model_aggregate["mean_macro_run_rmse"]) / baseline
    )
    model_aggregate["improvement_vs_all_on_pct"] = (
        100.0 * (all_on - model_aggregate["mean_macro_run_rmse"]) / all_on
    )
    method_risk = physics_regret.groupby("method", as_index=False).agg(
        mean_regret=("physics_regret", "mean"),
        mean_positive_regret=("positive_physics_regret", "mean"),
        maximum_regret=("physics_regret", "max"),
        harmful_run_fraction=("physics_regret", lambda x: float(np.mean(x > harm_margin))),
        improved_trajectory_seed_units=("physics_regret", lambda x: int(np.sum(x < 0))),
    )
    model_aggregate = model_aggregate.merge(method_risk, on="method", validate="one_to_one")
    seed_pivot = comparison.pivot(index="parent_seed", columns="method", values="macro_run_rmse")
    seed_pivot["priorcred_improvement_percent"] = (
        100.0 * (seed_pivot["data_only"] - seed_pivot["priorcred"]) / seed_pivot["data_only"]
    )
    seed_pivot = seed_pivot.reset_index()
    selected = controls[
        (controls["method"] == "priorcred")
        & (controls["priorcred_selected_candidates"] > 0)
    ]
    selected_scale_06_fraction = float(
        np.isclose(
            selected["priorcred_selected_time_scale_factor"].to_numpy(float),
            0.6,
            atol=1e-12,
        ).mean()
    )
    selected_family_match_fraction = float(
        (
            selected["priorcred_selected_candidate_family"] == selected["true_family"]
        ).mean()
    )

    write_json(output_root / "artifact_validation.json", artifacts)
    write_json(output_root / "verified_gate_decision.json", gate)
    write_json(output_root / "verified_statistical_summary.json", statistical)
    write_json(output_root / "runtime_summary.json", runtime)
    write_json(output_root / "controller_verification.json", controller)
    metrics.to_csv(output_root / "verified_metrics.csv", index=False)
    comparison.to_csv(output_root / "verified_model_comparison.csv", index=False)
    physics_regret.to_csv(output_root / "verified_physics_regret.csv", index=False)
    model_aggregate.to_csv(output_root / "model_aggregate.csv", index=False)
    seed_pivot.to_csv(output_root / "seed_level_comparison.csv", index=False)
    diagnostics.to_csv(output_root / "candidate_diagnostics.csv", index=False)
    family_metrics.to_csv(output_root / "credibility_by_true_family.csv", index=False)
    convergence.to_csv(output_root / "convergence_summary.csv", index=False)
    development.to_csv(output_root / "development_verification.csv", index=False)
    slices.to_csv(output_root / "verified_risk_slice_metrics.csv", index=False)
    frequency.to_csv(output_root / "verified_selected_candidate_frequency.csv", index=False)
    success.to_csv(output_root / "success_criteria.csv", index=False)

    aggregate_index = model_aggregate.set_index("method")
    lifecycle = slices[slices["dimension"] == "lifecycle_region"].set_index("value")
    family_risk = slices[slices["dimension"] == "true_family"].set_index("value")
    prior_wins = int(
        method_risk.set_index("method").loc["priorcred", "improved_trajectory_seed_units"]
    )
    all_on_risk = method_risk.set_index("method").loc["all_on"]
    analysis = f"""# EXP-007B verified analysis

## Validity and execution

EXP-007B is a valid completed execution of commit `{manifest['git_commit']}` on the required
Tesla T4. All {artifacts['artifact_inventory_records']} inventoried artifacts and all
{artifacts['bundle_entries']} lightweight-bundle entries passed size, SHA-256, and safe-path
checks. The frozen configuration, fresh simulator seed 920072, 64/16/16 split, scenario, feature
cache, metadata, and five neural seeds all match the committed inputs. Every development gate
was reproduced before test access, every seed completed, and no failed model, OOM retry, or
non-finite training value was found.

The complete run took `{runtime['manifest_elapsed_minutes']:.1f}` minutes: approximately
`{runtime['sum_per_seed_training_minutes']:.1f}` minutes in the five seed jobs and
`{runtime['artifact_finalization_seconds_after_last_seed']/60:.1f}` minutes after the last seed
for aggregation, hashing, plots, and export. The short runtime is credible because this is an
8,268-row synthetic feature-sequence experiment with a 22,625-parameter backbone, not raw-signal
training.

## Confirmatory endpoint

The causal controller reduced mean macro-run RMSE from `{data_macro:.6f}` to `{prior_macro:.6f}`
normalized RUL, a `{relative_improvement:.6f}%` improvement. This misses the preregistered
`1.0%` minimum, so the combined confirmation gate **fails**. The controller passed every safety
and exposure constraint: mean positive regret `{mean_positive:.6f}` (limit `0.005`), harmful-run
fraction `{harmful_fraction:.4f}` (limit `0.10` at regret margin `0.01`), maximum run regret
`{maximum_regret:.6f}` (limit `0.05`), and pooled coverage `{coverage:.4f}` (allowed
`0.05-0.90`). The verified decision is `stop_and_preserve_exp007b_negative_result`.

The controller improved {prior_wins}/80 complete trajectory-seed units and four of five
seed-level macro averages, but seed 3042 regressed by
`{abs(seed_pivot.set_index('parent_seed').loc[3042, 'priorcred_improvement_percent']):.3f}%`.
The signed-regret bootstrap 95% interval is
`[{regret_ci['mean_regret_ci_lower_95']:.6f}, {regret_ci['mean_regret_ci_upper_95']:.6f}]`, which
crosses zero. Thus the modest average gain is neither large enough for the frozen gate nor
statistically stable in this hierarchy.

Always-on physics achieved `{aggregate_index.loc['all_on','improvement_vs_data_pct']:.3f}%`
average improvement, more than the controller, but its harmful-run fraction was
`{all_on_risk['harmful_run_fraction']:.4f}`, above the `0.10` safety limit. Oracle selection
shows substantial headroom (`{aggregate_index.loc['oracle','improvement_vs_data_pct']:.2f}%`
improvement), confirming that intervention choice matters even though the learned selector did
not capture enough of it.

## Credibility, lifecycle, and selection behavior

Safe-intervention discrimination was weak: mean seed AUROC was
`{statistical['mean_seed_auroc']:.6f} +/- {statistical['std_seed_auroc']:.6f}`, with bootstrap
95% interval `[{statistical['auroc_ci_lower_95']:.6f},
{statistical['auroc_ci_upper_95']:.6f}]`. Mean Brier score `{statistical['mean_seed_brier']:.6f}`
was worse than the constant-prevalence reference `{statistical['mean_prevalence_brier']:.6f}`;
only one seed beat that reference. AUROC was secondary in EXP-007B, but this near-chance ranking
helps explain why development improvement did not generalize.

Intervention coverage varied sharply by neural seed, from
`{controller['selected_rows']/controller['rows_verified']:.4f}` pooled but
`{metrics['all_on_fraction'].min():.4f}` to `{metrics['all_on_fraction'].max():.4f}` per seed.
`{selected_scale_06_fraction:.1%}` of selected samples used the 0.60 time-scale prior, and only
`{selected_family_match_fraction:.1%}` selected the simulator's named true family. Family match
is not the optimization target, but the concentration shows that the controller mainly learned
a conservative short-time-scale correction rather than recovering the generating law.

The lifecycle result is asymmetric. Early-life intervention increased mean segment regret by
`{lifecycle.loc['early','mean_regret']:.6f}`, while middle and late life improved it by
`{-lifecycle.loc['middle','mean_regret']:.6f}` and `{-lifecycle.loc['late','mean_regret']:.6f}`.
Gamma and linear-increasing trajectories had positive mean full-run regret
(`{family_risk.loc['gamma','mean_regret']:.6f}` and
`{family_risk.loc['linear_increasing','mean_regret']:.6f}`), whereas progressive and step-like
trajectories benefited. These are opened-test diagnostics and cannot be used to retune a new
confirmatory method.

## Training behavior and interpretation

All 325 expected fits have finite histories and separate nonnegative data, prior-value,
prior-rate, and monotonic losses. Median recorded training length was
`{convergence['epochs'].median():.0f}` epochs and the maximum was
`{int(convergence['epochs'].max())}`; the five final backbones selected best epochs
{sorted(runtime['backbone_best_epochs'].values())}. This is consistent with functioning early
stopping and checkpoint reuse, not skipped training.

EXP-007B therefore supports a narrower finding than hoped: causal abstention reduced the tail
risk of physics intervention, but the frozen controller did not deliver the preregistered average
RUL improvement on a fresh population. Higher-fidelity or real-bearing confirmation is not yet
authorized by this gate.
"""
    (output_root / "analysis.md").write_text(analysis, encoding="utf-8")

    issues = f"""# EXP-007B issues

1. **The primary improvement gate failed.** Verified improvement is
   `{relative_improvement:.6f}%`, below the frozen `1.0%` requirement.
2. **The average gain is uncertain.** The signed-regret bootstrap interval
   `[{regret_ci['mean_regret_ci_lower_95']:.6f}, {regret_ci['mean_regret_ci_upper_95']:.6f}]`
   crosses zero, and seed 3042 regressed.
3. **Credibility does not generalize strongly.** Mean AUROC is
   `{statistical['mean_seed_auroc']:.6f}` and mean Brier score is worse than prevalence; test
   coverage ranges from `{metrics['all_on_fraction'].min():.2%}` to
   `{metrics['all_on_fraction'].max():.2%}` across seeds.
4. **Risk is lifecycle- and family-dependent.** Early-life, gamma, and linear-increasing slices
   show positive mean regret despite acceptable pooled full-run safety.
5. **The fresh test population is now open.** Threshold, feature, blend, family, or loss changes
   motivated by EXP-007B must be treated as exploratory and confirmed on another untouched
   simulator population.
6. **External validity remains limited.** This experiment uses normalized RUL from controlled
   synthetic feature trajectories; it does not establish performance on raw vibration,
   real-bearing failures, MATLAB simulation, or ANSYS physics.
"""
    (output_root / "issues.md").write_text(issues, encoding="utf-8")

    recommendations = """# EXP-007B recommendations

1. Preserve EXP-007B as a valid negative confirmation. Do not relabel the completed run or
   relax its 1% gate after seeing the test result.
2. Do not proceed directly to the planned higher-fidelity/real-bearing EXP-008. The current
   controller first needs a development-only redesign that improves benefit without losing the
   demonstrated tail-risk control.
3. Use EXP-007B only to formulate hypotheses. Candidate directions are direct regret or
   benefit prediction, partial pooling of thresholds across neural seeds, and lifecycle-aware
   abstention. Select among them using development trajectories only, not these 16 test runs.
4. Strengthen the next protocol with multiple newly generated simulator population seeds.
   Five neural initializations quantify optimization variation but do not quantify variation in
   the test population itself.
5. Predefine both an average-benefit interval and safety bounds. Retain macro trajectory RMSE,
   mean positive regret, harm rate, maximum regret, and coverage so a method cannot pass by
   simply abstaining.
6. Once a revised controller passes a fresh synthetic confirmation, use MATLAB or ANSYS to
   create an independently governed degradation benchmark, then freeze the method before any
   real-bearing evaluation. ANSYS is valuable at that stage, not as a substitute for repairing
   the selector's generalization.
7. A publishable narrative is possible even if the method remains negative: matched physics
   laws can cause negative transfer, causal selective intervention controls tail harm, and
   apparent development gains can fail on a fresh degradation population. That claim requires
   the next study to quantify population-seed uncertainty explicitly.
"""
    (output_root / "recommendations.md").write_text(recommendations, encoding="utf-8")

    previous_aggregate = pd.read_csv(
        ROOT / "results" / "analyzed" / "EXP-007A" / "model_aggregate.csv"
    ).set_index("method")
    previous_regret = pd.read_csv(
        ROOT / "results" / "analyzed" / "EXP-007A" / "verified_physics_regret.csv"
    )
    previous_prior = previous_regret[previous_regret["method"] == "priorcred"]
    previous_metrics = pd.read_csv(
        ROOT / "results" / "analyzed" / "EXP-007A" / "verified_metrics.csv"
    )
    stage_comparison = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-007A",
                "test_population": "opened_seed_920071",
                "controller": "complete_trajectory_selector",
                "mean_seed_auroc": float(previous_metrics["auroc"].mean()),
                "data_only_macro_run_rmse": float(
                    previous_aggregate.loc["data_only", "mean_macro_run_rmse"]
                ),
                "controller_macro_run_rmse": float(
                    previous_aggregate.loc["priorcred", "mean_macro_run_rmse"]
                ),
                "relative_improvement_percent": float(
                    previous_aggregate.loc["priorcred", "improvement_vs_data_pct"]
                ),
                "mean_positive_regret": float(
                    previous_prior["positive_physics_regret"].mean()
                ),
                "maximum_regret": float(previous_prior["physics_regret"].max()),
                "gate": "failed",
                "comparability_note": "Different controller and opened test population; descriptive only",
            },
            {
                "experiment_id": "EXP-007B",
                "test_population": "fresh_seed_920072",
                "controller": "causal_prefix_selector_blend_0.50",
                "mean_seed_auroc": statistical["mean_seed_auroc"],
                "data_only_macro_run_rmse": data_macro,
                "controller_macro_run_rmse": prior_macro,
                "relative_improvement_percent": relative_improvement,
                "mean_positive_regret": mean_positive,
                "maximum_regret": maximum_regret,
                "gate": "failed",
                "comparability_note": "Fresh confirmatory population; primary result",
            },
        ]
    )
    stage_comparison.to_csv(
        ROOT / "results" / "comparisons" / "exp007a_vs_exp007b.csv", index=False
    )
    publication_path = ROOT / "results" / "comparisons" / "publication_stage_status.csv"
    publication = pd.read_csv(publication_path)
    publication = publication[publication["experiment_id"] != "EXP-007B"]
    publication = pd.concat(
        [
            publication,
            pd.DataFrame(
                [
                    {
                        "experiment_id": "EXP-007B",
                        "status": "completed",
                        "primary_endpoint": "Fresh causal controller risk and macro-run RMSE",
                        "gate": "failed",
                        "decision": (
                            "Safety gates passed but improvement was below 1%; preserve negative "
                            "result and do not escalate fidelity"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    publication.to_csv(publication_path, index=False)

    print(
        json.dumps(
            {
                "status": "verified_gate_failed",
                "output": str(output_root),
                "relative_macro_rmse_improvement_percent": relative_improvement,
                "mean_positive_regret": mean_positive,
                "harmful_run_fraction": harmful_fraction,
                "maximum_regret": maximum_regret,
                "coverage": coverage,
                "mean_seed_auroc_secondary": statistical["mean_seed_auroc"],
                "artifact_records": artifacts["artifact_inventory_records"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
