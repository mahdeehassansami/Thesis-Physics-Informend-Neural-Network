from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

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
ANALYZED = ROOT / "results" / "analyzed" / "EXP-007"
COMPARISONS = ROOT / "results" / "comparisons"
SEEDS = [42, 1042, 2042, 3042, 4042]
UNIT_KEYS = [
    "seed",
    "run_id",
    "candidate_spec",
    "true_family",
    "candidate_family",
    "corruption_type",
    "condition_id",
    "validity_label",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        mask = (probability >= edges[index]) & upper
        if mask.any():
            result += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return result


def aggregate_units(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(UNIT_KEYS, as_index=False)
        .agg(
            credibility=("credibility", "mean"),
            raw_credibility=("credibility_raw", "mean"),
            fallback_fraction=("fallback", "mean"),
            target_rul=("target_rul", "mean"),
            covariate_shift_score=("covariate_shift_score", "mean"),
            operation_shift_score=("operation_shift_score", "mean"),
        )
    )


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["validity_label"].to_numpy(dtype=int)
    probability = frame["credibility"].to_numpy(dtype=float)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "ece": expected_calibration_error(target, probability),
        "fallback_fraction": float(frame["fallback_fraction"].mean()),
        "all_on_fraction": float(1.0 - frame["fallback_fraction"].mean()),
    }


def hierarchical_bootstrap(units: pd.DataFrame, replicates: int = 2000) -> dict[str, float | int]:
    runs = sorted(units["run_id"].unique())
    seeds = sorted(units["seed"].unique())
    grouped = {
        (seed, run): group[["validity_label", "credibility"]].to_numpy(dtype=float)
        for (seed, run), group in units.groupby(["seed", "run_id"])
    }
    rng = np.random.default_rng(42007)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_runs = rng.choice(runs, len(runs), replace=True)
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        seed_aurocs = []
        for seed in sampled_seeds:
            sample = np.concatenate([grouped[(seed, run)] for run in sampled_runs])
            seed_aurocs.append(roc_auc_score(sample[:, 0], sample[:, 1]))
        estimates.append(float(np.mean(seed_aurocs)))
    return {
        "replicates": replicates,
        "mean": float(np.mean(estimates)),
        "median": float(np.median(estimates)),
        "ci_lower_95": float(np.quantile(estimates, 0.025)),
        "ci_upper_95": float(np.quantile(estimates, 0.975)),
    }


def validate_artifacts(run: Path) -> dict[str, object]:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "experiment_config.yaml").read_text(encoding="utf-8"))
    split = json.loads((run / "data_split.json").read_text(encoding="utf-8"))
    local_config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    local_config["repository"]["expected_commit"] = manifest["git_commit"]
    local_split = json.loads((ROOT / "configs" / "publication_data_split.json").read_text(encoding="utf-8"))[
        "controlled_synthetic"
    ]
    records = list(csv.DictReader((run / "artifact_inventory.csv").open(encoding="utf-8", newline="")))
    inventory_failures = []
    for record in records:
        path = run / record["relative_path"]
        if not path.is_file():
            inventory_failures.append(f"missing:{record['relative_path']}")
        elif path.stat().st_size != int(record["bytes"]):
            inventory_failures.append(f"size:{record['relative_path']}")
        elif sha256_file(path) != record["sha256"]:
            inventory_failures.append(f"sha256:{record['relative_path']}")
    with zipfile.ZipFile(run / "codex_results_bundle.zip") as archive:
        bundle_names = archive.namelist()
        unsafe = [name for name in bundle_names if Path(name).is_absolute() or ".." in Path(name).parts]
        binary = [name for name in bundle_names if name.endswith((".pt", ".pth", ".pkl", ".joblib"))]
    repository_notebook = json.loads((ROOT / "notebooks" / "train_models_colab.ipynb").read_text(encoding="utf-8"))
    executed_notebook = json.loads((run / "executed_notebook.ipynb").read_text(encoding="utf-8"))
    repository_source = [(cell["cell_type"], "".join(cell.get("source", []))) for cell in repository_notebook["cells"]]
    executed_source = [(cell["cell_type"], "".join(cell.get("source", []))) for cell in executed_notebook["cells"]]
    local_cache = ROOT / "data" / "processed_features" / "publication" / "exp006" / "controlled_synthetic_features.csv"
    result = {
        "experiment_id_valid": manifest["experiment_id"] == "EXP-007",
        "status_completed": manifest["status"] == "completed",
        "commit_identity_valid": (run / "git_commit.txt").read_text(encoding="utf-8").strip() == manifest["git_commit"],
        "config_hash_valid": json_sha256(config) == manifest["config_sha256"],
        "config_matches_pinned_source": config == local_config,
        "split_hash_valid": json_sha256(split) == manifest["split_sha256"],
        "split_matches_source": split == local_split,
        "dataset_hash_valid": sha256_file(local_cache) == manifest["dataset_fingerprint"],
        "requested_completed_seeds_valid": manifest["requested_seeds"] == manifest["completed_seeds"] == SEEDS,
        "failure_report_empty": json.loads((run / "failure_report.json").read_text(encoding="utf-8")) == [],
        "environment_matches_manifest": json.loads((run / "environment.txt").read_text(encoding="utf-8")) == manifest["environment"],
        "t4_cuda_valid": manifest["environment"]["cuda_available"] and "T4" in manifest["environment"]["gpu_name"],
        "notebook_source_matches": repository_source == executed_source,
        "notebook_hash_valid": sha256_file(run / "executed_notebook.ipynb") == manifest["executed_notebook_sha256"],
        "inventory_count_valid": len(records) == manifest["artifact_count_excluding_manifest_inventory_bundle"],
        "inventory_hash_valid": sha256_file(run / "artifact_inventory.csv") == manifest["artifact_inventory_sha256"],
        "inventory_failures": inventory_failures,
        "bundle_safe": not unsafe,
        "bundle_contains_no_binaries": not binary,
        "bundle_file_count": len(bundle_names),
    }
    result["all_identity_and_integrity_checks_passed"] = all(
        value is True
        for key, value in result.items()
        if key.endswith(("_valid", "_matches", "_safe")) or key in {"status_completed", "failure_report_empty", "bundle_contains_no_binaries"}
    ) and not inventory_failures
    return result


def verify_rul_metrics(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "seed",
        "method",
        "run_id",
        "sample_index",
        "true_family",
        "candidate_family",
        "candidate_spec",
        "corruption_type",
        "validity_label",
        "target_rul",
        "predicted_rul",
        "absolute_error",
    ]
    frame = pd.read_csv(run / "rul_predictions.csv", usecols=columns)
    if frame.duplicated(["seed", "method", "run_id", "sample_index", "candidate_spec"]).any():
        raise ValueError("RUL predictions contain duplicate identifiers.")
    comparison_rows = []
    for (seed, method), group in frame.groupby(["seed", "method"]):
        target = group["target_rul"].to_numpy(dtype=float)
        prediction = group["predicted_rul"].to_numpy(dtype=float)
        run_rmse = [
            math.sqrt(mean_squared_error(run_group["target_rul"], run_group["predicted_rul"]))
            for _, run_group in group.groupby("run_id")
        ]
        comparison_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "mae": float(mean_absolute_error(target, prediction)),
                "mse": float(mean_squared_error(target, prediction)),
                "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
                "r2": float(r2_score(target, prediction)),
                "macro_run_rmse": float(np.mean(run_rmse)),
            }
        )
    verified = pd.DataFrame(comparison_rows).sort_values(["seed", "method"]).reset_index(drop=True)
    saved = pd.read_csv(run / "model_comparison.csv").sort_values(["seed", "method"]).reset_index(drop=True)
    for column in ("mae", "mse", "rmse", "r2", "macro_run_rmse"):
        if not np.allclose(verified[column], saved[column], rtol=0.0, atol=1e-12):
            raise ValueError(f"Saved RUL metric does not reproduce: {column}")
    regret_rows = []
    for (seed, validity), subset in frame.groupby(["seed", "validity_label"]):
        baseline = subset[subset["method"] == "data_only"]
        baseline_macro = np.mean(
            [
                math.sqrt(mean_squared_error(group["target_rul"], group["predicted_rul"]))
                for _, group in baseline.groupby("run_id")
            ]
        )
        for method, group in subset.groupby("method"):
            macro = np.mean(
                [
                    math.sqrt(mean_squared_error(run_group["target_rul"], run_group["predicted_rul"]))
                    for _, run_group in group.groupby("run_id")
                ]
            )
            regret_rows.append(
                {
                    "seed": int(seed),
                    "validity_label": int(validity),
                    "method": method,
                    "macro_run_rmse": float(macro),
                    "physics_regret": float(macro - baseline_macro),
                }
            )
    frame["lifecycle_region"] = pd.cut(
        1.0 - frame["target_rul"],
        [-1e-9, 1.0 / 3.0, 2.0 / 3.0, 1.0000001],
        labels=["early", "middle", "late"],
    )
    lifecycle = (
        frame.groupby(["seed", "method", "lifecycle_region"], observed=True)["absolute_error"]
        .mean()
        .reset_index(name="mae")
    )
    return verified, pd.DataFrame(regret_rows), lifecycle


def convergence_summary(run: Path) -> pd.DataFrame:
    history = pd.read_csv(run / "training_history.csv")
    rows = []
    for (recorded_seed, phase), group in history.groupby(["seed", "phase"]):
        best = group.loc[group["validation_mse"].idxmin()]
        rows.append(
            {
                "recorded_seed": int(recorded_seed),
                "phase": phase,
                "epochs": int(len(group)),
                "best_epoch": int(best["epoch"]),
                "best_validation_mse": float(best["validation_mse"]),
                "final_validation_mse": float(group.iloc[-1]["validation_mse"]),
                "final_train_mse": float(group.iloc[-1]["train_mse"]),
                "final_validation_to_train_ratio": float(
                    group.iloc[-1]["validation_mse"] / max(group.iloc[-1]["train_mse"], 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["recorded_seed", "phase"])


def main(run: Path) -> None:
    run = run.resolve()
    ANALYZED.mkdir(parents=True, exist_ok=True)
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    validation = validate_artifacts(run)
    (ANALYZED / "artifact_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["all_identity_and_integrity_checks_passed"]:
        raise RuntimeError("EXP-007 artifact validation failed; see artifact_validation.json.")

    credibility = pd.read_csv(run / "credibility_predictions.csv")
    if credibility.duplicated(["seed", "run_id", "sample_index", "candidate_spec"]).any():
        raise ValueError("Credibility predictions contain duplicate identifiers.")
    units = aggregate_units(credibility)
    if len(units) != 800 or set(units["seed"]) != set(SEEDS):
        raise ValueError("Expected 800 trajectory/candidate/seed units.")
    per_seed_rows = []
    for seed, group in units.groupby("seed"):
        per_seed_rows.append({"scope": "seed", "seed": int(seed), **classification_metrics(group)})
    per_seed = pd.DataFrame(per_seed_rows)
    mean_metrics = {
        column: float(per_seed[column].mean())
        for column in ("auroc", "auprc", "brier", "ece", "fallback_fraction", "all_on_fraction")
    }
    std_metrics = {
        f"{column}_std": float(per_seed[column].std(ddof=1))
        for column in ("auroc", "auprc", "brier", "ece", "fallback_fraction", "all_on_fraction")
    }
    ensemble = (
        units.groupby(
            [
                "run_id",
                "candidate_spec",
                "true_family",
                "candidate_family",
                "corruption_type",
                "condition_id",
                "validity_label",
            ],
            as_index=False,
        )
        .agg(credibility=("credibility", "mean"), fallback_fraction=("fallback_fraction", "mean"))
    )
    ensemble_metrics = classification_metrics(ensemble)
    bootstrap = hierarchical_bootstrap(units)
    saved_statistics = json.loads((run / "statistical_summary.json").read_text(encoding="utf-8"))
    verified_rows = per_seed.to_dict(orient="records")
    verified_rows.extend(
        [
            {"scope": "seed_mean", "seed": "all", **mean_metrics, **std_metrics},
            {"scope": "mean_probability_ensemble", "seed": "all", **ensemble_metrics},
            {
                "scope": "saved_cross_seed_pooled",
                "seed": "all",
                "auroc": saved_statistics["auroc"],
                "auprc": saved_statistics["auprc"],
                "brier": saved_statistics["brier"],
                "ece": saved_statistics["ece"],
            },
        ]
    )
    verified_credibility = pd.DataFrame(verified_rows)
    verified_credibility.to_csv(ANALYZED / "verified_metrics.csv", index=False)

    family_rows = []
    for family, subset in units.groupby("true_family"):
        values = [roc_auc_score(group["validity_label"], group["credibility"]) for _, group in subset.groupby("seed")]
        family_rows.append(
            {
                "true_family": family,
                "mean_seed_auroc": float(np.mean(values)),
                "seed_auroc_std": float(np.std(values, ddof=1)),
                "minimum_seed_auroc": float(np.min(values)),
                "maximum_seed_auroc": float(np.max(values)),
            }
        )
    pd.DataFrame(family_rows).to_csv(ANALYZED / "credibility_by_family.csv", index=False)

    corruption_rows = []
    for corruption in ("wrong_progression_family", "time_scale_fast", "time_scale_slow"):
        subset = units[(units["validity_label"] == 1) | (units["corruption_type"] == corruption)]
        values = [roc_auc_score(group["validity_label"], group["credibility"]) for _, group in subset.groupby("seed")]
        corruption_rows.append(
            {
                "negative_class": corruption,
                "mean_seed_auroc": float(np.mean(values)),
                "seed_auroc_std": float(np.std(values, ddof=1)),
                "minimum_seed_auroc": float(np.min(values)),
                "maximum_seed_auroc": float(np.max(values)),
            }
        )
    pd.DataFrame(corruption_rows).to_csv(ANALYZED / "credibility_by_corruption.csv", index=False)

    shift_rows = []
    for seed in SEEDS:
        seed_dir = run / "seeds" / f"seed_{seed:05d}"
        for partition, filename in (
            ("train", "train_credibility_predictions.csv"),
            ("validation", "validation_credibility_predictions.csv"),
            ("test", "test_credibility_predictions.csv"),
        ):
            frame = pd.read_csv(
                seed_dir / filename,
                usecols=[
                    "run_id",
                    "sample_index",
                    "covariate_shift_score",
                    "operation_shift_score",
                    "credibility",
                    "fallback",
                    "validity_label",
                    "candidate_spec",
                    "true_family",
                    "candidate_family",
                    "corruption_type",
                    "condition_id",
                ],
            )
            sample = frame.drop_duplicates(["run_id", "sample_index"])
            frame["seed"] = seed
            partition_units = aggregate_units(frame.assign(credibility_raw=frame["credibility"], target_rul=0.0))
            shift_rows.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "covariate_shift_median": float(sample["covariate_shift_score"].median()),
                    "operation_shift_median": float(sample["operation_shift_score"].median()),
                    "auroc": float(roc_auc_score(partition_units["validity_label"], partition_units["credibility"])),
                    "fallback_fraction": float(frame["fallback"].mean()),
                }
            )
    pd.DataFrame(shift_rows).to_csv(ANALYZED / "shift_and_collapse.csv", index=False)

    verified_models, regret, lifecycle = verify_rul_metrics(run)
    verified_models.to_csv(ANALYZED / "verified_model_comparison.csv", index=False)
    regret.to_csv(ANALYZED / "verified_physics_regret.csv", index=False)
    lifecycle.to_csv(ANALYZED / "lifecycle_metrics.csv", index=False)
    convergence = convergence_summary(run)
    convergence.to_csv(ANALYZED / "convergence_summary.csv", index=False)
    recovery = pd.read_csv(run / "parameter_recovery.csv")
    recovery_summary = (
        recovery.groupby(["seed", "true_family"])["absolute_scale_error"]
        .agg(["mean", "median", "std", "max"])
        .reset_index()
    )
    recovery_summary.to_csv(ANALYZED / "parameter_recovery_summary.csv", index=False)

    collapsed_seeds = int((per_seed["fallback_fraction"] > 0.90).sum())
    prevalence = float(units["validity_label"].mean())
    null_brier = prevalence * (1.0 - prevalence)
    corrected = {
        "experiment_id": "EXP-007",
        "saved_pooled_auroc": saved_statistics["auroc"],
        "corrected_mean_seed_auroc": mean_metrics["auroc"],
        "corrected_seed_auroc_std": std_metrics["auroc_std"],
        "mean_probability_ensemble_auroc": ensemble_metrics["auroc"],
        "hierarchical_bootstrap": bootstrap,
        "mean_seed_auprc": mean_metrics["auprc"],
        "chance_auprc": prevalence,
        "mean_seed_brier": mean_metrics["brier"],
        "constant_prevalence_brier": null_brier,
        "collapsed_all_off_seeds": collapsed_seeds,
        "seeds": len(SEEDS),
        "decision": "stop_and_diagnose_identifiability",
    }
    (ANALYZED / "corrected_statistical_summary.json").write_text(
        json.dumps(corrected, indent=2), encoding="utf-8"
    )

    criteria = pd.DataFrame(
        [
            ["identity_integrity", True, "All manifest, commit, config, split, cache, notebook, inventory, and bundle checks passed."],
            ["all_five_seeds_completed", True, "Five of five seeds completed with no recorded failures."],
            ["mean_seed_auroc_at_least_0p80", mean_metrics["auroc"] >= 0.80, f"Corrected mean seed AUROC={mean_metrics['auroc']:.6f}."],
            ["hierarchical_ci_lower_above_0p50", bootstrap["ci_lower_95"] > 0.50, f"Corrected 95% interval=[{bootstrap['ci_lower_95']:.6f}, {bootstrap['ci_upper_95']:.6f}]."],
            ["anti_collapse_per_seed", collapsed_seeds == 0, f"{collapsed_seeds}/5 seeds exceeded 90% all-off fallback."],
            ["corrupt_priors_create_negative_transfer_stress", False, "All-on corrupt-prior macro regret was negative on average; the benchmark did not create the intended harm endpoint."],
            ["proceed_to_exp008", False, "The locked AUROC and anti-collapse gates failed."],
        ],
        columns=["criterion", "passed", "evidence"],
    )
    criteria.to_csv(ANALYZED / "success_criteria.csv", index=False)

    model_summary = verified_models.groupby("method")["macro_run_rmse"].agg(["mean", "std"]).sort_values("mean")
    all_on_corrupt = regret[(regret["method"] == "all_on") & (regret["validity_label"] == 0)]["physics_regret"].mean()
    priorcred_corrupt = regret[(regret["method"] == "priorcred_thresholded") & (regret["validity_label"] == 0)]["physics_regret"].mean()
    analysis = f"""# EXP-007 independent analysis

Status: **valid completed run; predeclared scientific gate failed**

## Facts verified from artifacts

- The run used exact clean commit `{json.loads((run / 'run_manifest.json').read_text())['git_commit']}` on a Tesla T4.
- Configuration, immutable 24/8/8 split, controlled-cache fingerprint, executed-notebook source,
  artifact inventory, and lightweight ZIP all match. All five seeds completed; no model failure
  or OOM was recorded.
- The complete run took {json.loads((run / 'run_manifest.json').read_text())['elapsed_seconds']:.1f} seconds.

## Independently recomputed credibility result

The saved report pooled differently calibrated probabilities across seeds and reported AUROC
`{saved_statistics['auroc']:.6f}`. That is not the locked seed-level aggregation. Recomputing
AUROC within each seed gives mean `{mean_metrics['auroc']:.6f}` (SD
`{std_metrics['auroc_std']:.6f}`) with a trajectory-first, seed-second bootstrap 95% interval
`[{bootstrap['ci_lower_95']:.6f}, {bootstrap['ci_upper_95']:.6f}]`. The mean-probability
five-seed ensemble AUROC is `{ensemble_metrics['auroc']:.6f}`. All remain below the required
`0.80` point estimate.

Mean AUPRC is `{mean_metrics['auprc']:.6f}` versus class prevalence `{prevalence:.2f}`. Mean
Brier score is `{mean_metrics['brier']:.6f}`, worse than the constant-prevalence reference
`{null_brier:.6f}`. Four of five seeds exceed 90% all-off fallback; three are exactly all-off
at the trajectory-candidate threshold level. The pooled 85.8% fallback rate therefore hides
seed-level collapse.

The diagnostic is strongly family- and corruption-dependent. Gamma mean AUROC is about 0.385,
and valid-versus-slow-time-scale AUROC is about 0.459, both worse than chance. Linear and
progressive families are easier, but their seed variation is large.

## Failure mechanism

Training uses only one speed/SNR condition. StandardScaler assigns scale 1 to those zero-
variance features, making median operation-shift score rise from about 0.31 in training to
69.4 in validation and 138.8 in test. Median covariate-shift score similarly rises from about
0.87 to 32.1 and 64.25. The credibility classifier therefore extrapolates far beyond its
training support. Validation thresholds yield roughly half fallback, but four test seeds
collapse almost completely off.

The target is also misaligned with the intended negative-transfer question. Corrupt priors did
not generally harm RUL prediction: mean all-on corrupt-prior macro regret is
`{all_on_corrupt:.6f}`, and PriorCred corrupt-prior regret is `{priorcred_corrupt:.6f}` (negative
means improvement over data-only). Validation-selected scalar blending ranks first at mean
macro RMSE `{model_summary.iloc[0]['mean']:.6f}`, while data-only is
`{model_summary.loc['data_only', 'mean']:.6f}`. Because even wrong priors usually help this weak
backbone, law-correctness labels are not a usable proxy for intervention harm in this setup.

Parameter time-scale recovery is poor: mean absolute factor error is
`{recovery['absolute_scale_error'].mean():.6f}`. Final backbones also vary substantially: best
epochs are 103, 21, 160, 1, and 1, and final validation-to-training MSE ratios range from about
2.7 to 26.4.

## Conclusion

H1 is not supported, anti-collapse fails, and the benchmark does not express the required
negative-transfer stress. Do not proceed to EXP-008 or claim credibility-guided physics. This
is a useful negative feasibility result and a precise diagnosis of identifiability, scaling,
calibration, and target-alignment problems.
"""
    (ANALYZED / "analysis.md").write_text(analysis, encoding="utf-8")

    issues = """# EXP-007 issues

1. The required AUROC >= 0.80 gate failed under both corrected seed aggregation and a mean-probability ensemble.
2. Four of five seeds exceed 90% all-off fallback; pooled averaging concealed this collapse.
3. Zero-variance source speed/SNR features make validation/test shift scores tens to hundreds of nominal standard deviations.
4. Slow time-scale corruption and gamma-family applicability are worse than chance.
5. Corrupt priors usually improve rather than harm the data-only backbone, so the benchmark does not instantiate the negative-transfer endpoint.
6. Mean Brier score is worse than a constant class-prevalence predictor, showing poor calibration.
7. The saved gate pools probabilities across seeds instead of aggregating seed-level AUROC as required by the protocol.
8. Seed 4042 job_result reports AUROC 0.597733, while its saved predictions reproduce 0.574142; serialized predictions are authoritative.
9. Cross-fit histories record derived optimization seeds (for example 143 and 244) without a separate parent experiment-seed field.
10. Time-scale parameter recovery and backbone convergence are unstable.
"""
    (ANALYZED / "issues.md").write_text(issues, encoding="utf-8")

    recommendations = """# EXP-007 recommendations

1. Freeze EXP-007 as a valid negative feasibility result. Do not tune it against the observed test set and do not start EXP-008.
2. Amend the protocol before a corrective experiment to distinguish law correctness from counterfactual intervention harm while preserving the applicability-versus-weight distinction.
3. Replace zero-variance StandardScaler shift features with predeclared physically scaled condition deltas or a source design containing multiple operating conditions.
4. Run an oracle-evidence ceiling and feature ablations on the now-open EXP-007 data only for diagnosis. These results cannot validate a redesigned method.
5. On development simulations, require corrupt priors to produce measurable positive regret before training a negative-transfer prevention mechanism.
6. Use MATLAB to generate a larger multi-condition development population and a new sealed test population with a new seed after the method and gate code are frozen.
7. Correct hierarchical seed/trajectory aggregation, per-seed anti-collapse checks, parent-seed recording, probability serialization, and calibration baselines before another run.
8. Keep ANSYS deferred; the current blocker is statistical identifiability and benchmark design, not missing finite-element fidelity.
"""
    (ANALYZED / "recommendations.md").write_text(recommendations, encoding="utf-8")

    stage = pd.DataFrame(
        [
            ["EXP-006", "completed", "Data/physics qualification", "passed", "Controlled labeled benchmark established"],
            ["EXP-007", "completed", "Synthetic credibility feasibility", "failed", "Stop and diagnose identifiability; do not proceed to EXP-008"],
        ],
        columns=["experiment_id", "status", "primary_endpoint", "gate", "decision"],
    )
    stage.to_csv(COMPARISONS / "publication_stage_status.csv", index=False)
    print(json.dumps(corrected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    arguments = parser.parse_args()
    main(arguments.run)
