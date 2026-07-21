from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "saved results" / "run_07a" / "colab_run_01" / "experiment_outputs_exp007a"
)
OUTPUT = ROOT / "results" / "analyzed" / "EXP-007A" / "open_diagnostics"
FAMILIES = ["linear_increasing", "progressively_increasing", "step_like", "gamma"]
HARM_MARGIN = 0.01
MEAN_POSITIVE_REGRET_LIMIT = 0.005
HARM_RATE_LIMIT = 0.10
MINIMUM_COVERAGE = 0.05
MAXIMUM_COVERAGE = 0.90
CHECKPOINTS_PER_TRAJECTORY = 12
MAXIMUM_PHYSICS_BLEND = 0.50


@dataclass
class FittedRiskModel:
    name: str
    estimator: Any
    scaler: StandardScaler | None
    feature_names: list[str]


def _load_config(artifact_root: Path) -> dict[str, Any]:
    return yaml.safe_load((artifact_root / "experiment_config.yaml").read_text(encoding="utf-8"))


def _load_partition(artifact_root: Path, seeds: Iterable[int], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = artifact_root / "seeds" / f"seed_{int(seed):05d}" / filename
        frame = pd.read_csv(path)
        frame["parent_seed"] = int(seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _family_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    matrix = np.column_stack(
        [(frame["candidate_family"].to_numpy() == family).astype(float) for family in FAMILIES]
    )
    return matrix, [f"candidate_family__{family}" for family in FAMILIES]


def _matrix(frame: pd.DataFrame, numeric: list[str]) -> tuple[np.ndarray, list[str]]:
    family, family_names = _family_matrix(frame)
    matrix = np.concatenate([frame[numeric].to_numpy(dtype=float), family], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite causal selector evidence encountered.")
    return matrix, [*numeric, *family_names]


def _checkpoint_rows(frame: pd.DataFrame, count: int = CHECKPOINTS_PER_TRAJECTORY) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    keys = ["parent_seed", "run_id", "candidate_spec"]
    for _, group in frame.groupby(keys, sort=False):
        ordered = group.sort_values("sample_index")
        positions = np.unique(np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int))
        pieces.append(ordered.iloc[positions])
    return pd.concat(pieces, ignore_index=True)


def _fit_model(
    model_name: str,
    frame: pd.DataFrame,
    numeric: list[str],
    random_seed: int,
) -> FittedRiskModel:
    sampled = _checkpoint_rows(frame)
    x, names = _matrix(sampled, numeric)
    if model_name == "logistic_safe":
        scaler = StandardScaler().fit(x)
        estimator = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=3000,
            random_state=random_seed,
        ).fit(scaler.transform(x), sampled["safe_to_apply"].to_numpy(dtype=int))
        return FittedRiskModel(model_name, estimator, scaler, names)
    if model_name == "extra_trees_safe":
        estimator = ExtraTreesClassifier(
            n_estimators=256,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ).fit(x, sampled["safe_to_apply"].to_numpy(dtype=int))
        return FittedRiskModel(model_name, estimator, None, names)
    if model_name == "extra_trees_regret":
        regret = sampled["physics_regret"].to_numpy(dtype=float)
        weights = 1.0 + 4.0 * (regret > HARM_MARGIN).astype(float) + 2.0 * np.clip(regret, 0.0, 0.25)
        estimator = ExtraTreesRegressor(
            n_estimators=256,
            max_depth=12,
            min_samples_leaf=8,
            random_state=random_seed,
            n_jobs=-1,
        ).fit(x, regret, sample_weight=weights)
        return FittedRiskModel(model_name, estimator, None, names)
    raise ValueError(f"Unknown risk model: {model_name}")


def _risk_score(fit: FittedRiskModel, frame: pd.DataFrame, numeric: list[str]) -> np.ndarray:
    matrix, names = _matrix(frame, numeric)
    if names != fit.feature_names:
        raise ValueError("Causal selector feature order changed.")
    if fit.scaler is not None:
        matrix = fit.scaler.transform(matrix)
    if fit.name.endswith("_safe"):
        return 1.0 - fit.estimator.predict_proba(matrix)[:, 1]
    return fit.estimator.predict(matrix)


def _best_candidate_rows(frame: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    scored = frame.copy()
    scored["predicted_risk"] = np.asarray(risk, dtype=float)
    keys = ["parent_seed", "run_id", "sample_index"]
    order = scored.sort_values([*keys, "predicted_risk", "candidate_spec"])
    return order.groupby(keys, as_index=False, sort=False).head(1).reset_index(drop=True)


def _run_metrics(best: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, float]]:
    work = best.copy()
    selected = work["predicted_risk"].to_numpy(dtype=float) <= threshold
    work["selected"] = selected
    physics_blend = np.clip(
        work["data_rul"]
        + MAXIMUM_PHYSICS_BLEND * (work["physics_rul"] - work["data_rul"]),
        0.0,
        1.0,
    )
    work["controlled_rul"] = np.where(selected, physics_blend, work["data_rul"])
    rows: list[dict[str, Any]] = []
    for (seed, run_id), group in work.groupby(["parent_seed", "run_id"], sort=True):
        target = group["target_rul"].to_numpy(dtype=float)
        data = group["data_rul"].to_numpy(dtype=float)
        control = group["controlled_rul"].to_numpy(dtype=float)
        data_rmse = float(math.sqrt(np.mean(np.square(data - target))))
        control_rmse = float(math.sqrt(np.mean(np.square(control - target))))
        regret = control_rmse - data_rmse
        rows.append(
            {
                "parent_seed": int(seed),
                "run_id": str(run_id),
                "true_family": str(group["true_family"].iloc[0]),
                "condition_id": str(group["condition_id"].iloc[0]),
                "data_only_rmse": data_rmse,
                "controlled_rmse": control_rmse,
                "control_regret": regret,
                "positive_control_regret": max(0.0, regret),
                "harmful_control": int(regret > HARM_MARGIN),
                "coverage": float(group["selected"].mean()),
            }
        )
    runs = pd.DataFrame(rows)
    summary = {
        "macro_run_rmse": float(runs["controlled_rmse"].mean()),
        "data_only_macro_run_rmse": float(runs["data_only_rmse"].mean()),
        "mean_control_regret": float(runs["control_regret"].mean()),
        "mean_positive_control_regret": float(runs["positive_control_regret"].mean()),
        "harm_rate": float(runs["harmful_control"].mean()),
        "maximum_control_regret": float(runs["control_regret"].max()),
        "coverage": float(work["selected"].mean()),
    }
    return runs, summary


def _threshold_grid(best: pd.DataFrame) -> np.ndarray:
    values = best["predicted_risk"].to_numpy(dtype=float)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 101))
    all_off = np.nextafter(float(values.min()), -np.inf)
    return np.unique(np.concatenate([[all_off], quantiles]))


def _choose_threshold(best: pd.DataFrame) -> tuple[float, dict[str, float], bool]:
    candidates: list[tuple[float, dict[str, float]]] = []
    for threshold in _threshold_grid(best):
        _, summary = _run_metrics(best, float(threshold))
        feasible = (
            summary["mean_positive_control_regret"] <= MEAN_POSITIVE_REGRET_LIMIT
            and summary["harm_rate"] <= HARM_RATE_LIMIT
            and MINIMUM_COVERAGE <= summary["coverage"] <= MAXIMUM_COVERAGE
        )
        if feasible:
            candidates.append((float(threshold), summary))
    if not candidates:
        threshold = float(np.nextafter(best["predicted_risk"].min(), -np.inf))
        _, summary = _run_metrics(best, threshold)
        return threshold, summary, False
    threshold, summary = min(
        candidates,
        key=lambda item: (
            item[1]["macro_run_rmse"],
            item[1]["mean_positive_control_regret"],
            -item[1]["coverage"],
        ),
    )
    return threshold, summary, True


def _fold_map(run_ids: Iterable[str], seed: int, folds: int = 5) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[str]] = {family: [] for family in FAMILIES}
    for run_id in sorted(set(run_ids)):
        family = next(family for family in FAMILIES if family.split("_")[0] in run_id or (
            family == "progressively_increasing" and "progressive" in run_id
        ) or (family == "step_like" and "step" in run_id))
        by_family[family].append(run_id)
    result: dict[str, int] = {}
    for family, family_runs in by_family.items():
        shuffled = np.asarray(family_runs, dtype=object)
        rng.shuffle(shuffled)
        for index, run_id in enumerate(shuffled.tolist()):
            result[str(run_id)] = index % folds
    return result


def _evaluate_development(
    development: pd.DataFrame,
    numeric: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_names = ["logistic_safe", "extra_trees_safe", "extra_trees_regret"]
    fold_rows: list[dict[str, Any]] = []
    run_frames: list[pd.DataFrame] = []
    for parent_seed in seeds:
        seed_frame = development[development["parent_seed"] == parent_seed].copy()
        mapping = _fold_map(seed_frame["run_id"].unique(), parent_seed + 70070)
        seed_frame["diagnostic_fold"] = seed_frame["run_id"].map(mapping).astype(int)
        for model_index, model_name in enumerate(model_names):
            for outer_fold in range(5):
                calibration_fold = (outer_fold + 1) % 5
                fit_frame = seed_frame[
                    ~seed_frame["diagnostic_fold"].isin([outer_fold, calibration_fold])
                ]
                calibration = seed_frame[seed_frame["diagnostic_fold"] == calibration_fold]
                held_out = seed_frame[seed_frame["diagnostic_fold"] == outer_fold]
                fit = _fit_model(
                    model_name,
                    fit_frame,
                    numeric,
                    parent_seed + model_index * 1000 + outer_fold,
                )
                calibration_best = _best_candidate_rows(
                    calibration, _risk_score(fit, calibration, numeric)
                )
                threshold, calibration_summary, feasible = _choose_threshold(calibration_best)
                held_out_best = _best_candidate_rows(held_out, _risk_score(fit, held_out, numeric))
                held_out_runs, held_out_summary = _run_metrics(held_out_best, threshold)
                fold_rows.append(
                    {
                        "parent_seed": parent_seed,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "fit_trajectories": int(fit_frame["run_id"].nunique()),
                        "calibration_trajectories": int(calibration["run_id"].nunique()),
                        "held_out_trajectories": int(held_out["run_id"].nunique()),
                        "threshold": threshold,
                        "calibration_feasible": feasible,
                        **{f"calibration_{key}": value for key, value in calibration_summary.items()},
                        **{f"held_out_{key}": value for key, value in held_out_summary.items()},
                    }
                )
                held_out_runs["model"] = model_name
                held_out_runs["outer_fold"] = outer_fold
                held_out_runs["threshold"] = threshold
                held_out_runs["calibration_feasible"] = feasible
                run_frames.append(held_out_runs)
    return pd.DataFrame(fold_rows), pd.concat(run_frames, ignore_index=True)


def _model_summary(folds: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in runs.groupby("model"):
        metrics = {
                "model": model,
                "trajectory_seed_units": int(len(group)),
                "macro_run_rmse": float(group["controlled_rmse"].mean()),
                "data_only_macro_run_rmse": float(group["data_only_rmse"].mean()),
                "relative_improvement_percent": float(
                    100.0 * (group["data_only_rmse"].mean() - group["controlled_rmse"].mean())
                    / group["data_only_rmse"].mean()
                ),
                "mean_control_regret": float(group["control_regret"].mean()),
                "mean_positive_control_regret": float(group["positive_control_regret"].mean()),
                "harm_rate": float(group["harmful_control"].mean()),
                "maximum_control_regret": float(group["control_regret"].max()),
                "mean_coverage": float(group["coverage"].mean()),
                "feasible_fold_fraction": float(
                    folds.loc[folds["model"] == model, "calibration_feasible"].mean()
                ),
            }
        metrics["development_eligible"] = bool(
            metrics["relative_improvement_percent"] > 0.0
            and metrics["mean_positive_control_regret"] <= MEAN_POSITIVE_REGRET_LIMIT
            and metrics["harm_rate"] <= HARM_RATE_LIMIT
            and metrics["feasible_fold_fraction"] >= 0.70
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        ["development_eligible", "macro_run_rmse", "mean_positive_control_regret", "model"],
        ascending=[False, True, True, True],
    )


def _opened_test_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    seeds: list[int],
    selected_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    run_frames: list[pd.DataFrame] = []
    for parent_seed in seeds:
        fit_frame = train[train["parent_seed"] == parent_seed]
        calibration = validation[validation["parent_seed"] == parent_seed]
        held_out = test[test["parent_seed"] == parent_seed]
        fit = _fit_model(selected_model, fit_frame, numeric, parent_seed + 9070)
        calibration_best = _best_candidate_rows(
            calibration, _risk_score(fit, calibration, numeric)
        )
        threshold, calibration_summary, feasible = _choose_threshold(calibration_best)
        test_best = _best_candidate_rows(held_out, _risk_score(fit, held_out, numeric))
        test_runs, test_summary = _run_metrics(test_best, threshold)
        summaries.append(
            {
                "parent_seed": parent_seed,
                "model": selected_model,
                "threshold": threshold,
                "calibration_feasible": feasible,
                **{f"calibration_{key}": value for key, value in calibration_summary.items()},
                **{f"opened_test_{key}": value for key, value in test_summary.items()},
            }
        )
        test_runs["model"] = selected_model
        test_runs["threshold"] = threshold
        run_frames.append(test_runs)
    return pd.DataFrame(summaries), pd.concat(run_frames, ignore_index=True)


def _classification_audit(test: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    keys = ["parent_seed", "run_id", "candidate_spec", "true_family", "candidate_family"]
    units = test.groupby(keys, as_index=False).agg(
        safe_to_apply=("safe_to_apply", "first"),
        credibility=("credibility", "first"),
        physics_regret=("physics_regret", "first"),
    )
    rows: list[dict[str, Any]] = []
    for dimension in ["parent_seed", "true_family", "candidate_family"]:
        for value, group in units.groupby(dimension):
            target = group["safe_to_apply"].to_numpy(dtype=int)
            probability = group["credibility"].to_numpy(dtype=float)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "units": int(len(group)),
                    "safe_fraction": float(target.mean()),
                    "auroc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else np.nan,
                    "auprc": float(average_precision_score(target, probability)),
                    "brier": float(brier_score_loss(target, probability)),
                    "mean_positive_candidate_regret": float(np.maximum(group["physics_regret"], 0.0).mean()),
                    "maximum_candidate_regret": float(group["physics_regret"].max()),
                }
            )
    return pd.DataFrame(rows)


def _tail_tables(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    units = test.groupby(
        ["parent_seed", "run_id", "candidate_spec", "true_family", "candidate_family", "time_scale_factor", "condition_id"],
        as_index=False,
    ).agg(physics_regret=("physics_regret", "first"))
    units["positive_physics_regret"] = units["physics_regret"].clip(lower=0.0)
    tables: dict[str, pd.DataFrame] = {}
    for dimension in ["parent_seed", "true_family", "candidate_family", "time_scale_factor", "condition_id"]:
        tables[dimension] = units.groupby(dimension, as_index=False).agg(
            units=("physics_regret", "size"),
            harmful_fraction=("physics_regret", lambda values: float((values > HARM_MARGIN).mean())),
            mean_regret=("physics_regret", "mean"),
            mean_positive_regret=("positive_physics_regret", "mean"),
            maximum_regret=("physics_regret", "max"),
        )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open, non-confirmatory EXP-007A tail and causal risk-control diagnostics."
    )
    parser.add_argument("artifact_root", nargs="?", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    manifest = json.loads((artifact_root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "EXP-007A":
        raise ValueError("The diagnostic input is not EXP-007A.")
    config = _load_config(artifact_root)
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    numeric = list(config["credibility"]["numeric_evidence"])
    forbidden = set(config["credibility"]["forbidden_inputs"])
    if forbidden & set(numeric):
        raise ValueError("A forbidden field entered the causal diagnostic features.")

    train = _load_partition(artifact_root, seeds, "train_counterfactual_evidence.csv")
    validation = _load_partition(artifact_root, seeds, "validation_counterfactual_evidence.csv")
    test = _load_partition(artifact_root, seeds, "test_credibility_predictions.csv")
    development = pd.concat([train, validation], ignore_index=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    classification = _classification_audit(test, config)
    classification.to_csv(OUTPUT / "opened_test_classification_slices.csv", index=False)
    for dimension, table in _tail_tables(test).items():
        table.to_csv(OUTPUT / f"opened_test_candidate_tail_by_{dimension}.csv", index=False)

    folds, development_runs = _evaluate_development(development, numeric, seeds)
    summary = _model_summary(folds, development_runs)
    folds.to_csv(OUTPUT / "development_nested_grouped_folds.csv", index=False)
    development_runs.to_csv(OUTPUT / "development_nested_grouped_run_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "development_risk_control_summary.csv", index=False)
    selected_model = str(summary.iloc[0]["model"])

    opened_summary, opened_runs = _opened_test_audit(
        train, validation, test, numeric, seeds, selected_model
    )
    opened_summary.to_csv(OUTPUT / "opened_test_selected_method_by_seed.csv", index=False)
    opened_runs.to_csv(OUTPUT / "opened_test_selected_method_run_metrics.csv", index=False)

    within_unit_score_counts = (
        test.groupby(["parent_seed", "run_id", "candidate_spec"])["credibility"].nunique()
    )
    report = {
        "experiment_id": "EXP-007A",
        "status": "open_data_diagnostic_only",
        "confirmation_use_allowed": False,
        "source_git_commit": manifest.get("git_commit"),
        "causality_audit": {
            "existing_selector_feature_aggregation": "complete_trajectory_mean",
            "existing_score_unique_values_per_trajectory_candidate_max": int(
                within_unit_score_counts.max()
            ),
            "existing_selector_is_prefix_local": False,
            "finding": (
                "EXP-007A computes rolling evidence causally, then averages it across the complete "
                "trajectory before fitting and inference; the resulting intervention score is not "
                "available at the earlier prediction times to which it is applied."
            ),
        },
        "development_design": {
            "held_out_unit": "complete_trajectory across all candidates and prefixes",
            "outer_folds": 5,
            "per_outer_fold": "48 fit trajectories, 16 threshold-calibration trajectories, 16 held-out trajectories",
            "test_population_used_for_model_choice": False,
            "mean_positive_regret_limit": MEAN_POSITIVE_REGRET_LIMIT,
            "harm_margin": HARM_MARGIN,
            "harm_rate_limit": HARM_RATE_LIMIT,
            "coverage_interval": [MINIMUM_COVERAGE, MAXIMUM_COVERAGE],
            "maximum_physics_blend": MAXIMUM_PHYSICS_BLEND,
        },
        "selected_development_method": selected_model,
        "selected_development_metrics": summary.iloc[0].to_dict(),
        "opened_test_role": "exploratory stress audit only; not confirmation and not used for method selection",
        "methodology_references": [
            {
                "title": "Conformal Risk Control",
                "url": "https://arxiv.org/abs/2208.02814",
                "use": "risk-control framing and explicit abstention conservativeness",
                "claim_excluded": "No conformal guarantee is claimed for this diagnostic selector.",
            },
            {
                "title": "Selective Classification via One-Sided Prediction",
                "url": "https://proceedings.mlr.press/v130/gangrade21a.html",
                "use": "coverage-versus-risk abstention framing",
            },
        ],
    }
    (OUTPUT / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# EXP-007A opened diagnostics\n\n"
        "These results are exploratory. EXP-007A's test population is open and cannot confirm "
        "a revised selector. Model selection in this directory uses only trajectory-grouped "
        "development folds; the old test is reported separately as a stress audit.\n\n"
        "The audit also corrects a methodological description: EXP-007A's evidence columns were "
        "constructed causally, but its complete-trajectory mean score was not a causal prefix-local "
        "decision. The proposed next phase therefore operates at each observed prefix and abstains "
        "to the exact data-only parent when its development-calibrated risk condition is not met.\n\n"
        "This is a selective risk-control diagnostic, not a conformal guarantee. The available "
        "validation sample is too small to support the desired finite-sample claim under the current "
        "loss and exchangeability assumptions.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
