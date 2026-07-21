from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = (
    ROOT / "saved results" / "run_07a" / "colab_run_01" / "experiment_outputs_exp007a"
)
OUTPUT = ROOT / "results" / "analyzed" / "EXP-007A" / "open_diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Opened EXP-007A lifecycle regret diagnostic.")
    parser.add_argument("artifact_root", nargs="?", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    controls = pd.read_csv(args.artifact_root.resolve() / "control_predictions.csv")
    selected = controls[controls["method"].isin(["data_only", "priorcred"])].copy()
    selected["lifecycle_region"] = np.select(
        [selected["target_rul"] > 2.0 / 3.0, selected["target_rul"] > 1.0 / 3.0],
        ["early", "middle"],
        default="late",
    )
    keys = [
        "parent_seed",
        "run_id",
        "true_family",
        "condition_id",
        "lifecycle_region",
    ]
    rows: list[dict] = []
    for identity, group in selected.groupby(keys, sort=True):
        metrics: dict[str, float] = {}
        for method in ("data_only", "priorcred"):
            part = group[group["method"] == method]
            metrics[f"{method}_rmse"] = float(
                math.sqrt(np.mean(np.square(part["predicted_rul"] - part["target_rul"])))
            )
        regret = metrics["priorcred_rmse"] - metrics["data_only_rmse"]
        prior = group[group["method"] == "priorcred"]
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                **metrics,
                "physics_regret": regret,
                "positive_physics_regret": max(0.0, regret),
                "harmful_region": int(regret > 0.01),
                "intervention_coverage": float(
                    (prior["priorcred_selected_candidates"] > 0).mean()
                ),
            }
        )
    units = pd.DataFrame(rows)
    summary = units.groupby("lifecycle_region", as_index=False).agg(
        trajectory_seed_regions=("physics_regret", "size"),
        mean_regret=("physics_regret", "mean"),
        mean_positive_regret=("positive_physics_regret", "mean"),
        harmful_fraction=("harmful_region", "mean"),
        maximum_regret=("physics_regret", "max"),
        mean_intervention_coverage=("intervention_coverage", "mean"),
    )
    order = pd.CategoricalDtype(["early", "middle", "late"], ordered=True)
    summary["lifecycle_region"] = summary["lifecycle_region"].astype(order)
    summary = summary.sort_values("lifecycle_region")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units.to_csv(OUTPUT / "opened_test_priorcred_lifecycle_run_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "opened_test_priorcred_lifecycle_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
