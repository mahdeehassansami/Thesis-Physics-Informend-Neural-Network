from __future__ import annotations

from pathlib import Path
import copy

import numpy as np
import pandas as pd

from thesis_work.exp7a_harm_credibility import (
    _control_base,
    apply_credibility,
    candidate_specs,
    fit_credibility_estimator,
)
from thesis_work.exp7b_causal_risk_control import (
    load_exp7b_config,
    validate_exp7b_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _evidence(config: dict, partition: str, run_count: int) -> pd.DataFrame:
    rows: list[dict] = []
    candidates = candidate_specs(config)
    for run_index in range(run_count):
        run_id = f"{partition}_run_{run_index:02d}"
        true_family = config["physics_intervention"]["candidate_families"][run_index % 4]
        for candidate_index, candidate in enumerate(candidates):
            safe = int(
                candidate["candidate_family"] == true_family
                and candidate["time_scale_factor"] == 1.0
            )
            for sample_index in range(8):
                target = 1.0 - sample_index / 7.0
                data_rul = float(np.clip(target + 0.05, 0.0, 1.0))
                physics_rul = (
                    float(np.clip(target + 0.01, 0.0, 1.0))
                    if safe
                    else float(np.clip(target + 0.20, 0.0, 1.0))
                )
                row = {
                    "parent_seed": 42,
                    "partition": partition,
                    "run_id": run_id,
                    "sample_index": sample_index,
                    "elapsed_minutes": sample_index * 10.0,
                    "condition_id": f"condition_{run_index % 4}",
                    "true_family": true_family,
                    "candidate_spec": candidate["candidate_spec"],
                    "candidate_family": candidate["candidate_family"],
                    "time_scale_factor": candidate["time_scale_factor"],
                    "law_correctness": bool(candidate["candidate_family"] == true_family),
                    "safe_to_apply": safe,
                    "harmful_intervention": 1 - safe,
                    "physics_regret": -0.04 if safe else 0.15,
                    "data_only_rmse": 0.05,
                    "physics_rmse": 0.01 if safe else 0.20,
                    "target_rul": target,
                    "data_rul": data_rul,
                    "physics_rul": physics_rul,
                }
                for feature_index, name in enumerate(
                    config["credibility"]["numeric_evidence"]
                ):
                    if name == "time_scale_factor":
                        value = candidate["time_scale_factor"]
                    elif name == "causal_prefix_length_log":
                        value = float(np.log1p(sample_index + 1))
                    else:
                        value = (
                            0.05 + 0.005 * feature_index + 0.001 * sample_index
                            if safe
                            else 1.0 + 0.02 * feature_index + 0.003 * candidate_index
                        )
                    row[name] = value
                rows.append(row)
    return pd.DataFrame(rows)


def test_exp007b_cache_and_seal_are_qualified() -> None:
    config = load_exp7b_config(ROOT / "configs" / "experiment.yaml")
    qualification = validate_exp7b_config(config, ROOT)
    assert qualification["status"] == "qualified"
    assert qualification["rows"] == 8268
    assert qualification["runs"] == 96
    frame = pd.read_csv(ROOT / config["data"]["feature_cache"])
    assert set(frame.loc[frame["official_partition"] == "test", "simulator_seed"]) == {
        920072
    }
    assert not (frame["simulator_seed"] == 920071).any()


def test_prefix_local_selector_qualifies_and_selects_at_most_one_candidate() -> None:
    config = load_exp7b_config(ROOT / "configs" / "experiment.yaml")
    train = _evidence(config, "train_oof", 12)
    validation = _evidence(config, "validation", 16)
    fit = fit_credibility_estimator(train, validation, config, 42)
    assert fit.calibrator is None
    assert fit.threshold_qualification is not None
    assert fit.threshold_qualification["passed"] is True
    scored = apply_credibility(fit, validation, config)
    assert (
        scored.groupby(["run_id", "candidate_spec"])["credibility"].nunique() > 1
    ).any()
    controls = _control_base(scored, 42, config)
    priorcred = controls[controls["method"] == "priorcred"]
    assert priorcred["priorcred_selected_candidates"].max() <= 1
    assert priorcred["priorcred_selected_candidates"].min() >= 0
    assert priorcred["predicted_rul"].between(0.0, 1.0).all()


def test_infeasible_selector_threshold_is_exact_all_off() -> None:
    config = copy.deepcopy(load_exp7b_config(ROOT / "configs" / "experiment.yaml"))
    config["credibility"]["risk_constraints"]["minimum_intervention_coverage"] = 1.1
    train = _evidence(config, "train_oof", 12)
    validation = _evidence(config, "validation", 16)
    fit = fit_credibility_estimator(train, validation, config, 42)
    assert fit.threshold_qualification is not None
    assert fit.threshold_qualification["passed"] is False
    assert fit.threshold_qualification["intervention_coverage"] == 0.0
    scored = apply_credibility(fit, validation, config)
    controls = _control_base(scored, 42, config)
    priorcred = controls[controls["method"] == "priorcred"]
    assert priorcred["priorcred_fallback"].all()
    assert np.allclose(priorcred["predicted_rul"], priorcred["data_rul"])
