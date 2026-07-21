from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_work.exp7a_harm_credibility import (
    FAMILIES,
    _condition_evidence,
    _credibility_matrix,
    _fit_degradation_proxy,
    _prior_arrays,
    aggregate_credibility_units,
    build_counterfactual_evidence,
    candidate_specs,
    fit_training_context,
    load_exp7a_config,
    validate_exp7a_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture_frame(config: dict, runs_per_family: int = 2) -> pd.DataFrame:
    rows: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(1, runs_per_family + 1):
            run_id = f"fixture_{family_index}_{replicate}"
            for sample_index in range(12):
                lifecycle = sample_index / 11.0
                damage = {
                    "linear_increasing": lifecycle,
                    "progressively_increasing": lifecycle**2,
                    "step_like": np.floor(lifecycle * 4.0) / 4.0,
                    "gamma": lifecycle**1.5,
                }[family]
                row = {
                    "run_id": run_id,
                    "official_partition": "train",
                    "condition_id": f"condition_{replicate}",
                    "sample_index": sample_index,
                    "elapsed_minutes": sample_index * 10.0,
                    "rul_norm": 1.0 - lifecycle,
                    "degradation_family": family,
                    "degradation_value": damage,
                    "load_n": 4200.0 + 600.0 * replicate,
                    "speed_rpm": 2700.0 + 180.0 * replicate,
                    "snr_db": 6.0 + 2.0 * replicate,
                }
                for index, column in enumerate(
                    config["data"]["signal_feature_columns"], start=1
                ):
                    row[column] = index + damage * (family_index + 1) + replicate * 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_design_is_multicondition_and_cache_pending() -> None:
    config = load_exp7a_config(ROOT / "configs" / "experiment.yaml")
    qualification = validate_exp7a_config(config, ROOT)
    assert qualification["status"] == "design_valid_cache_pending"
    assert qualification["split_counts"] == {
        "train": 64,
        "validation": 16,
        "test": 16,
    }
    scenarios = pd.read_csv(ROOT / config["data"]["scenario_file"])
    train = scenarios[scenarios["publication_split"] == "train"]
    assert train["OC_load_mean"].nunique() == 4
    assert train["OC_f_set"].nunique() == 4
    assert train["SD_SNR"].nunique() == 4
    assert set(
        scenarios.loc[scenarios["publication_split"] == "test", "simulator_seed"]
    ) == {920071}


def test_candidate_pool_is_truth_independent() -> None:
    config = load_exp7a_config(ROOT / "configs" / "experiment.yaml")
    candidates = candidate_specs(config)
    assert len(candidates) == 12
    assert {item["candidate_family"] for item in candidates} == set(FAMILIES)
    assert {item["time_scale_factor"] for item in candidates} == {0.6, 1.0, 1.6}
    assert all("true_family" not in item for item in candidates)


def test_physics_prior_is_finite_and_condition_scaled() -> None:
    config = load_exp7a_config(ROOT / "configs" / "experiment.yaml")
    frame = _fixture_frame(config)
    context = fit_training_context(frame, config)
    arrays = {
        "time": np.linspace(0.0, 1.0, 10).reshape(-1, 1),
        "load_n": np.linspace(4000.0, 6000.0, 10),
        "speed_rpm": np.linspace(2700.0, 3300.0, 10),
    }
    prior_rul, prior_rate = _prior_arrays(
        arrays, candidate_specs(config)[0], context, config
    )
    assert prior_rul.shape == prior_rate.shape == (10, 1)
    assert np.isfinite(prior_rul).all() and np.isfinite(prior_rate).all()
    assert np.all((prior_rul >= 0.0) & (prior_rul <= 1.0))
    conditions = _condition_evidence(frame, config)
    assert np.isfinite(conditions.to_numpy()).all()
    assert conditions["condition_distance"].max() < 2.0


def test_counterfactual_target_uses_actual_model_regret_not_law_correctness() -> None:
    config = load_exp7a_config(ROOT / "configs" / "experiment.yaml")
    proxy_config = copy.deepcopy(config)
    proxy_config["degradation_proxy"]["estimators"] = 8
    frame = _fixture_frame(config)
    context = fit_training_context(frame, config)
    proxy = _fit_degradation_proxy(frame, proxy_config, 42)
    samples = frame[frame["sample_index"] >= 7].copy()
    data = samples[["run_id", "sample_index", "elapsed_minutes", "rul_norm"]].rename(
        columns={"rul_norm": "target_rul"}
    )
    data["predicted_rul"] = data["target_rul"]
    candidate_rows: list[pd.DataFrame] = []
    for index, candidate in enumerate(candidate_specs(config)):
        part = data.rename(columns={"predicted_rul": "physics_rul"}).copy()
        if index % 2:
            part["physics_rul"] = np.clip(part["target_rul"] + 0.25, 0.0, 1.0)
        part["candidate_spec"] = candidate["candidate_spec"]
        part["candidate_family"] = candidate["candidate_family"]
        part["time_scale_factor"] = candidate["time_scale_factor"]
        candidate_rows.append(part)
    evidence = build_counterfactual_evidence(
        data,
        pd.concat(candidate_rows, ignore_index=True),
        frame,
        context,
        proxy,
        proxy_config,
        "train_oof",
    )
    evidence["parent_seed"] = 42
    units = aggregate_credibility_units(evidence, config)
    assert set(units["safe_to_apply"]) == {0, 1}
    assert not np.array_equal(
        units["safe_to_apply"], units["law_correctness"].astype(int)
    )
    matrix, names = _credibility_matrix(units, config)
    assert np.isfinite(matrix).all()
    assert not set(config["credibility"]["forbidden_inputs"]) & set(names)
