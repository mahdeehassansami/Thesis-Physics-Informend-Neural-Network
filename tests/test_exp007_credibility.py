from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from thesis_work.exp7_credibility import (
    BackboneFit,
    FAMILIES,
    _candidate_specs,
    _credibility_matrix,
    _fit_degradation_proxy,
    apply_credibility,
    build_credibility_evidence,
    fit_credibility_estimator,
    fit_template_library,
    load_exp7_config,
)
from thesis_work.sequence_models import LSTMRUL


ROOT = Path(__file__).resolve().parents[1]


def _fixture_frame(config: dict, runs_per_family: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(FAMILIES):
        for run_index in range(runs_per_family):
            run_id = f"fixture_{family_index}_{run_index}"
            for sample_index in range(12):
                lifecycle = sample_index / 11.0
                if family == "linear_increasing":
                    degradation = lifecycle
                elif family == "progressively_increasing":
                    degradation = lifecycle**2
                elif family == "step_like":
                    degradation = np.floor(lifecycle * 4.0) / 4.0
                else:
                    degradation = lifecycle**1.4
                row: dict[str, object] = {
                    "run_id": run_id,
                    "sample_index": sample_index,
                    "elapsed_minutes": sample_index * 10.0,
                    "condition_id": "fixture",
                    "degradation_family": family,
                    "degradation_value": degradation,
                    "rul_norm": 1.0 - lifecycle,
                }
                for number, column in enumerate(config["data"]["feature_columns"], start=1):
                    row[column] = number + degradation * (family_index + 1) + run_index * 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def _fit_fixture(config: dict, frame: pd.DataFrame) -> BackboneFit:
    columns = config["data"]["feature_columns"]
    scaler = StandardScaler().fit(frame[columns].to_numpy(dtype=float))
    return BackboneFit(
        model=LSTMRUL(len(columns), 8),
        scaler=scaler,
        time_scale_minutes=110.0,
        history=pd.DataFrame(),
        parameter_count=0,
        best_epoch=0,
        best_validation_mse=0.0,
    )


def test_exp007_config_keeps_truth_out_of_credibility_inputs() -> None:
    config = load_exp7_config(ROOT / "configs" / "experiment_exp007.yaml")
    forbidden = set(config["corruptions"]["forbidden_credibility_inputs"])
    evidence = set(config["credibility"]["numeric_evidence"]) | set(
        config["credibility"]["categorical_evidence"]
    )
    assert config["experiment"]["id"] == "EXP-007"
    assert len(config["training"]["seeds"]) == 5
    assert not forbidden & evidence
    assert config["data"]["target_test_access"] == "evaluation_only"


def test_partition_corruptions_are_predeclared_and_truth_independent() -> None:
    config = load_exp7_config(ROOT / "configs" / "experiment_exp007.yaml")
    train = _candidate_specs("linear_increasing", "train", config)
    test = _candidate_specs("linear_increasing", "test", config)
    assert sum(item["validity_label"] for item in train) == 3
    assert sum(item["validity_label"] for item in test) == 3
    assert len(train) == len(test) == 20
    assert {item["time_scale_factor"] for item in train} != {
        item["time_scale_factor"] for item in test
    }
    assert {item["candidate_family"] for item in train} == set(FAMILIES)
    alternate_truth = _candidate_specs("gamma", "train", config)
    assert [
        (item["candidate_spec"], item["candidate_family"], item["time_scale_factor"])
        for item in train
    ] == [
        (item["candidate_spec"], item["candidate_family"], item["time_scale_factor"])
        for item in alternate_truth
    ]
    assert all(
        item["corruption_type"] == "wrong_progression_family"
        for item in train
        if item["candidate_family"] != "linear_increasing"
    )


def test_empirical_templates_are_monotone_and_bounded() -> None:
    config = load_exp7_config(ROOT / "configs" / "experiment_exp007.yaml")
    library = fit_template_library(_fixture_frame(config), config)
    for family in FAMILIES:
        assert np.all(np.diff(library.mean[family]) >= -1e-12)
        assert np.all((library.mean[family] >= 0.0) & (library.mean[family] <= 1.0))
        assert library.mean[family][-1] == 1.0


def test_evidence_generation_does_not_expose_truth_to_estimator() -> None:
    config = load_exp7_config(ROOT / "configs" / "experiment_exp007.yaml")
    frame = _fixture_frame(config)
    fit = _fit_fixture(config, frame)
    proxy_config = dict(config)
    proxy_config["degradation_proxy"] = dict(config["degradation_proxy"])
    proxy_config["degradation_proxy"]["estimators"] = 8
    proxy = _fit_degradation_proxy(frame, proxy_config, seed=42)
    templates = fit_template_library(frame, config)
    prediction_rows = []
    for (run_id, sample_index, elapsed, target) in frame.loc[
        frame["sample_index"] >= 7,
        ["run_id", "sample_index", "elapsed_minutes", "rul_norm"],
    ].itertuples(index=False, name=None):
        prediction_rows.append(
            {
                "run_id": run_id,
                "sample_index": sample_index,
                "elapsed_minutes": elapsed,
                "target_rul": target,
                "backbone_rul": np.clip(target + 0.03, 0.0, 1.0),
            }
        )
    evidence = build_credibility_evidence(
        pd.DataFrame(prediction_rows), frame, fit, proxy, templates, config, "train"
    )
    matrix, names = _credibility_matrix(evidence, config)
    forbidden = set(config["corruptions"]["forbidden_credibility_inputs"])
    assert matrix.shape[0] == len(evidence)
    assert np.isfinite(matrix).all()
    assert not forbidden & set(names)
    assert set(evidence["validity_label"]) == {0, 1}


def test_credibility_fit_is_frozen_and_finite() -> None:
    config = load_exp7_config(ROOT / "configs" / "experiment_exp007.yaml")
    frame = _fixture_frame(config)
    fit = _fit_fixture(config, frame)
    proxy_config = dict(config)
    proxy_config["degradation_proxy"] = dict(config["degradation_proxy"])
    proxy_config["degradation_proxy"]["estimators"] = 8
    proxy = _fit_degradation_proxy(frame, proxy_config, seed=42)
    templates = fit_template_library(frame, config)
    predictions = frame.loc[frame["sample_index"] >= 7, ["run_id", "sample_index", "elapsed_minutes", "rul_norm"]].rename(
        columns={"rul_norm": "target_rul"}
    )
    predictions["backbone_rul"] = np.clip(predictions["target_rul"] + 0.02, 0.0, 1.0)
    train = build_credibility_evidence(predictions, frame, fit, proxy, templates, config, "train")
    validation = build_credibility_evidence(predictions, frame, fit, proxy, templates, config, "validation")
    estimator = fit_credibility_estimator(train, validation, config, seed=42)
    scored = apply_credibility(estimator, validation, config)
    assert 0.0 <= estimator.threshold <= 1.0
    assert np.isfinite(scored["credibility"]).all()
    assert scored["credibility"].between(0.0, 1.0).all()
