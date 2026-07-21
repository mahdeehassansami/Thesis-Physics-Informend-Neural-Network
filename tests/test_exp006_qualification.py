from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from thesis_work.exp6_data_qualification import (
    physics_applicability_rows,
    qualification_paths,
    validate_feature_cache,
    validate_publication_splits,
    validate_scenario_design,
)
from thesis_work.multi_dataset import SIGNAL_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def test_exp006_controlled_design_matches_immutable_split() -> None:
    config = json.loads(
        (ROOT / "configs" / "exp006_data_qualification.json").read_text(
            encoding="utf-8"
        )
    )
    split = json.loads(
        (ROOT / "configs" / "publication_data_split.json").read_text(
            encoding="utf-8"
        )
    )
    scenarios = pd.read_csv(
        ROOT / "configs" / "exp006_controlled_simulation_scenarios.csv"
    )

    split_summary = validate_publication_splits(split)
    scenario_summary = validate_scenario_design(scenarios, split, config)

    assert split_summary["supplied_synthetic_v2"]["total_runs"] == 40
    assert split_summary["controlled_synthetic"] == {
        "train_runs": 24,
        "validation_runs": 8,
        "test_runs": 8,
        "total_runs": 40,
    }
    assert scenario_summary["scenarios"] == 40
    assert scenario_summary["family_counts"] == {
        "gamma": 10,
        "linear_increasing": 10,
        "progressively_increasing": 10,
        "step_like": 10,
    }
    json.dumps(scenario_summary)


def _feature_rows(run_id: str, truth: bool, family: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index in range(10):
        row: dict[str, object] = {
            "dataset": "fixture",
            "run_id": run_id,
            "official_partition": "train",
            "sample_index": sample_index,
            "elapsed_minutes": sample_index * 10.0,
            "elapsed_seconds": sample_index * 600.0,
            "rul_minutes": (9 - sample_index) * 10.0,
            "rul_norm": 1.0 - sample_index / 9.0,
            "truth_available": truth,
            "degradation_family": family,
            "fault_location": "outer" if truth else "",
            "degradation_value": sample_index / 9.0 if truth else np.nan,
            "sampling_hz": 4096.0,
            "load_n": 4500.0,
            "speed_rpm": 3000.0,
        }
        for feature_number, feature in enumerate(SIGNAL_FEATURES, start=1):
            row[feature] = feature_number + 0.01 * sample_index
        rows.append(row)
    return rows


def test_feature_cache_validation_preserves_truth_boundary(tmp_path: Path) -> None:
    supplied_path = tmp_path / "supplied.csv"
    pd.DataFrame(_feature_rows("synthetic_train_001", False)).to_csv(
        supplied_path, index=False
    )
    supplied = validate_feature_cache(
        supplied_path,
        expected_run_ids={"synthetic_train_001"},
        truth_expected=False,
    )
    assert supplied["runs"] == 1
    assert supplied["truth_available"] is False

    controlled_path = tmp_path / "controlled.csv"
    frames = []
    families = {
        "a": "linear_increasing",
        "b": "progressively_increasing",
        "c": "step_like",
        "d": "gamma",
    }
    for run_id, family in families.items():
        frames.extend(_feature_rows(run_id, True, family))
    pd.DataFrame(frames).to_csv(controlled_path, index=False)
    controlled = validate_feature_cache(
        controlled_path,
        expected_run_ids=set(families),
        truth_expected=True,
        expected_families=set(families.values()),
    )
    assert controlled["runs"] == 4
    assert controlled["degradation_families"] == sorted(families.values())


def test_feature_cache_rejects_nonterminal_rul(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame(_feature_rows("run", False))
    frame.loc[frame.index[-1], "rul_norm"] = 0.2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not terminate"):
        validate_feature_cache(path, {"run"}, truth_expected=False)


def test_physics_applicability_is_explicit_and_masked() -> None:
    priors = yaml.safe_load(
        (ROOT / "configs" / "physics_priors.yaml").read_text(encoding="utf-8")
    )
    rows = physics_applicability_rows(priors)
    assert set(rows["prior"]) == set(priors["priors"])
    assert set(rows["missing_input_action"]) == {"mask_prior"}
    paris_ims = rows[
        rows["prior"].eq("paris_crack_growth") & rows["dataset"].eq("ims")
    ].iloc[0]
    assert paris_ims["applicability"] == "unidentifiable_from_supplied_sensors"


def test_exp006_paths_cannot_escape_repository() -> None:
    paths = qualification_paths(ROOT)
    assert paths.config_path == ROOT / "configs" / "exp006_data_qualification.json"
    with pytest.raises(ValueError, match="escapes"):
        qualification_paths(ROOT, "../outside.json")
