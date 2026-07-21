from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from thesis_work.multi_dataset import SIGNAL_FEATURES


FEATURE_CACHE_REQUIRED_COLUMNS = {
    "dataset",
    "run_id",
    "official_partition",
    "sample_index",
    "elapsed_minutes",
    "elapsed_seconds",
    "rul_minutes",
    "rul_norm",
    "truth_available",
    "degradation_family",
    "fault_location",
    "degradation_value",
    "sampling_hz",
    "load_n",
    "speed_rpm",
    *SIGNAL_FEATURES,
}

SCENARIO_REQUIRED_COLUMNS = {
    "scenario_id",
    "publication_split",
    "condition_id",
    "replicate_within_family",
    "number",
    "simulation_name",
    "BP_name",
    "BP_d",
    "BP_D",
    "BP_n_roller",
    "BP_alpha",
    "BP_C",
    "BP_p",
    "BP_b_form",
    "OC_load_mean",
    "OC_load_std",
    "OC_a_ISO",
    "OC_f_set",
    "OC_f_d",
    "OC_f_m",
    "OC_f_sampling",
    "OC_T_measure_deg",
    "OC_T_measure_acc",
    "SD_degradation_progression",
    "SD_gamma_process_alpha",
    "SD_gamma_process_beta",
    "SD_slip_mean",
    "SD_SDOF_m",
    "SD_SDOF_k",
    "SD_SDOF_c",
    "SD_SNR",
}


@dataclass(frozen=True)
class QualificationPaths:
    project_root: Path
    config_path: Path
    split_path: Path
    priors_path: Path
    scenario_path: Path
    supplied_cache_path: Path
    supplied_metadata_path: Path
    controlled_results_path: Path
    controlled_cache_path: Path
    controlled_metadata_path: Path
    output_dir: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repository_path(root: Path, value: str) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Configured path escapes the repository: {value}") from error
    return path


def qualification_paths(
    project_root: str | Path,
    config_path: str | Path = "configs/exp006_data_qualification.json",
    output_dir: str | Path = "results/analyzed/EXP-006",
) -> QualificationPaths:
    root = Path(project_root).resolve()
    config_file = _resolve_repository_path(root, str(config_path))
    config = _load_json(config_file)
    inputs = config["inputs"]
    derived = config["derived_outputs"]
    return QualificationPaths(
        project_root=root,
        config_path=config_file,
        split_path=_resolve_repository_path(root, inputs["split_file"]),
        priors_path=_resolve_repository_path(root, inputs["physics_prior_file"]),
        scenario_path=_resolve_repository_path(root, inputs["scenario_file"]),
        supplied_cache_path=_resolve_repository_path(
            root, derived["supplied_feature_cache"]
        ),
        supplied_metadata_path=_resolve_repository_path(
            root, derived["supplied_metadata"]
        ),
        controlled_results_path=_resolve_repository_path(
            root, derived["controlled_simulator_results"]
        ),
        controlled_cache_path=_resolve_repository_path(
            root, derived["controlled_feature_cache"]
        ),
        controlled_metadata_path=_resolve_repository_path(
            root, derived["controlled_metadata"]
        ),
        output_dir=_resolve_repository_path(root, str(output_dir)),
    )


def validate_publication_splits(split: dict[str, Any]) -> dict[str, Any]:
    if int(split.get("schema_version", 0)) != 1:
        raise ValueError("publication_data_split.json must use schema_version 1.")
    checked: dict[str, Any] = {}
    for dataset in ("supplied_synthetic_v2", "controlled_synthetic", "pronostia"):
        definition = split[dataset]
        memberships = {
            name: [str(value) for value in definition[f"{name}_runs"]]
            for name in ("train", "validation", "test")
        }
        for name, runs in memberships.items():
            if not runs or len(runs) != len(set(runs)):
                raise ValueError(f"{dataset}.{name}_runs must be nonempty and unique.")
        names = tuple(memberships)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = sorted(set(memberships[left]) & set(memberships[right]))
                if overlap:
                    raise ValueError(
                        f"{dataset} has {left}/{right} overlap: {overlap}"
                    )
        checked[dataset] = {
            "train_runs": len(memberships["train"]),
            "validation_runs": len(memberships["validation"]),
            "test_runs": len(memberships["test"]),
            "total_runs": sum(len(values) for values in memberships.values()),
        }

    if checked["supplied_synthetic_v2"] != {
        "train_runs": 20,
        "validation_runs": 8,
        "test_runs": 12,
        "total_runs": 40,
    }:
        raise ValueError("The supplied synthetic split must be 20/8/12.")
    if checked["controlled_synthetic"] != {
        "train_runs": 24,
        "validation_runs": 8,
        "test_runs": 8,
        "total_runs": 40,
    }:
        raise ValueError("The controlled synthetic split must be 24/8/8.")
    if checked["pronostia"] != {
        "train_runs": 3,
        "validation_runs": 3,
        "test_runs": 11,
        "total_runs": 17,
    }:
        raise ValueError("The PRONOSTIA split must preserve 6 learning and 11 test runs.")
    return checked


def validate_scenario_design(
    frame: pd.DataFrame, split: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    missing = sorted(SCENARIO_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Scenario design is missing columns: {missing}")
    expected_count = int(config["controlled_benchmark"]["expected_scenarios"])
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} scenarios, found {len(frame)}.")
    for column in ("scenario_id", "number", "simulation_name"):
        if frame[column].duplicated().any():
            raise ValueError(f"Scenario column {column} must be unique.")
    if not frame["scenario_id"].astype(str).equals(
        frame["simulation_name"].astype(str)
    ):
        raise ValueError("scenario_id and simulation_name must match exactly.")

    expected_families = set(config["controlled_benchmark"]["progression_families"])
    actual_families = set(frame["SD_degradation_progression"].astype(str))
    if actual_families != expected_families:
        raise ValueError(
            f"Progression families differ: {sorted(actual_families)} versus "
            f"{sorted(expected_families)}"
        )
    counts = frame.groupby(
        ["SD_degradation_progression", "publication_split"]
    ).size()
    expected_split_counts = config["controlled_benchmark"][
        "split_counts_per_family"
    ]
    for family in expected_families:
        for partition, expected in expected_split_counts.items():
            actual = int(counts.get((family, partition), 0))
            if actual != int(expected):
                raise ValueError(
                    f"{family}/{partition} has {actual} rather than {expected} scenarios."
                )

    declared_membership = set()
    for partition in ("train", "validation", "test"):
        expected_ids = set(
            split["controlled_synthetic"][f"{partition}_runs"]
        )
        actual_ids = set(
            frame.loc[
                frame["publication_split"].eq(partition), "scenario_id"
            ].astype(str)
        )
        if actual_ids != expected_ids:
            raise ValueError(
                f"Controlled {partition} scenario IDs do not match the split file."
            )
        declared_membership |= expected_ids
    if declared_membership != set(frame["scenario_id"].astype(str)):
        raise ValueError("Not every controlled scenario is present in the split file.")

    gamma = frame[frame["SD_degradation_progression"].eq("gamma")]
    if gamma[["SD_gamma_process_alpha", "SD_gamma_process_beta"]].isna().any().any():
        raise ValueError("Gamma scenarios require alpha and beta parameters.")
    nongamma = frame[~frame["SD_degradation_progression"].eq("gamma")]
    if nongamma[["SD_gamma_process_alpha", "SD_gamma_process_beta"]].notna().any().any():
        raise ValueError("Non-gamma scenarios must not invent gamma parameters.")

    conditions = []
    for partition, partition_frame in frame.groupby("publication_split", sort=True):
        conditions.append(
            {
                "publication_split": str(partition),
                "load_mean_n": sorted(
                    float(value) for value in partition_frame["OC_load_mean"].unique()
                ),
                "speed_hz": sorted(
                    float(value) for value in partition_frame["OC_f_set"].unique()
                ),
                "snr_db": sorted(
                    float(value) for value in partition_frame["SD_SNR"].unique()
                ),
            }
        )
    return {
        "scenarios": len(frame),
        "families": sorted(actual_families),
        "family_counts": {
            str(key): int(value)
            for key, value in frame["SD_degradation_progression"]
            .value_counts()
            .sort_index()
            .items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in frame["publication_split"]
            .value_counts()
            .sort_index()
            .items()
        },
        "condition_design": conditions,
    }


def _truth_values(series: pd.Series) -> set[bool]:
    if pd.api.types.is_bool_dtype(series):
        return {bool(value) for value in series.dropna().unique()}
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized truth_available values: {unknown}")
    return {mapping[value] for value in normalized}


def validate_feature_cache(
    path: str | Path,
    expected_run_ids: set[str],
    truth_expected: bool,
    expected_families: set[str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_csv(path)
    missing = sorted(FEATURE_CACHE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing feature-cache columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    if frame[["run_id", "sample_index"]].duplicated().any():
        raise ValueError(f"{path} contains duplicate run/sample identifiers.")
    actual_runs = set(frame["run_id"].astype(str))
    if actual_runs != expected_run_ids:
        missing_runs = sorted(expected_run_ids - actual_runs)
        extra_runs = sorted(actual_runs - expected_run_ids)
        raise ValueError(
            f"{path} run IDs differ; missing={missing_runs}, extra={extra_runs}."
        )
    if _truth_values(frame["truth_available"]) != {truth_expected}:
        raise ValueError(f"{path} has incorrect truth availability flags.")

    numeric_features = frame[SIGNAL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_features).all():
        raise ValueError(f"{path} contains non-finite signal features.")

    short_runs: list[str] = []
    snapshot_counts: dict[str, int] = {}
    for run_id, run in frame.groupby("run_id", sort=True):
        ordered = run.sort_values("sample_index")
        indices = ordered["sample_index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(len(ordered))):
            raise ValueError(f"{run_id} sample indices are not contiguous from zero.")
        elapsed = ordered["elapsed_seconds"].to_numpy(dtype=float)
        if len(elapsed) > 1 and not np.all(np.diff(elapsed) > 0):
            raise ValueError(f"{run_id} elapsed time is not strictly increasing.")
        rul = ordered["rul_norm"].to_numpy(dtype=float)
        if not np.isclose(rul[0], 1.0, atol=1e-10):
            raise ValueError(f"{run_id} does not begin at normalized RUL 1.")
        if not np.isclose(rul[-1], 0.0, atol=1e-10):
            raise ValueError(f"{run_id} does not terminate at normalized RUL 0.")
        if np.any(np.diff(rul) > 1e-10):
            raise ValueError(f"{run_id} normalized RUL is not nonincreasing.")
        if len(ordered) <= 8:
            short_runs.append(str(run_id))
        snapshot_counts[str(run_id)] = len(ordered)

    families: list[str] = []
    if truth_expected:
        if frame["degradation_value"].isna().any():
            raise ValueError("Controlled truth must include every degradation value.")
        families = sorted(
            value
            for value in frame["degradation_family"].dropna().astype(str).unique()
            if value
        )
        if expected_families is not None and set(families) != expected_families:
            raise ValueError(
                f"Controlled feature families {families} do not match "
                f"{sorted(expected_families)}."
            )
    else:
        disclosed = frame["degradation_family"].dropna().astype(str).str.strip()
        if disclosed.ne("").any():
            raise ValueError("Supplied v2 cache must not infer withheld progression labels.")
        if frame["degradation_value"].notna().any():
            raise ValueError("Supplied v2 cache must not invent hidden degradation values.")

    values = list(snapshot_counts.values())
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "runs": len(snapshot_counts),
        "snapshots": len(frame),
        "minimum_snapshots_per_run": min(values),
        "maximum_snapshots_per_run": max(values),
        "runs_not_supporting_sequence_length_8": short_runs,
        "truth_available": truth_expected,
        "degradation_families": families,
    }


def physics_applicability_rows(priors: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prior_name, prior in priors["priors"].items():
        required = prior.get("required_observations", [])
        for dataset, status in prior["applicability"].items():
            rows.append(
                {
                    "prior": prior_name,
                    "tier": prior["tier"],
                    "dataset": dataset,
                    "applicability": status,
                    "required_observation_count": len(required),
                    "required_observations": "|".join(required),
                    "missing_input_action": priors["availability_policy"][
                        "missing_required_input_action"
                    ],
                    "fallback": priors["availability_policy"]["fallback"],
                }
            )
    return pd.DataFrame(rows).sort_values(["prior", "dataset"]).reset_index(drop=True)


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _relative_inventory(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required EXP-006 file is missing: {path}")
        inventory[name] = {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def run_qualification(paths: QualificationPaths) -> dict[str, Any]:
    config = _load_json(paths.config_path)
    split = _load_json(paths.split_path)
    priors = yaml.safe_load(paths.priors_path.read_text(encoding="utf-8"))
    scenarios = pd.read_csv(paths.scenario_path)

    split_summary = validate_publication_splits(split)
    scenario_summary = validate_scenario_design(scenarios, split, config)

    supplied_expected = set()
    supplied_definition = split["supplied_synthetic_v2"]
    for partition in ("train", "validation", "test"):
        supplied_expected.update(supplied_definition[f"{partition}_runs"])
    supplied_summary = validate_feature_cache(
        paths.supplied_cache_path,
        expected_run_ids=supplied_expected,
        truth_expected=False,
    )

    controlled_expected = set(scenarios["scenario_id"].astype(str))
    controlled_summary = validate_feature_cache(
        paths.controlled_cache_path,
        expected_run_ids=controlled_expected,
        truth_expected=True,
        expected_families=set(
            config["controlled_benchmark"]["progression_families"]
        ),
    )

    input_paths = {
        "supplied_data": _resolve_repository_path(
            paths.project_root, config["inputs"]["supplied_data"]
        ),
        "supplied_documentation": _resolve_repository_path(
            paths.project_root, config["inputs"]["supplied_documentation"]
        ),
        "simulator_archive": _resolve_repository_path(
            paths.project_root, config["inputs"]["simulator_archive"]
        ),
        "simulator_documentation": _resolve_repository_path(
            paths.project_root, config["inputs"]["simulator_documentation"]
        ),
        "experiment_config": paths.config_path,
        "publication_split": paths.split_path,
        "physics_priors": paths.priors_path,
        "controlled_scenarios": paths.scenario_path,
        "supplied_metadata": paths.supplied_metadata_path,
        "controlled_metadata": paths.controlled_metadata_path,
        "supplied_feature_cache": paths.supplied_cache_path,
        "controlled_feature_cache": paths.controlled_cache_path,
    }
    inventory = _relative_inventory(paths.project_root, input_paths)

    simulator_manifests = sorted(
        paths.controlled_results_path.rglob("exp006_simulator_run.json")
    )
    if len(simulator_manifests) != 1:
        raise ValueError(
            "Expected exactly one controlled simulator run manifest under "
            f"{paths.controlled_results_path}, found {len(simulator_manifests)}."
        )
    simulator_run = _load_json(simulator_manifests[0])
    if bool(simulator_run.get("smoke_only")):
        raise ValueError("The final EXP-006 qualification cannot use only a smoke run.")
    if int(simulator_run["scenario_count"]) != 40:
        raise ValueError("The controlled simulator run must include all 40 scenarios.")
    if int(simulator_run["seed"]) != int(
        config["randomness"]["scenario_generation_seed"]
    ):
        raise ValueError("The controlled simulator seed differs from the configuration.")

    applicability = physics_applicability_rows(priors)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    applicability_path = paths.output_dir / "physics_applicability.csv"
    applicability.to_csv(applicability_path, index=False)

    dataset_manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-006",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(paths.project_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matlab": simulator_run["matlab_version"],
        },
        "inventory": inventory,
        "split_summary": split_summary,
        "scenario_summary": scenario_summary,
        "supplied_cache": supplied_summary,
        "controlled_cache": controlled_summary,
        "simulator_run_manifest": {
            "relative_path": simulator_manifests[0]
            .relative_to(paths.project_root)
            .as_posix(),
            "sha256": sha256_file(simulator_manifests[0]),
            "content": simulator_run,
        },
        "split_sha256": inventory["publication_split"]["sha256"],
        "physics_priors_sha256": inventory["physics_priors"]["sha256"],
        "scenario_sha256": inventory["controlled_scenarios"]["sha256"],
    }
    manifest_path = paths.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    criteria = [
        {
            "criterion": "Supplied v2 28/12 partition and fingerprint verified",
            "passed": supplied_summary["runs"] == 40,
            "evidence": (
                f"40 runs, {supplied_summary['snapshots']} snapshots, "
                f"SHA-256 {supplied_summary['sha256']}"
            ),
        },
        {
            "criterion": "Supplied hidden progression/fault labels not invented",
            "passed": not supplied_summary["truth_available"],
            "evidence": "Documentation withholds both labels; exported truth_available=false.",
        },
        {
            "criterion": "Official simulator executed all controlled scenarios",
            "passed": int(simulator_run["scenario_count"]) == 40,
            "evidence": (
                f"MATLAB {simulator_run['matlab_version']}; seed "
                f"{simulator_run['seed']}; {simulator_run['elapsed_seconds']:.3f} s"
            ),
        },
        {
            "criterion": "Controlled truth is complete for four progression families",
            "passed": controlled_summary["truth_available"]
            and len(controlled_summary["degradation_families"]) == 4,
            "evidence": (
                f"{controlled_summary['runs']} runs, "
                f"{controlled_summary['snapshots']} snapshots, families "
                f"{controlled_summary['degradation_families']}"
            ),
        },
        {
            "criterion": "Immutable complete-trajectory splits pass",
            "passed": True,
            "evidence": json.dumps(split_summary, sort_keys=True),
        },
        {
            "criterion": "No neural training occurred",
            "passed": config["experiment"]["neural_training"] is False,
            "evidence": "EXP-006 contains MATLAB simulation, feature extraction, and validation only.",
        },
    ]
    criteria_frame = pd.DataFrame(criteria)
    criteria_path = paths.output_dir / "success_criteria.csv"
    criteria_frame.to_csv(criteria_path, index=False)
    all_passed = bool(criteria_frame["passed"].all())

    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-006",
        "status": "completed" if all_passed else "partial",
        "all_success_criteria_passed": all_passed,
        "central_finding": (
            "The supplied v2 data support RUL evaluation but not known-truth physics "
            "applicability because progression and fault labels are intentionally withheld. "
            "The official MATLAB simulator supplies those labels for the new controlled set."
        ),
        "ansys_required_for_next_stage": False,
        "ansys_decision": (
            "Not required for EXP-007 generator-family credibility. Consider ANSYS later "
            "only for an independently parameterized contact/crack mechanics validation."
        ),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "physics_applicability_sha256": sha256_file(applicability_path),
        "success_criteria_sha256": sha256_file(criteria_path),
    }
    summary_path = paths.output_dir / "qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write_reports(
        paths=paths,
        summary=summary,
        dataset_manifest=dataset_manifest,
        applicability=applicability,
    )
    return summary


def _write_reports(
    paths: QualificationPaths,
    summary: dict[str, Any],
    dataset_manifest: dict[str, Any],
    applicability: pd.DataFrame,
) -> None:
    supplied = dataset_manifest["supplied_cache"]
    controlled = dataset_manifest["controlled_cache"]
    simulator = dataset_manifest["simulator_run_manifest"]["content"]
    analysis = f"""# EXP-006 data and physics-identifiability qualification

Status: **{summary['status']}**

## Outcome

EXP-006 passed its qualification gates without neural-network training. The supplied
Bearings with Varying Degradation Behaviors v2 file was preserved unchanged and exported
to a compact derived feature cache. It contains {supplied['runs']} trajectories and
{supplied['snapshots']} vibration snapshots.

The supplied dataset cannot be used as known-truth evidence for physics applicability.
Its documentation intentionally withholds the degradation progression and fault type, and
the exported cache therefore records `truth_available=false` rather than inferring labels.

The official CC BY 4.0 MATLAB simulator ran all {controlled['runs']} predeclared controlled
scenarios with seed {simulator['seed']} on {simulator['matlab_version']}. The controlled
cache contains {controlled['snapshots']} snapshots and retains the progression family,
hidden degradation value, fault location, bearing parameters, operating conditions, and
simulation details. This is the known-truth benchmark for EXP-007.

## Important data limitation

The supplied v2 data contain trajectories with as few as
{supplied['minimum_snapshots_per_run']} snapshots. The frozen sequence length of eight from
Runs 4/5 cannot create samples for these runs. EXP-007 must declare a controlled sequence
policy or a causal variable-length model; it may not silently discard the short lives.

## Physics conclusion

Only the simulator progression family is known truth in the controlled benchmark. Paris
crack growth, ISO 281/L10-Miner, and temperature-lubrication terms remain conditional or
unidentified for the current real datasets. A low residual to those equations must not be
reported as proof that their physical assumptions are valid.

## ANSYS decision

ANSYS is not required for EXP-007, whose falsifiable target is progression-family
applicability and negative-transfer prevention. It may add a later, independent validation
layer if we design geometry/load-specific contact stress and crack-growth simulations with
measured units and a simulation-to-real gap analysis.
"""
    (paths.output_dir / "analysis.md").write_text(analysis, encoding="utf-8")

    issues = f"""# EXP-006 issues and limitations

- Supplied v2 progression-family and fault labels are intentionally unavailable.
- {len(supplied['runs_not_supporting_sequence_length_8'])} supplied trajectories have at
  most eight snapshots and cannot support the old fixed sequence construction.
- The simulator is distributed primarily as MATLAB P-code. Its inputs and outputs are
  auditable, but internal implementation lines cannot be independently inspected.
- Simulator truth establishes controlled synthetic validity, not real bearing-mechanics
  validity or a resolved simulation-to-real domain gap.
- The dataset documentation describes dynamic load rating as 32,000 N, while the stored
  supplied structures and official template use 32,500 N. Both values must remain visible;
  do not silently reconcile them.
"""
    (paths.output_dir / "issues.md").write_text(issues, encoding="utf-8")

    masked = applicability[
        applicability["applicability"].astype(str).str.contains(
            "unavailable|unsupported|unidentifiable|not_observed", case=False, regex=True
        )
    ]
    recommendations = f"""# EXP-006 recommendations

1. Proceed to EXP-007 synthetic credibility feasibility using the controlled cache and its
   immutable 24/8/8 split.
2. Pair every controlled trajectory with its correct progression prior and predeclared
   wrong-family/parameter corruptions; keep ordinary operation/noise shift distinct from
   invalid physics.
3. Resolve the short-trajectory policy on validation data before model training.
4. Keep the supplied v2 dataset as a separate RUL generalization benchmark, never as
   physics-validity ground truth.
5. Mask unavailable conditional physics. The applicability table contains {len(masked)}
   dataset/prior combinations explicitly marked unavailable, unsupported, unidentifiable,
   or unobserved.
6. Defer ANSYS until the progression-credibility mechanism passes the EXP-007 synthetic
   discrimination gate.
"""
    (paths.output_dir / "recommendations.md").write_text(
        recommendations, encoding="utf-8"
    )
