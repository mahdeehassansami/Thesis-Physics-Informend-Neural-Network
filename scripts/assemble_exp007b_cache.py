from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = ROOT / "data" / "processed_features" / "publication" / "exp007a" / "multicondition_features.csv"
DEFAULT_FRESH_TEST = ROOT / "saved results" / "run_07b" / "fresh_test_cache" / "fresh_test_features.csv"
DEFAULT_FRESH_METADATA = ROOT / "saved results" / "run_07b" / "fresh_test_cache" / "fresh_test_metadata.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed_features" / "publication" / "exp007b" / "multicondition_features.csv"
PREREGISTRATION_COMMIT = "b8e4b1e18845e7056fd70c6956426483360975f3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the frozen EXP-007B development/fresh-test cache.")
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--fresh-test", type=Path, default=DEFAULT_FRESH_TEST)
    parser.add_argument("--fresh-metadata", type=Path, default=DEFAULT_FRESH_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    development_path = args.development.resolve()
    fresh_test_path = args.fresh_test.resolve()
    fresh_metadata_path = args.fresh_metadata.resolve()
    output_path = args.output.resolve()
    scenario_path = ROOT / "configs" / "exp007b_multicondition_scenarios.csv"
    split_path = ROOT / "configs" / "exp007b_data_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha256_file(scenario_path) != split["scenario_sha256"]:
        raise ValueError("EXP-007B scenario hash differs from its preregistration.")

    old = pd.read_csv(development_path)
    development = old[old["official_partition"].isin(["train", "validation"])].copy()
    if set(development["simulator_seed"].astype(int)) != {420071}:
        raise ValueError("EXP-007B development rows must be the frozen seed-420071 cache.")
    if (old["official_partition"] == "test").sum() == 0:
        raise ValueError("Source cache identity is incomplete; expected the excluded EXP-007A test rows.")
    fresh = pd.read_csv(fresh_test_path)
    if set(fresh["official_partition"]) != {"test"} or set(fresh["simulator_seed"].astype(int)) != {920072}:
        raise ValueError("Fresh test cache is not the sealed seed-920072 population.")
    if set(development.columns) != set(fresh.columns):
        missing_left = sorted(set(fresh.columns) - set(development.columns))
        missing_right = sorted(set(development.columns) - set(fresh.columns))
        raise ValueError(f"Cache schemas differ: missing development={missing_left}, missing test={missing_right}")
    fresh = fresh[development.columns]
    combined = pd.concat([development, fresh], ignore_index=True)
    combined["dataset"] = "exp007b_multicondition_synthetic"
    combined = combined.sort_values(["official_partition", "run_id", "sample_index"]).reset_index(drop=True)
    expected = {
        run_id: partition
        for partition in ("train", "validation", "test")
        for run_id in split[f"{partition}_runs"]
    }
    observed = dict(combined[["run_id", "official_partition"]].drop_duplicates().itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("Assembled cache membership differs from the preregistered split.")
    if combined[["run_id", "sample_index"]].duplicated().any():
        raise ValueError("Assembled cache contains duplicate sample identities.")
    if set(combined.loc[combined["official_partition"] == "test", "simulator_seed"].astype(int)) != {920072}:
        raise ValueError("The opened EXP-007A test population entered EXP-007B.")

    fresh_metadata = json.loads(fresh_metadata_path.read_text(encoding="utf-8"))
    if fresh_metadata.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("Fresh test metadata does not identify the preregistration commit.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    combined.to_csv(temporary, index=False)
    temporary.replace(output_path)
    run_sizes = combined.groupby("run_id").size()
    metadata = {
        "schema_version": 1,
        "experiment_id": "EXP-007B",
        "protocol_version": "0.3.0",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "dataset_id": "exp007b_multicondition_synthetic",
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": int(combined["run_id"].nunique()),
        "snapshot_count": int(len(combined)),
        "minimum_snapshots_per_run": int(run_sizes.min()),
        "maximum_snapshots_per_run": int(run_sizes.max()),
        "progression_families": sorted(combined["degradation_family"].unique().tolist()),
        "development_simulator_seed": 420071,
        "sealed_test_simulator_seed": 920072,
        "opened_exp007a_test_seed_excluded": 920071,
        "sealed_test_generated_separately": True,
        "development_source_csv": str(development_path),
        "development_source_sha256": sha256_file(development_path),
        "fresh_test_source_csv": str(fresh_test_path),
        "fresh_test_source_sha256": sha256_file(fresh_test_path),
        "fresh_test_metadata_sha256": sha256_file(fresh_metadata_path),
        "scenario_sha256": split["scenario_sha256"],
        "split_sha256": sha256_file(split_path),
        "degradation_family_disclosed": True,
        "physics_truth_available": True,
        "output_columns": combined.columns.tolist(),
        "source_simulator_license": "CC BY 4.0",
        "source_simulator_citation": (
            "Mauthe, Hagmeyer, and Zeiler (2025), DOI "
            "10.3850/978-981-94-3281-3_ESREL-SRA-E2025-P8028-cd"
        ),
    }
    metadata_path = output_path.with_name("multicondition_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "feature_sha256": sha256_file(output_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "rows": len(combined),
                "runs": combined["run_id"].nunique(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
