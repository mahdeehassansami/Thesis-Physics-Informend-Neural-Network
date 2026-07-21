from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from generate_exp007a_design import FIELDNAMES, build_rows


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "configs" / "exp007b_multicondition_scenarios.csv"
SPLIT_PATH = ROOT / "configs" / "exp007b_data_split.json"
DEVELOPMENT_SEED = 420071
FRESH_TEST_SEED = 920072


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = build_rows()
    for row in rows:
        if row["publication_split"] == "test":
            row["simulator_seed"] = FRESH_TEST_SEED
            row["sealed_test"] = "true"
        else:
            row["simulator_seed"] = DEVELOPMENT_SEED
            row["sealed_test"] = "false"
    if len(rows) != 96 or len({row["scenario_id"] for row in rows}) != 96:
        raise RuntimeError("EXP-007B requires 96 unique frozen scenarios.")
    SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCENARIO_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    scenario_sha = _sha256(SCENARIO_PATH)
    split = {
        "schema_version": 1,
        "created_for": "EXP-007B",
        "design_policy": (
            "Reuse the exact EXP-007A development trajectories; replace the opened EXP-007A "
            "test population with a separately generated fresh sealed simulator seed."
        ),
        "scenario_file": SCENARIO_PATH.relative_to(ROOT).as_posix(),
        "scenario_sha256": scenario_sha,
        "development_simulator_seed": DEVELOPMENT_SEED,
        "sealed_test_simulator_seed": FRESH_TEST_SEED,
        "exp007a_opened_test_seed_excluded": 920071,
        "train_runs": [row["scenario_id"] for row in rows if row["publication_split"] == "train"],
        "validation_runs": [
            row["scenario_id"] for row in rows if row["publication_split"] == "validation"
        ],
        "test_runs": [row["scenario_id"] for row in rows if row["publication_split"] == "test"],
        "sealed_test": True,
        "test_access": "evaluation_only_after_development_target_and_selector_qualification",
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scenario_file": str(SCENARIO_PATH),
                "scenario_sha256": scenario_sha,
                "split_file": str(SPLIT_PATH),
                "development_seed": DEVELOPMENT_SEED,
                "fresh_test_seed": FRESH_TEST_SEED,
                "counts": {
                    partition: sum(row["publication_split"] == partition for row in rows)
                    for partition in ("train", "validation", "test")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
