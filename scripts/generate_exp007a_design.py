from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "configs" / "exp007a_multicondition_scenarios.csv"
SPLIT_PATH = ROOT / "configs" / "exp007a_data_split.json"

FAMILIES = {
    "linear": "linear_increasing",
    "progressive": "progressively_increasing",
    "step": "step_like",
    "gamma": "gamma",
}

TRAIN_CONDITIONS = [
    (3800, 80, 42, 6),
    (3800, 120, 48, 10),
    (3800, 160, 54, 14),
    (3800, 200, 60, 8),
    (4600, 120, 42, 14),
    (4600, 160, 48, 8),
    (4600, 200, 54, 6),
    (4600, 80, 60, 10),
    (5400, 160, 42, 8),
    (5400, 200, 48, 14),
    (5400, 80, 54, 10),
    (5400, 120, 60, 6),
    (6200, 200, 42, 10),
    (6200, 80, 48, 6),
    (6200, 120, 54, 8),
    (6200, 160, 60, 14),
]

VALIDATION_CONDITIONS = [
    (4200, 100, 45, 7),
    (5000, 140, 51, 11),
    (5800, 180, 57, 13),
    (5000, 180, 45, 9),
]

TEST_CONDITIONS = [
    (4400, 110, 47, 8),
    (5200, 150, 53, 12),
    (6000, 190, 59, 7),
    (5600, 110, 47, 13),
]

FIELDNAMES = [
    "scenario_id",
    "publication_split",
    "condition_id",
    "replicate_within_family",
    "simulator_seed",
    "sealed_test",
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
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_id(partition: str, values: tuple[int, int, int, int]) -> str:
    load, load_std, speed, snr = values
    return f"{partition}_l{load}_ls{load_std}_s{speed}_n{snr}"


def _row(
    *,
    family_short: str,
    family: str,
    partition: str,
    replicate: int,
    number: int,
    condition: tuple[int, int, int, int],
) -> dict[str, Any]:
    scenario_id = f"pca_{family_short}_{partition}_{replicate:02d}"
    load, load_std, speed, snr = condition
    gamma_alpha = 30 if family == "gamma" and replicate % 2 else 45
    return {
        "scenario_id": scenario_id,
        "publication_split": partition,
        "condition_id": _condition_id(partition, condition),
        "replicate_within_family": replicate,
        "simulator_seed": 920071 if partition == "test" else 420071,
        "sealed_test": str(partition == "test").lower(),
        "number": number,
        "simulation_name": scenario_id,
        "BP_name": "NU204-E-XL-TVP2",
        "BP_d": 7.5,
        "BP_D": 34,
        "BP_n_roller": 12,
        "BP_alpha": 0,
        "BP_C": 32500,
        "BP_p": 3.3333333333333335,
        "BP_b_form": 1.35,
        "OC_load_mean": load,
        "OC_load_std": load_std,
        "OC_a_ISO": 1,
        "OC_f_set": speed,
        "OC_f_d": round(0.8 + 0.2 * ((replicate - 1) % 4), 1),
        "OC_f_m": 1,
        "OC_f_sampling": 15626,
        "OC_T_measure_deg": 10080,
        "OC_T_measure_acc": 1,
        "SD_degradation_progression": family,
        "SD_gamma_process_alpha": gamma_alpha if family == "gamma" else "",
        "SD_gamma_process_beta": 10 if family == "gamma" else "",
        # The official documentation gives an interpretation but no supported range.
        # Keep the 1% value that completed all EXP-006 trajectories; condition diversity
        # comes from load, speed, noise, load variation, and frequency modulation.
        "SD_slip_mean": 0.01,
        "SD_SDOF_m": 15000,
        "SD_SDOF_k": 3.5e12,
        "SD_SDOF_c": 1.0e7,
        "SD_SNR": snr,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    number = 1
    partition_conditions = {
        "train": TRAIN_CONDITIONS,
        "validation": VALIDATION_CONDITIONS,
        "test": TEST_CONDITIONS,
    }
    for family_short, family in FAMILIES.items():
        for partition, conditions in partition_conditions.items():
            for replicate, condition in enumerate(conditions, start=1):
                rows.append(
                    _row(
                        family_short=family_short,
                        family=family,
                        partition=partition,
                        replicate=replicate,
                        number=number,
                        condition=condition,
                    )
                )
                number += 1
    return rows


def main() -> None:
    rows = build_rows()
    if len(rows) != 96:
        raise RuntimeError(f"Expected 96 scenarios, found {len(rows)}.")
    identities = [row["scenario_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("EXP-007A scenario identifiers are not unique.")
    SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCENARIO_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    split = {
        "schema_version": 1,
        "created_for": "EXP-007A",
        "design_policy": (
            "Complete simulator trajectories; multi-condition source support; separate "
            "development and fresh sealed-test simulator RNG seeds."
        ),
        "scenario_file": SCENARIO_PATH.relative_to(ROOT).as_posix(),
        "scenario_sha256": _sha256(SCENARIO_PATH),
        "development_simulator_seed": 420071,
        "sealed_test_simulator_seed": 920071,
        "train_runs": [row["scenario_id"] for row in rows if row["publication_split"] == "train"],
        "validation_runs": [
            row["scenario_id"] for row in rows if row["publication_split"] == "validation"
        ],
        "test_runs": [row["scenario_id"] for row in rows if row["publication_split"] == "test"],
        "sealed_test": True,
        "test_access": "evaluation_only_after_development_target_and_method_freeze",
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scenario_file": str(SCENARIO_PATH),
                "scenario_sha256": split["scenario_sha256"],
                "split_file": str(SPLIT_PATH),
                "counts": {
                    "train": len(split["train_runs"]),
                    "validation": len(split["validation_runs"]),
                    "test": len(split["test_runs"]),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
