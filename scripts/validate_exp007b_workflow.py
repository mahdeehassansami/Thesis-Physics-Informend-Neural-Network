from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thesis_work.exp7b_causal_risk_control import (
    load_exp7b_config,
    validate_exp7b_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.yaml"
NOTEBOOK = ROOT / "notebooks" / "train_models_colab.ipynb"


def main() -> None:
    config = load_exp7b_config(CONFIG)
    qualification = validate_exp7b_config(config, ROOT)
    if qualification["status"] != "qualified":
        raise RuntimeError("EXP-007B frozen cache is not qualified.")
    frame = pd.read_csv(ROOT / config["data"]["feature_cache"])
    test = frame[frame["official_partition"] == "test"]
    development = frame[frame["official_partition"] != "test"]
    if set(test["simulator_seed"].astype(int)) != {920072}:
        raise RuntimeError("EXP-007B fresh test seed changed.")
    if set(development["simulator_seed"].astype(int)) != {420071}:
        raise RuntimeError("EXP-007B development seed changed.")
    if (frame["simulator_seed"].astype(int) == 920071).any():
        raise RuntimeError("The opened EXP-007A test entered EXP-007B.")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    sections = [
        "Mount Drive",
        "Install",
        "Verify the T4",
        "configuration",
        "Train, qualify",
        "Final",
        "Download",
    ]
    positions = [source.index(section) for section in sections]
    if positions != sorted(positions):
        raise RuntimeError("EXP-007B notebook sections are incomplete or out of order.")
    required = [
        "run_exp7b_experiment",
        "validate_exp7b_runtime",
        "expected_commit.txt",
        "nvidia-smi",
        "experiment_outputs_exp007b",
    ]
    missing = [value for value in required if value not in code]
    if missing:
        raise RuntimeError(f"EXP-007B notebook is missing controller calls: {missing}")
    if "class " in code or "def " in code:
        raise RuntimeError("Notebook duplicates implementation logic from src/.")
    print(
        json.dumps(
            {
                "status": "passed_without_full_training",
                "experiment_id": "EXP-007B",
                "cache": qualification,
                "development_runs": int(development["run_id"].nunique()),
                "fresh_test_runs": int(test["run_id"].nunique()),
                "opened_test_seed_excluded": True,
                "notebook_cells": len(notebook["cells"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
