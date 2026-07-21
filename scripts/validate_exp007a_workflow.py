from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thesis_work.exp7a_harm_credibility import (
    FAMILIES,
    _fit_data_only,
    _fit_degradation_proxy,
    _fit_physics_intervention,
    _predict_model,
    build_counterfactual_evidence,
    candidate_specs,
    load_exp7a_config,
    validate_exp7a_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.yaml"
NOTEBOOK = ROOT / "notebooks" / "train_models_colab.ipynb"


def fixture(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for replicate in range(1, 4):
            run_id = f"smoke_{family_index}_{replicate}"
            for sample in range(10):
                lifecycle = sample / 9.0
                damage = {
                    "linear_increasing": lifecycle,
                    "progressively_increasing": lifecycle**2,
                    "step_like": np.floor(lifecycle * 4.0) / 4.0,
                    "gamma": lifecycle**1.5,
                }[family]
                row = {
                    "run_id": run_id,
                    "official_partition": "train" if replicate < 3 else "validation",
                    "condition_id": f"smoke_condition_{replicate}",
                    "sample_index": sample,
                    "elapsed_minutes": sample * 8.0,
                    "rul_norm": 1.0 - lifecycle,
                    "degradation_family": family,
                    "degradation_value": damage,
                    "load_n": 4000.0 + replicate * 500.0,
                    "speed_rpm": 2700.0 + replicate * 180.0,
                    "snr_db": 6.0 + replicate * 2.0,
                }
                for index, column in enumerate(
                    config["data"]["signal_feature_columns"], start=1
                ):
                    row[column] = index + (family_index + 1) * damage + 0.02 * replicate
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_exp7a_config(CONFIG)
    qualification = validate_exp7a_config(config, ROOT)
    if qualification["status"] != "qualified":
        raise RuntimeError("Expected the frozen EXP-007A cache to be qualified.")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    required_sections = [
        "Mount Drive",
        "Install",
        "Verify the T4",
        "configuration",
        "Train, qualify",
        "Final",
        "Download",
    ]
    positions = [source.index(section) for section in required_sections]
    if positions != sorted(positions):
        raise RuntimeError("EXP-007A notebook sections are incomplete or out of order.")
    if "run_exp7a_experiment" not in code or "nvidia-smi" not in code:
        raise RuntimeError("EXP-007A notebook is not a thin complete controller.")
    install_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and '"pip", "install"' in "".join(cell.get("source", []))
    ]
    if (
        len(install_cells) != 1
        or "cwd=CLONE" not in install_cells[0]
        or '"pip", "install", "-q"' in install_cells[0]
    ):
        raise RuntimeError("EXP-007A pip installation is not visible and clone-relative.")
    if "class " in code or "def " in code:
        raise RuntimeError("Notebook duplicates Python implementation logic.")

    smoke = copy.deepcopy(config)
    smoke["backbone"].update(
        {"hidden_dim": 8, "epochs": 1, "patience": 1, "batch_size": 32}
    )
    smoke["physics_intervention"].update({"fine_tune_epochs": 1, "patience": 1})
    smoke["degradation_proxy"]["estimators"] = 8
    frame = fixture(smoke)
    train = frame[frame["official_partition"] == "train"].copy()
    validation = frame[frame["official_partition"] == "validation"].copy()
    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp007a_smoke_", dir=tmp_root) as temporary:
        artifact = Path(temporary)
        parent = _fit_data_only(train, validation, smoke, 42, 42, artifact / "data", "smoke")
        candidate = candidate_specs(smoke)[0]
        physics = _fit_physics_intervention(
            parent,
            train,
            validation,
            candidate,
            smoke,
            artifact / "physics",
            "smoke",
            43,
        )
        device = torch.device("cpu")
        parent.model.to(device)
        physics.model.to(device)
        data_prediction = _predict_model(parent, validation, smoke, device)
        physics_prediction = _predict_model(physics, validation, smoke, device).rename(
            columns={"predicted_rul": "physics_rul"}
        )
        physics_prediction["candidate_spec"] = candidate["candidate_spec"]
        physics_prediction["candidate_family"] = candidate["candidate_family"]
        physics_prediction["time_scale_factor"] = candidate["time_scale_factor"]
        proxy = _fit_degradation_proxy(train, smoke, 42)
        evidence = build_counterfactual_evidence(
            data_prediction,
            physics_prediction,
            validation,
            parent.context,
            proxy,
            smoke,
            "validation",
        )
        if not np.isfinite(
            evidence[
                smoke["credibility"]["numeric_evidence"]
                + ["data_only_rmse", "physics_rmse", "physics_regret"]
            ].to_numpy(dtype=float)
        ).all():
            raise RuntimeError("Tiny EXP-007A intervention smoke produced non-finite values.")
        physics_history = pd.read_csv(artifact / "physics" / "history.csv")
        if physics_history["physics_value_loss"].isna().any():
            raise RuntimeError("Physics intervention smoke did not record separate physics loss.")

    print(
        json.dumps(
            {
                "status": "passed_without_full_training",
                "design": qualification,
                "tiny_smoke_rows": len(evidence),
                "tiny_smoke_epochs": 1,
                "tiny_smoke_metrics_are_experiment_results": False,
                "notebook_cells": len(notebook["cells"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
