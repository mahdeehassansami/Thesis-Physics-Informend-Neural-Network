from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from thesis_work.multi_dataset import (
    MODEL_FEATURES,
    PHYSICS_COLUMNS,
    SIGNAL_FEATURES,
    prepare_sequence_dataset,
)
from thesis_work.sequence_models import build_model, calculate_loss


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for run_number, run_id in enumerate(("train", "validation", "test"), start=1):
        for index in range(12):
            elapsed_norm = index / 11
            row = {
                "dataset": "smoke",
                "run_id": run_id,
                "sample_index": index,
                "elapsed_seconds": float(index * 60),
                "elapsed_norm": elapsed_norm,
                "rul_norm": 1.0 - elapsed_norm,
                "health_indicator": elapsed_norm,
                "temperature_c": 35.0 + 20.0 * elapsed_norm,
                "ambient_temperature_c": 25.0,
                "temperature_delta_c": 10.0 + 20.0 * elapsed_norm,
                "load_n": 4000.0,
                "speed_rpm": 1800.0,
                "contact_pressure_mpa": 200.0,
                "dynamic_capacity_n": 30000.0,
                "fatigue_limit_n": 1800.0,
                "viscosity_ref_cst": 80.0,
                "viscosity_required_cst": 25.0,
                "contamination_factor": 0.8,
                "cycles_per_time_unit": 400000.0,
                "temperature_available": 1.0,
                "load_available": 1.0,
                "contact_pressure_available": 1.0,
            }
            for feature_number, name in enumerate(SIGNAL_FEATURES, start=1):
                row[name] = (
                    0.1 * feature_number
                    + elapsed_norm
                    + 0.01 * run_number
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    missing = set(MODEL_FEATURES + PHYSICS_COLUMNS) - set(frame.columns)
    assert not missing, missing
    return frame


def validate_notebooks() -> None:
    for relative in (
        "Thesis_v3_with_extra_graphs_tables.ipynb",
        "notebooks/train_models_colab.ipynb",
    ):
        notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) >= 10
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{relative}:cell-{index}")


def validate_models() -> None:
    config = json.loads(
        (ROOT / "configs" / "colab_experiments.json").read_text(encoding="utf-8")
    )
    dataset_config = {
        "name": "smoke",
        "split": {
            "strategy": "run_ids",
            "train_runs": ["train"],
            "validation_runs": ["validation"],
            "test_runs": ["test"],
        },
    }
    prepared = prepare_sequence_dataset(
        synthetic_frame(), dataset_config, sequence_length=3
    )
    batch = next(iter(DataLoader(prepared.train, batch_size=4, shuffle=False)))
    feature_indices = {
        name: index for index, name in enumerate(prepared.feature_columns)
    }
    profiles = {
        "fnn": "data_only",
        "cnn": "data_only",
        "lstm": "data_only",
        "attnpinn": "attnpinn_medium",
        "weak_pinn": "weak_medium",
        "strong_pinn": "strong_medium",
    }
    for name, profile in profiles.items():
        model = build_model(
            name,
            input_dim=len(prepared.feature_columns),
            model_config=config["models"][name],
        )
        model.train()
        total, components, prediction = calculate_loss(
            model=model,
            batch=batch,
            weights=config["weight_profiles"][profile],
            feature_indices=feature_indices,
            physics=config["physics"],
        )
        assert prediction.shape == batch["target"].shape
        assert torch.isfinite(total), (name, components)
        total.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ), name
        print(name, float(total.detach()), sorted(components))


if __name__ == "__main__":
    validate_notebooks()
    validate_models()
    print("Colab workflow validation passed.")
