from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from thesis_work.exp7_credibility import (
    BackboneFit,
    FAMILIES,
    _credibility_matrix,
    _fit_degradation_proxy,
    _load_split,
    _sequence_arrays,
    build_credibility_evidence,
    fit_template_library,
    load_exp7_config,
    validate_exp7_config,
)
from thesis_work.sequence_models import LSTMRUL


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
NOTEBOOK_PATH = ROOT / "notebooks" / "train_models_colab.ipynb"


def main() -> None:
    config = load_exp7_config(CONFIG_PATH)
    qualification = validate_exp7_config(config, ROOT)
    split = _load_split(config, ROOT)
    frame = pd.read_csv(qualification["feature_path"])
    columns = config["data"]["feature_columns"]
    train = frame[frame["run_id"].isin(split["train_runs"])].copy()
    validation = frame[frame["run_id"].isin(split["validation_runs"])].copy()
    test = frame[frame["run_id"].isin(split["test_runs"])].copy()

    scaler = StandardScaler().fit(train[columns].to_numpy(dtype=float))
    time_scale = max(float(train["elapsed_minutes"].max()), 1.0)
    arrays = {
        name: _sequence_arrays(
            subset,
            scaler,
            columns,
            int(config["data"]["sequence_length"]),
            int(config["data"]["sequence_stride"]),
            time_scale,
        )
        for name, subset in (("train", train), ("validation", validation), ("test", test))
    }
    assert all(len(value["target"]) > 0 for value in arrays.values())
    assert all(value["x"].shape[1:] == (8, len(columns)) for value in arrays.values())
    assert set(arrays["test"]["run_id"]) == set(split["test_runs"])

    model = LSTMRUL(len(columns), int(config["backbone"]["hidden_dim"]))
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(arrays["test"]["x"][:8]),
            torch.from_numpy(arrays["test"]["time"][:8]),
        )
    assert prediction.shape == (8, 1)
    assert torch.isfinite(prediction).all()

    smoke_config = dict(config)
    smoke_config["degradation_proxy"] = dict(config["degradation_proxy"])
    smoke_config["degradation_proxy"]["estimators"] = 8
    proxy = _fit_degradation_proxy(train, smoke_config, seed=42)
    templates = fit_template_library(train, config)
    assert set(templates.mean) == set(FAMILIES)
    assert all(np.all(np.diff(templates.mean[family]) >= -1e-12) for family in FAMILIES)
    fit = BackboneFit(
        model=model,
        scaler=scaler,
        time_scale_minutes=time_scale,
        history=pd.DataFrame(),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        best_epoch=0,
        best_validation_mse=float("nan"),
    )
    mock_predictions = pd.DataFrame(
        {
            "run_id": arrays["test"]["run_id"],
            "sample_index": arrays["test"]["sample_index"],
            "elapsed_minutes": arrays["test"]["elapsed_minutes"],
            "target_rul": arrays["test"]["target"].reshape(-1),
            "backbone_rul": np.clip(
                prediction.detach().numpy().reshape(-1).mean()
                + np.zeros(len(arrays["test"]["target"])),
                0.0,
                1.0,
            ),
        }
    )
    evidence = build_credibility_evidence(
        mock_predictions, test, fit, proxy, templates, config, "test"
    )
    matrix, evidence_names = _credibility_matrix(evidence, config)
    forbidden = set(config["corruptions"]["forbidden_credibility_inputs"])
    assert not forbidden & set(evidence_names)
    assert np.isfinite(matrix).all()
    assert evidence.groupby(["run_id", "sample_index"])["validity_label"].sum().eq(3).all()
    assert evidence.groupby(["run_id", "sample_index"]).size().eq(20).all()

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert notebook["nbformat"] == 4
    assert "EXP-007" in source
    assert "expected_commit.txt" in source
    assert "run_exp7_experiment" in source
    assert "experiment_outputs_exp007" in source
    assert "nvidia-smi" in source
    assert "class " not in code and "def " not in code
    print(
        json.dumps(
            {
                "status": "passed_without_training",
                "qualification": qualification,
                "sequence_counts": {key: len(value["target"]) for key, value in arrays.items()},
                "smoke_evidence_rows": len(evidence),
                "credibility_feature_count": matrix.shape[1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
