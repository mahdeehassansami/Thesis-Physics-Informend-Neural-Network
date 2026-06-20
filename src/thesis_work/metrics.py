from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_label(run_ids: list[str]) -> str:
    return " + ".join(run_ids)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluation_row(spec: dict[str, object], model_name: str, y_true, y_pred) -> dict[str, object]:
    row = {
        "Experiment": spec["name"],
        "Train runs": run_label(spec["train_runs"]),
        "Validation run": run_label(spec["validation_runs"]),
        "Test run": run_label(spec["test_runs"]),
        "Model": model_name,
    }
    row.update(regression_metrics(y_true, y_pred))
    return row
