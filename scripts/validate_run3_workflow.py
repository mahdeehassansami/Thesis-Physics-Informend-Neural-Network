from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pandas as pd

from thesis_work.multi_dataset import PHYSICS_COLUMNS, SIGNAL_FEATURES
from thesis_work.run3_calibration import run_run3_experiment


ROOT = Path(__file__).resolve().parents[1]


def synthetic_ims_frame() -> pd.DataFrame:
    run_ids = (
        "ims_ds1_b4",
        "ims_ds3_b3",
        "ims_ds1_b3",
        "ims_ds2_b1",
    )
    rows = []
    for run_number, run_id in enumerate(run_ids, start=1):
        for sample_index in range(14):
            elapsed_norm = sample_index / 13
            row = {
                "dataset": "ims",
                "run_id": run_id,
                "sample_index": sample_index,
                "elapsed_seconds": float(sample_index * 60),
                "elapsed_norm": elapsed_norm,
                "rul_norm": 1.0 - elapsed_norm,
                "health_indicator": elapsed_norm,
                "temperature_c": 0.0,
                "ambient_temperature_c": 0.0,
                "temperature_delta_c": 0.0,
                "load_n": 26689.0,
                "speed_rpm": 2000.0,
                "contact_pressure_mpa": 200.0,
                "dynamic_capacity_n": 50000.0,
                "fatigue_limit_n": 3000.0,
                "viscosity_ref_cst": 100.0,
                "viscosity_required_cst": 30.0,
                "contamination_factor": 0.7,
                "cycles_per_time_unit": 26000.0,
                "temperature_available": 0.0,
                "load_available": 1.0,
                "contact_pressure_available": 1.0,
            }
            for feature_number, name in enumerate(SIGNAL_FEATURES, start=1):
                row[name] = (
                    0.05 * feature_number
                    + 0.4 * elapsed_norm
                    + 0.01 * run_number
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in PHYSICS_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame


def main() -> None:
    config = json.loads(
        (ROOT / "configs" / "colab_experiments.json").read_text(encoding="utf-8")
    )
    smoke = copy.deepcopy(config)
    smoke["runtime"] = {
        "require_cuda": False,
        "required_gpu_name_contains": None,
    }
    smoke["training"].update(
        {
            "sequence_length": 3,
            "epochs": 2,
            "patience": 1,
            "batch_size": 64,
            "seed_repeats": 1,
            "seeds": [42],
            "gradient_diagnostics_interval": 1,
        }
    )
    smoke["calibration"]["paris_weights"] = [0.003, 0.01]
    smoke["calibration"]["miner_weights"] = [0.001]

    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        temporary = Path(directory)
        cache = temporary / "feature_cache"
        output = temporary / "experiment_outputs_run_03"
        cache.mkdir()
        output.mkdir()
        synthetic_ims_frame().to_csv(cache / "ims_features.csv", index=False)
        results, grid = run_run3_experiment(
            config=smoke,
            project_root=ROOT,
            cache_dir=cache,
            output_root=output,
        )
        assert len(grid) == 2
        assert set(grid["selection_split"]) == {"validation"}
        assert not grid["test_evaluated"].any()
        assert len(results) == 3
        assert set(results["model"]) == {"lstm", "weak_pinn", "strong_pinn"}
        assert set(results["seed"]) == {42}
        calibration = output / "ims" / "calibration"
        assert len(list(calibration.glob("*/seed_*/validation_predictions.csv"))) == 2
        assert len(list(calibration.glob("*/seed_*/predictions.csv"))) == 1
        selected = json.loads(
            (calibration / "selected_profile.json").read_text(encoding="utf-8")
        )
        assert selected["test_was_used_for_selection"] is False
        manifest = json.loads(
            (output / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert manifest["calibration_candidates"] == 2
        assert (output / "artifact_inventory.csv").exists()
        selected_prediction = next(calibration.glob("*/seed_*/predictions.csv"))
        prediction_columns = pd.read_csv(selected_prediction, nrows=1).columns
        assert "target_rul_seconds" in prediction_columns
        selected_history = next(calibration.glob("*/seed_*/history.csv"))
        history_columns = pd.read_csv(selected_history, nrows=1).columns
        assert "weighted_paris_crack_growth" in history_columns
        assert "gradient_physics_weighted" in history_columns
    print("Run 3 validation-only workflow smoke test passed.")


if __name__ == "__main__":
    main()
