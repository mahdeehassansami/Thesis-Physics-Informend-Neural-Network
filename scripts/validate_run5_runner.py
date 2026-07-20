from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from thesis_work.multi_dataset import SIGNAL_FEATURES
from thesis_work.run5_baseline_normalization import run_run5_experiment


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame() -> pd.DataFrame:
    rows = []
    run_ids = ["ims_ds1_b3", "ims_ds1_b4", "ims_ds2_b1", "ims_ds3_b3"]
    for run_number, run_id in enumerate(run_ids, 1):
        for sample_index in range(15):
            elapsed_norm = sample_index / 14
            row = {
                "dataset": "ims",
                "run_id": run_id,
                "sample_index": sample_index,
                "elapsed_seconds": float(sample_index * 600),
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
                "cycles_per_time_unit": 20000.0,
                "temperature_available": 0.0,
                "load_available": 1.0,
                "contact_pressure_available": 1.0,
            }
            for feature_number, feature in enumerate(SIGNAL_FEATURES, 1):
                row[feature] = (
                    0.1 * run_number * feature_number
                    + 0.02 * feature_number * elapsed_norm
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads(
        (ROOT / "configs" / "colab_experiments.json").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["runtime"] = {
        "require_cuda": False,
        "require_expected_commit": False,
        "require_clean_git": False,
    }
    config["training"].update(
        {
            "epochs": 1,
            "patience": 1,
            "sequence_length": 3,
            "seeds": [42],
            "seed_repeats": 1,
            "batch_size": 32,
            "gradient_diagnostics_interval": 1,
        }
    )
    config["preprocessing"]["prefix_samples"] = 3
    config["cross_bearing"]["expected_jobs"] = 12

    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
        temporary = Path(temporary)
        cache_dir = temporary / "feature_cache"
        output_dir = temporary / "experiment_outputs_run_05"
        cache_dir.mkdir()
        output_dir.mkdir()
        cache_path = cache_dir / "ims_features.csv"
        synthetic_frame().to_csv(cache_path, index=False)
        config["cross_bearing"]["expected_feature_cache_sha256"] = hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest()

        results, folds, aggregate = run_run5_experiment(
            config,
            ROOT,
            cache_dir,
            output_dir,
        )
        assert len(results) == 12 and set(results.status) == {"ok"}
        assert len(folds) == 12 and len(aggregate) == 3
        assert len(list(output_dir.rglob("predictions.csv"))) == 12
        assert len(list(output_dir.rglob("final_predictions.csv"))) == 12
        assert (output_dir / "codex_results_bundle.zip").is_file()
        manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert manifest["completed_jobs"] == 12
        for path in output_dir.rglob("predictions.csv"):
            predictions = pd.read_csv(path)
            assert set(predictions["run_id"]) == set(predictions["bearing_run_id"])
            assert set(predictions["experiment_run_id"]) == {"run_05"}

    print("Run 5 one-epoch synthetic runner smoke test passed; metrics discarded.")


if __name__ == "__main__":
    main()
