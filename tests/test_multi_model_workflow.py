from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from thesis_work.multi_dataset import SIGNAL_FEATURES, apply_initial_baseline_normalization
from thesis_work.run4_cross_bearing import _annotate
from thesis_work.sequence_models import build_model
ROOT=Path(__file__).resolve().parents[1]

def test_experiment_config_has_distinct_physics_variants():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    assert {"attnpinn","weak_pinn","strong_pinn"} <= set(config["models"])
    assert config["weight_profiles"]["weak_low"] != config["weight_profiles"]["weak_high"]
    assert config["weight_profiles"]["strong_low"] != config["weight_profiles"]["strong_high"]
    assert config["physics"]["crack_growth"]["paris_exponent"]==7.5

def test_all_model_families_produce_bounded_rul():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8")); x=torch.randn(2,3,20); t=torch.tensor([[.25],[.75]])
    for name in ("fnn","cnn","lstm","attnpinn","weak_pinn","strong_pinn"):
        model=build_model(name,20,config["models"][name]); prediction=model(x,t)
        assert prediction.shape==(2,1) and torch.all((prediction>=0)&(prediction<=1))

def test_run5_changes_only_preprocessing_and_prediction_identity():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    run4=json.loads((ROOT/"configs"/"colab_experiments_run_04.json").read_text(encoding="utf-8"))
    assert config["experiment"]["id"]=="EXP-005" and config["run_label"]=="run_05"
    assert config["training"]["seeds"]==[42,1042,2042] and config["cross_bearing"]["expected_jobs"]==36
    folds=config["cross_bearing"]["folds"]; runs=set(config["cross_bearing"]["all_runs"])
    assert len(folds)==4
    assert sorted(next(iter(f["test_runs"])) for f in folds)==sorted(runs)
    assert sorted(next(iter(f["validation_runs"])) for f in folds)==sorted(runs)
    enabled=[n for n,m in config["models"].items() if m.get("enabled")]
    assert enabled==["lstm","weak_pinn","strong_pinn"]
    assert config["models"]["strong_pinn"]["profiles"]==["strong_paris_0p003_miner_0p003"]
    assert config["cross_bearing"]["folds"]==run4["cross_bearing"]["folds"]
    assert config["training"]==run4["training"]
    assert config["models"]==run4["models"]
    assert config["weight_profiles"]==run4["weight_profiles"]
    assert config["physics"]==run4["physics"]
    assert config["preprocessing"]["prefix_samples"]==config["training"]["sequence_length"]==8
    assert config["preprocessing"]["uses_targets"] is False


def test_baseline_normalization_is_fixed_before_first_prediction():
    rows=[]
    for run_number,run_id in enumerate(("bearing_a","bearing_b"),1):
        for sample_index in range(12):
            row={"run_id":run_id,"sample_index":sample_index,"rul_norm":1-sample_index/11}
            for feature_number,feature in enumerate(SIGNAL_FEATURES,1):
                row[feature]=run_number*feature_number+0.1*sample_index
            rows.append(row)
    frame=pd.DataFrame(rows)
    config={
        "strategy":"per_run_initial_robust_relative",
        "prefix_samples":8,
        "feature_columns":list(SIGNAL_FEATURES),
        "mad_consistency_constant":1.4826,
        "absolute_scale_floor":1e-8,
        "require_prefix_before_first_prediction":True,
    }
    transformed,metadata=apply_initial_baseline_normalization(frame,config,sequence_length=8)
    changed=frame.copy()
    changed.loc[changed.sample_index>=8,SIGNAL_FEATURES]=1e9
    transformed_changed,metadata_changed=apply_initial_baseline_normalization(changed,config,sequence_length=8)
    assert metadata["uses_targets"] is False and metadata["uses_failure_time"] is False
    assert metadata["run_statistics"]==metadata_changed["run_statistics"]
    early=frame.sample_index<8
    np.testing.assert_allclose(
        transformed.loc[early,SIGNAL_FEATURES],
        transformed_changed.loc[early,SIGNAL_FEATURES],
    )
    np.testing.assert_allclose(transformed["rul_norm"],frame["rul_norm"])


def test_prediction_annotation_keeps_physical_bearing_id(tmp_path):
    path=tmp_path/"predictions.csv"
    pd.DataFrame({
        "run_id":["ims_ds1_b3","ims_ds1_b3"],
        "target_rul":[1.0,0.5],
        "predicted_rul":[0.9,0.4],
        "target_rul_seconds":[100.0,50.0],
        "predicted_rul_seconds":[90.0,40.0],
    }).to_csv(path,index=False)
    config={"experiment":{"id":"EXP-005"},"run_label":"run_05"}
    fold={"fold_id":"fold_01_test_ims_ds1_b3"}
    annotated=_annotate(path,config,fold,"lstm","data_only",1,42,"best_validation")
    assert set(annotated.run_id)=={"ims_ds1_b3"}
    assert set(annotated.bearing_run_id)=={"ims_ds1_b3"}
    assert set(annotated.experiment_run_id)=={"run_05"}
