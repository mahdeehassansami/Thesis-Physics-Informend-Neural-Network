from __future__ import annotations
import copy, hashlib, json, tempfile
from pathlib import Path
import pandas as pd
from thesis_work.multi_dataset import PHYSICS_COLUMNS, SIGNAL_FEATURES
from thesis_work.run4_cross_bearing import run_run4_experiment

ROOT=Path(__file__).resolve().parents[1]

def synthetic_frame():
    runs=["ims_ds1_b3","ims_ds1_b4","ims_ds2_b1","ims_ds3_b3"]; rows=[]
    for number,run_id in enumerate(runs,1):
        for i in range(12):
            n=i/11
            row={"dataset":"ims","run_id":run_id,"sample_index":i,"elapsed_seconds":float(i*60),"elapsed_norm":n,"rul_norm":1-n,"health_indicator":n,"temperature_c":0.,"ambient_temperature_c":0.,"temperature_delta_c":0.,"load_n":26689.,"speed_rpm":2000.,"contact_pressure_mpa":200.,"dynamic_capacity_n":50000.,"fatigue_limit_n":3000.,"viscosity_ref_cst":100.,"viscosity_required_cst":30.,"contamination_factor":.7,"cycles_per_time_unit":26000.,"temperature_available":0.,"load_available":1.,"contact_pressure_available":1.}
            for j,name in enumerate(SIGNAL_FEATURES,1): row[name]=.05*j+.4*n+.01*number
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    config=copy.deepcopy(config); config["runtime"]={"require_cuda":False,"require_expected_commit":False,"require_clean_git":False}; config["training"].update({"epochs":2,"patience":1,"sequence_length":3,"seeds":[42],"seed_repeats":1,"batch_size":32,"gradient_diagnostics_interval":1}); config["cross_bearing"]["expected_jobs"]=12
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as d:
        d=Path(d); cache=d/"feature_cache"; out=d/"experiment_outputs_run_04"; cache.mkdir(); out.mkdir()
        frame=synthetic_frame(); path=cache/"ims_features.csv"; frame.to_csv(path,index=False)
        config["cross_bearing"]["expected_feature_cache_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        results,folds,aggregate=run_run4_experiment(config,ROOT,cache,out)
        print(results.to_string(index=False))
        assert len(results)==12 and set(results.status)=={"ok"}, results.status.value_counts()
        assert len(folds)==12 and len(aggregate)==3
        assert (out/"codex_results_bundle.zip").exists()
        assert (out/"run_manifest.json").exists()
        assert len(list(out.rglob("final_predictions.csv")))==12
        assert len(list(out.rglob("job_result.json")))==12
    print("Run 4 cross-bearing workflow smoke test passed.")

if __name__=="__main__": main()
