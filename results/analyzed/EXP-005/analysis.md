# Run 5 analysis

## Validity decision

EXP-005 is a valid completed controlled preprocessing experiment. All 36 jobs completed on a Tesla T4 in 51.9 minutes with no failures. The exact clean Git commit `359fd3ceb0df7314e0714468414aba52c95b7783` exists locally and its 30 committed source blobs match the saved source manifest. Configuration, split, preprocessing, feature-cache, executed-notebook, artifact-inventory, and bundle checks passed. All best-checkpoint and final-epoch metrics were independently reproduced from predictions.

The Run 4 prediction-identity defect is corrected. Every test and validation prediction file preserves the physical trajectory in both `run_id` and `bearing_run_id` and stores `run_05` separately in `experiment_run_id`; no identity issues were found.

## Primary outcome

- Weak-PINN/high remained first by equal-bearing macro normalized RMSE at `0.288104`, an improvement of 8.3% over Run 4. Worst-bearing RMSE improved to `0.479774` and between-bearing RMSE SD to `0.167423`.
- Strong-PINN ranked second at macro RMSE `0.311587`. It improved over Run 4 but won no fold.
- LSTM ranked third at macro RMSE `0.371753`. Its worst-bearing RMSE and between-bearing variation improved, but its overall RMSE worsened because performance on IMS-DS2/B1 collapsed.
- Macro R2 remained negative for all models: Weak-PINN `-0.270`, Strong-PINN `-0.461`, and LSTM `-0.811`. Run 5 therefore does not establish generally reliable cross-bearing prediction.

Per-bearing winners were ims_ds1_b3: weak_pinn (RMSE 0.170740); ims_ds1_b4: weak_pinn (RMSE 0.126815); ims_ds2_b1: lstm (RMSE 0.301256); ims_ds3_b3: lstm (RMSE 0.449622).

## Hypothesis and predeclared criteria

The Run 5 hypothesis is only partially supported and the full success criterion failed. Weak-PINN passed the aggregate macro-RMSE, worst-bearing, and between-bearing-variation thresholds, but improved only `2/4` folds rather than at least three.

Late-life behavior was mixed: Weak-PINN worst absolute late-life bias improved from `0.704754` to `0.663586`, but macro late-life MAE worsened from `0.290534` to `0.330923`, and absolute late-life bias improved in only `2/4` folds. A conservative decision is therefore that the late-life condition did not pass.

## Diagnostics and publication relevance

The transformation reproduced the label-free covariate result: mean signal-feature Wasserstein shift fell from `1.4482` to `0.9342` (35.5% reduction), with a reduction in every fold. Reduced covariate discrepancy did not translate uniformly into lower RUL error.

The strongest counterexample is IMS-DS2/B1: Weak-PINN RMSE increased from `0.207321` to `0.375087`, and LSTM increased from `0.144667` to `0.301256`. Conversely, Weak-PINN improved dramatically on IMS-DS1/B3 (`0.444780` to `0.170740`) and all models improved or remained close on the difficult IMS-DS3/B3 fold. The preprocessing effect is therefore model- and domain-dependent rather than universally stabilizing.

Final-epoch minus best-validation test RMSE changes were: lstm mean -0.0521, range [-0.1411, +0.0254]; strong_pinn mean +0.0430, range [-0.1083, +0.1691]; weak_pinn mean -0.0074, range [-0.0911, +0.0333]. Best-validation checkpoint reporting remains necessary.

This controlled negative/partial result is useful for the publication pivot: a fixed normalization rule and fixed physics weighting can both help some bearings while harming others even when measured covariate shift decreases. That directly motivates studying identifiability- or uncertainty-aware mechanisms that learn when a preprocessing assumption or physical prior is trustworthy.

## Next decision

Do not automatically prepare Run 6 or tune another fixed weight grid. Freeze Runs 4 and 5 as the controlled diagnostic baseline pair. The next repository task should be a formal 2021–2026 novelty matrix and publication research protocol that defines the central method, datasets, modern baselines, uncertainty treatment, ablations, and locked evaluation before further model implementation.
