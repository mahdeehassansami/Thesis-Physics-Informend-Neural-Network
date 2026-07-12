# Thesis Walkthrough: Section-by-Section Study Notes

Updated: 2026-07-10

These notes replace the older walkthrough and reflect the current thesis source and PDF state. They are written as study notes for presentation, viva preparation, and quick revision. The thesis source itself is not edited by this file.

## Current Thesis Position

The thesis investigates normalized remaining useful life (RUL) prediction for selected rolling-element bearing runs from the NASA IMS dataset. The work compares a proposed weak physics-informed neural network with three data-driven baselines: a data-only feed-forward neural network, an LSTM sequence model, and a 1D CNN sequence model.

The most recent evaluation uses 12 complete-run train-validation-test split configurations and three random seeds per split. That gives 36 evaluated runs per model. The data-only feed-forward baseline has the lowest mean RMSE, but the proposed PINN is close. Because the standard deviations are large and only four independent bearing trajectories are used, the thesis takes a conservative position: the weak physics-informed formulation is competitive in some splits, but it is not statistically proven to be superior.

Key final results:

| Model | MAE mean +/- SD | RMSE mean +/- SD | R2 mean +/- SD |
|---|---:|---:|---:|
| Data-only FNN | 0.168 +/- 0.136 | 0.209 +/- 0.156 | 0.184 +/- 0.885 |
| Proposed PINN | 0.178 +/- 0.148 | 0.216 +/- 0.170 | 0.095 +/- 0.998 |
| LSTM | 0.188 +/- 0.072 | 0.230 +/- 0.078 | 0.256 +/- 0.474 |
| CNN | 0.203 +/- 0.089 | 0.235 +/- 0.095 | 0.199 +/- 0.632 |

## Recent Developments Reflected Here

- The experiment description is updated from the older limited split explanation to the final 12 split configurations with three repeated random seeds.
- The results now report mean +/- standard deviation across 36 runs per model.
- The uncertainty discussion now includes block-bootstrap RMSE intervals over contiguous prediction-error blocks.
- The thesis no longer claims that the proposed PINN is the best overall model. It presents a conservative, statistically careful interpretation.
- The graph discussion reflects the revised figure style: larger plots, black labels, readable legends, and non-heatmap model-comparison figures.
- The final appendix section now uses supplied appendix pages: Appendix A, Appendix B, Appendix C, and Appendix D. Page 47 is a horizontal landscape table page.

## How To Use These Notes

For a defense, explain each section in this order:

1. What problem the section addresses.
2. What technical choice was made.
3. Why that choice matters.
4. What result or limitation should be remembered.

The safest overall defense line is: this thesis provides a reproducible local pipeline and a conservative comparison, not a claim that the proposed PINN is universally superior.

---

# Front Matter

## Abstract

The abstract states the full story in compressed form. Bearing faults can begin as small localized defects and grow into vibration, heat, misalignment, secondary damage, and shutdown. Because of this, RUL prediction is important in prognostics and health management.

The method uses selected IMS run-to-failure bearing data. The pipeline loads the raw data locally, extracts time-domain and envelope spectral features, creates normalized RUL targets, builds a PCA-based health indicator, and evaluates four models: data-only FNN, proposed DeepXDE PINN, LSTM, and CNN.

The final abstract result is important: the data-only FNN has the lowest mean RMSE, the proposed PINN is second and close, and the LSTM and CNN are comparable. The standard deviations are large, so the abstract does not overclaim. It says the weak physics-informed model is competitive in some split settings but not robustly superior.

Defense line: the abstract is intentionally conservative because the four bearing trajectories limit statistical strength.

## Acknowledgement

This section is nontechnical. It thanks Allah, the supervisor, family, and the researchers who made the IMS dataset public. It gives academic and personal context but is normally not discussed in technical defense questions.

## Nomenclature

The nomenclature defines repeated terms such as RUL, PHM, IMS, FNN, LSTM, CNN, PINN, PCA, HI, RMSE, MAE, R2, FFT, FTF, BPFO, BPFI, and BSF. In a defense, use this page to keep terminology consistent.

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

This section explains why bearing RUL prediction matters. Rolling-element bearings support shafts and carry loads in rotating machines. A small defect can create repeated impacts, vibration growth, heat, secondary damage, and shutdown.

Traditional maintenance is limited. Reactive maintenance waits for failure. Preventive maintenance changes components at fixed intervals and may be too early or too late. Condition-based maintenance and PHM are better because they use evidence from the machine condition.

The section also explains why bearing RUL is hard. Degradation is nonlinear, vibration can remain low for a long time, and the final rise near failure can be sharp. Different bearings can degrade differently even in the same test rig.

Defense line: the motivation is not just machine learning; it is the maintenance need to predict failure before shutdown.

## 1.2 Problem Statement

The problem is to predict normalized RUL for selected IMS bearing runs and test whether a physics-informed model generalizes better across complete bearing runs than comparable data-driven neural baselines.

The key point is complete-run generalization. The model is not tested by randomly mixing snapshots from the same bearing into train and test. Instead, entire bearing trajectories are held out. This is stricter and closer to the real maintenance problem.

Defense line: a useful RUL model should transfer to a different bearing run, not memorize one degradation trajectory.

## 1.3 Objectives of the Study

The objectives are to extract degradation-relevant vibration features, build normalized RUL targets and a PCA health indicator, design a DeepXDE physics-informed model using weak monotonicity and fault-frequency energy priors, and compare it with FNN, LSTM, and CNN baselines.

The objectives connect signal processing, machine learning, and interpretation. The thesis is not only model training; it also builds the data pipeline and explains why the features are relevant.

## 1.4 Scope

The thesis uses four selected IMS bearing runs: `ds2_b1`, `ds1_b3`, `ds1_b4`, and `ds3_b3`. It uses vibration data only. It does not include temperature, oil debris, acoustic emission, motor current, or industrial field data.

The physics-informed part is weak by design. It uses monotonic RUL behavior and fault-frequency envelope energy priors, not a full bearing degradation law. Therefore the thesis should not be presented as a complete physical bearing-failure model.

## 1.5 Importance of the Study

The importance is that the work combines interpretable vibration features, neural RUL prediction, and a reproducible local workflow. RMS, kurtosis, crest factor, and envelope energy have physical meaning, and the models test whether those features can predict normalized RUL across held-out bearing runs.

The final interpretation is cautious. The data-only baseline has the lowest mean RMSE, while the proposed PINN is close but less consistent. This is still useful because it shows where the current weak physical prior helps and where it is not enough.

## 1.6 Limitations

The major limitation is the small number of independent complete bearing runs. The thesis expands the evaluation to 12 split configurations and three seeds per split, but this does not create new independent bearing failures.

Other limitations are the weak physical prior, hand-engineered features, normalized RUL labels based on the final timestamp, and one local hardware/software environment.

Defense line: repeated seeds improve validation, but they do not replace additional run-to-failure bearings.

## 1.7 Thesis Outline

The outline tells the reader how the thesis is organized: literature review, methodology, implementation, results, conclusion, references, and appendices.

---

# Chapter 2: Literature Review

## 2.1 Introduction

This chapter places the thesis in the existing research context. It moves from bearing failure mechanisms to predictive maintenance, deep learning, PINNs, and recent physics-informed bearing work.

## 2.2.1 Rolling-Element Bearings: Fundamentals and Failure Modes

This subsection explains bearing components: inner race, outer race, rolling elements, and cage. Defects can come from fatigue, lubrication failure, contamination, overload, misalignment, electrical pitting, poor installation, or manufacturing variation.

Fault type affects vibration signature. Outer-race, inner-race, rolling-element, and cage faults generate different impact and modulation patterns. This is why characteristic fault frequencies are meaningful in feature extraction.

## 2.2.2 Traditional Predictive Maintenance Techniques for Bearings

This subsection reviews condition monitoring and PHM. The usual pipeline is sensing, preprocessing, feature extraction, health indicator construction, diagnosis, and RUL estimation.

Vibration is emphasized because bearing faults create impacts and modulations. Time-domain features summarize amplitude and impulsiveness, while envelope-domain features can reveal periodic fault-related energy.

The IMS dataset is useful because it is public and run-to-failure. The caution is that public bearing datasets are small, so evaluation design matters.

## 2.2.3 Deep Learning in Predictive Maintenance

Deep learning can capture nonlinear relationships between vibration features and RUL. FNNs fit engineered feature vectors, CNNs learn local sequence patterns, and LSTMs model temporal dependence.

The weakness is distribution shift. A model trained on one bearing may fail on another. This is why the thesis uses complete held-out bearing runs rather than random snapshot splitting.

## 2.2.4 Physics-Informed Neural Networks

PINNs combine data loss with physical residuals. In classical PINNs, the residual often comes from a known differential equation. Bearing RUL prediction is harder because there is no simple complete governing equation for feature-level degradation.

The thesis therefore uses weaker physics-informed learning: monotonic RUL behavior and fault-frequency energy consistency. This makes the model physically guided but not a complete physics simulator.

## 2.2.5 PINNs for Bearing Predictive Maintenance

Recent studies show that physics-informed or knowledge-informed learning can improve interpretability and guide models when data are limited. They also show that the physics term must match the real process. A poor or weak prior may not improve prediction.

Defense line: adding physics is not automatically beneficial; the physical prior must be relevant, correctly weighted, and validated.

## 2.3 Research Gap

The research gap is that many bearing RUL studies differ in preprocessing, labeling, splitting, and evaluation. Some random splits may leak trajectory information and overstate performance. Physics-informed RUL models are promising but not guaranteed to improve complete-run generalization.

The thesis addresses a narrow question: how does a weak physics-informed RUL model behave against common neural baselines on complete held-out IMS bearing runs, under a reproducible local pipeline and repeated-seed validation?

---

# Chapter 3: Dataset and Research Methodology

## 3.1 Introduction

This chapter explains the pipeline from raw IMS vibration files to final metrics and figures. It includes dataset setup, feature extraction, RUL labeling, preprocessing, PCA health indicators, model definitions, experiment design, and evaluation metrics.

## 3.2 Experimental Setup

The IMS data were generated using four Rexnord ZA-2115 double-row bearings on a shaft running at 2000 RPM under a 6000 lb radial load. Vibration was measured using PCB accelerometers. The first dataset has two axes per bearing; the second and third datasets have one channel per bearing.

Defense line: the data come from a controlled run-to-failure bearing test rig, not from simulated signals.

## 3.3 Dataset Description

Each raw IMS file contains about one second of vibration data with 20,480 samples at 20 kHz. The project uses local folders under `data/raw`.

Final selected runs:

| Run | Dataset | Bearing | Samples |
|---|---|---:|---:|
| `ds2_b1` | `2nd_test` | 1 | 984 |
| `ds1_b3` | `1st_test` | 3 | 2156 |
| `ds1_b4` | `1st_test` | 4 | 2156 |
| `ds3_b3` | `3rd_test` | 3 | 6324 |

The run lengths are imbalanced, so the final design uses balanced sampling within training runs to prevent the longest run from dominating.

## 3.4 Feature Extraction

Each vibration snapshot is converted into time-domain and envelope spectral features.

Time-domain features include RMS, standard deviation, peak-to-peak, kurtosis, crest factor, and mean absolute value. These represent vibration energy, spread, impulsiveness, and peak behavior.

Fault-frequency features use envelope spectral energy around FTF, BPFO, BPFI, and BSF harmonics. Their combined value is stored as `E_kin`.

Fault-frequency values used:

| Frequency | Meaning | Value |
|---|---|---:|
| FTF | Cage / fundamental train | 14.78 Hz |
| BPFO | Outer race | 236.40 Hz |
| BPFI | Inner race | 296.93 Hz |
| BSF | Rolling element | 139.92 Hz |

Defense line: the model inputs are not arbitrary; the features are tied to bearing vibration behavior.

## 3.5 RUL Target Labeling

RUL is computed from elapsed time to the final snapshot of each run. The target is normalized so it starts near 1 and ends near 0. This makes runs with different durations comparable.

Important limitation: normalized RUL is a relative life fraction, not an independently measured failure threshold in hours.

## 3.6 Preprocessing

The pipeline normalizes features using the early healthy baseline of each run. The first 5 percent of samples, with at least 20 samples, are treated as baseline. Features are converted to relative increases, transformed with `log1p`, and smoothed using a rolling median.

This reduces scale differences between runs, but it cannot remove all cross-bearing variation.

## 3.7 PCA Health Indicator

A PCA health indicator checks whether the extracted features contain a degradation trend. The first principal component is scaled between 0 and 1 and flipped if needed so that higher values mean greater damage.

PCA-HI monotonicity scores:

| Run | Score | Interpretation |
|---|---:|---|
| `ds1_b3` | 0.766 | Strong degradation consistency |
| `ds1_b4` | 0.691 | Moderate or weaker consistency |
| `ds2_b1` | 0.803 | Strong degradation consistency |
| `ds3_b3` | 0.864 | Strong degradation consistency |

The PCA-HI is used for explanation, not as the final target.

## 3.8 Model Definitions

Four models are tested.

| Model | Input | Role |
|---|---|---|
| Data-only FNN | Single feature vector | Main data-only baseline |
| Proposed PINN | Single feature vector with weak residuals | Proposed model |
| LSTM | Sliding window of length 20 | Sequence baseline |
| CNN | Sliding window of length 20 | Sequence baseline |

The proposed PINN adds two weak residuals to the supervised RUL loss: a monotonicity residual and a fault-frequency energy residual. The idea is that RUL should generally decrease over time and higher fault-frequency energy should be consistent with greater damage.

## 3.9 Experiment Design

The final design uses all 12 complete-run train-validation-test configurations. For each split, one bearing is the test run, one different bearing is the validation run, and the remaining two are training runs. Each model is trained three times per split with different random seeds.

| Split | Train runs | Validation | Test |
|---|---|---|---|
| S01 | `ds1_b4 + ds3_b3` | `ds1_b3` | `ds2_b1` |
| S02 | `ds1_b3 + ds3_b3` | `ds1_b4` | `ds2_b1` |
| S03 | `ds1_b3 + ds1_b4` | `ds3_b3` | `ds2_b1` |
| S04 | `ds1_b4 + ds3_b3` | `ds2_b1` | `ds1_b3` |
| S05 | `ds2_b1 + ds3_b3` | `ds1_b4` | `ds1_b3` |
| S06 | `ds2_b1 + ds1_b4` | `ds3_b3` | `ds1_b3` |
| S07 | `ds1_b3 + ds3_b3` | `ds2_b1` | `ds1_b4` |
| S08 | `ds2_b1 + ds3_b3` | `ds1_b3` | `ds1_b4` |
| S09 | `ds2_b1 + ds1_b3` | `ds3_b3` | `ds1_b4` |
| S10 | `ds1_b3 + ds1_b4` | `ds2_b1` | `ds3_b3` |
| S11 | `ds2_b1 + ds1_b4` | `ds1_b3` | `ds3_b3` |
| S12 | `ds2_b1 + ds1_b3` | `ds1_b4` | `ds3_b3` |

Defense line: 12 splits cover every choice of held-out test run and validation run among the four selected trajectories.

## 3.10 Evaluation Metrics

The thesis uses MAE, RMSE, and R2. RMSE is the main ranking criterion because large RUL errors are costly in maintenance planning. R2 gives another view of fit quality, and negative R2 means the model is worse than predicting the mean target.

Statistical validation is reported as mean +/- standard deviation over 12 split configurations and three seeds per split. Prediction uncertainty is estimated using 95 percent block-bootstrap intervals over contiguous prediction-error blocks.

---

# Chapter 4: Local Implementation and Experimental Setup

## 4.1 Local Project Structure

The work began as a Google Colab notebook but was converted into a local Python project. The code reads data from `data/raw`, stores features in `data/processed_features`, and writes outputs to `outputs`.

The package under `src/thesis_work` separates configuration, loading, feature extraction, preprocessing, model training, metrics, reports, visualization, and CLI commands.

## 4.2 Reproducibility Commands

Main commands:

```bash
uv sync --extra dev
uv run thesis-work validate-data
uv run thesis-work extract-features
uv run thesis-work run
uv run thesis-work regenerate-figures
uv run pytest -q
```

The full run trains all four models across 12 splits and three seeds. Final repeated-seed rows are saved in `outputs/tables/split_seed_results_table.csv`, and per-split mean results are saved in `outputs/tables/final_results_table.csv`.

## 4.3 Software Environment

The implementation uses Python 3.11 with NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, DeepXDE, PyTorch, and tqdm. The thesis is compiled with MiKTeX on Windows.

The reported local PC uses an AMD Ryzen 7 5800H CPU, 16 GB RAM, Windows 11, and an NVIDIA RTX 3060 Laptop GPU with 4 GB VRAM.

## 4.4 Data Validation

The validation command checks that expected data folders exist and that sample files have the expected column format. This prevents silent errors from wrong IMS extraction or folder placement.

## 4.5 Feature Cache

Feature extraction is expensive because every vibration snapshot must be read and transformed. Cached feature tables make it possible to regenerate reports without repeating raw signal processing every time.

## 4.6 Visual Reporting

The final figures use consistent colors and black text for labels, ticks, legends, and annotations. The models use stable colors across plots: gray for data-only FNN, blue for proposed PINN, green for LSTM, and orange-red for CNN. The revised figures are larger and more readable in the PDF.

---

# Chapter 5: Results and Discussion

## 5.1 Feature Behavior Over Bearing Life

This section shows how RMS, kurtosis, crest factor, and combined fault-frequency envelope energy evolve over normalized bearing life. The trends differ across bearings, which explains why complete-run generalization is difficult.

Defense line: the feature plots show the core challenge. A degradation pattern learned from one bearing does not always match another bearing.

## 5.2 Main Prediction Results

The final model comparison is based on 36 evaluated runs per model. The data-only FNN has the lowest average RMSE, followed closely by the proposed PINN.

The safest interpretation is that no model is statistically dominant. The difference between the FNN and PINN mean RMSE values is small relative to their standard deviations.

## 5.3 Statistical Validation and Uncertainty

This section directly responds to the criticism about limited sample size and lack of statistical validation. The thesis now reports:

- 12 complete-run split configurations,
- 3 random seeds per split,
- 36 evaluated runs per model,
- mean and standard deviation for all metrics,
- block-bootstrap uncertainty intervals for RMSE.

Important defense point: this improves validation but does not remove the core limitation that there are only four independent bearing trajectories.

## 5.4 Experiment-Wise Comparison

Different models win different split configurations by RMSE:

| Winner | Splits |
|---|---|
| LSTM | S01, S02, S03, S12 |
| Data-only FNN | S04, S07, S11 |
| Proposed PINN | S06, S09 |
| CNN | S05, S08, S10 |

This supports the conservative conclusion. The proposed PINN wins some splits but not enough to claim overall superiority.

## 5.5 True Versus Predicted RUL

The thesis shows representative prediction curves for four splits so each selected test bearing appears once: S01, S04, S08, and S12.

These plots are useful during presentation because they show model behavior over life, not just table values. They also reveal whether a model predicts the general downward RUL trend or fails at particular life stages.

## 5.6 Error Analysis

The error analysis includes mean smoothed absolute error over life, signed error distribution, and block-bootstrap RMSE uncertainty intervals.

Positive signed error means the model overestimated RUL. In maintenance, overestimating RUL can be risky because it may delay replacement or inspection.

## 5.7 Discussion of the Physics-Informed Model

The proposed PINN uses two weak physical ideas: RUL should generally decrease with time, and fault-frequency energy should relate to damage. These assumptions are reasonable, but incomplete.

The monotonicity residual may not fully match noisy feature behavior. The fault-energy residual may be affected by sensor direction, resonance, load-zone behavior, and different fault timing. Therefore the physical prior can help in some splits but fail in others.

Defense line: physics-informed learning is useful only when the prior is close enough to the real degradation process.

## 5.8 Practical Implications

The thesis does not recommend deploying any tested model as a final industrial RUL predictor. More run-to-failure data, uncertainty-aware prediction, stronger physics, and decision rules are needed for field use.

The practical value is the workflow: interpretable features, full-run validation, repeated seeds, uncertainty reporting, and conservative interpretation.

---

# Chapter 6: Conclusion

## 6.1 Summary

The thesis builds a local RUL prediction workflow for selected IMS bearing runs. It extracts vibration features, constructs normalized RUL labels, builds a PCA health indicator, trains four neural models, and generates tables and figures.

The proposed PINN is competitive but not clearly superior. The data-only FNN has the lowest mean RMSE across the final repeated-seed evaluation.

## 6.2 Contributions

Main contributions:

- local reproducible IMS RUL prediction pipeline,
- documented time-domain and fault-frequency envelope feature extraction,
- PCA health-indicator analysis,
- DeepXDE physics-informed RUL model with weak residuals,
- comparison against FNN, LSTM, and CNN baselines,
- 12 complete-run split configurations with three repeated random seeds,
- uncertainty analysis using block-bootstrap RMSE intervals.

## 6.3 Limitations

The main limitation is the small number of complete failure runs. The thesis has many snapshots, but snapshots from the same bearing trajectory are time-dependent and not independent bearing failures.

Other limitations are normalized RUL labeling, weak physical priors, engineered features instead of raw waveform learning, one implementation setup, and limited hyperparameter tuning.

## 6.4 Future Work

Future work should test stronger physics-informed formulations, uncertainty-aware RUL prediction, Weibull or crack-growth inspired priors, multi-task health indicator and RUL prediction, additional datasets such as PRONOSTIA, time-frequency features, adaptive envelope bands, raw waveform models, and broader hyperparameter tuning.

## 6.5 Final Conclusion

The final conclusion is balanced. Physics-informed RUL prediction is promising, but weak physics alone did not prove superior to the data-only baseline. A strong bearing prognostics claim requires more independent run-to-failure data and stronger validation.

---

# Appendices

The final appendix section uses supplied appendix pages with matching table-of-contents entries.

## Appendix A: Cost Estimation of the Project

This appendix lists software, data, hardware, and documentation resources. Most items are free/open source or already available, so the total estimated cost is negligible.

## Appendix B: Gantt Chart of the Project

This appendix provides the project schedule page.

## Appendix C: Originality and AI Reports

This appendix includes the Turnitin originality and AI report pages.

## Appendix D: CO-PO-K-P-A Mapping

This appendix includes the CO-PO-K-P-A mapping pages. The final page, printed page 47, is now a horizontal landscape table page so the title, student ID, and table are readable.

---

# Quick Viva Answers

## Did you use the 12-split configuration?

Yes. The final evaluation uses all 12 complete-run train-validation-test configurations across the four selected bearing trajectories.

## Did you repeat training with random seeds?

Yes. Each model is trained three times per split using different random seeds, giving 36 evaluated runs per model.

## Do the results prove the PINN is better?

No. The proposed PINN is competitive and wins some split configurations, but the data-only FNN has the lowest mean RMSE overall. Because the standard deviations are large, the thesis does not claim statistically significant superiority.

## Why not use random snapshot splits?

Random snapshot splits can leak information from the same degradation trajectory into both training and testing. Complete-run splits are stricter because the model must transfer to a held-out bearing trajectory.

## What is the biggest limitation?

Only four selected complete bearing runs are used. Repeated seeds and 12 splits improve validation, but they do not create more independent bearing failures.

## What is the main contribution?

The main contribution is a reproducible local workflow that combines interpretable bearing features, weak physics-informed learning, complete-run validation, repeated-seed reporting, and uncertainty-aware interpretation.

