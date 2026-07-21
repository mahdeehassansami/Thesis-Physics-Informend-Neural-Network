# Thesis Work: Local NASA IMS Bearing RUL Pipeline

This project is the authoritative, local, testable version of the original Colab notebook:

`Thesis_v3_with_extra_graphs_tables.ipynb`

The notebook is kept as prior-work context only. Final tables, figures, manifests, and LaTeX assets should be generated from this Python project. The code no longer mounts Google Drive. It reads IMS data from `data/raw/`, caches extracted features in `data/processed_features/`, and writes tables/figures to `outputs/`.

## Data Layout

Your current extracted datasets are already in the expected location:

```text
data/raw/
  1st_test/
  2nd_test/
  3rd_test/
```

The loader also supports zip files named `1st_test.zip`, `2nd_test.zip`, and `3rd_test.zip` in `data/raw/`, but extracted folders are preferred for your current setup.

## Setup

```powershell
uv sync --extra dev
```

## Useful Commands

Validate that local IMS data can be read:

```powershell
uv run thesis-work validate-data
```

Extract and cache features for all thesis runs:

```powershell
uv run thesis-work extract-features
```

Generate features, PCA-HI analysis, and static thesis tables/figures without neural training:

```powershell
uv run thesis-work run --skip-training
```

Run the full pipeline:

```powershell
uv run thesis-work run
```

The full pipeline trains the DeepXDE data-only baseline, the proposed DeepXDE physics-informed model, the LSTM baseline, and the CNN baseline. It can take a long time on CPU.

Regenerate polished figures from cached full-run data without retraining:

```powershell
uv run thesis-work regenerate-figures
```

This command also refreshes `thesis/latex/assets/images/` with the exact PNG and PDF filenames used by the LaTeX manuscript.

For a quick smoke run, use fewer files and shorter training:

```powershell
uv run thesis-work run --max-files 40 --baseline-iterations 10 --pinn-iterations 10 --sequence-epochs 1 --sequence-length 5
```

## Outputs

Generated artifacts are written under:

```text
outputs/
  figures/
  tables/
```

The full run writes `outputs/tables/run_manifest.json`; cached figure regeneration writes `outputs/tables/figure_manifest.json`. These record seeds, run options, package versions, raw-data counts, feature-cache hashes, and result-table hashes so future figure changes can be traced.

LaTeX-ready figures are mirrored to:

```text
thesis/latex/assets/images/
```

Feature caches are written under:

```text
data/processed_features/
```

## Current publication experiment

The active Colab protocol is `EXP-007B`, preregistered in
`research/EXP007B_PROTOCOL.md`. It confirms a prefix-local abstaining physics controller on a
fresh seed-920072 simulator test population; the opened EXP-007A test is diagnostic-only and is
excluded from the active cache.

The notebook remains a thin controller over `src/thesis_work/`. Before Colab handoff, validate
the workflow with:

```powershell
uv run python scripts/validate_exp007b_workflow.py
```

The prepared `Upload` directory contains the notebook, exact pushed commit, and the lightweight
feature cache needed by Colab. Full checkpoints and run outputs remain in Google Drive; return
the lightweight result bundle under `results/incoming/` for independent verification.
