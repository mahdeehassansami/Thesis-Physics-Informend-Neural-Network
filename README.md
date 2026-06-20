# Thesis Work: Local NASA IMS Bearing RUL Pipeline

This project is a local, testable version of the original Colab notebook:

`Thesis_v3_with_extra_graphs_tables.ipynb`

The code no longer mounts Google Drive. It reads IMS data from `data/raw/`, caches extracted features in `data/processed_features/`, and writes tables/figures to `outputs/`.

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

Feature caches are written under:

```text
data/processed_features/
```
