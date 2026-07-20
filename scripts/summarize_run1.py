from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "saved results" / "run_01"
OUTPUTS = RUN / "experiment_outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


results = pd.read_csv(OUTPUTS / "all_model_comparisons.csv")
ranking = results.sort_values(["dataset", "rmse"]).copy()
ranking["rank_by_rmse"] = ranking.groupby("dataset")["rmse"].rank(
    method="dense"
)
ranking.to_csv(RUN / "run_01_model_ranking.csv", index=False)

epoch_rows = []
for history_path in OUTPUTS.glob("*/*__*/history.csv"):
    history = pd.read_csv(history_path)
    epoch_rows.append(
        {
            "dataset": history_path.parts[-3],
            "model_profile": history_path.parts[-2],
            "epochs_completed": len(history),
            "best_epoch": int(history.loc[history["validation_mse"].idxmin(), "epoch"]),
            "best_validation_mse": float(history["validation_mse"].min()),
            "final_validation_mse": float(history["validation_mse"].iloc[-1]),
        }
    )
pd.DataFrame(epoch_rows).sort_values(
    ["dataset", "model_profile"]
).to_csv(RUN / "run_01_training_epochs.csv", index=False)

manifest_rows = []
for path in sorted(OUTPUTS.rglob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "relative_path": path.relative_to(RUN).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
pd.DataFrame(manifest_rows).to_csv(RUN / "artifact_manifest.csv", index=False)

best = ranking.groupby("dataset", as_index=False).first()
sensitivity_path = OUTPUTS / "ims" / "sensitivity" / "sensitivity_results.csv"
sensitivity = pd.read_csv(sensitivity_path)

lines = [
    "# Run 1 analysis",
    "",
    "Run 1 is preserved as a diagnostic pilot, not final thesis evidence. The code supplied ",
    "`elapsed_norm` to every model while defining `rul_norm = 1 - elapsed_norm`, creating ",
    "direct target leakage. Run 2 removes that leakage and uses training-only time scaling.",
    "",
    "## Execution",
    "",
    f"- Successful model/profile runs: {len(results)} of {len(results)}.",
    f"- Sum of recorded model-training time: {results['seconds'].sum():.1f} seconds.",
    "- Maximum epochs: 150; early stopping patience: 20.",
    "- Compact sequences: IMS 8,474 training sequences, PRONOSTIA 4,220, and KAIST 80.",
    "- One seed per model/profile; Run 2 uses three comparable seeds.",
    "",
    "## Best observed result per dataset",
    "",
]
for row in best.itertuples(index=False):
    lines.append(
        f"- {row.dataset}: {row.model}/{row.weight_profile}, "
        f"RMSE={row.rmse:.6f}, MAE={row.mae:.6f}, R2={row.r2:.6f}."
    )
lines.extend(
    [
        "",
        "## Interpretation",
        "",
        "- LSTM ranked first on IMS and PRONOSTIA, but those scores are inflated by time leakage.",
        "- Every KAIST model had negative R2. The single-run temporal test covers only the final-life ",
        "  target range and all models extrapolated poorly; weak-PINN/high was merely the least poor.",
        "- Strong-PINN medium/high collapsed toward RUL near 1 on IMS and PRONOSTIA because the Paris ",
        "  residual was orders of magnitude larger than the data loss.",
        "- The sensitivity sweep confirms that poor-lubrication multiplier and Paris-loss weight had ",
        "  the largest apparent effect, but trial-specific random seeds confounded Run 1 sensitivity.",
        f"- Sensitivity trials preserved: {len(sensitivity)}.",
        "",
        "## Why it ran quickly",
        "",
        "The models never processed 28 GB of raw vibration waveforms in Colab. They used small cached ",
        "feature tables, short three-step sequences, batch size 128, hidden width 128, one seed, and ",
        "early stopping. Most IMS/PRONOSTIA runs stopped after about 21-50 epochs; KAIST has only one ",
        "trajectory and 80 training sequences. Fast execution was therefore expected, while leakage ",
        "made convergence even easier.",
    ]
)
(RUN / "RUN_01_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Saved Run 1 analysis and manifest to {RUN}")
