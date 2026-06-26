from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pathlib import Path

from thesis_work.config import (
    CNN_BASELINE_NAME,
    DATA_BASELINE_NAME,
    EXPERIMENTS,
    FEATURE_COLS_MULTI,
    LSTM_BASELINE_NAME,
    MODEL_ORDER,
    MODEL_COLS,
    PROPOSED_MODEL_NAME,
    ProjectPaths,
    TARGET_COL,
    TARGET_RUNS,
)
from thesis_work.ims import FAULT_FREQS
from thesis_work.metrics import run_label
from thesis_work.viz_style import (
    COLORS,
    MODEL_PALETTE,
    RUN_PALETTE,
    add_header,
    apply_theme,
    model_handles,
    new_figure,
    polish_axes,
    top_legend,
)

LATEX_FIGURE_NAME_OVERRIDES = {
    "pca_hi_ablation.png": "pca_health_indicator.png",
}

MODEL_SHORT_LABELS = {
    DATA_BASELINE_NAME: "Data-only FNN",
    PROPOSED_MODEL_NAME: "Proposed PINN",
    LSTM_BASELINE_NAME: "LSTM",
    CNN_BASELINE_NAME: "CNN",
}


def save_table(paths: ProjectPaths, df: pd.DataFrame, filename: str) -> None:
    path = paths.tables / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def latex_figure_name(filename: str) -> str:
    name = Path(filename).name
    if name in LATEX_FIGURE_NAME_OVERRIDES:
        return LATEX_FIGURE_NAME_OVERRIDES[name]
    if name.startswith("fig_"):
        return name[4:]
    return name


def save_current_figure(paths: ProjectPaths, filename: str, dpi: int = 320) -> None:
    fig = plt.gcf()
    path = paths.figures / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")

    latex_name = latex_figure_name(filename)
    thesis_png = paths.thesis_images / latex_name
    thesis_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(thesis_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(thesis_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved thesis figure: {thesis_png}")
    print(f"Saved thesis figure: {thesis_png.with_suffix('.pdf')}")
    plt.close(fig)


def plot_pca_hi(paths: ProjectPaths, pca_hi_df: pd.DataFrame, pca_hi_summary: pd.DataFrame) -> None:
    fig, ax = new_figure(10.5, 5.4)
    for run_id in pca_hi_summary["Run"]:
        run = pca_hi_df[pca_hi_df["run_id"] == run_id].sort_values("life_norm_for_plot")
        color = RUN_PALETTE.get(run_id, COLORS["blue"])
        ax.plot(run["life_norm_for_plot"], run["pca_hi_smooth"], linewidth=2.4, color=color, label=f"{run_id} PCA-HI")
        ax.plot(
            run["life_norm_for_plot"],
            run["damage_norm"],
            linestyle="--",
            color=COLORS["slate_dark"],
            alpha=0.68,
            linewidth=1.5,
            label=f"{run_id} damage = 1 - RUL",
        )
    polish_axes(ax, "Normalized lifetime", "Normalized value", grid_axis="both")
    top_legend(ax, ncol=2)
    add_header(
        fig,
        "PCA health indicator tracks degradation over bearing life",
        "Dashed line shows damage = 1 - normalized RUL; solid line shows smoothed PCA-HI.",
    )
    save_current_figure(paths, "pca_hi_ablation.png")


def plot_final_predictions(paths: ProjectPaths, experiments, prediction_store) -> None:
    for spec in experiments:
        exp_name = spec["name"]
        preds = prediction_store[exp_name]
        fig, ax = new_figure(10.5, 5.2)
        ax.plot(preds["x_true"], preds["true"], label="True normalized RUL", linewidth=2.7, color=COLORS["ink"])
        for model_name in MODEL_ORDER:
            if model_name not in preds:
                continue
            pred = preds[model_name]
            x_pred = preds["x_by_model"][model_name]
            if len(x_pred) != len(pred):
                raise ValueError(f"Misaligned x/pred lengths for {exp_name} {model_name}.")
            ax.plot(x_pred, pred, label=model_name, color=MODEL_PALETTE[model_name], alpha=0.9, linewidth=1.9)
        polish_axes(ax, "Normalized test-life index", "Normalized RUL", grid_axis="both")
        top_legend(ax, ncol=3)
        add_header(
            fig,
            f"{exp_name}: true versus predicted RUL",
            "Normalized remaining useful life on the held-out test run.",
        )
        save_current_figure(paths, f'{exp_name.lower().replace(" ", "_")}_true_vs_predicted_rul.png')


def prediction_store_to_table(prediction_store: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp_name, preds in prediction_store.items():
        for x, y in zip(preds["x_true"], preds["true"]):
            rows.append(
                {
                    "Experiment": exp_name,
                    "Model": "True normalized RUL",
                    "x": float(x),
                    "y": float(y),
                }
            )
        for model_name in MODEL_ORDER:
            if model_name not in preds:
                continue
            for x, y in zip(preds["x_by_model"][model_name], preds[model_name]):
                rows.append(
                    {
                        "Experiment": exp_name,
                        "Model": model_name,
                        "x": float(x),
                        "y": float(y),
                    }
                )
    return pd.DataFrame(rows)


def plot_final_predictions_from_table(paths: ProjectPaths, prediction_table: pd.DataFrame) -> None:
    for exp_name, exp_df in prediction_table.groupby("Experiment", sort=True):
        fig, ax = new_figure(10.5, 5.2)
        true_df = exp_df[exp_df["Model"] == "True normalized RUL"].sort_values("x")
        ax.plot(true_df["x"], true_df["y"], label="True normalized RUL", linewidth=2.7, color=COLORS["ink"])
        for model_name in MODEL_ORDER:
            model_df = exp_df[exp_df["Model"] == model_name].sort_values("x")
            if model_df.empty:
                continue
            ax.plot(model_df["x"], model_df["y"], label=model_name, color=MODEL_PALETTE[model_name], alpha=0.9, linewidth=1.9)
        polish_axes(ax, "Normalized test-life index", "Normalized RUL", grid_axis="both")
        top_legend(ax, ncol=3)
        add_header(
            fig,
            f"{exp_name}: true versus predicted RUL",
            "Normalized remaining useful life on the held-out test run.",
        )
        save_current_figure(paths, f'{exp_name.lower().replace(" ", "_")}_true_vs_predicted_rul.png')


def plot_metric_comparisons(paths: ProjectPaths, final_results: pd.DataFrame) -> None:
    metric_specs = [
        ("RMSE", "RMSE", "Lower is better", "fig_metric_comparison_rmse.png"),
        ("MAE", "MAE", "Lower is better", "fig_metric_comparison_mae.png"),
        ("R2", "R2", "Higher is better", "fig_metric_comparison_r2.png"),
    ]
    for metric_col, ylabel, subtitle, filename in metric_specs:
        fig, ax = new_figure(10.8, 5.4)
        plot_df = final_results.copy()
        sns.barplot(
            data=plot_df,
            x="Experiment",
            y=metric_col,
            hue="Model",
            hue_order=[m for m in MODEL_ORDER if m in plot_df["Model"].unique()],
            palette=MODEL_PALETTE,
            ax=ax,
            edgecolor=COLORS["ink"],
            linewidth=0.7,
        )
        if metric_col == "R2":
            ax.axhline(0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
        polish_axes(ax, "Experiment", ylabel)
        top_legend(ax, ncol=2)
        add_header(fig, f"{ylabel} by model and experiment", subtitle)
        save_current_figure(paths, filename)


def plot_average_model_performance(paths: ProjectPaths, average: pd.DataFrame) -> None:
    for metric_col, ylabel, filename, title, subtitle in [
        ("Average RMSE", "Average RMSE", "fig_average_rmse_by_model.png", "Average RMSE by model", "Lower is better across the three experiments."),
        ("Average MAE", "Average MAE", "fig_average_mae_by_model.png", "Average MAE by model", "Lower is better across the three experiments."),
        ("Average R2", "Average R2", "fig_average_r2_by_model.png", "Average R2 by model", "Higher is better across the three experiments."),
    ]:
        ordered = average.set_index("Model").loc[[m for m in MODEL_ORDER if m in average["Model"].values]].reset_index()
        plot_df = ordered.sort_values(metric_col, ascending=True if metric_col != "Average R2" else False)
        fig, ax = new_figure(10.2, 5.2)
        sns.barplot(
            data=plot_df,
            y="Model",
            x=metric_col,
            hue="Model",
            palette=MODEL_PALETTE,
            legend=False,
            ax=ax,
            edgecolor=COLORS["ink"],
            linewidth=0.8,
        )
        if metric_col == "Average R2":
            ax.axvline(0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
        for patch in ax.patches:
            width = patch.get_width()
            ax.text(
                width + (0.015 if width >= 0 else -0.015),
                patch.get_y() + patch.get_height() / 2,
                f"{width:.3f}",
                va="center",
                ha="left" if width >= 0 else "right",
                fontsize=9.6,
                color=COLORS["ink"],
            )
        polish_axes(ax, ylabel, "")
        add_header(fig, title, subtitle)
        save_current_figure(paths, filename)


def plot_model_ranking(paths: ProjectPaths, ranking: pd.DataFrame) -> None:
    fig, ax = new_figure(10.4, 5.2)
    plot_df = ranking.copy()
    sns.barplot(
        data=plot_df,
        x="Experiment",
        y="Rank by RMSE",
        hue="Model",
        hue_order=[m for m in MODEL_ORDER if m in plot_df["Model"].unique()],
        palette=MODEL_PALETTE,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.7,
    )
    ax.set_yticks([1, 2, 3, 4])
    polish_axes(ax, "Experiment", "Rank by RMSE")
    top_legend(ax, ncol=2)
    add_header(fig, "Model ranking by RMSE", "Lower rank indicates lower prediction error in that experiment.")
    save_current_figure(paths, "fig_model_ranking_by_rmse.png")


def plot_methodology_flowchart(paths: ProjectPaths) -> None:
    fig, ax = new_figure(8.4, 8.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Local IMS data", "Read extracted run-to-failure vibration folders from data/raw."),
        ("Feature extraction", "Compute time-domain and envelope fault-frequency energy features."),
        ("Preprocessing", "Normalize features by each run's early healthy baseline and build RUL labels."),
        ("Health indicator", "Fit PCA-HI and compare it with normalized damage."),
        ("Model comparison", "Train data-only, physics-informed, LSTM, and CNN RUL models."),
        ("Thesis outputs", "Write reproducible tables, figures, manifests, and LaTeX assets."),
    ]
    box_color = "#F8FAFC"
    edge_color = COLORS["slate_dark"]
    y_positions = np.linspace(0.84, 0.16, len(steps))
    for idx, ((title, subtitle), y_pos) in enumerate(zip(steps, y_positions)):
        patch = FancyBboxPatch(
            (0.12, y_pos - 0.055),
            0.76,
            0.095,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=edge_color,
            facecolor=box_color,
        )
        ax.add_patch(patch)
        ax.text(0.16, y_pos + 0.012, title, ha="left", va="center", fontsize=12.0, weight="bold", color=COLORS["ink"])
        ax.text(0.16, y_pos - 0.022, subtitle, ha="left", va="center", fontsize=9.6, color=COLORS["muted"])
        if idx < len(steps) - 1:
            arrow = FancyArrowPatch(
                (0.5, y_pos - 0.062),
                (0.5, y_positions[idx + 1] + 0.052),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.0,
                color=COLORS["slate_dark"],
            )
            ax.add_patch(arrow)
    add_header(fig, "Methodology workflow", "Local project stages used to generate the thesis results.")
    save_current_figure(paths, "methodology_flowchart.png")


def plot_experiment_split_matrix(paths: ProjectPaths, experiment_split: pd.DataFrame) -> None:
    fig, ax = new_figure(9.4, 4.8)
    runs = [run_id for run_id, _, _ in TARGET_RUNS]
    role_colors = {
        "Train": COLORS["blue"],
        "Validation": COLORS["gold"],
        "Test": COLORS["vermillion"],
        "Unused": "#F2F4F7",
    }
    role_text_colors = {
        "Train": "#FFFFFF",
        "Validation": COLORS["ink"],
        "Test": "#FFFFFF",
        "Unused": COLORS["muted"],
    }
    ax.set_xlim(0, len(runs))
    ax.set_ylim(0, len(EXPERIMENTS))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(runs)) + 0.5)
    ax.set_xticklabels(runs)
    ax.set_yticks(np.arange(len(EXPERIMENTS)) + 0.5)
    ax.set_yticklabels([spec["name"] for spec in EXPERIMENTS])
    for row_idx, spec in enumerate(EXPERIMENTS):
        for col_idx, run_id in enumerate(runs):
            role = "Unused"
            if run_id in spec["train_runs"]:
                role = "Train"
            elif run_id in spec["validation_runs"]:
                role = "Validation"
            elif run_id in spec["test_runs"]:
                role = "Test"
            rect = Rectangle((col_idx, row_idx), 1, 1, facecolor=role_colors[role], edgecolor=COLORS["panel"], linewidth=2.0)
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, row_idx + 0.5, role, ha="center", va="center", fontsize=10.0, weight="bold", color=role_text_colors[role])
    polish_axes(ax, "Bearing run", "Experiment", grid_axis="")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=COLORS["ink"], label=role) for role, color in role_colors.items() if role != "Unused"]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, borderaxespad=0)
    add_header(fig, "Experiment split matrix", "Each experiment holds out complete bearing runs for validation and testing.")
    save_current_figure(paths, "experiment_split_matrix.png")


def plot_feature_correlation_heatmap(paths: ProjectPaths, proc_df: pd.DataFrame) -> None:
    corr_cols = [col for col in [*MODEL_COLS, TARGET_COL] if col in proc_df.columns]
    label_map = {
        "rul_norm": "RUL",
        "mean_abs": "mean abs",
        "crest_factor": "crest factor",
    }
    corr = proc_df[corr_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).corr(method="spearman")
    corr = corr.rename(index=label_map, columns=label_map)
    fig, ax = new_figure(9.0, 7.6)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        linecolor=COLORS["panel"],
        cbar_kws={"label": "Spearman correlation"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    add_header(fig, "Feature correlation structure", "Spearman correlation among processed features and normalized RUL.")
    save_current_figure(paths, "feature_correlation_heatmap.png")


def plot_best_model_by_experiment(paths: ProjectPaths, final_results: pd.DataFrame) -> None:
    best = final_results.sort_values(["Experiment", "RMSE"]).groupby("Experiment", as_index=False).first()
    best = best.copy()
    best["Model label"] = best["Model"].map(MODEL_SHORT_LABELS).fillna(best["Model"])
    fig, ax = new_figure(8.8, 5.0)
    x_pos = np.arange(len(best))
    bars = ax.bar(
        x_pos,
        best["RMSE"].to_numpy(dtype=float),
        color=[MODEL_PALETTE[model] for model in best["Model"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
        width=0.58,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best["Experiment"])
    for patch, (_, row) in zip(bars, best.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.008,
            f"{row['Model label']}\n{row['RMSE']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.6,
            color=COLORS["ink"],
        )
    polish_axes(ax, "Experiment", "Lowest RMSE")
    winning_models = [model for model in MODEL_ORDER if model in set(best["Model"])]
    ax.legend(handles=model_handles(winning_models), loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=len(winning_models), borderaxespad=0)
    add_header(fig, "Lowest-RMSE model in each experiment", "Winners differ across held-out bearing splits.")
    save_current_figure(paths, "best_model_by_experiment.png")


def prediction_errors_from_table(prediction_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp_name, exp_df in prediction_table.groupby("Experiment", sort=True):
        true_df = exp_df[exp_df["Model"] == "True normalized RUL"].sort_values("x")
        if true_df.empty:
            continue
        x_true = true_df["x"].to_numpy(dtype=float)
        y_true = true_df["y"].to_numpy(dtype=float)
        for model_name in MODEL_ORDER:
            model_df = exp_df[exp_df["Model"] == model_name].sort_values("x")
            if model_df.empty:
                continue
            x_model = model_df["x"].to_numpy(dtype=float)
            y_pred = model_df["y"].to_numpy(dtype=float)
            y_aligned = np.interp(x_model, x_true, y_true)
            for x_value, pred_value, true_value in zip(x_model, y_pred, y_aligned):
                signed_error = float(pred_value - true_value)
                rows.append(
                    {
                        "Experiment": exp_name,
                        "Model": model_name,
                        "Model label": MODEL_SHORT_LABELS.get(model_name, model_name),
                        "x": float(x_value),
                        "true": float(true_value),
                        "predicted": float(pred_value),
                        "signed_error": signed_error,
                        "absolute_error": abs(signed_error),
                    }
                )
    return pd.DataFrame(rows)


def plot_absolute_error_over_life(paths: ProjectPaths, error_df: pd.DataFrame) -> None:
    apply_theme()
    experiments = [spec["name"] for spec in EXPERIMENTS if spec["name"] in set(error_df["Experiment"])]
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10.2, 8.8), sharex=True, constrained_layout=False)
    axes = np.atleast_1d(axes)
    for ax, exp_name in zip(axes, experiments):
        exp_df = error_df[error_df["Experiment"] == exp_name]
        for model_name in MODEL_ORDER:
            model_df = exp_df[exp_df["Model"] == model_name].sort_values("x")
            if model_df.empty:
                continue
            window = max(7, int(len(model_df) * 0.03))
            smooth_error = model_df["absolute_error"].rolling(window=window, min_periods=1, center=True).mean()
            ax.plot(model_df["x"], smooth_error, color=MODEL_PALETTE[model_name], linewidth=1.8, label=model_name)
        ax.text(0.01, 0.88, exp_name, transform=ax.transAxes, fontsize=11.2, weight="bold", color=COLORS["ink"])
        polish_axes(ax, None, "Absolute error", grid_axis="both")
    axes[-1].set_xlabel("Normalized test-life index")
    top_legend(axes[0], ncol=2)
    add_header(fig, "Smoothed absolute prediction error over test life", "Errors are aligned to each model's prediction positions.")
    save_current_figure(paths, "absolute_error_over_life.png")


def plot_prediction_error_distribution(paths: ProjectPaths, error_df: pd.DataFrame) -> None:
    fig, ax = new_figure(10.2, 5.4)
    order = [MODEL_SHORT_LABELS[m] for m in MODEL_ORDER if m in set(error_df["Model"])]
    palette = {MODEL_SHORT_LABELS[m]: MODEL_PALETTE[m] for m in MODEL_ORDER if m in set(error_df["Model"])}
    sns.boxplot(
        data=error_df,
        y="Model label",
        x="signed_error",
        order=order,
        hue="Model label",
        palette=palette,
        legend=False,
        ax=ax,
        linewidth=0.9,
        fliersize=1.5,
    )
    ax.axvline(0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
    polish_axes(ax, "Predicted RUL - true RUL", "")
    add_header(fig, "Signed prediction error distribution", "Positive values indicate overestimated remaining useful life.")
    save_current_figure(paths, "prediction_error_distribution.png")


def plot_prediction_error_analysis(paths: ProjectPaths, prediction_table: pd.DataFrame) -> None:
    error_df = prediction_errors_from_table(prediction_table)
    if error_df.empty:
        return
    summary = (
        error_df.groupby(["Experiment", "Model", "Model label"], as_index=False)
        .agg(
            mean_signed_error=("signed_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            max_absolute_error=("absolute_error", "max"),
        )
    )
    save_table(paths, summary, "prediction_error_summary_table.csv")
    plot_absolute_error_over_life(paths, error_df)
    plot_prediction_error_distribution(paths, error_df)


def make_dataset_summary(multi_df: pd.DataFrame) -> pd.DataFrame:
    run_metadata = {run_id: {"IMS dataset": dataset, "Bearing number": bearing} for run_id, dataset, bearing in TARGET_RUNS}
    usage_by_run = {run_id: [] for run_id in run_metadata}
    for spec in EXPERIMENTS:
        for role_key, role_label in [
            ("train_runs", "Train"),
            ("validation_runs", "Validation"),
            ("test_runs", "Test"),
        ]:
            for run_id in spec[role_key]:
                usage_by_run.setdefault(run_id, []).append(f"{spec['name']} {role_label}")

    rows = []
    for run_id, _, _ in TARGET_RUNS:
        run_df = multi_df[multi_df["run_id"] == run_id]
        rows.append(
            {
                "Run ID": run_id,
                "IMS dataset": run_metadata[run_id]["IMS dataset"],
                "Bearing number": run_metadata[run_id]["Bearing number"],
                "Failure type if available": "Not specified in cleaned notebook",
                "Number of samples": int(len(run_df)),
                "Used in experiments": "; ".join(usage_by_run.get(run_id, [])),
                "Notes": "Run-to-failure IMS bearing run used in cross-bearing experiments",
            }
        )
    return pd.DataFrame(rows)


def save_static_tables_and_figures(
    paths: ProjectPaths,
    multi_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    pca_hi_df: pd.DataFrame,
    pca_hi_summary: pd.DataFrame,
    final_results: pd.DataFrame | None,
    prediction_store: dict[str, dict[str, object]] | None,
) -> None:
    plot_methodology_flowchart(paths)

    dataset_summary = make_dataset_summary(multi_df)
    save_table(paths, dataset_summary, "dataset_summary_table.csv")
    fig, ax = new_figure(8.6, 5.0)
    sns.barplot(
        data=dataset_summary,
        x="Run ID",
        y="Number of samples",
        hue="Run ID",
        palette=RUN_PALETTE,
        legend=False,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, value + 70, f"{value:,.0f}", ha="center", va="bottom", fontsize=9.8, color=COLORS["ink"])
    polish_axes(ax, "Bearing run", "Number of vibration snapshots")
    add_header(fig, "Snapshot coverage by bearing run", "Counts reflect the extracted IMS folders used in the local pipeline.")
    save_current_figure(paths, "fig_dataset_sample_counts.png")

    feature_description = pd.DataFrame(
        [
            ("elapsed_scaled", "time feature", "Normalized elapsed operating time used as temporal input"),
            ("rms", "time-domain", "Signal energy amplitude through root mean square vibration"),
            ("std", "time-domain", "Vibration dispersion around the mean"),
            ("ptp", "time-domain", "Peak-to-peak vibration range"),
            ("kurtosis", "time-domain", "Impulsiveness of vibration signal"),
            ("crest_factor", "time-domain", "Peak amplitude relative to RMS"),
            ("mean_abs", "time-domain", "Mean absolute vibration amplitude"),
            ("E_FTF", "physics-informed frequency feature", "Envelope spectral energy around cage fault train frequency harmonics"),
            ("E_BPFO", "physics-informed frequency feature", "Envelope spectral energy around outer-race fault frequency harmonics"),
            ("E_BPFI", "physics-informed frequency feature", "Envelope spectral energy around inner-race fault frequency harmonics"),
            ("E_BSF", "physics-informed frequency feature", "Envelope spectral energy around rolling-element fault frequency harmonics"),
            ("E_kin", "physics-informed frequency feature", "Combined bearing kinematic fault-frequency spectral energy"),
            ("sigma_H_norm", "auxiliary placeholder", "Constant placeholder; not a Hertzian contact stress calculation"),
        ],
        columns=["Feature name", "Domain", "Physical meaning"],
    )
    feature_description["Used by models"] = "yes"
    feature_description = feature_description[feature_description["Feature name"].isin(FEATURE_COLS_MULTI)]
    save_table(paths, feature_description, "feature_description_table.csv")

    fault_component_map = {
        "FTF": ("FTF", "Cage / fundamental train"),
        "BPFO": ("BPFO", "Outer race"),
        "BPFI": ("BPFI", "Inner race"),
        "BSF": ("BSF", "Rolling element"),
    }
    fault_frequency = pd.DataFrame(
        [
            {
                "Frequency name": name,
                "Symbol": symbol,
                "Value in Hz": float(FAULT_FREQS[name]),
                "Related bearing component": component,
                "Extracted feature": f"E_{name}",
            }
            for name, (symbol, component) in fault_component_map.items()
        ]
    )
    save_table(paths, fault_frequency, "fault_frequency_table.csv")
    fig, ax = new_figure(8.8, 5.0)
    frequency_palette = {
        "FTF": COLORS["blue"],
        "BPFO": COLORS["gold"],
        "BPFI": COLORS["emerald"],
        "BSF": COLORS["violet"],
    }
    sns.barplot(
        data=fault_frequency,
        x="Symbol",
        y="Value in Hz",
        hue="Symbol",
        palette=frequency_palette,
        legend=False,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.8,
    )
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, value + 8, f"{value:.1f}", ha="center", va="bottom", fontsize=9.8, color=COLORS["ink"])
    polish_axes(ax, "Fault-frequency symbol", "Frequency (Hz)")
    add_header(fig, "Bearing fault-frequency markers", "Kinematic frequencies used for envelope spectral-energy features.")
    save_current_figure(paths, "fig_fault_frequencies.png")

    experiment_purpose = {
        "Exp 1": "Cross-dataset/cross-bearing transfer with ds3_b3 held out for testing",
        "Exp 2": "Alternative training-bearing combination with the same ds3_b3 test run",
        "Exp 3": "Cross-dataset transfer to ds1_b3 with balanced training runs",
    }
    experiment_split = pd.DataFrame(
        [
            {
                "Experiment": spec["name"],
                "Train runs": run_label(spec["train_runs"]),
                "Validation run": run_label(spec["validation_runs"]),
                "Test run": run_label(spec["test_runs"]),
                "Purpose": experiment_purpose.get(spec["name"], "Cross-bearing RUL generalization experiment"),
            }
            for spec in EXPERIMENTS
        ]
    )
    save_table(paths, experiment_split, "experiment_split_table.csv")
    plot_experiment_split_matrix(paths, experiment_split)

    model_summary = pd.DataFrame(
        [
            (DATA_BASELINE_NAME, "Feed-forward neural network", "Single snapshot feature vector", "Data-only baseline"),
            (PROPOSED_MODEL_NAME, "DeepXDE physics-informed neural network", "Single snapshot feature vector with time and fault-frequency energy features", "Proposed model"),
            (LSTM_BASELINE_NAME, "Recurrent neural network", "Sliding-window sequence of feature vectors", "Sequence baseline"),
            (CNN_BASELINE_NAME, "1D convolutional neural network", "Sliding-window sequence of feature vectors", "Sequence baseline"),
        ],
        columns=["Model", "Model type", "Input format", "Role in thesis"],
    )
    model_summary["Target"] = "Normalized RUL"
    save_table(paths, model_summary, "model_summary_table.csv")

    pca_table = pca_hi_summary.rename(
        columns={
            "Run": "Bearing run",
            "Monotonic increase score": "Monotonicity score",
            "Spearman corr(PCA-HI, damage)": "Correlation with damage",
        }
    )[["Bearing run", "Monotonicity score", "Correlation with damage"]]
    pca_table["Interpretation"] = np.where(
        (pca_table["Correlation with damage"] >= 0.7) & (pca_table["Monotonicity score"] >= 0.7),
        "Strongly degradation-consistent PCA-HI behavior",
        "Moderately or weakly degradation-consistent PCA-HI behavior",
    )
    save_table(paths, pca_table, "pca_hi_ablation_table.csv")
    plot_pca_hi(paths, pca_hi_df, pca_hi_summary)
    plot_feature_correlation_heatmap(paths, proc_df)

    for feature_col, y_label, filename in [
        ("rms", "RMS", "fig_feature_evolution_rms.png"),
        ("kurtosis", "Kurtosis", "fig_feature_evolution_kurtosis.png"),
        ("crest_factor", "Crest factor", "fig_feature_evolution_crest_factor.png"),
        ("E_kin", "Combined fault-frequency energy E_kin", "fig_feature_evolution_e_kin.png"),
    ]:
        fig, ax = new_figure(10.2, 5.2)
        for run_id, _, _ in TARGET_RUNS:
            run = proc_df[proc_df["run_id"] == run_id].sort_values("elapsed_scaled")
            if not run.empty:
                ax.plot(
                    normalized_life_for_run(run),
                    run[feature_col],
                    label=run_id,
                    color=RUN_PALETTE.get(run_id, COLORS["blue"]),
                    linewidth=1.9,
                    alpha=0.92,
                )
        polish_axes(ax, "Normalized lifetime", y_label, grid_axis="both")
        top_legend(ax, ncol=4)
        add_header(fig, f"{y_label} evolution over bearing life", "Features are normalized by each run's early healthy baseline.")
        save_current_figure(paths, filename)

    if final_results is None:
        return

    save_table(paths, final_results, "final_results_table.csv")
    plot_metric_comparisons(paths, final_results)
    average = (
        final_results.groupby("Model", as_index=False)
        .agg({"MAE": "mean", "RMSE": "mean", "R2": "mean"})
        .rename(columns={"MAE": "Average MAE", "RMSE": "Average RMSE", "R2": "Average R2"})
    )
    save_table(paths, average, "average_model_performance_table.csv")
    plot_average_model_performance(paths, average)

    ranking = final_results[["Experiment", "Model", "RMSE"]].copy()
    ranking["Rank by RMSE"] = ranking.groupby("Experiment")["RMSE"].rank(method="dense", ascending=True).astype(int)
    ranking = ranking.sort_values(["Experiment", "Rank by RMSE"]).reset_index(drop=True)
    save_table(paths, ranking, "model_ranking_by_rmse.csv")
    plot_model_ranking(paths, ranking)
    plot_best_model_by_experiment(paths, final_results)

    if prediction_store is not None:
        plot_final_predictions(paths, EXPERIMENTS, prediction_store)
        plot_prediction_error_analysis(paths, prediction_store_to_table(prediction_store))


def normalized_life_for_run(run_df: pd.DataFrame) -> pd.Series:
    values = run_df["elapsed_scaled"].astype(float)
    return (values - values.min()) / (values.max() - values.min() + 1e-12)
