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


def apply_large_split_chart_fonts(ax, xlabel: str, ylabel: str, legend_cols: int = 4) -> None:
    ax.set_xlabel(xlabel, fontsize=14.5, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14.5, labelpad=8)
    ax.tick_params(axis="both", labelsize=11.8, colors=COLORS["ink"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_color(COLORS["ink"])
    for tick in ax.get_yticklabels():
        tick.set_color(COLORS["ink"])
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    top_legend(ax, ncol=legend_cols)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12.2)
            text.set_color(COLORS["ink"])
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(12.2)
            legend.get_title().set_color(COLORS["ink"])


def draw_grouped_split_bars(
    ax,
    grouped: pd.DataFrame,
    value_col: str,
    split_order: list[str],
    model_order: list[str],
    error_col: str | None = None,
) -> None:
    x_pos = np.arange(len(split_order))
    bar_width = min(0.18, 0.78 / max(1, len(model_order)))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2.0) * bar_width
    for idx, model_name in enumerate(model_order):
        model_df = grouped[grouped["Model"] == model_name].set_index("Experiment").reindex(split_order)
        values = model_df[value_col].astype(float).to_numpy()
        errors = None
        if error_col is not None and error_col in model_df.columns:
            errors = model_df[error_col].fillna(0.0).astype(float).to_numpy()
        ax.bar(
            x_pos + offsets[idx],
            values,
            width=bar_width,
            yerr=errors,
            capsize=2.8 if errors is not None else 0,
            label=MODEL_SHORT_LABELS.get(model_name, model_name),
            color=MODEL_PALETTE[model_name],
            edgecolor=COLORS["ink"],
            linewidth=0.75,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(split_order)


def plot_metric_comparisons(paths: ProjectPaths, final_results: pd.DataFrame) -> None:
    metric_specs = [
        ("RMSE", "RMSE", "Lower values are better.", "fig_metric_comparison_rmse.png", "RMSE by model and split configuration"),
        ("MAE", "MAE", "Lower values are better.", "fig_metric_comparison_mae.png", "MAE by model and split configuration"),
        ("R2", "R2", "Higher values are better.", "fig_metric_comparison_r2.png", "R2 by model and split configuration"),
    ]
    split_order = [spec["name"] for spec in EXPERIMENTS if spec["name"] in set(final_results["Experiment"])]
    model_order = [m for m in MODEL_ORDER if m in set(final_results["Model"])]

    for metric_col, ylabel, subtitle, filename, title in metric_specs:
        grouped = (
            final_results.groupby(["Experiment", "Model"], as_index=False)
            .agg(Value=(metric_col, "mean"), SD=(metric_col, "std"))
            .fillna({"SD": 0.0})
        )
        fig, ax = new_figure(13.6, 6.1)
        draw_grouped_split_bars(ax, grouped, "Value", split_order, model_order, error_col="SD")
        if metric_col == "R2":
            ax.axhline(0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
        polish_axes(ax, "Split configuration", ylabel)
        apply_large_split_chart_fonts(ax, "Split configuration", ylabel)
        add_header(fig, title, subtitle)
        save_current_figure(paths, filename)


def plot_average_model_performance(paths: ProjectPaths, average: pd.DataFrame) -> None:
    metric_specs = [
        ("Average RMSE", "SD RMSE", "Average RMSE", "fig_average_rmse_by_model.png", "Average RMSE by model", "Mean +/- one SD across the 12 split configurations and repeated seeds."),
        ("Average MAE", "SD MAE", "Average MAE", "fig_average_mae_by_model.png", "Average MAE by model", "Mean +/- one SD across the 12 split configurations and repeated seeds."),
        ("Average R2", "SD R2", "Average R2", "fig_average_r2_by_model.png", "Average R2 by model", "Mean +/- one SD across the 12 split configurations and repeated seeds."),
    ]
    for metric_col, sd_col, xlabel, filename, title, subtitle in metric_specs:
        ordered = average.set_index("Model").loc[[m for m in MODEL_ORDER if m in average["Model"].values]].reset_index()
        ordered = ordered.sort_values(metric_col, ascending=True if metric_col != "Average R2" else False).reset_index(drop=True)
        fig, ax = new_figure(10.8, 5.2)
        values = ordered[metric_col].to_numpy(dtype=float)
        errors = ordered[sd_col].to_numpy(dtype=float) if sd_col in ordered.columns else np.zeros(len(ordered))
        y_pos = np.arange(len(ordered))
        ax.barh(
            y_pos,
            values,
            xerr=errors,
            color=[MODEL_PALETTE[m] for m in ordered["Model"]],
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            capsize=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_SHORT_LABELS.get(model, model) for model in ordered["Model"]])
        ax.invert_yaxis()
        if metric_col == "Average R2":
            ax.axvline(0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
        max_extent = float(np.nanmax(np.abs(values) + np.nan_to_num(errors))) if len(values) else 1.0
        label_pad = max(0.012, max_extent * 0.035)
        for idx, row in ordered.iterrows():
            value = float(row[metric_col])
            sd = float(row[sd_col]) if sd_col in ordered.columns else 0.0
            label = f"{value:.3f} +/- {sd:.3f}" if sd_col in ordered.columns else f"{value:.3f}"
            xpos = value + (sd if value >= 0 else -sd) + (label_pad if value >= 0 else -label_pad)
            ax.text(
                xpos,
                idx,
                label,
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9.5,
                color=COLORS["ink"],
            )
        polish_axes(ax, xlabel, "")
        add_header(fig, title, subtitle)
        save_current_figure(paths, filename)


def plot_model_ranking(paths: ProjectPaths, ranking: pd.DataFrame) -> None:
    plot_df = ranking.copy()
    split_order = [spec["name"] for spec in EXPERIMENTS if spec["name"] in set(plot_df["Experiment"])]
    model_order = [m for m in MODEL_ORDER if m in set(plot_df["Model"])]

    fig, ax = new_figure(13.6, 5.9)
    draw_grouped_split_bars(ax, plot_df, "Rank by RMSE", split_order, model_order)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0, 4.35)
    polish_axes(ax, "Split configuration", "Rank by RMSE")
    apply_large_split_chart_fonts(ax, "Split configuration", "Rank by RMSE")
    add_header(fig, "Model ranking by RMSE", "Lower rank indicates lower prediction error in that split configuration.")
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
    fig, ax = new_figure(9.8, max(5.4, 0.46 * len(EXPERIMENTS) + 1.6))
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
            ax.text(col_idx + 0.5, row_idx + 0.5, role, ha="center", va="center", fontsize=8.6, weight="bold", color=role_text_colors[role])
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
    ax.tick_params(axis="x", rotation=45, colors=COLORS["ink"])
    ax.tick_params(axis="y", rotation=0, colors=COLORS["ink"])
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(COLORS["ink"])
        cbar.ax.tick_params(colors=COLORS["ink"])
    add_header(fig, "Feature correlation structure", "Spearman correlation among processed features and normalized RUL.")
    save_current_figure(paths, "feature_correlation_heatmap.png")


def plot_best_model_by_experiment(paths: ProjectPaths, final_results: pd.DataFrame) -> None:
    best = final_results.sort_values(["Experiment", "RMSE"]).groupby("Experiment", as_index=False).first()
    best = best.copy()
    best["Model label"] = best["Model"].map(MODEL_SHORT_LABELS).fillna(best["Model"])
    fig, ax = new_figure(10.0, 5.2)
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
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
        tick.set_color(COLORS["ink"])
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
    if not experiments:
        return

    common_x = np.linspace(0.0, 1.0, 140)
    rows: list[dict[str, object]] = []
    for exp_name in experiments:
        exp_df = error_df[error_df["Experiment"] == exp_name]
        for model_name in MODEL_ORDER:
            model_df = exp_df[exp_df["Model"] == model_name].sort_values("x")
            if model_df.empty:
                continue
            window = max(7, int(len(model_df) * 0.03))
            smooth_error = model_df["absolute_error"].rolling(window=window, min_periods=1, center=True).mean().to_numpy(dtype=float)
            x_values = model_df["x"].to_numpy(dtype=float)
            interp_error = np.interp(common_x, x_values, smooth_error)
            for x_value, error_value in zip(common_x, interp_error):
                rows.append(
                    {
                        "Experiment": exp_name,
                        "Model": model_name,
                        "Model label": MODEL_SHORT_LABELS.get(model_name, model_name),
                        "x": float(x_value),
                        "absolute_error": float(error_value),
                    }
                )
    if not rows:
        return

    aligned = pd.DataFrame(rows)
    summary = (
        aligned.groupby(["Model", "Model label", "x"], as_index=False)
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            q25_absolute_error=("absolute_error", lambda values: float(np.quantile(values, 0.25))),
            q75_absolute_error=("absolute_error", lambda values: float(np.quantile(values, 0.75))),
        )
    )

    fig, ax = new_figure(12.4, 6.8)
    for model_name in MODEL_ORDER:
        model_summary = summary[summary["Model"] == model_name].sort_values("x")
        if model_summary.empty:
            continue
        x = model_summary["x"].to_numpy(dtype=float)
        mean_error = model_summary["mean_absolute_error"].to_numpy(dtype=float)
        q25 = model_summary["q25_absolute_error"].to_numpy(dtype=float)
        q75 = model_summary["q75_absolute_error"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean_error,
            color=MODEL_PALETTE[model_name],
            linewidth=2.2,
            label=MODEL_SHORT_LABELS.get(model_name, model_name),
        )
        ax.fill_between(x, q25, q75, color=MODEL_PALETTE[model_name], alpha=0.14, linewidth=0)
    polish_axes(ax, "Normalized test-life index", "Absolute error", grid_axis="both")
    top_legend(ax, ncol=4)
    add_header(
        fig,
        "Mean absolute prediction error over test life",
        "One graph summarizes the 12 split configurations; shaded bands show the interquartile range across splits.",
    )
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



def aggregate_seed_results(final_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Experiment", "Train runs", "Validation run", "Test run", "Model"]
    if final_results.empty:
        return final_results.copy()
    if "Seed repeat" not in final_results.columns:
        split_results = final_results.copy()
        split_results["Seed repeats"] = 1
        split_results["MAE SD"] = 0.0
        split_results["RMSE SD"] = 0.0
        split_results["R2 SD"] = 0.0
        return split_results[group_cols + ["Seed repeats", "MAE", "MAE SD", "RMSE", "RMSE SD", "R2", "R2 SD"]]

    grouped = final_results.groupby(group_cols, as_index=False)
    split_results = grouped.agg(
        **{
            "Seed repeats": ("Seed repeat", "nunique"),
            "MAE": ("MAE", "mean"),
            "MAE SD": ("MAE", "std"),
            "RMSE": ("RMSE", "mean"),
            "RMSE SD": ("RMSE", "std"),
            "R2": ("R2", "mean"),
            "R2 SD": ("R2", "std"),
        }
    )
    return split_results.fillna(0.0)


def make_statistical_validation_summary(final_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        final_results.groupby("Model", as_index=False)
        .agg(
            Split_configurations=("Experiment", "nunique"),
            Model_runs=("RMSE", "count"),
            Mean_MAE=("MAE", "mean"),
            SD_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            SD_RMSE=("RMSE", "std"),
            Min_RMSE=("RMSE", "min"),
            Max_RMSE=("RMSE", "max"),
            Mean_R2=("R2", "mean"),
            SD_R2=("R2", "std"),
        )
        .fillna(0.0)
    )
    if "Seed repeat" in final_results.columns:
        seed_counts = final_results.groupby("Model")["Seed repeat"].nunique().rename("Seed_repeats")
        summary = summary.merge(seed_counts, on="Model", how="left")
    else:
        summary["Seed_repeats"] = 1
    summary["Seed_repeats"] = summary["Seed_repeats"].fillna(1).astype(int)
    ordered_cols = [
        "Model",
        "Split_configurations",
        "Seed_repeats",
        "Model_runs",
        "Mean_MAE",
        "SD_MAE",
        "Mean_RMSE",
        "SD_RMSE",
        "Min_RMSE",
        "Max_RMSE",
        "Mean_R2",
        "SD_R2",
    ]
    return summary[ordered_cols]


def plot_statistical_validation_summary(paths: ProjectPaths, summary: pd.DataFrame) -> None:
    ordered = summary.set_index("Model").loc[[m for m in MODEL_ORDER if m in summary["Model"].values]].reset_index()
    ordered["Model label"] = ordered["Model"].map(MODEL_SHORT_LABELS).fillna(ordered["Model"])
    fig, ax = new_figure(10.2, 5.2)
    y_pos = np.arange(len(ordered))
    ax.barh(
        y_pos,
        ordered["Mean_RMSE"].to_numpy(dtype=float),
        xerr=ordered["SD_RMSE"].to_numpy(dtype=float),
        color=[MODEL_PALETTE[m] for m in ordered["Model"]],
        edgecolor=COLORS["ink"],
        linewidth=0.9,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["Model label"])
    ax.invert_yaxis()
    for idx, row in ordered.iterrows():
        ax.text(
            float(row["Mean_RMSE"] + row["SD_RMSE"]) + 0.01,
            idx,
            f'{row["Mean_RMSE"]:.3f} +/- {row["SD_RMSE"]:.3f}',
            va="center",
            ha="left",
            fontsize=9.5,
            color=COLORS["ink"],
        )
    polish_axes(ax, "Mean RMSE +/- one SD", "")
    add_header(fig, "Repeated-seed statistical validation", "Summary across 12 split configurations and three random seeds per split.")
    save_current_figure(paths, "statistical_validation_rmse_sensitivity.png")


def block_bootstrap_metric_intervals(
    error_df: pd.DataFrame,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    group_cols = ["Experiment", "Model", "Model label"]
    for (exp_name, model_name, model_label), group in error_df.groupby(group_cols, sort=True):
        ordered = group.sort_values("x")
        errors = ordered["signed_error"].to_numpy(dtype=float)
        n = len(errors)
        if n == 0:
            continue
        block_len = max(10, int(round(n * 0.05)))
        starts = np.arange(0, n, block_len)
        blocks = [np.arange(start, min(start + block_len, n)) for start in starts]
        n_blocks = int(np.ceil(n / block_len))
        mae_samples = []
        rmse_samples = []
        for _ in range(n_bootstrap):
            chosen = rng.integers(0, len(blocks), size=n_blocks)
            idx = np.concatenate([blocks[i] for i in chosen])[:n]
            sample_errors = errors[idx]
            mae_samples.append(float(np.mean(np.abs(sample_errors))))
            rmse_samples.append(float(np.sqrt(np.mean(sample_errors**2))))
        mae_samples = np.asarray(mae_samples)
        rmse_samples = np.asarray(rmse_samples)
        point_mae = float(np.mean(np.abs(errors)))
        point_rmse = float(np.sqrt(np.mean(errors**2)))
        rows.append(
            {
                "Experiment": exp_name,
                "Model": model_name,
                "Model label": model_label,
                "Samples": n,
                "Block length": block_len,
                "MAE": point_mae,
                "MAE CI low": float(np.quantile(mae_samples, 0.025)),
                "MAE CI high": float(np.quantile(mae_samples, 0.975)),
                "RMSE": point_rmse,
                "RMSE CI low": float(np.quantile(rmse_samples, 0.025)),
                "RMSE CI high": float(np.quantile(rmse_samples, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def plot_bootstrap_rmse_intervals(paths: ProjectPaths, uncertainty: pd.DataFrame) -> None:
    if uncertainty.empty:
        return
    apply_theme()
    plot_df = uncertainty.copy()
    plot_df["Model label"] = plot_df["Model"].map(MODEL_SHORT_LABELS).fillna(plot_df["Model label"])
    experiments = [spec["name"] for spec in EXPERIMENTS if spec["name"] in set(plot_df["Experiment"])]
    models = [m for m in MODEL_ORDER if m in set(plot_df["Model"])]
    if not experiments or not models:
        return

    fig, ax = new_figure(12.4, 7.4)
    base_y = np.arange(len(experiments), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(models)) if len(models) > 1 else np.array([0.0])
    max_high = 0.0
    for offset, model_name in zip(offsets, models):
        model_df = plot_df[plot_df["Model"] == model_name].set_index("Experiment")
        x_values = []
        y_values = []
        lower_errors = []
        upper_errors = []
        for idx, exp_name in enumerate(experiments):
            if exp_name not in model_df.index:
                continue
            row = model_df.loc[exp_name]
            rmse = float(row["RMSE"])
            low = float(row["RMSE CI low"])
            high = float(row["RMSE CI high"])
            x_values.append(rmse)
            y_values.append(base_y[idx] + offset)
            lower_errors.append(max(0.0, rmse - low))
            upper_errors.append(max(0.0, high - rmse))
            max_high = max(max_high, high)
        ax.errorbar(
            x_values,
            y_values,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            color=MODEL_PALETTE[model_name],
            ecolor=MODEL_PALETTE[model_name],
            markeredgecolor=COLORS["ink"],
            markersize=5.2,
            capsize=3.0,
            linewidth=1.15,
            label=MODEL_SHORT_LABELS.get(model_name, model_name),
        )
    ax.set_yticks(base_y)
    ax.set_yticklabels(experiments)
    ax.invert_yaxis()
    if max_high > 0:
        ax.set_xlim(left=0.0, right=max_high * 1.12)
    polish_axes(ax, "RMSE with 95% block-bootstrap interval", "Split configuration", grid_axis="x")
    top_legend(ax, ncol=4)
    add_header(
        fig,
        "Prediction uncertainty by split configuration",
        "One graph shows all split configurations; intervals resample contiguous error blocks within each test run.",
    )
    save_current_figure(paths, "prediction_uncertainty_rmse_intervals.png")

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
    uncertainty = block_bootstrap_metric_intervals(error_df)
    save_table(paths, uncertainty, "prediction_uncertainty_bootstrap_table.csv")
    plot_bootstrap_rmse_intervals(paths, uncertainty)
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
    fig, ax = new_figure(10.0, 5.2)
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
        "Holdout ds2_b1": "Leave-one-bearing-out test on ds2_b1 with balanced training runs",
        "Holdout ds1_b3": "Leave-one-bearing-out test on ds1_b3 with balanced training runs",
        "Holdout ds1_b4": "Leave-one-bearing-out test on ds1_b4 with balanced training runs",
        "Holdout ds3_b3": "Leave-one-bearing-out test on ds3_b3 with balanced training runs",
    }
    experiment_split = pd.DataFrame(
        [
            {
                "Experiment": spec["name"],
                "Train runs": run_label(spec["train_runs"]),
                "Validation run": run_label(spec["validation_runs"]),
                "Test run": run_label(spec["test_runs"]),
                "Purpose": f"Test {run_label(spec['test_runs'])}; validation {run_label(spec['validation_runs'])}",
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

    seed_results = final_results.copy()
    if "Seed repeat" in seed_results.columns:
        save_table(paths, seed_results, "split_seed_results_table.csv")
    split_results = aggregate_seed_results(seed_results)
    save_table(paths, split_results, "final_results_table.csv")

    validation_summary = make_statistical_validation_summary(seed_results)
    save_table(paths, validation_summary, "statistical_validation_summary.csv")
    save_table(paths, validation_summary, "model_seed_split_summary_table.csv")
    plot_statistical_validation_summary(paths, validation_summary)
    plot_metric_comparisons(paths, seed_results)

    average = validation_summary.rename(
        columns={
            "Mean_MAE": "Average MAE",
            "SD_MAE": "SD MAE",
            "Mean_RMSE": "Average RMSE",
            "SD_RMSE": "SD RMSE",
            "Mean_R2": "Average R2",
            "SD_R2": "SD R2",
            "Model_runs": "Runs",
            "Split_configurations": "Split configurations",
            "Seed_repeats": "Seed repeats",
        }
    )[
        [
            "Model",
            "Split configurations",
            "Seed repeats",
            "Runs",
            "Average MAE",
            "SD MAE",
            "Average RMSE",
            "SD RMSE",
            "Average R2",
            "SD R2",
        ]
    ]
    save_table(paths, average, "average_model_performance_table.csv")
    plot_average_model_performance(paths, average)

    ranking = split_results[["Experiment", "Model", "RMSE"]].copy()
    ranking["Rank by RMSE"] = ranking.groupby("Experiment")["RMSE"].rank(method="dense", ascending=True).astype(int)
    ranking = ranking.sort_values(["Experiment", "Rank by RMSE"]).reset_index(drop=True)
    save_table(paths, ranking, "model_ranking_by_rmse.csv")
    plot_model_ranking(paths, ranking)
    plot_best_model_by_experiment(paths, split_results)

    if prediction_store is not None:
        plot_final_predictions(paths, EXPERIMENTS, prediction_store)
        plot_prediction_error_analysis(paths, prediction_store_to_table(prediction_store))


def normalized_life_for_run(run_df: pd.DataFrame) -> pd.Series:
    values = run_df["elapsed_scaled"].astype(float)
    return (values - values.min()) / (values.max() - values.min() + 1e-12)



