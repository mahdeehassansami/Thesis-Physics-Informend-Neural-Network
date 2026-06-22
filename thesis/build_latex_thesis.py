from __future__ import annotations

import math
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
LATEX = THESIS / "latex"
SECTIONS = LATEX / "sections"
IMAGES = LATEX / "assets" / "images"
TABLES = ROOT / "outputs" / "tables"
FEATURES = ROOT / "data" / "processed_features" / "all_runs_features.csv"

TITLE = "Physics-Informed Neural Networks for Remaining-Useful-Life Prediction of Rolling-Element Bearings"
AUTHOR = "Mahdee Hassan Sami"
STUDENT_ID = "2003101"
SESSION = "2024-2025"
COURSE = "ME 494"
SUPERVISOR = "Dr. Md. Abu Mowazzem Hossain"
SUPERVISOR_DESIGNATION = "Professor"

DATA_MODEL = "Data-only neural baseline"
PINN_MODEL = "Proposed DeepXDE Physics-Informed RUL Model"
LSTM_MODEL = "LSTM baseline"
CNN_MODEL = "CNN baseline"
MODEL_ORDER = [DATA_MODEL, PINN_MODEL, LSTM_MODEL, CNN_MODEL]
MODEL_SHORT = {
    DATA_MODEL: "Data-only FNN",
    PINN_MODEL: "Proposed PINN",
    LSTM_MODEL: "LSTM",
    CNN_MODEL: "CNN",
}

COLORS = {
    "ink": "#18202A",
    "muted": "#667085",
    "grid": "#E4EAF2",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "gold": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "teal": "#00A6A6",
}

MODEL_PALETTE = {
    DATA_MODEL: "#7A869A",
    PINN_MODEL: COLORS["blue"],
    LSTM_MODEL: COLORS["green"],
    CNN_MODEL: COLORS["red"],
}

RUN_PALETTE = {
    "ds2_b1": COLORS["blue"],
    "ds1_b3": COLORS["gold"],
    "ds1_b4": COLORS["green"],
    "ds3_b3": COLORS["purple"],
}

FEATURE_COLS = [
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "crest_factor",
    "mean_abs",
    "E_FTF",
    "E_BPFO",
    "E_BPFI",
    "E_BSF",
    "E_kin",
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(content).strip() + "\n"
    path.write_text(text, encoding="utf-8")


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def table_tex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    colspec: str,
    size: str = r"\small",
    digits: int = 3,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        f"    {size}",
        r"    \renewcommand{\arraystretch}{1.25}",
        rf"    \caption{{{latex_escape(caption)}}}",
        rf"    \label{{{label}}}",
        rf"    \begin{{tabularx}}{{\textwidth}}{{{colspec}}}",
        r"        \toprule",
        "        " + " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"        \midrule",
    ]
    for _, row in df.iterrows():
        values = [latex_escape(fmt(row[col], digits=digits)) for col in df.columns]
        lines.append("        " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"\end{table}",
            r"\FloatBarrier",
        ]
    )
    return "\n".join(lines)


def figure_tex(filename: str, caption: str, label: str, width: str = r"0.92\textwidth") -> str:
    return textwrap.dedent(
        rf"""
        \begin{{figure}}[htbp]
            \centering
            \includegraphics[width={width}]{{{filename}}}
            \caption{{{caption}}}
            \label{{{label}}}
        \end{{figure}}
        \FloatBarrier
        """
    ).strip()


def setup_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        },
    )


def polish(ax, xlabel: str, ylabel: str, grid_axis: str = "y") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if grid_axis in {"x", "both"}:
        ax.xaxis.grid(True)
    if grid_axis in {"y", "both"}:
        ax.yaxis.grid(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(length=0)


def save_fig(fig: plt.Figure, filename: str) -> str:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / filename
    fig.savefig(path, bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return filename


def generate_dataset_sample_counts(dataset: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    plot_df = dataset.sort_values("Number of samples", ascending=True)
    sns.barplot(
        data=plot_df,
        x="Number of samples",
        y="Run ID",
        hue="Run ID",
        palette=RUN_PALETTE,
        legend=False,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.6,
    )
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + max(plot_df["Number of samples"]) * 0.015, patch.get_y() + patch.get_height() / 2, f"{int(width)}", va="center", fontsize=8.5)
    polish(ax, "Number of vibration snapshots", "Bearing run", "x")
    return save_fig(fig, "dataset_sample_counts.pdf")


def generate_fault_frequency_plot(freq: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    freq = freq.copy()
    freq["Value in Hz"] = freq["Value in Hz"].astype(float)
    colors = [COLORS["blue"], COLORS["gold"], COLORS["green"], COLORS["red"]]
    ax.vlines(freq["Frequency name"], 0, freq["Value in Hz"], color=colors, linewidth=2.5)
    ax.scatter(freq["Frequency name"], freq["Value in Hz"], s=75, color=colors, edgecolor=COLORS["ink"], linewidth=0.6, zorder=3)
    for x, y in zip(freq["Frequency name"], freq["Value in Hz"]):
        ax.text(x, y + 10, f"{y:.1f}", ha="center", va="bottom", fontsize=8.5)
    polish(ax, "Fault-frequency component", "Frequency (Hz)", "y")
    ax.set_ylim(0, freq["Value in Hz"].max() * 1.2)
    return save_fig(fig, "fault_frequencies.pdf")


def generate_split_matrix(split: pd.DataFrame) -> str:
    runs = ["ds2_b1", "ds1_b3", "ds1_b4", "ds3_b3"]
    role_value = {"": 0, "Train": 1, "Val": 2, "Test": 3}
    values = pd.DataFrame(0, index=split["Experiment"], columns=runs)
    labels = pd.DataFrame("", index=split["Experiment"], columns=runs)
    for _, row in split.iterrows():
        exp = row["Experiment"]
        for run in [r.strip() for r in row["Train runs"].split("+")]:
            values.loc[exp, run] = role_value["Train"]
            labels.loc[exp, run] = "Train"
        for run in [r.strip() for r in row["Validation run"].split("+")]:
            values.loc[exp, run] = role_value["Val"]
            labels.loc[exp, run] = "Val"
        for run in [r.strip() for r in row["Test run"].split("+")]:
            values.loc[exp, run] = role_value["Test"]
            labels.loc[exp, run] = "Test"
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    cmap = sns.color_palette(["#FFFFFF", COLORS["sky"], COLORS["gold"], COLORS["green"]], as_cmap=True)
    sns.heatmap(
        values,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 9.5, "fontweight": "bold", "color": COLORS["ink"]},
    )
    ax.set_xlabel("Bearing run")
    ax.set_ylabel("Experiment")
    return save_fig(fig, "experiment_split_matrix.pdf")


def generate_methodology_flowchart() -> str:
    steps = [
        ("Raw IMS vibration data", r"\texttt{data/raw/1st\_test}, \texttt{2nd\_test}, and \texttt{3rd\_test}", COLORS["blue"]),
        ("Data validation", "Snapshot counts, timestamp order, and channel-to-bearing mapping", COLORS["sky"]),
        ("Feature extraction", "RMS, kurtosis, crest factor, and envelope fault-frequency energy", COLORS["gold"]),
        ("RUL labeling and preprocessing", "Healthy-baseline normalization and normalized RUL target", COLORS["green"]),
        ("PCA-HI analysis", "Degradation consistency check from engineered vibration features", COLORS["purple"]),
        ("Model comparison", "Data-only FNN, DeepXDE PINN, LSTM, and 1D CNN", COLORS["red"]),
        ("Evaluation and reporting", "MAE, RMSE, R-squared, rankings, and prediction curves", COLORS["muted"]),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box_x, box_w, box_h = 0.22, 0.68, 0.086
    start_y, gap = 0.86, 0.034
    center_x = box_x + box_w / 2
    for idx, (title, subtitle, accent) in enumerate(steps, start=1):
        y = start_y - (idx - 1) * (box_h + gap)
        box = FancyBboxPatch(
            (box_x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            linewidth=1.0,
            edgecolor=COLORS["grid"],
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((box_x, y), 0.016, box_h, boxstyle="round,pad=0.012,rounding_size=0.022", linewidth=0, facecolor=accent))
        circle = plt.Circle((0.14, y + box_h / 2), 0.027, color=accent, ec="white", lw=1.2)
        ax.add_patch(circle)
        ax.text(0.14, y + box_h / 2, f"{idx}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        ax.text(box_x + 0.043, y + box_h * 0.62, title, ha="left", va="center", fontsize=9.8, fontweight="bold", color=COLORS["ink"])
        ax.text(box_x + 0.043, y + box_h * 0.31, "\n".join(textwrap.wrap(subtitle.replace("\\texttt{", "").replace("}", ""), 58)), ha="left", va="center", fontsize=8.0, color=COLORS["muted"], linespacing=1.1)
        if idx < len(steps):
            next_y = y - gap
            ax.add_patch(FancyArrowPatch((center_x, y - 0.005), (center_x, next_y + 0.005), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=COLORS["muted"]))
    return save_fig(fig, "methodology_flowchart.pdf")


def normalize_for_plot(series: pd.Series) -> pd.Series:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    lo, hi = values.min(), values.max()
    if not math.isfinite(float(hi - lo)) or abs(float(hi - lo)) < 1e-12:
        return values * 0
    return (values - lo) / (hi - lo)


def generate_feature_evolution(features: pd.DataFrame, feature: str, label: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for run_id, run_df in features.groupby("run_id", sort=False):
        ordered = run_df.sort_values("time_norm")
        x = 1.0 - ordered["rul_norm"].astype(float)
        window = max(9, min(101, len(ordered) // 35 * 2 + 1))
        y = normalize_for_plot(ordered[feature]).rolling(window, min_periods=1, center=True).median()
        ax.plot(x, y, label=run_id, color=RUN_PALETTE.get(run_id, COLORS["blue"]), linewidth=1.6, alpha=0.95)
    polish(ax, "Normalized lifetime", "Normalized feature value", "both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"])
    return save_fig(fig, filename)


def generate_correlation_heatmap(features: pd.DataFrame) -> str:
    cols = FEATURE_COLS + ["rul_norm"]
    corr = features[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
        annot_kws={"fontsize": 6.3},
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return save_fig(fig, "feature_correlation_heatmap.pdf")


def generate_pca_health_indicator(features: pd.DataFrame) -> str:
    selected = features[features["run_id"].isin(["ds1_b3", "ds3_b3"])].copy()
    x = selected[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    pcs = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(x)).ravel()
    selected["pca_hi"] = pcs
    damage = 1.0 - selected["rul_norm"].astype(float)
    if pd.Series(pcs).corr(damage, method="spearman") < 0:
        selected["pca_hi"] = -selected["pca_hi"]
    selected["pca_hi_scaled"] = normalize_for_plot(selected["pca_hi"])
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for run_id, run_df in selected.groupby("run_id", sort=False):
        ordered = run_df.sort_values("time_norm")
        x_axis = 1.0 - ordered["rul_norm"].astype(float)
        window = max(11, min(101, len(ordered) // 35 * 2 + 1))
        ax.plot(x_axis, ordered["pca_hi_scaled"].rolling(window, min_periods=1, center=True).median(), color=RUN_PALETTE[run_id], linewidth=1.8, label=f"{run_id} PCA-HI")
        ax.plot(x_axis, 1.0 - ordered["rul_norm"].astype(float), color=RUN_PALETTE[run_id], linewidth=1.0, linestyle="--", alpha=0.65, label=f"{run_id} damage")
    polish(ax, "Normalized lifetime", "Normalized value", "both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    return save_fig(fig, "pca_health_indicator.pdf")


def generate_metric_comparison(final_results: pd.DataFrame, metric: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    plot_df = final_results.copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)
    palette = {MODEL_SHORT[k]: v for k, v in MODEL_PALETTE.items()}
    sns.barplot(
        data=plot_df,
        x="Experiment",
        y=metric,
        hue="Model",
        hue_order=[MODEL_SHORT[m] for m in MODEL_ORDER],
        palette=palette,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.55,
    )
    if metric == "R2":
        ax.axhline(0, color=COLORS["ink"], linewidth=0.9, linestyle=":")
        ylabel = r"$R^2$"
    else:
        ylabel = metric
    polish(ax, "Experiment", ylabel, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False)
    return save_fig(fig, filename)


def generate_average_metric(average: pd.DataFrame, metric: str, filename: str, higher_is_better: bool = False) -> str:
    plot_df = average.copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)
    plot_df = plot_df.sort_values(metric, ascending=not higher_is_better)
    palette = {MODEL_SHORT[k]: v for k, v in MODEL_PALETTE.items()}
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    sns.barplot(
        data=plot_df,
        x=metric,
        y="Model",
        hue="Model",
        palette=palette,
        legend=False,
        ax=ax,
        edgecolor=COLORS["ink"],
        linewidth=0.55,
    )
    for patch in ax.patches:
        width = patch.get_width()
        offset = (plot_df[metric].max() - plot_df[metric].min()) * 0.03 + 0.005
        ax.text(width + offset, patch.get_y() + patch.get_height() / 2, f"{width:.3f}", va="center", fontsize=8.2)
    polish(ax, metric, "Model", "x")
    if metric == "Average R2":
        ax.set_xlabel(r"Average $R^2$")
    return save_fig(fig, filename)


def generate_model_ranking(ranking: pd.DataFrame) -> str:
    plot_df = ranking.copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)
    palette = {MODEL_SHORT[k]: v for k, v in MODEL_PALETTE.items()}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sns.scatterplot(
        data=plot_df,
        x="Experiment",
        y="Rank by RMSE",
        hue="Model",
        hue_order=[MODEL_SHORT[m] for m in MODEL_ORDER],
        palette=palette,
        s=95,
        edgecolor=COLORS["ink"],
        linewidth=0.6,
        ax=ax,
        zorder=3,
    )
    for model, model_df in plot_df.groupby("Model"):
        ordered = model_df.sort_values("Experiment")
        ax.plot(ordered["Experiment"], ordered["Rank by RMSE"], color=palette[model], linewidth=1.1, alpha=0.55)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.6, 4.4)
    polish(ax, "Experiment", "Rank by RMSE", "both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=2, frameon=False)
    return save_fig(fig, "model_ranking_by_rmse.pdf")


def generate_best_model(final_results: pd.DataFrame) -> str:
    best = final_results.loc[final_results.groupby("Experiment")["RMSE"].idxmin()].copy()
    best["Model"] = best["Model"].map(MODEL_SHORT)
    palette = {MODEL_SHORT[k]: v for k, v in MODEL_PALETTE.items()}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sns.barplot(data=best, x="Experiment", y="RMSE", hue="Model", palette=palette, ax=ax, edgecolor=COLORS["ink"], linewidth=0.55)
    for patch in ax.patches:
        if patch.get_height() > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.006, f"{patch.get_height():.3f}", ha="center", va="bottom", fontsize=8.2)
    polish(ax, "Experiment", "RMSE", "y")
    ax.set_ylim(0, best["RMSE"].max() * 1.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
    return save_fig(fig, "best_model_by_experiment.pdf")


def load_prediction_errors(prediction_series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp, exp_df in prediction_series.groupby("Experiment"):
        true_df = exp_df[exp_df["Model"] == "True normalized RUL"].sort_values("x")
        x_true = true_df["x"].to_numpy()
        y_true = true_df["y"].to_numpy()
        for model, model_df in exp_df[exp_df["Model"] != "True normalized RUL"].groupby("Model"):
            ordered = model_df.sort_values("x")
            y_interp = np.interp(ordered["x"].to_numpy(), x_true, y_true)
            error = ordered["y"].to_numpy() - y_interp
            stride = max(1, len(error) // 1600)
            for x, err in zip(ordered["x"].to_numpy()[::stride], error[::stride]):
                rows.append({"Experiment": exp, "Model": model, "x": float(x), "Error": float(err), "Absolute error": abs(float(err))})
    return pd.DataFrame(rows)


def generate_prediction_curves(prediction_series: pd.DataFrame) -> list[str]:
    saved = []
    for exp, exp_df in prediction_series.groupby("Experiment", sort=True):
        fig, ax = plt.subplots(figsize=(7.2, 3.7))
        true_df = exp_df[exp_df["Model"] == "True normalized RUL"].sort_values("x")
        stride_true = max(1, len(true_df) // 1800)
        ax.plot(true_df["x"].iloc[::stride_true], true_df["y"].iloc[::stride_true], color=COLORS["ink"], linewidth=2.0, label="True RUL")
        for model in MODEL_ORDER:
            model_df = exp_df[exp_df["Model"] == model].sort_values("x")
            if model_df.empty:
                continue
            stride = max(1, len(model_df) // 1800)
            ax.plot(model_df["x"].iloc[::stride], model_df["y"].iloc[::stride], color=MODEL_PALETTE[model], linewidth=1.25, alpha=0.92, label=MODEL_SHORT[model])
        polish(ax, "Normalized test-life index", "Normalized RUL", "both")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False)
        filename = f"{exp.lower().replace(' ', '_')}_true_vs_predicted_rul.pdf"
        saved.append(save_fig(fig, filename))
    return saved


def generate_absolute_error(errors: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True, sharey=True)
    for ax, (exp, exp_df) in zip(axes, errors.groupby("Experiment", sort=True)):
        for model, model_df in exp_df.groupby("Model"):
            ordered = model_df.sort_values("x")
            smooth = ordered["Absolute error"].rolling(35, min_periods=1, center=True).median()
            ax.plot(ordered["x"], smooth, color=MODEL_PALETTE[model], linewidth=1.1, alpha=0.92, label=MODEL_SHORT[model])
        ax.set_ylabel(exp)
        ax.grid(True, axis="both", alpha=0.45)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False)
    axes[-1].set_xlabel("Normalized test-life index")
    fig.text(0.015, 0.5, "Smoothed absolute error", rotation="vertical", va="center", color=COLORS["ink"])
    return save_fig(fig, "absolute_error_over_life.pdf")


def generate_error_distribution(errors: pd.DataFrame) -> str:
    plot_df = errors.copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)
    palette = {MODEL_SHORT[k]: v for k, v in MODEL_PALETTE.items()}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.boxplot(data=plot_df, x="Model", y="Error", hue="Model", palette=palette, showfliers=False, legend=False, ax=ax)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.9, linestyle=":")
    polish(ax, "Model", "Prediction error", "y")
    ax.tick_params(axis="x", rotation=12)
    return save_fig(fig, "prediction_error_distribution.pdf")


def prepare_figures() -> dict[str, str]:
    setup_plot_style()
    dataset = pd.read_csv(TABLES / "dataset_summary_table.csv")
    freq = pd.read_csv(TABLES / "fault_frequency_table.csv")
    split = pd.read_csv(TABLES / "experiment_split_table.csv")
    final_results = pd.read_csv(TABLES / "final_results_table.csv")
    average = pd.read_csv(TABLES / "average_model_performance_table.csv")
    ranking = pd.read_csv(TABLES / "model_ranking_by_rmse.csv")
    predictions = pd.read_csv(TABLES / "prediction_series.csv")
    features = pd.read_csv(FEATURES)

    generated = {
        "dataset": generate_dataset_sample_counts(dataset),
        "faults": generate_fault_frequency_plot(freq),
        "split": generate_split_matrix(split),
        "methodology": generate_methodology_flowchart(),
        "corr": generate_correlation_heatmap(features),
        "pca": generate_pca_health_indicator(features),
        "rms": generate_feature_evolution(features, "rms", "RMS", "feature_evolution_rms.pdf"),
        "kurtosis": generate_feature_evolution(features, "kurtosis", "Kurtosis", "feature_evolution_kurtosis.pdf"),
        "crest": generate_feature_evolution(features, "crest_factor", "Crest factor", "feature_evolution_crest_factor.pdf"),
        "ekin": generate_feature_evolution(features, "E_kin", "Combined envelope fault-frequency energy", "feature_evolution_e_kin.pdf"),
        "rmse": generate_metric_comparison(final_results, "RMSE", "metric_comparison_rmse.pdf"),
        "mae": generate_metric_comparison(final_results, "MAE", "metric_comparison_mae.pdf"),
        "r2": generate_metric_comparison(final_results, "R2", "metric_comparison_r2.pdf"),
        "avg_rmse": generate_average_metric(average, "Average RMSE", "average_rmse_by_model.pdf"),
        "avg_mae": generate_average_metric(average, "Average MAE", "average_mae_by_model.pdf"),
        "avg_r2": generate_average_metric(average, "Average R2", "average_r2_by_model.pdf", higher_is_better=True),
        "ranking": generate_model_ranking(ranking),
        "best": generate_best_model(final_results),
    }
    for prediction_file in generate_prediction_curves(predictions):
        generated[prediction_file] = prediction_file
    errors = load_prediction_errors(predictions)
    generated["abs_error"] = generate_absolute_error(errors)
    generated["err_dist"] = generate_error_distribution(errors)

    logo_source = THESIS / "example" / "assets" / "images" / "cuet.png"
    if logo_source.exists():
        shutil.copy2(logo_source, IMAGES / "cuet.png")
    return generated


def build_tables() -> dict[str, str]:
    dataset = pd.read_csv(TABLES / "dataset_summary_table.csv")
    dataset = dataset[["Run ID", "IMS dataset", "Bearing number", "Number of samples", "Used in experiments"]]
    features = pd.read_csv(TABLES / "feature_description_table.csv")
    freq = pd.read_csv(TABLES / "fault_frequency_table.csv")
    freq["Value in Hz"] = freq["Value in Hz"].map(lambda x: f"{float(x):.2f}")
    pca = pd.read_csv(TABLES / "pca_hi_ablation_table.csv")
    pca[["Monotonicity score", "Correlation with damage"]] = pca[["Monotonicity score", "Correlation with damage"]].round(3)
    models = pd.read_csv(TABLES / "model_summary_table.csv")
    split = pd.read_csv(TABLES / "experiment_split_table.csv")
    final_results = pd.read_csv(TABLES / "final_results_table.csv")
    final_short = final_results[["Experiment", "Model", "MAE", "RMSE", "R2"]].copy()
    final_short["Model"] = final_short["Model"].map(MODEL_SHORT)
    final_short[["MAE", "RMSE", "R2"]] = final_short[["MAE", "RMSE", "R2"]].round(3)
    average = pd.read_csv(TABLES / "average_model_performance_table.csv")
    average["Model"] = average["Model"].map(MODEL_SHORT)
    average[["Average MAE", "Average RMSE", "Average R2"]] = average[["Average MAE", "Average RMSE", "Average R2"]].round(3)
    ranking = pd.read_csv(TABLES / "model_ranking_by_rmse.csv")
    ranking["Model"] = ranking["Model"].map(MODEL_SHORT)
    ranking["Rank by RMSE"] = ranking["Rank by RMSE"].astype(int)
    ranking["RMSE"] = ranking["RMSE"].round(3)
    return {
        "dataset": table_tex(dataset, "Bearing runs used in the final experiments.", "tab:dataset_summary", r"@{}l l c c Y@{}"),
        "features": table_tex(features, "Extracted features used by the neural models.", "tab:feature_set", r"@{}l l Y c@{}", r"\scriptsize"),
        "freq": table_tex(freq, "Bearing fault frequencies used for envelope spectral-energy extraction.", "tab:fault_frequencies", r"@{}l l c Y l@{}"),
        "pca": table_tex(pca, "PCA health-indicator monotonicity and damage correlation.", "tab:pca_hi", r"@{}l c c Y@{}"),
        "models": table_tex(models, "Model definitions and roles in the experiment.", "tab:model_summary", r"@{}Y Y Y Y l@{}", r"\scriptsize"),
        "split": table_tex(split, "Train-validation-test split for each experiment.", "tab:experiment_split", r"@{}l l l l Y@{}", r"\scriptsize"),
        "final": table_tex(final_short, "Final prediction metrics for all models and experiments.", "tab:final_results", r"@{}l l c c c@{}"),
        "average": table_tex(average, "Average model performance across the three experiments.", "tab:average_results", r"@{}l c c c@{}"),
        "ranking": table_tex(ranking[["Experiment", "Model", "RMSE", "Rank by RMSE"]], "Model ranking by RMSE.", "tab:ranking", r"@{}l l c c@{}"),
    }


def build_bibliography() -> str:
    return r"""
@article{rycerz2017,
  author = {Rycerz, Pawel and Olver, Andrew and Kadiric, Amir},
  title = {Propagation of surface initiated rolling contact fatigue cracks in bearing steel},
  journal = {International Journal of Fatigue},
  volume = {97},
  pages = {29--38},
  year = {2017},
  doi = {10.1016/j.ijfatigue.2016.12.004}
}

@article{sadeghi2009,
  author = {Sadeghi, Farshid and Jalalahmadi, Behrooz and Slack, Trevor S. and Raje, Nihar and Arakere, Nagaraj K.},
  title = {A Review of Rolling Contact Fatigue},
  journal = {Journal of Tribology},
  volume = {131},
  number = {4},
  year = {2009},
  doi = {10.1115/1.3209132}
}

@article{li2000,
  author = {Li, Y. and Zhang, C. and Kurfess, T. R. and Danyluk, S. and Liang, S. Y.},
  title = {Diagnostics and prognostics of a single surface defect on roller bearings},
  journal = {Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science},
  volume = {214},
  number = {9},
  pages = {1173--1185},
  year = {2000},
  doi = {10.1243/0954406001523614}
}

@article{xu2023,
  author = {Xu, Funing and Ding, Ning and Li, Nan and Liu, Long and Hou, Nan and Xu, Na and Guo, Weimin and Tian, Linan and Xu, Huixia and Wu, Chi-Man Lawrence and Wu, Xiaofeng and Chen, Xiangfeng},
  title = {A review of bearing failure modes, mechanisms and causes},
  journal = {Engineering Failure Analysis},
  volume = {152},
  pages = {107518},
  year = {2023},
  doi = {10.1016/j.engfailanal.2023.107518}
}

@article{wu2022,
  author = {Wu, Guoguo and Yan, Tanyi and Yang, Guolai and Chai, Hongqiang and Cao, Chuanchuan},
  title = {A Review on Rolling Bearing Fault Signal Detection Methods Based on Different Sensors},
  journal = {Sensors},
  volume = {22},
  number = {21},
  pages = {8330},
  year = {2022},
  doi = {10.3390/s22218330}
}

@article{kannan2024,
  author = {Kannan, Vigneshwar and Zhang, Tieling and Li, Huaizhong},
  title = {A Review of the Intelligent Condition Monitoring of Rolling Element Bearings},
  journal = {Machines},
  volume = {12},
  number = {7},
  pages = {484},
  year = {2024},
  doi = {10.3390/machines12070484}
}

@article{qiu2006,
  author = {Qiu, Hai and Lee, Jay and Lin, Jing and Yu, Gang},
  title = {Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics},
  journal = {Journal of Sound and Vibration},
  volume = {289},
  number = {4--5},
  pages = {1066--1090},
  year = {2006},
  doi = {10.1016/j.jsv.2005.03.007}
}

@inproceedings{nectoux2012,
  author = {Nectoux, Patrick and Gouriveau, Rafael and Medjaher, Kamal and Ramasso, Emmanuel and Chebel-Morello, Brigitte and Zerhouni, Noureddine and Varnier, Christophe},
  title = {{PRONOSTIA}: An experimental platform for bearings accelerated degradation tests},
  booktitle = {IEEE International Conference on Prognostics and Health Management},
  pages = {1--8},
  year = {2012},
  url = {https://hal.science/hal-00719503}
}

@article{jardine2006,
  author = {Jardine, Andrew K. S. and Lin, Daming and Banjevic, Dragan},
  title = {A review on machinery diagnostics and prognostics implementing condition-based maintenance},
  journal = {Mechanical Systems and Signal Processing},
  volume = {20},
  number = {7},
  pages = {1483--1510},
  year = {2006},
  doi = {10.1016/j.ymssp.2005.09.012}
}

@article{lei2018,
  author = {Lei, Yaguo and Li, Naipeng and Guo, Liang and Li, Ningbo and Yan, Tao and Lin, Jing},
  title = {Machinery health prognostics: A systematic review from data acquisition to {RUL} prediction},
  journal = {Mechanical Systems and Signal Processing},
  volume = {104},
  pages = {799--834},
  year = {2018},
  doi = {10.1016/j.ymssp.2017.11.016}
}

@article{huang2020,
  author = {Huang, Wei and Farahat, Ahmed and Gupta, Chetan},
  title = {Similarity-based Feature Extraction from Vibration Data for Prognostics},
  journal = {Annual Conference of the PHM Society},
  volume = {12},
  number = {1},
  year = {2020},
  doi = {10.36001/phmconf.2020.v12i1.1298}
}

@inproceedings{babu2016,
  author = {Sateesh Babu, Giduthuri and Zhao, Peilin and Li, Xiao-Li},
  title = {Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life},
  booktitle = {Database Systems for Advanced Applications},
  pages = {214--228},
  year = {2016},
  doi = {10.1007/978-3-319-32025-0_14}
}

@article{berghout2022,
  author = {Berghout, Tarek and Mouss, Leila-Hayet and Bentrcia, Toufik and Benbouzid, Mohamed},
  title = {A Semi-Supervised Deep Transfer Learning Approach for Rolling-Element Bearing Remaining Useful Life Prediction},
  journal = {IEEE Transactions on Energy Conversion},
  volume = {37},
  number = {2},
  pages = {1200--1210},
  year = {2022},
  doi = {10.1109/TEC.2021.3116423}
}

@article{ayman2025,
  author = {Ayman, Ahmed and Onsy, Ahmed and Attallah, Omneya and Brooks, Hadley and Morsi, Iman},
  title = {Feature learning for bearing prognostics: A comprehensive review of machine/deep learning methods, challenges, and opportunities},
  journal = {Measurement},
  volume = {245},
  pages = {116589},
  year = {2025},
  doi = {10.1016/j.measurement.2024.116589}
}

@article{farooq2024,
  author = {Farooq, Umer and Ademola, Moses and Shaalan, Abdu},
  title = {Comparative Analysis of Machine Learning Models for Predictive Maintenance of Ball Bearing Systems},
  journal = {Electronics},
  volume = {13},
  number = {2},
  pages = {438},
  year = {2024},
  doi = {10.3390/electronics13020438}
}

@article{han2024,
  author = {Han, Kaixu and Wang, Wenhao and Guo, Jun},
  title = {Research on a Bearing Fault Diagnosis Method Based on a {CNN-LSTM-GRU} Model},
  journal = {Machines},
  volume = {12},
  number = {12},
  pages = {927},
  year = {2024},
  doi = {10.3390/machines12120927}
}

@article{yang2024,
  author = {Yang, Lei and Jiang, Yibo and Zeng, Kang and Peng, Tao},
  title = {Rolling Bearing Remaining Useful Life Prediction Based on {CNN-VAE-MBiLSTM}},
  journal = {Sensors},
  volume = {24},
  number = {10},
  pages = {2992},
  year = {2024},
  doi = {10.3390/s24102992}
}

@article{wang2025,
  author = {Wang, Chenyang and Jiang, Wanlu and Shi, Lei and Zhang, Liang},
  title = {Rolling bearing remaining useful life prediction using deep learning based on high-quality representation},
  journal = {Scientific Reports},
  volume = {15},
  number = {1},
  year = {2025},
  doi = {10.1038/s41598-025-93165-4}
}

@article{raissi2019,
  author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  title = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  journal = {Journal of Computational Physics},
  volume = {378},
  pages = {686--707},
  year = {2019},
  doi = {10.1016/j.jcp.2018.10.045}
}

@article{lawal2022,
  author = {Lawal, Zaharaddeen Karami and Yassin, Hayati and Lai, Daphne Teck Ching and Che Idris, Azam},
  title = {Physics-Informed Neural Network ({PINN}) Evolution and Beyond: A Systematic Literature Review and Bibliometric Analysis},
  journal = {Big Data and Cognitive Computing},
  volume = {6},
  number = {4},
  pages = {140},
  year = {2022},
  doi = {10.3390/bdcc6040140}
}

@article{ren2025,
  author = {Ren, Zhiyuan and Zhou, Shijie and Liu, Dong and Liu, Qihe},
  title = {Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing},
  journal = {Applied Sciences},
  volume = {15},
  number = {14},
  pages = {8092},
  year = {2025},
  doi = {10.3390/app15148092}
}

@article{chen2022,
  author = {Chen, Xuefeng and Ma, Meng and Zhao, Zhibin and Zhai, Zhi and Mao, Zhu},
  title = {Physics-Informed Deep Neural Network for Bearing Prognosis with Multisensory Signals},
  journal = {Journal of Dynamics, Monitoring and Diagnostics},
  pages = {200--207},
  year = {2022},
  doi = {10.37965/jdmd.2022.54}
}

@article{parziale2023,
  author = {Parziale, Marc and Lomazzi, Luca and Giglio, Marco and Cadini, Francesco},
  title = {Physics-Informed Neural Networks for the Condition Monitoring of Rotating Shafts},
  journal = {Sensors},
  volume = {24},
  number = {1},
  pages = {207},
  year = {2023},
  doi = {10.3390/s24010207}
}

@article{herwig2025,
  author = {Herwig, N. and Borghesani, P. and Smith, W. and Peng, Z.},
  title = {Signal processing- and physics-informed neural network for explainable bearing condition monitoring},
  journal = {Mechanical Systems and Signal Processing},
  volume = {235},
  pages = {112925},
  year = {2025},
  doi = {10.1016/j.ymssp.2025.112925}
}

@article{zhong2025,
  author = {Zhong, Jingshu and Zheng, Yu and Ruan, Chengtao and Chen, Liang and Bao, Xiangyu and Lyu, Lyu},
  title = {{M-IPISincNet}: An explainable multi-source physics-informed neural network based on improved {SincNet} for rolling bearings fault diagnosis},
  journal = {Information Fusion},
  volume = {115},
  pages = {102761},
  year = {2025},
  doi = {10.1016/j.inffus.2024.102761}
}

@article{vonhahn2022,
  author = {von Hahn, Tim and Mechefske, Chris K.},
  title = {Knowledge informed machine learning using a {Weibull}-based loss function},
  journal = {arXiv preprint arXiv:2201.01769},
  year = {2022},
  doi = {10.48550/arXiv.2201.01769}
}

@article{lu2021,
  author = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  title = {{DeepXDE}: A Deep Learning Library for Solving Differential Equations},
  journal = {SIAM Review},
  volume = {63},
  number = {1},
  pages = {208--228},
  year = {2021},
  doi = {10.1137/19M1274067}
}

@article{paris1963,
  author = {Paris, P. and Erdogan, F.},
  title = {A Critical Analysis of Crack Propagation Laws},
  journal = {Journal of Basic Engineering},
  volume = {85},
  number = {4},
  pages = {528--533},
  year = {1963},
  doi = {10.1115/1.3656900}
}

@inproceedings{saxena2008,
  author = {Saxena, Abhinav and Celaya, Jose and Balaban, Edward and Goebel, Kai and Saha, Bhaskar and Saha, Sankalita and Schwabacher, Mark},
  title = {Metrics for evaluating performance of prognostic techniques},
  booktitle = {2008 International Conference on Prognostics and Health Management},
  pages = {1--17},
  year = {2008},
  doi = {10.1109/PHM.2008.4711436}
}
"""


def build_main_tex() -> str:
    return r"""
\documentclass[12pt,a4paper]{report}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{placeins}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{enumitem}
\usepackage[nottoc]{tocbibind}
\usepackage{cite}

\graphicspath{{assets/images/}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

\titleformat{\chapter}[display]
{\centering\bfseries\fontsize{22pt}{22pt}\selectfont}
{Chapter \thechapter}
{0pt}
{}
[\vspace{1em}]

\titlespacing*{\chapter}{0pt}{6pt}{8pt}

\titleformat{\section}
{\fontsize{13pt}{13pt}\selectfont\MakeUppercase}
{\thesection}
{1em}
{}

\titlespacing*{\section}{0pt}{6pt}{6pt}

\titleformat{\subsection}
{\bfseries\fontsize{12pt}{12pt}\selectfont}
{\thesubsection}
{1em}
{}

\titleformat{\subsubsection}
{\itshape\fontsize{12pt}{12pt}\selectfont}
{\thesubsubsection}
{1em}
{}

\setlength{\parskip}{12pt}
\setlength{\parindent}{0pt}

\geometry{
    left=1.2in,
    right=0.8in,
    top=1in,
    bottom=0.8in
}

\onehalfspacing
\pagestyle{plain}
\sloppy

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    filecolor=black,
    urlcolor=black,
    citecolor=black,
    pdftitle={Physics-Informed Neural Networks for Remaining-Useful-Life Prediction of Rolling-Element Bearings},
    pdfauthor={Mahdee Hassan Sami}
}

\begin{document}

\pagenumbering{gobble}

\input{sections/cover}
\input{sections/information}

\clearpage
\pagenumbering{roman}

\phantomsection
\input{sections/abstract}
\addcontentsline{toc}{chapter}{Abstract}

\phantomsection
\input{sections/acknowledgement}
\addcontentsline{toc}{chapter}{Acknowledgement}

\tableofcontents
\addcontentsline{toc}{chapter}{\contentsname}
\listoffigures
\listoftables

\phantomsection
\input{sections/nomenclature}
\addcontentsline{toc}{chapter}{Nomenclature}

\newpage
\pagenumbering{arabic}
\renewcommand{\thechapter}{\arabic{chapter}}
\renewcommand{\thesection}{\thechapter.\arabic{section}}
\setcounter{chapter}{0}

\input{sections/introduction}
\input{sections/literature_review}
\input{sections/methodology}
\input{sections/implementation}
\input{sections/results}
\input{sections/conclusion}

\renewcommand{\bibname}{References}
\bibliographystyle{IEEEtran}
\bibliography{references}

\input{sections/appendices}

\end{document}
"""


def build_cover() -> str:
    return r"""
\begin{titlepage}
    \centering
    \singlespacing

    \begin{flushright}
        \normalsize Course no: ME-494
    \end{flushright}

    \vspace{0.5cm}

    {\Large \bfseries PHYSICS-INFORMED NEURAL NETWORKS FOR REMAINING-USEFUL-LIFE PREDICTION OF ROLLING-ELEMENT BEARINGS \par}

    \vspace{0.6cm}

    \includegraphics[width=3.5cm]{cuet.png}

    \vspace{0.6cm}

    {\normalsize A thesis submitted in partial fulfillment of \\
    requirements for the degree of \par}

    \vspace{0.6cm}

    {\Large Bachelor of Science \par}

    \vspace{0.2cm}

    {\normalsize in \par}

    \vspace{0.2cm}

    {\Large Mechanical Engineering \par}

    \vspace{0.6cm}

    Submitted By:\\
    Mahdee Hassan Sami\\
    Student ID: 2003101

    \vspace{0.5cm}

    Supervised By:\\
    Dr. Md. Abu Mowazzem Hossain\\
    Professor

    \vfill

    {\large Department of Mechanical Engineering \par}

    \vspace{0.5cm}

    \hrule height 0.5pt

    \vspace{0.3cm}

    {\large Chittagong University of Engineering \& Technology \\
    Chattogram-4349, Bangladesh \par}
\end{titlepage}
"""


def build_information() -> str:
    return r"""
\begin{titlepage}
\begin{center}
    \textbf{\large CHITTAGONG UNIVERSITY OF ENGINEERING \\ AND TECHNOLOGY, CHATTOGRAM}

    \vspace{0.5em}

    (PROJECT AND THESIS)

    \vspace{0.5em}

    \textbf{COURSE NO: ME 494}
\end{center}

\vspace{1.5cm}

\noindent
\textbf{Title of Project:} Physics-Informed Neural Networks for Remaining-Useful-Life Prediction of Rolling-Element Bearings

\vspace{1.2cm}

\noindent
\textbf{Name of the Student:} Mahdee Hassan Sami

\vspace{1.2cm}

\noindent
\textbf{Student ID:} 2003101 \hfill \textbf{Session:} 2024-2025

\vspace{1.2cm}

\noindent
\textbf{Name of the Department:} Mechanical Engineering \hfill \textbf{Program:} B.Sc. Engg.

\vspace{1.2cm}

\noindent
\begin{tabular}{@{}ll}
    \textbf{Name of the Supervisor:} & Dr. Md. Abu Mowazzem Hossain \\
                                     & Professor \\
                                     & Department of Mechanical Engineering, CUET
\end{tabular}

\vspace{3.0cm}

\noindent
Signature of the Student:

\vspace{2.0cm}

\noindent
Signature of the Supervisor:
\end{titlepage}
"""


def build_abstract() -> str:
    return r"""
\chapter*{Abstract}

\noindent Rolling-element bearings are small components, but their condition often decides whether a rotating machine can keep running safely. A bearing fault can begin as a local surface defect and then grow into severe vibration, heat, loss of alignment, secondary damage, and unplanned shutdown. Remaining useful life prediction is therefore a central problem in prognostics and health management. This thesis studies bearing remaining useful life prediction using the NASA IMS run-to-failure bearing data and compares a physics-informed neural network against three data-driven baselines.

The original proposal for this work expected a physics-informed model to outperform a purely data-driven model. The completed local experiments gave a more measured result. The proposed DeepXDE physics-informed model performed best in one of the three cross-bearing experiments, but the LSTM baseline gave the lowest average RMSE and highest average $R^2$ across the full experiment set. This does not make the physics-informed approach unimportant. It shows that weak physical regularization, when applied to noisy run-to-failure vibration features, must be designed and weighted carefully. A simple monotonicity and fault-frequency prior can help in some transfer settings, but it may also restrict the network when the degradation path of the test bearing differs from the training bearings.

The final pipeline reads the IMS data locally from extracted folders, removes the dependency on Google Drive, extracts time-domain and envelope spectral features, constructs normalized RUL targets, builds a PCA-based health indicator, and evaluates four models: a feed-forward neural baseline, the proposed DeepXDE physics-informed model, an LSTM sequence model, and a 1D CNN sequence model. The experiments use three train-validation-test splits designed to test cross-bearing and cross-dataset transfer. The best average RMSE was achieved by the LSTM baseline at 0.162, followed by the data-only feed-forward baseline at 0.185, the proposed physics-informed model at 0.223, and the CNN baseline at 0.235. The work contributes a reproducible local project structure, a documented feature and evaluation pipeline, and an honest analysis of where the investigated physics-informed formulation helped and where it did not.

\textbf{Keywords:} remaining useful life, rolling-element bearing, prognostics and health management, physics-informed neural network, IMS bearing dataset, vibration analysis, LSTM, CNN.
"""


def build_acknowledgement() -> str:
    return r"""
\chapter*{Acknowledgement}

\noindent First, I offer my gratitude to Allah for giving me the strength, patience, and opportunity to complete this thesis work. I am deeply grateful to my undergraduate thesis supervisor, Dr. Md. Abu Mowazzem Hossain, Professor, Department of Mechanical Engineering, Chittagong University of Engineering \& Technology, for his supervision, advice, and encouragement throughout this project.

I also thank my family for their constant support during the long process of learning the theory, writing code, running experiments, correcting mistakes, and preparing this thesis. Their patience made the work easier to continue when the results did not come out as simply as expected.

Finally, I acknowledge the researchers who made the IMS bearing dataset publicly available. Open run-to-failure datasets make it possible for students and researchers to test prognostic methods in a reproducible way.
"""


def build_nomenclature() -> str:
    return r"""
\chapter*{Nomenclature}

\begin{tabularx}{\textwidth}{@{}lX@{}}
\toprule
\textbf{Symbol or term} & \textbf{Meaning} \\
\midrule
ANN & Artificial neural network \\
BPFO & Ball pass frequency outer race \\
BPFI & Ball pass frequency inner race \\
BSF & Ball spin frequency \\
CNN & Convolutional neural network \\
DeepXDE & Python library used for physics-informed neural networks \\
FFT & Fast Fourier transform \\
FNN & Feed-forward neural network \\
FTF & Fundamental train frequency \\
HI & Health indicator \\
IMS & Intelligent Maintenance Systems \\
LSTM & Long short-term memory network \\
MAE & Mean absolute error \\
PCA & Principal component analysis \\
PHM & Prognostics and health management \\
PINN & Physics-informed neural network \\
REB & Rolling-element bearing \\
RMS & Root mean square \\
RMSE & Root mean squared error \\
RUL & Remaining useful life \\
$R^2$ & Coefficient of determination \\
\bottomrule
\end{tabularx}
"""


def build_introduction() -> str:
    return r"""
\chapter{Introduction}
\label{chapter:introduction}

\section{Background and Motivation}

Rolling-element bearings support shafts, carry loads, and reduce friction in motors, pumps, turbines, gearboxes, fans, compressors, and many other machines. Because they sit directly in the load path, a bearing defect rarely remains an isolated issue for long. A small defect on a raceway or rolling element can create repeated impacts, excite vibration, produce heat, damage nearby parts, and eventually force the machine out of service. In industrial maintenance, the cost of a bearing failure is often much larger than the cost of the bearing itself.

Traditional maintenance strategies do not fully solve the problem. Reactive maintenance waits for failure and accepts the risk of unplanned shutdown. Preventive maintenance replaces parts on a fixed schedule, but that schedule is usually conservative. It can remove healthy components too early or miss faults that develop faster than expected. Condition-based maintenance improves on both approaches by measuring the actual condition of the machine and using that evidence to support maintenance decisions. Prognostics and health management extends this idea by estimating not only whether a component is damaged, but how much useful life may remain before failure \cite{jardine2006,lei2018}.

Remaining useful life prediction is difficult for bearings because degradation is not a simple straight line. Vibration can remain low for a long portion of the test, then change rapidly near the end of life. The measured signal depends on fault type, speed, load, sensor placement, noise, resonance, lubrication, and bearing geometry \cite{wu2022,kannan2024}. A model trained on one bearing may not behave well on another bearing, even when both bearings come from the same test rig. This is why benchmark run-to-failure datasets such as the IMS bearing dataset are useful. They allow researchers to study the same raw measurements and compare methods under repeatable conditions \cite{qiu2006}.

Deep learning methods have become common in bearing diagnosis and RUL prediction because they can model nonlinear patterns in vibration signals. CNNs can learn local patterns from windows of data, LSTMs can model temporal dependence, and feed-forward networks can fit engineered feature vectors \cite{babu2016,berghout2022,ayman2025}. These methods can work well when the training and test data are similar. Their weakness is that they usually learn only from examples. When labeled run-to-failure data are limited or the test bearing degrades differently from the training bearings, a data-only model may produce predictions that are numerically plausible but physically questionable.

Physics-informed neural networks offer a way to include prior engineering knowledge in the learning process. In their most established form, PINNs train a neural network while penalizing violations of governing equations or physical constraints \cite{raissi2019}. Bearing RUL prediction does not usually provide a complete closed-form degradation equation, but it does provide useful partial knowledge. Damage should generally increase with operating time, RUL should generally decrease, and bearing geometry gives characteristic fault frequencies associated with cage, inner-race, outer-race, and rolling-element defects. This thesis studies whether such weak physical information can improve RUL prediction on the IMS data.

\section{Problem Statement}

The problem studied in this thesis is the prediction of normalized remaining useful life for selected bearings in the IMS run-to-failure dataset. The main question is whether a physics-informed neural network using time and fault-frequency information can generalize better across bearing runs than comparable data-driven neural baselines.

This question is practical as well as methodological. In a real machine, the future failure path is unknown. A maintenance engineer may have historical data from one bearing, but the next bearing may not degrade in the same way. A useful RUL model must therefore do more than memorize one run. It must learn a relationship between measured condition and remaining life that transfers, at least partly, across bearings.

\section{Objectives of the Study}

The objectives of this thesis are:

\begin{enumerate}[label=\arabic*.]
    \item To build a local, reproducible processing pipeline for the IMS bearing data without depending on Google Colab or Google Drive.
    \item To extract time-domain vibration features and physics-related envelope spectral features from selected IMS bearing runs.
    \item To construct normalized RUL targets and a PCA-based health indicator for degradation analysis.
    \item To design and evaluate a DeepXDE physics-informed neural model for RUL prediction.
    \item To compare the proposed physics-informed model with a data-only neural baseline, an LSTM baseline, and a 1D CNN baseline.
    \item To discuss the results honestly, including the cases where the proposed model does not outperform the baselines.
\end{enumerate}

\section{Scope}

This study is limited to the IMS bearing data placed locally under \texttt{data/raw/1st\_test}, \texttt{data/raw/2nd\_test}, and \texttt{data/raw/3rd\_test}. Four bearing runs are used in the final experiment design: \texttt{ds2\_b1}, \texttt{ds1\_b3}, \texttt{ds1\_b4}, and \texttt{ds3\_b3}. The work uses vibration signals only. It does not add temperature, oil debris, acoustic emission, or motor-current data.

The models predict normalized RUL from extracted features. The thesis does not claim field deployment readiness. The data come from a controlled test rig, and the operational conditions are much simpler than many industrial machines. The physical information in the PINN is also intentionally weak. It uses monotonicity and fault-frequency energy priors rather than a full contact mechanics or crack growth model.

\section{Importance of the Study}

This work connects three parts of the bearing prognostics problem in one local workflow: signal processing, machine learning, and physical interpretation. The feature extraction step keeps the connection to vibration analysis through RMS, kurtosis, crest factor, and envelope spectral energies. The neural models test whether those features can support RUL prediction across bearing runs. The physics-informed model tests whether simple physical priors can regularize the learning process.

The final result is not a simple success story for the proposed model. That is important. A thesis result should report what the experiment shows, not what the proposal expected. The completed run suggests that LSTM sequence modeling captured the degradation trend more consistently than the proposed weakly physics-informed feed-forward model. At the same time, the PINN was competitive in Experiment 2 and ranked first there by RMSE. This mixed behavior gives a clearer direction for future work than an overconfident positive claim would.

\section{Limitations}

The main limitations are:

\begin{enumerate}[label=\arabic*.]
    \item The IMS dataset contains a small number of run-to-failure records, so the experiment design is constrained.
    \item The selected physics prior is not a full bearing degradation law. It is a weak regularizer based on monotonic RUL behavior and fault-frequency energy.
    \item The model uses hand-engineered features, not raw vibration waveforms.
    \item The RUL labels are derived from elapsed time to the end of each run, which assumes the last available file represents failure or near-failure.
    \item The experiments are run on one local PC, so training time and numerical results may vary on other machines.
\end{enumerate}

\section{Thesis Outline}

Chapter~\ref{chapter:literature} reviews bearing failure, vibration-based condition monitoring, RUL prediction, deep learning baselines, and physics-informed learning. Chapter~\ref{chapter:methodology} describes the IMS dataset, feature extraction, preprocessing, health indicator construction, model architectures, and evaluation metrics. Chapter~\ref{chapter:implementation} describes the local implementation, experiment splits, software environment, and reproducibility setup. Chapter~\ref{chapter:results} presents the results and discusses model behavior across the three experiments. Chapter~\ref{chapter:conclusion} summarizes the findings, contributions, limitations, and future work.
"""


def build_literature_review() -> str:
    return r"""
\chapter{Literature Review}
\label{chapter:literature}

\section{Introduction}

This chapter reviews the research background for bearing RUL prediction with physics-informed neural networks. The discussion follows the structure of the thesis proposal, but it is updated to reflect the completed experiments. It begins with rolling-element bearing failure modes and vibration-based health monitoring, then reviews predictive maintenance methods, deep learning for bearing prognosis, general PINN theory, and recent physics-informed learning work in bearing condition monitoring. The chapter closes by identifying the research gap addressed in this thesis.

\section{Literature Survey}

\subsection{Rolling-Element Bearings: Fundamentals and Failure Modes}

A rolling-element bearing contains an inner race, an outer race, rolling elements, and a cage. The rolling elements transfer load through small contact regions, so repeated stress cycles can initiate subsurface or surface-initiated fatigue. Defects may also arise from lubrication failure, contamination, overload, misalignment, electrical pitting, poor installation, or manufacturing variation. Once a defect begins, repeated rolling contact can grow it into a spall or crack that changes the vibration response of the machine \cite{rycerz2017,sadeghi2009,xu2023}.

The vibration signature depends on the component that is damaged. An outer-race defect produces impacts when rolling elements pass over a fixed damaged region. An inner-race defect rotates with the shaft and is affected by the load zone. Rolling-element and cage defects produce different modulation patterns. These mechanisms explain why bearing monitoring often uses characteristic frequencies derived from bearing geometry and shaft speed. The frequencies are not perfect fault labels, but they provide physically meaningful regions in the spectrum and envelope spectrum \cite{li2000,wu2022}.

The thesis proposal emphasized that bearing prognostics should not treat vibration features as arbitrary numbers. That point remains important in the completed work. RMS, kurtosis, crest factor, and envelope energy have mechanical meaning, even when they are later passed into neural models. Using physically interpretable features also makes it easier to see why a model succeeds or fails on a held-out bearing run.

\subsection{Traditional Predictive Maintenance Techniques for Bearings}

Predictive maintenance is normally discussed as part of condition-based maintenance and prognostics and health management. Jardine et al. \cite{jardine2006} describe the broader machinery diagnostics and prognostics framework, while Lei et al. \cite{lei2018} review the full path from data acquisition to RUL prediction. In a bearing application, the workflow usually includes sensor measurement, preprocessing, feature extraction, health indicator construction, fault diagnosis, and future-life estimation.

Vibration remains one of the most widely used signals for bearing health monitoring because bearing faults generate impacts and modulations in the mechanical response. Time-domain features such as RMS and kurtosis summarize energy and impulsiveness. Frequency-domain and envelope-domain features can reveal periodic impact patterns linked to bearing kinematics. Similarity-based and health-indicator methods then convert these features into a degradation measure or RUL estimate \cite{huang2020}.

The IMS bearing dataset is one of the common public run-to-failure datasets used for this type of work. It is associated with the wavelet-filter bearing prognostics study of Qiu et al. \cite{qiu2006}. The PRONOSTIA platform is another well-known accelerated bearing degradation benchmark \cite{nectoux2012}. Public datasets are valuable because they let different researchers test methods on the same raw measurements. At the same time, public datasets are small compared with industrial variability, so the train-test split must be designed carefully. Random snapshot splits can overstate performance by allowing information from the same run-to-failure trajectory into both training and testing.

\subsection{Deep Learning in Predictive Maintenance}

Deep learning has become popular in bearing prognostics because it can fit nonlinear relationships between sensor measurements and degradation labels. A feed-forward neural network can work directly with engineered feature vectors. A CNN can detect local patterns from a sequence or transformed signal representation. An LSTM can model temporal dependence and is therefore naturally suited to degradation trajectories. Babu et al. \cite{babu2016} showed early CNN-based RUL regression, and later studies extended deep learning for transfer learning, high-quality representation learning, and hybrid sequence architectures \cite{berghout2022,ayman2025,wang2025}.

Recent studies also combine convolutional, recurrent, and autoencoder structures for bearing health monitoring and RUL prediction \cite{farooq2024,han2024,yang2024}. These models are attractive because inference is fast after training and because they can learn patterns that are difficult to specify manually. However, their performance depends strongly on the similarity between the training and test distributions. In bearing RUL prediction, this is a serious issue because two bearings under similar nominal conditions may fail at different times and with different feature trajectories.

For this reason, model evaluation should hold out complete bearing runs when the research question is transfer across bearings. A random split of individual snapshots may measure interpolation within one degradation curve rather than generalization to another bearing. The experiment design in this thesis uses complete held-out runs for validation and testing.

\subsection{Physics-Informed Neural Networks}

Physics-informed neural networks were introduced as a way to train neural networks under physical residual constraints \cite{raissi2019}. In the usual PINN setting, a neural network approximates a field variable, and automatic differentiation is used to compute derivatives that appear in a governing equation. The training loss combines data mismatch and a physics residual. This has made PINNs attractive in fluid mechanics, heat transfer, wave propagation, and inverse problems where governing equations are available.

The PINN literature has expanded quickly. Reviews by Lawal et al. \cite{lawal2022} and Ren et al. \cite{ren2025} describe methodological growth, optimization issues, and applications across scientific computing. The strength of a PINN is not simply that it contains a neural network and a physical equation. Its value depends on whether the physical residual is relevant, correctly weighted, and numerically trainable. Poorly chosen physical constraints can slow training or add bias.

Bearing RUL prediction is not a textbook PINN problem because the degradation process is not governed by a single known partial differential equation. Rolling contact fatigue, lubrication, load distribution, resonance, material variation, and measurement noise interact in ways that are difficult to reduce to one residual. Crack-growth laws such as Paris and Erdogan \cite{paris1963} show how mechanics can describe damage propagation in some contexts, but a complete bearing RUL equation for vibration features is rarely available in practice. This motivates weaker physics-informed learning: monotonic damage behavior, lifetime distributions, fault-frequency consistency, and other engineering priors.

\subsection{Physics-Informed Neural Networks for Bearing Predictive Maintenance}

Recent bearing-related studies have begun to use physics-informed or knowledge-informed learning. Chen et al. \cite{chen2022} proposed a physics-informed deep neural network for bearing prognosis with multisensory signals. Parziale et al. \cite{parziale2023} applied PINNs to condition monitoring of rotating shafts. Herwig et al. \cite{herwig2025} developed a signal-processing and physics-informed network for explainable bearing condition monitoring. Zhong et al. \cite{zhong2025} used a multi-source physics-informed SincNet structure for rolling bearing fault diagnosis. von Hahn and Mechefske \cite{vonhahn2022} investigated a knowledge-informed loss based on Weibull lifetime behavior.

These studies show why the approach is promising. Physics can improve interpretability, guide learning when data are limited, and discourage predictions that conflict with engineering expectations. They also show why the approach must be tested carefully. A physical prior only helps when it matches the measured data closely enough. If the prior is too weak, the model may behave like a data-only network. If it is too strong or mismatched, it may prevent the model from fitting valid degradation patterns.

\section{Research Gap}

The literature shows three gaps that motivate this thesis. First, many bearing RUL studies report strong results, but their preprocessing, RUL labeling, train-test split, and evaluation protocol differ. This makes comparisons difficult. Second, some studies use random sample splits that can leak run-specific degradation information into both training and testing. Third, physics-informed learning for bearing prognosis is still developing, and there is no guarantee that a weak physical prior improves prediction on complete held-out bearing runs.

This thesis addresses a narrower and reproducible question: how does a weak physics-informed RUL model behave when compared with common neural baselines on complete held-out IMS bearing runs? The study uses a local project structure, explicit feature extraction, a defined RUL target, saved tables and figures, and three train-validation-test splits. The goal is not to claim a universal best model. The goal is to test a specific physics-informed formulation and report what the completed experiment shows.
"""


def build_methodology(tables: dict[str, str]) -> str:
    return (
        r"""
\chapter{Dataset and Research Methodology}
\label{chapter:methodology}

\section{Introduction}

This chapter describes the dataset, signal-processing steps, RUL target construction, model definitions, experiment design, and evaluation metrics used in the study. The workflow begins with raw IMS vibration files and ends with model rankings, prediction curves, and error analysis. Figure~\ref{fig:methodology_workflow} summarizes the local pipeline.

"""
        + figure_tex("methodology_flowchart.pdf", "End-to-end local workflow used for IMS bearing RUL prediction.", "fig:methodology_workflow", r"0.82\textwidth")
        + r"""

\section{Dataset Description}

The IMS bearing dataset contains run-to-failure vibration measurements collected from a bearing test rig. The raw data are stored as timestamped text files. Each file contains a vibration snapshot sampled at high frequency. In this project, the extracted dataset folders are stored locally under \texttt{data/raw}.

The local pipeline uses three IMS folders: \texttt{1st\_test}, \texttt{2nd\_test}, and \texttt{3rd\_test}. The first test contains eight columns because two accelerometer channels are available for each of four bearings. The second and third tests contain four columns, one for each bearing. The implementation maps each selected column to a bearing and averages two axes when the first test provides x and y directions for the target bearing.

"""
        + tables["dataset"]
        + r"""

Figure~\ref{fig:dataset_sample_counts} shows the number of snapshots available for each selected run. The difference in run length is large. The third dataset bearing run has 6324 samples, while the second dataset bearing run has 984 samples. This imbalance affects model training and is one reason why Experiment 3 uses balanced sampling for its training runs.

"""
        + figure_tex("dataset_sample_counts.pdf", "Snapshot coverage for the four bearing runs used in the final experiments.", "fig:dataset_sample_counts")
        + r"""

\section{Feature Extraction}

Each vibration snapshot is converted into a feature vector. The time-domain features are RMS, standard deviation, peak-to-peak value, kurtosis, crest factor, and mean absolute value. These features summarize amplitude, spread, impulsiveness, and peak behavior. They are widely used in bearing condition monitoring because they are simple and physically interpretable.

The physics-related features are calculated from envelope spectral energy near bearing fault frequencies. The signal is centered, transformed with the Hilbert transform to obtain the envelope, centered again, windowed with a Hann window, and transformed with a real FFT. For each fault frequency, the pipeline sums spectral energy in a 5 Hz band around the first four harmonics. The four energy features are $E_{FTF}$, $E_{BPFO}$, $E_{BPFI}$, and $E_{BSF}$. Their sum is stored as $E_{kin}$.

The fault-frequency equations used by the implementation are
\begin{equation}
FTF = \frac{f_r}{2}\left(1-\frac{d}{D}\cos\theta\right),
\end{equation}
\begin{equation}
BPFO = \frac{n f_r}{2}\left(1-\frac{d}{D}\cos\theta\right),
\end{equation}
\begin{equation}
BPFI = \frac{n f_r}{2}\left(1+\frac{d}{D}\cos\theta\right),
\end{equation}
\begin{equation}
BSF = \frac{D f_r}{2d}\left(1-\left(\frac{d}{D}\cos\theta\right)^2\right),
\end{equation}
where $f_r$ is shaft frequency, $n$ is the number of rolling elements, $d$ is rolling-element diameter, $D$ is pitch diameter, and $\theta$ is contact angle.

"""
        + tables["features"]
        + tables["freq"]
        + figure_tex("fault_frequencies.pdf", "Bearing fault-frequency values used as markers for envelope spectral-energy extraction.", "fig:fault_frequencies", r"0.82\textwidth")
        + r"""

\section{RUL Target Labeling}

For each selected bearing run, elapsed time is computed from the timestamp of the first snapshot. RUL is computed by subtracting the current timestamp from the timestamp of the final snapshot. The model target is normalized RUL:
\begin{equation}
RUL_{norm}(t) = \frac{RUL(t)}{\max(RUL)}.
\end{equation}
This gives a target near 1 at the beginning of the run and near 0 at the end. The normalization makes results comparable across runs with different durations. It also means that the model predicts relative life fraction, not absolute hours.

\section{Preprocessing}

Feature values can differ greatly between bearing runs. To reduce the effect of run-specific scale, each run is normalized using its early healthy baseline. The first 5 percent of samples, with a minimum of 20 samples, are treated as the healthy baseline. For each model feature, the median baseline value is used. The feature is converted to a nonnegative relative increase, transformed with \texttt{log1p}, and smoothed with a rolling median of 15 samples.

Elapsed time is converted to \texttt{elapsed\_scaled} by dividing elapsed hours by 1200. The model input vector contains \texttt{elapsed\_scaled}, the normalized feature set, and \texttt{sigma\_H\_norm}. The final variable is a constant placeholder kept for compatibility with the original notebook structure. It is not treated as a real Hertzian contact stress calculation in the final interpretation.

\section{PCA Health Indicator}

A PCA-based health indicator is built to inspect whether the extracted features contain a monotonic degradation signal. The model feature matrix is standardized and projected onto the first principal component. The resulting component is scaled between 0 and 1. If the Spearman correlation with damage is negative, the indicator is flipped so that higher values correspond to greater damage.

The health indicator is not used as the final RUL target. It is used as an explanatory tool. It helps show whether the engineered features broadly track degradation.

"""
        + tables["pca"]
        + figure_tex("pca_health_indicator.pdf", "Smoothed PCA health indicator compared with normalized damage on held-out test runs.", "fig:pca_health_indicator")
        + r"""

\section{Model Definitions}

Four models are evaluated. The data-only neural baseline is a feed-forward neural network implemented in DeepXDE \cite{lu2021}. It uses three hidden layers with 64 neurons per layer and tanh activation. A sigmoid output transform constrains predictions between 0 and 1. It is trained with mean squared error.

The proposed physics-informed model uses the same basic feed-forward architecture but adds weak physical residuals. The first residual penalizes positive RUL slope with respect to time beyond a small tolerance, because RUL should generally decrease as life is consumed. The second residual connects high $E_{kin}$ to damage by encouraging the predicted damage term $1 - RUL$ to be consistent with fault-frequency energy. The model loss can be written as
\begin{equation}
\mathcal{L} = \mathcal{L}_{data} + \lambda_m \mathcal{L}_{mono} + \lambda_e \mathcal{L}_{energy},
\end{equation}
where $\mathcal{L}_{data}$ is the supervised RUL loss, $\mathcal{L}_{mono}$ is the monotonicity residual, $\mathcal{L}_{energy}$ is the fault-energy residual, and $\lambda_m$ and $\lambda_e$ are small regularization weights.

The LSTM baseline uses sliding windows of feature vectors. It reads a sequence of length 20 and predicts the RUL at the end of the sequence. The architecture has one LSTM layer with 64 hidden units followed by a small dense head and a sigmoid output. The CNN baseline uses the same sequence windows but applies 1D convolutions across the time axis.

"""
        + tables["models"]
        + r"""

\section{Experiment Design}

The three experiments hold out complete bearing runs for validation and testing. This is stricter than randomly splitting snapshots because the model must transfer across bearing runs. Table~\ref{tab:experiment_split} and Figure~\ref{fig:experiment_split_matrix} show the split design.

"""
        + tables["split"]
        + figure_tex("experiment_split_matrix.pdf", "Experiment split matrix showing how each bearing run is used.", "fig:experiment_split_matrix", r"0.82\textwidth")
        + r"""

\section{Evaluation Metrics}

The models are evaluated with MAE, RMSE, and $R^2$. MAE is the average absolute prediction error. RMSE penalizes larger errors more strongly because the error is squared before averaging. $R^2$ measures how much variance in the target is explained by the prediction. A negative $R^2$ means the model performs worse than predicting the mean target value.

The thesis uses RMSE as the main ranking metric because large RUL errors are especially undesirable in maintenance planning. MAE and $R^2$ are still reported to give a fuller view of model behavior. The metric definitions follow common prognostics evaluation practice \cite{saxena2008}.
"""
    )


def build_implementation() -> str:
    return r"""
\chapter{Local Implementation and Experimental Setup}
\label{chapter:implementation}

\section{Local Project Structure}

The original work began as a Google Colab notebook. For this thesis, the workflow was converted into a local Python project. The code no longer mounts Google Drive. It reads data from \texttt{data/raw}, caches extracted features in \texttt{data/processed\_features}, and writes tables and figures to \texttt{outputs}.

The main package is under \texttt{src/thesis\_work}. It contains separate modules for configuration, IMS data loading, feature extraction, preprocessing, model training, metrics, report generation, and command-line execution. This structure makes the experiment easier to test and rerun than a single notebook.

\section{Reproducibility Commands}

The project uses \texttt{uv} for dependency management. The main commands are:

\begin{verbatim}
uv sync --extra dev
uv run thesis-work validate-data
uv run thesis-work extract-features
uv run thesis-work run
uv run thesis-work regenerate-figures
uv run pytest -q
\end{verbatim}

The full run trains all four models for all three experiments. The command can take a long time on CPU, so the project also supports shorter smoke-test arguments. The final results in this thesis come from the completed full run saved in \texttt{outputs/tables/final\_results\_table.csv}.

\section{Software Environment}

The implementation uses Python 3.11 and the following main libraries: NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, DeepXDE, PyTorch, and tqdm. The LaTeX thesis is compiled with MiKTeX using the Windows batch script included in the thesis source folder.

The local PC used for the completed run had an AMD Ryzen 7 5800H CPU with 8 cores and 16 logical processors, 16 GB RAM, Windows 11 Home Single Language, and an NVIDIA GeForce RTX 3060 Laptop GPU with 4 GB VRAM. Training and inference wall time is machine-specific and should be reported together with this hardware context.

\section{Data Validation}

The validation command checks that the expected folders are present and that the first snapshot in each dataset can be read with the correct number of columns. This step matters because the IMS text files may be stored either as extracted folders or zip files. The local loader supports both, but the final run used extracted folders under \texttt{data/raw}.

\section{Feature Cache}

Feature extraction is the slowest preprocessing step because every vibration snapshot must be read and transformed. The pipeline therefore caches per-run feature tables and a combined \texttt{all\_runs\_features.csv}. Cached features make it possible to regenerate figures and tables without reprocessing the raw data or retraining the models.

\section{Visual Reporting}

The final figures were generated with matplotlib and seaborn using a consistent color palette. Models use the same colors across all plots: gray for the data-only baseline, blue for the proposed PINN, green for LSTM, and orange-red for CNN. Bearing runs also use fixed colors. In the LaTeX thesis, plot files avoid embedded chart titles where possible because figure captions and labels provide the formal titles in the document.
"""


def build_results(tables: dict[str, str]) -> str:
    return (
        r"""
\chapter{Results and Discussion}
\label{chapter:results}

\section{Feature Behavior Over Bearing Life}

The extracted features do not evolve identically across bearing runs. This is expected because different bearings can have different fault initiation times and degradation rates. RMS and peak-to-peak values capture energy growth. Kurtosis and crest factor are more sensitive to impulsive behavior. $E_{kin}$ summarizes energy near the calculated fault-frequency harmonics.

Figures~\ref{fig:feature_rms} to \ref{fig:feature_ekin} show representative feature evolution over normalized lifetime. The plotted feature values are normalized for visualization so that their trends can be compared across bearings.

"""
        + figure_tex("feature_evolution_rms.pdf", "RMS evolution over normalized bearing life.", "fig:feature_rms")
        + figure_tex("feature_evolution_kurtosis.pdf", "Kurtosis evolution over normalized bearing life.", "fig:feature_kurtosis")
        + figure_tex("feature_evolution_crest_factor.pdf", "Crest factor evolution over normalized bearing life.", "fig:feature_crest")
        + figure_tex("feature_evolution_e_kin.pdf", "Combined fault-frequency envelope energy evolution over normalized bearing life.", "fig:feature_ekin")
        + r"""

The plots show why cross-bearing RUL prediction is difficult. Some features rise smoothly in one run and irregularly in another. A model trained on one degradation shape may misread a different shape. This is the central challenge behind the results that follow.

Figure~\ref{fig:feature_correlation} shows the Spearman correlation structure among the extracted features and normalized RUL. The time-domain amplitude features are strongly related to each other, while the fault-frequency energy terms add a different but still related view of degradation.

"""
        + figure_tex("feature_correlation_heatmap.pdf", "Spearman correlation structure of extracted features and normalized RUL.", "fig:feature_correlation", r"0.85\textwidth")
        + r"""

\section{Main Prediction Results}

Table~\ref{tab:final_results} gives the full model results for all three experiments.

"""
        + tables["final"]
        + r"""

The average performance across the three experiments is shown in Table~\ref{tab:average_results}. On average, the LSTM baseline is the strongest model. It has the lowest average RMSE at 0.162 and the highest average $R^2$ at 0.646. The data-only feed-forward baseline is second by average RMSE at 0.185. The proposed physics-informed model has an average RMSE of 0.223. The CNN baseline has the weakest average RMSE at 0.235, although it performs best in Experiment 3.

"""
        + tables["average"]
        + figure_tex("average_rmse_by_model.pdf", "Average RMSE by model across the three experiments.", "fig:average_rmse", r"0.82\textwidth")
        + figure_tex("average_mae_by_model.pdf", "Average MAE by model across the three experiments.", "fig:average_mae", r"0.82\textwidth")
        + figure_tex("average_r2_by_model.pdf", "Average $R^2$ by model across the three experiments.", "fig:average_r2", r"0.82\textwidth")
        + r"""

These results do not support the original expectation that the proposed PINN would dominate the baselines. They show a more specific conclusion: the weak physics-informed formulation can help in one split, but it is not robustly better across all held-out bearing runs.

\section{Experiment-Wise Comparison}

The model ranking by RMSE is given in Table~\ref{tab:ranking} and Figure~\ref{fig:model_ranking}. The ranking figure is oriented so that ranks increase upward: rank 1 is at the bottom and rank 4 is at the top.

"""
        + tables["ranking"]
        + figure_tex("model_ranking_by_rmse.pdf", "Model ranking by RMSE for each experiment. Lower rank is better.", "fig:model_ranking", r"0.82\textwidth")
        + r"""

The ranking changes from experiment to experiment. In Experiment 1, the LSTM baseline performs best with RMSE 0.112. In Experiment 2, the proposed PINN performs best with RMSE 0.144, narrowly ahead of the data-only baseline. In Experiment 3, the CNN baseline performs best with RMSE 0.204, followed by the LSTM. The proposed PINN performs worst in Experiment 3, with negative $R^2$.

Figure~\ref{fig:best_model} makes this instability clear by showing only the lowest-RMSE model in each experiment. A single model does not win every split. The metric comparison plots in Figures~\ref{fig:metric_rmse} to \ref{fig:metric_r2} show the same pattern from different viewpoints.

"""
        + figure_tex("best_model_by_experiment.pdf", "Lowest-RMSE model in each experiment.", "fig:best_model", r"0.78\textwidth")
        + figure_tex("metric_comparison_rmse.pdf", "RMSE by model and experiment.", "fig:metric_rmse")
        + figure_tex("metric_comparison_mae.pdf", "MAE by model and experiment.", "fig:metric_mae")
        + figure_tex("metric_comparison_r2.pdf", "$R^2$ by model and experiment.", "fig:metric_r2")
        + r"""

\section{True Versus Predicted RUL}

Figures~\ref{fig:exp1_predictions} to \ref{fig:exp3_predictions} compare the true normalized RUL curve with model predictions on the held-out test runs.

"""
        + figure_tex("exp_1_true_vs_predicted_rul.pdf", "Experiment 1 true and predicted normalized RUL on the held-out test run.", "fig:exp1_predictions")
        + figure_tex("exp_2_true_vs_predicted_rul.pdf", "Experiment 2 true and predicted normalized RUL on the held-out test run.", "fig:exp2_predictions")
        + figure_tex("exp_3_true_vs_predicted_rul.pdf", "Experiment 3 true and predicted normalized RUL on the held-out test run.", "fig:exp3_predictions")
        + r"""

In Experiment 1, the LSTM follows the decreasing RUL trend more closely than the other models. The proposed PINN is smoother, but it does not match the test curve as well. In Experiment 2, the PINN is more competitive and gives the lowest RMSE. This suggests that the physics prior can help when the training bearings give a degradation pattern compatible with the held-out test bearing. In Experiment 3, the PINN struggles on \texttt{ds1\_b3}, while the CNN and LSTM sequence models handle the test run better.

\section{Error Analysis}

Figure~\ref{fig:absolute_error} shows smoothed absolute prediction error over normalized test life. The largest errors tend to appear when the degradation curve changes more sharply or when a model predicts RUL too conservatively near the end of life.

"""
        + figure_tex("absolute_error_over_life.pdf", "Smoothed absolute prediction error over normalized test life.", "fig:absolute_error")
        + figure_tex("prediction_error_distribution.pdf", "Signed prediction error distribution. Positive error means the model overestimated normalized RUL.", "fig:error_distribution")
        + r"""

The distribution confirms that the sequence models do not merely improve the average score. They also reduce the spread of errors in several cases. The proposed PINN has a wider error distribution because its weak physical prior does not fully capture the different degradation paths of the test bearings.

\section{Discussion of the Physics-Informed Model}

The proposed PINN includes two forms of physical regularization: a monotonicity residual and a spectral prior based on fault-frequency energy. Both are reasonable engineering assumptions, but both are incomplete.

The monotonicity residual assumes RUL should not increase with elapsed life. That is correct at the life-label level, but the model sees noisy features. If the feature pattern of a held-out bearing differs from the training bearings, a strict preference for smooth monotonic behavior can prevent the model from adapting to local signal changes. The implementation uses a tolerance and small weight to reduce this issue, but the result still depends on the split.

The spectral prior assumes that higher combined fault-frequency energy should correspond to greater damage. This is physically plausible, but envelope energy is not a perfect damage variable. It can be affected by resonance, noise, axis direction, load-zone behavior, and the timing of fault development. If $E_{kin}$ grows late or irregularly in one run, a model trained on another run may apply the prior at the wrong time.

The strongest lesson is that physics-informed learning is not automatically better because it contains engineering language. The prior must match the data-generating process closely enough to help. In this thesis, the prior is useful in Experiment 2 but not enough to beat sequence modeling overall.

\section{Why the LSTM Performed Well}

The LSTM baseline uses recent history instead of a single snapshot. This is important because degradation is a process. A feature value by itself may be ambiguous, but its recent trend gives more information. For example, moderate RMS may mean early healthy operation in one run or a temporary plateau after fault growth in another. A short sequence helps the model distinguish these cases.

The LSTM also has fewer assumptions about the exact physical relationship between $E_{kin}$ and damage. It learns temporal patterns directly from the feature sequences. This flexibility likely explains why it achieved the best average RMSE and $R^2$.

\section{Why the CNN Won Experiment 3}

The CNN baseline performs worst on average but best in Experiment 3. This result shows that model choice depends on the train-test pairing. A 1D CNN can detect local temporal patterns in the sliding feature window. In Experiment 3, those local patterns transferred well to \texttt{ds1\_b3}. In the other experiments, the same model did not generalize as strongly.

This does not mean the CNN is the best overall choice. It means that a single average metric can hide split-specific behavior. For a thesis, the split-wise analysis is therefore as important as the average table.

\section{Practical Implications}

For an engineering maintenance system, the safest conclusion is conservative. The current PINN formulation should not be deployed as a final RUL predictor. The LSTM baseline is stronger in the completed experiments, but it also needs more validation before field use. A practical system would need more run-to-failure data, multiple operating conditions, uncertainty estimates, and a clear maintenance decision rule.

The work still gives useful direction. Physics-informed features are valuable for interpretation, and the PINN framework remains promising. The next step is not to abandon physics. It is to improve the physical model, use better degradation constraints, and evaluate on more splits and datasets.
"""
    )


def build_conclusion() -> str:
    return r"""
\chapter{Conclusion}
\label{chapter:conclusion}

\section{Summary}

This thesis studied remaining useful life prediction for rolling-element bearings using the IMS run-to-failure dataset. A local Python project was built from the original notebook so the workflow can run without Google Drive. The pipeline reads raw IMS folders, extracts time-domain and envelope spectral features, constructs normalized RUL labels, builds a PCA health indicator, trains four neural models, and writes reproducible tables and figures.

The proposed model was a DeepXDE physics-informed neural network with weak physical residuals based on monotonic RUL behavior and fault-frequency energy. It was compared with a data-only feed-forward neural baseline, an LSTM sequence baseline, and a CNN sequence baseline across three cross-bearing experiments.

The results were mixed. The proposed PINN achieved the best RMSE in Experiment 2, but it did not perform best overall. The LSTM baseline had the best average performance, with average RMSE 0.162 and average $R^2$ 0.646. The data-only baseline was second by average RMSE. The proposed PINN ranked third by average RMSE, and the CNN ranked fourth, although the CNN performed best in Experiment 3.

\section{Contributions}

The main contributions of this thesis are:

\begin{enumerate}[label=\arabic*.]
    \item A complete local RUL prediction pipeline for the IMS bearing dataset.
    \item A documented feature extraction method combining time-domain vibration features and bearing fault-frequency envelope energy.
    \item A PCA-based health indicator analysis for degradation consistency.
    \item A DeepXDE physics-informed RUL model using monotonicity and spectral priors.
    \item A fair comparison with feed-forward, LSTM, and CNN baselines under complete held-out bearing splits.
    \item A clear discussion of the mismatch between the original proposal expectation and the final experimental evidence.
\end{enumerate}

\section{Limitations}

The study has several limitations. The IMS data provide only a small number of complete failure runs. The RUL target is normalized by the final timestamp, so it does not represent an independently measured failure threshold. The physical prior is weak and does not include a full contact fatigue model, lubrication model, temperature effect, or uncertainty estimate. The model is trained on engineered features rather than raw waveforms. Finally, the results are from one local implementation and one set of hyperparameters.

\section{Future Work}

Future work should test stronger physics-informed formulations. Possible directions include a learned degradation state constrained by monotonic damage growth, uncertainty-aware RUL prediction, Weibull or crack-growth inspired lifetime priors, and multi-task learning that predicts both health indicator and RUL. The experiments should also be repeated on PRONOSTIA and other bearing datasets so that conclusions do not depend on one benchmark.

The feature pipeline can also be improved. Time-frequency representations, adaptive envelope bands, bearing-specific resonance selection, and raw waveform sequence models may capture information lost by the current feature set. More careful hyperparameter tuning and repeated random seeds would make the comparison more statistically reliable.

\section{Final Conclusion}

The completed experiments show that physics-informed bearing RUL prediction is promising but not automatic. In this thesis, the proposed PINN improved the result in one experiment but did not beat the LSTM baseline overall. The best model was the one that used temporal context most effectively. A practical bearing prognostics system should therefore combine physical interpretation with sequence-aware learning, and it should be tested under complete run-level splits before strong claims are made.
"""


def build_appendices(tables: dict[str, str]) -> str:
    appendix_final_table = tables["final"].replace("tab:final_results", "tab:appendix_final_results")
    return (
        r"""
\appendix
\chapter{Reproducibility Notes}

\section{Final Result Table}

The main result table is repeated here for quick reference.

"""
        + appendix_final_table
        + r"""

\section{Commands Used for the Local Pipeline}

\begin{verbatim}
uv sync --extra dev
uv run thesis-work validate-data
uv run thesis-work extract-features
uv run thesis-work run
uv run thesis-work regenerate-figures
uv run pytest -q
\end{verbatim}

\section{Raw Data Location}

The raw IMS folders are expected in:

\begin{verbatim}
data/raw/
  1st_test/
  2nd_test/
  3rd_test/
\end{verbatim}

These folders are intentionally not part of the LaTeX source because they contain the local raw dataset.

\section{LaTeX Build}

The LaTeX source is stored under \texttt{thesis/latex}. The file \texttt{build\_pdf.bat} compiles the document into \texttt{physics\_informed\_bearing\_rul\_thesis.pdf} using MiKTeX.
"""
    )


def write_latex_sources() -> None:
    tables = build_tables()
    write(LATEX / "main.tex", build_main_tex())
    write(SECTIONS / "cover.tex", build_cover())
    write(SECTIONS / "information.tex", build_information())
    write(SECTIONS / "abstract.tex", build_abstract())
    write(SECTIONS / "acknowledgement.tex", build_acknowledgement())
    write(SECTIONS / "nomenclature.tex", build_nomenclature())
    write(SECTIONS / "introduction.tex", build_introduction())
    write(SECTIONS / "literature_review.tex", build_literature_review())
    write(SECTIONS / "methodology.tex", build_methodology(tables))
    write(SECTIONS / "implementation.tex", build_implementation())
    write(SECTIONS / "results.tex", build_results(tables))
    write(SECTIONS / "conclusion.tex", build_conclusion())
    write(SECTIONS / "appendices.tex", build_appendices(tables))
    write(LATEX / "references.bib", build_bibliography())
    write(
        LATEX / "build_pdf.bat",
        r"""
@echo off
setlocal
cd /d "%~dp0"

set JOB=physics_informed_bearing_rul_thesis
set "MIKTEX_BIN=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
set "PDFLATEX=pdflatex"
set "BIBTEX=bibtex"

where pdflatex >nul 2>nul
if errorlevel 1 (
    if exist "%MIKTEX_BIN%\pdflatex.exe" (
        set "PDFLATEX=%MIKTEX_BIN%\pdflatex.exe"
    ) else (
        echo pdflatex was not found. Make sure MiKTeX is installed and on PATH.
        pause
        exit /b 1
    )
)

where bibtex >nul 2>nul
if errorlevel 1 (
    if exist "%MIKTEX_BIN%\bibtex.exe" (
        set "BIBTEX=%MIKTEX_BIN%\bibtex.exe"
    ) else (
        echo bibtex was not found. Make sure MiKTeX is installed and on PATH.
        pause
        exit /b 1
    )
)

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%BIBTEX%" %JOB%
if errorlevel 1 goto fail

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error -jobname=%JOB% main.tex
if errorlevel 1 goto fail

echo.
echo Built %CD%\%JOB%.pdf
pause
exit /b 0

:fail
echo.
echo LaTeX build failed. Check %JOB%.log for details.
pause
exit /b 1
""",
    )
    write(
        LATEX / "README.md",
        r"""
# LaTeX Thesis Source

Run `build_pdf.bat` from this folder to compile the thesis with MiKTeX. The output file is `physics_informed_bearing_rul_thesis.pdf`.

The plots in `assets/images` are thesis-specific versions generated without embedded chart titles where practical. Figure captions in the LaTeX source provide the formal titles and labels.
""",
    )


def main() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    SECTIONS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    figures = prepare_figures()
    write_latex_sources()
    print(f"Wrote LaTeX thesis to {LATEX}")
    print(f"Generated {len(figures)} figure assets in {IMAGES}")
    print(f"Compile with {LATEX / 'build_pdf.bat'}")


if __name__ == "__main__":
    main()
