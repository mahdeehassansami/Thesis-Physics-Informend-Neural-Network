from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


FONT_FAMILY = ["Aptos", "Segoe UI", "DejaVu Sans", "Arial"]

COLORS = {
    "ink": "#20262E",
    "muted": "#68707C",
    "grid": "#D9DEE7",
    "panel": "#FFFFFF",
    "blue": "#3B6EA8",
    "blue_dark": "#234A73",
    "gold": "#C9972B",
    "gold_dark": "#816018",
    "olive": "#5F8E5E",
    "olive_dark": "#3C613B",
    "orange": "#D9793D",
    "orange_dark": "#93491F",
    "pink": "#C15E87",
    "pink_dark": "#843A58",
    "teal": "#3E8F91",
    "teal_dark": "#235C5E",
    "gray": "#9AA3AF",
    "gray_dark": "#5E6673",
}

RUN_PALETTE = {
    "ds2_b1": COLORS["blue"],
    "ds1_b3": COLORS["gold"],
    "ds1_b4": COLORS["olive"],
    "ds3_b3": COLORS["pink"],
}

MODEL_PALETTE = {
    "Data-only neural baseline": COLORS["gray"],
    "Proposed DeepXDE Physics-Informed RUL Model": COLORS["blue"],
    "LSTM baseline": COLORS["olive"],
    "CNN baseline": COLORS["orange"],
}


def apply_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font=FONT_FAMILY[0],
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.facecolor": COLORS["panel"],
            "figure.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.9,
            "axes.titlesize": 15,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.title_fontsize": 9,
        },
    )


def new_figure(width: float = 9.5, height: float = 5.2):
    apply_theme()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    return fig, ax


def add_header(fig, title: str, subtitle: str | None = None) -> None:
    fig.subplots_adjust(top=0.82)
    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=14.2, weight="bold", color=COLORS["ink"])
    if subtitle:
        fig.text(0.08, 0.922, subtitle, ha="left", va="top", fontsize=9.6, color=COLORS["muted"])


def polish_axes(ax, xlabel: str | None = None, ylabel: str | None = None, grid_axis: str = "y") -> None:
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.grid(False)
    if grid_axis in {"x", "both"}:
        ax.xaxis.grid(True)
    if grid_axis in {"y", "both"}:
        ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(length=0)


def top_legend(ax, ncol: int = 4, title: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title=title,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=min(ncol, max(1, len(labels))),
        borderaxespad=0,
        handlelength=2.2,
        columnspacing=1.2,
    )


def model_handles(models: list[str]) -> list[Patch]:
    return [Patch(facecolor=MODEL_PALETTE[m], edgecolor=COLORS["ink"], label=m) for m in models]
