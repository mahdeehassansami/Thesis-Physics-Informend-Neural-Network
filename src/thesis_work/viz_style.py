from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.patches import Patch


def register_latin_modern() -> None:
    font_dirs = [
        Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "fonts" / "opentype" / "public" / "lm",
        Path("/usr/share/texmf/fonts/opentype/public/lm"),
        Path("/usr/local/share/texmf/fonts/opentype/public/lm"),
    ]
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for font_file in font_dir.glob("lmroman*.otf"):
            font_manager.fontManager.addfont(str(font_file))


register_latin_modern()

FONT_FAMILY = ["Latin Modern Roman", "DejaVu Serif", "Times New Roman"]

COLORS = {
    "ink": "#000000",
    "muted": "#000000",
    "grid": "#D9DDE3",
    "panel": "#FFFFFF",
    "blue": "#0072B2",
    "blue_dark": "#00466E",
    "sky": "#56B4E9",
    "sky_dark": "#277DAA",
    "gold": "#E69F00",
    "gold_dark": "#9A6A00",
    "emerald": "#009E73",
    "emerald_dark": "#00664A",
    "vermillion": "#D55E00",
    "vermillion_dark": "#8F3F00",
    "violet": "#CC79A7",
    "violet_dark": "#88486C",
    "teal": "#00A6A6",
    "teal_dark": "#006B6B",
    "slate": "#7A869A",
    "slate_dark": "#475467",
}

RUN_PALETTE = {
    "ds2_b1": COLORS["blue"],
    "ds1_b3": COLORS["gold"],
    "ds1_b4": COLORS["emerald"],
    "ds3_b3": COLORS["violet"],
}

MODEL_PALETTE = {
    "Data-only neural baseline": COLORS["slate"],
    "Proposed DeepXDE Physics-Informed RUL Model": COLORS["blue"],
    "LSTM baseline": COLORS["emerald"],
    "CNN baseline": COLORS["vermillion"],
}


def apply_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font=FONT_FAMILY[0],
        rc={
            "font.family": "serif",
            "font.serif": FONT_FAMILY,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": COLORS["panel"],
            "figure.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["panel"],
            "text.color": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.85,
            "axes.linewidth": 1.05,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.labelcolor": COLORS["ink"],
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
        },
    )


def new_figure(width: float = 9.5, height: float = 5.2):
    apply_theme()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    return fig, ax


def add_header(fig, title: str, subtitle: str | None = None) -> None:
    _ = (title, subtitle)
    fig.subplots_adjust(top=0.90)


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
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(length=0)


def top_legend(ax, ncol: int = 4, title: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(
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
    for text in legend.get_texts():
        text.set_color(COLORS["ink"])
    if legend.get_title() is not None:
        legend.get_title().set_color(COLORS["ink"])


def model_handles(models: list[str]) -> list[Patch]:
    return [Patch(facecolor=MODEL_PALETTE[m], edgecolor=COLORS["ink"], label=m) for m in models]
