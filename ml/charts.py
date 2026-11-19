"""Charts of the evaluation results (matplotlib, saved as PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.evaluation import METRICS
from ml.reporting import MODEL_LABELS

# Categorical slots 1-3 of a validated colorblind-safe palette, fixed per entity.
MODEL_COLORS: dict[str, str] = {"majority": "#2a78d6", "tfidf": "#eb6834", "roberta": "#1baf7a"}
SETTING_COLORS: dict[str, str] = {"in_topic": "#2a78d6", "new_topic": "#eb6834"}
SETTING_LABELS: dict[str, str] = {
    "in_topic": "Seen topic (standard model)",
    "new_topic": "New topic (held out)",
}
METRIC_LABELS: dict[str, str] = {
    "f1": "F1 (stress)", "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply a light, recessive style to an axis."""
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def _bars(
    ax: plt.Axes, x: np.ndarray, value: np.ndarray, low: np.ndarray, high: np.ndarray,
    width: float, color: str, label: str,
) -> None:
    """Draw bars with CI error bars and a value label above each bar."""
    ax.bar(x, value, width=width, color=color, label=label, edgecolor="white", linewidth=2)
    ax.errorbar(x, value, yerr=[value - low, high - value], fmt="none",
                ecolor=TEXT, elinewidth=1, capsize=3)
    for xi, v, hi in zip(x, value, high, strict=True):
        ax.text(xi, hi + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=MUTED)


def _grouped(
    ax: plt.Axes, groups: list[pd.DataFrame], columns: list[str],
    colors: list[str], labels: list[str],
) -> np.ndarray:
    """Draw one bar series per group; ``columns`` are the metric columns along x."""
    base = np.arange(len(columns))
    width = 0.8 / len(groups)
    for i, (rows, color, label) in enumerate(zip(groups, colors, labels, strict=True)):
        offset = (i - (len(groups) - 1) / 2) * width
        _bars(ax, base + offset, *(rows[k].to_numpy() for k in ("value", "low", "high")),
              width=width, color=color, label=label)
    return base


def plot_model_comparison(table: pd.DataFrame, path: Path) -> None:
    """Grouped bars: each metric on the full test set, one bar per model."""
    standard = table[(table["setting"] == "standard") & (table["test_set"] == "all")]
    models = [m for m in MODEL_LABELS if m in set(standard["model"])]
    groups = []
    for model in models:
        row = standard[standard["model"] == model].iloc[0]
        groups.append(pd.DataFrame({
            "value": [row[m] for m in METRICS],
            "low": [row[f"{m}_low"] for m in METRICS],
            "high": [row[f"{m}_high"] for m in METRICS],
        }))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    base = _grouped(ax, groups, list(METRICS), [MODEL_COLORS[m] for m in models],
                    [MODEL_LABELS[m] for m in models])
    ax.set_xticks(base, [METRIC_LABELS[m] for m in METRICS])
    _style(ax, "Model comparison on the 715 test posts (95% bootstrap CI)", "Score")
    _legend(ax, len(models))
    _save(fig, path)


def plot_new_topic(table: pd.DataFrame, model: str, path: Path) -> None:
    """Bars per topic: stress F1 when the topic was seen in training vs held out."""
    rows = table[(table["model"] == model) & table["setting"].isin(SETTING_COLORS)]
    topics = sorted(rows["test_set"].unique())
    groups = []
    for setting in SETTING_COLORS:
        part = rows[rows["setting"] == setting].set_index("test_set").loc[topics]
        groups.append(pd.DataFrame({"value": part["f1"], "low": part["f1_low"],
                                    "high": part["f1_high"]}))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    base = _grouped(ax, groups, topics, list(SETTING_COLORS.values()),
                    list(SETTING_LABELS.values()))
    ax.set_xticks(base, topics)
    _style(ax, f"{MODEL_LABELS[model]}: stress F1 on seen vs new topics (95% CI)", "F1 (stress)")
    _legend(ax, 2)
    _save(fig, path)


def plot_latency(server_ms: np.ndarray, client_ms: np.ndarray, model: str, path: Path) -> None:
    """Histogram of round-trip prediction time with mean and p95 marked."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.hist(client_ms, bins=40, color=MODEL_COLORS.get(model, "#2a78d6"), edgecolor="white", linewidth=1)
    for value, name, style in ((np.mean(client_ms), "mean", "-"), (np.percentile(client_ms, 95), "p95", "--")):
        ax.axvline(value, color=TEXT, linestyle=style, linewidth=1)
        ax.text(value, ax.get_ylim()[1] * 0.95, f" {name} {value:.1f} ms", color=TEXT, fontsize=8, va="top")
    ax.set_title(f"{MODEL_LABELS.get(model, model)}: time per /predict request, {len(client_ms)} test posts",
                 loc="left", fontsize=12, color=TEXT, pad=12)
    ax.set_xlabel(f"Round-trip time (ms) · inside the API: mean {np.mean(server_ms):.1f} ms, "
                  f"p95 {np.percentile(server_ms, 95):.1f} ms", color=MUTED)
    ax.set_ylabel("Requests", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    _save(fig, path)


def plot_confusion(conf: pd.DataFrame, path: Path) -> None:
    """One 2x2 confusion matrix per model (rows = true label, columns = predicted)."""
    fig, axes = plt.subplots(1, len(conf), figsize=(3.4 * len(conf), 3.4), dpi=150, squeeze=False)
    labels = ["no stress", "stress"]
    for ax, (_, row) in zip(axes[0], conf.iterrows(), strict=True):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(matrix, cmap="Greens", vmin=0, vmax=matrix.sum())
        for (i, j), value in np.ndenumerate(matrix):
            share = value / matrix[i].sum() if matrix[i].sum() else 0
            ax.text(j, i, f"{value}\n{share:.0%} of row", ha="center", va="center", fontsize=9,
                    color="white" if value > matrix.sum() * 0.35 else TEXT)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("True", color=MUTED)
        ax.set_title(MODEL_LABELS.get(row["model"], row["model"]), loc="left", fontsize=11, color=TEXT)
        ax.tick_params(colors=MUTED, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    _save(fig, path)


def plot_calibration(bins: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Reliability diagram: predicted P(stress) vs observed stress rate."""
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Perfect calibration")
    for model, part in bins.groupby("model", sort=False):
        ece = summary.loc[summary["model"] == model, "ece"].iloc[0]
        ax.plot(part["mean_predicted"], part["observed_rate"], marker="o", markersize=4,
                color=MODEL_COLORS.get(model, TEXT), label=f"{MODEL_LABELS.get(model, model)} (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    _style(ax, "Calibration on the 715 test posts", "Observed share of stress posts")
    ax.set_xlabel("Mean predicted probability of stress", color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    _save(fig, path)


def _legend(ax: plt.Axes, ncol: int) -> None:
    """Place a frameless legend below the plot."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
