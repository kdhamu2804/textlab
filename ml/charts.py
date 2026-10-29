"""Charts of the evaluation results (matplotlib, saved as PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.evaluation import METRICS  # noqa: E402
from ml.reporting import MODEL_LABELS  # noqa: E402

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


def _legend(ax: plt.Axes, ncol: int) -> None:
    """Place a frameless legend below the plot."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
