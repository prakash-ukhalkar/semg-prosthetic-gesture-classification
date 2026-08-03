"""
sEMG Prosthetic Gesture Classification
Module: ml.cross_subject

Publication-quality visualisations for LOSO cross-subject generalisation results.
Generates fold accuracy/F1 line plots, subject ranking bar charts, metric distribution
box/violin plots, and aggregated confusion-matrix heatmaps.

All figures are saved in PNG (300 DPI), SVG, and PDF.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("semg_prosthetic_classification")

# ── house style ──────────────────────────────────────────────────────────────
_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Inter", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _save_fig(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    """Save a figure in PNG, SVG and PDF."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fold-wise line / bar plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_fold_metrics(
    fold_metrics: List[Dict[str, Any]],
    figures_dir: Path,
    model_name: str,
) -> None:
    """Line plots of Accuracy and Macro F1 across the 40 LOSO folds."""
    plt.rcParams.update(_STYLE)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics)
    subjects = df["Subject ID"].values

    # ── Accuracy ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(subjects.astype(str), df["Accuracy"] * 100,
           color="#7293CB", edgecolor="black", linewidth=0.6)
    mean_acc = df["Accuracy"].mean() * 100
    ax.axhline(mean_acc, color="#D35E60", ls="--", lw=1.5,
               label=f"Mean = {mean_acc:.2f}%")
    ax.set_xlabel("Held-Out Subject ID", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(
        f"{model_name} — LOSO Fold Accuracy (N = {len(subjects)} Subjects)",
        fontweight="bold", pad=12,
    )
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_fold_accuracy_{model_name.lower()}")

    # ── Macro F1 ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(subjects.astype(str), df["Macro F1"] * 100,
           color="#84BA5B", edgecolor="black", linewidth=0.6)
    mean_f1 = df["Macro F1"].mean() * 100
    ax.axhline(mean_f1, color="#D35E60", ls="--", lw=1.5,
               label=f"Mean = {mean_f1:.2f}%")
    ax.set_xlabel("Held-Out Subject ID", fontweight="bold")
    ax.set_ylabel("Macro F1 (%)", fontweight="bold")
    ax.set_title(
        f"{model_name} — LOSO Fold Macro F1 (N = {len(subjects)} Subjects)",
        fontweight="bold", pad=12,
    )
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_fold_macro_f1_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Subject ranking chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_subject_ranking(
    fold_metrics: List[Dict[str, Any]],
    figures_dir: Path,
    model_name: str,
) -> None:
    """Horizontal bar chart of subjects sorted by Macro F1."""
    plt.rcParams.update(_STYLE)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics).sort_values("Macro F1", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    colours = ["#D35E60" if v < df["Macro F1"].quantile(0.25)
               else "#84BA5B" if v > df["Macro F1"].quantile(0.75)
               else "#7293CB" for v in df["Macro F1"]]
    ax.barh(
        df["Subject ID"].astype(str),
        df["Macro F1"] * 100,
        color=colours,
        edgecolor="black",
        linewidth=0.6,
        height=0.7,
    )
    ax.set_xlabel("Macro F1 (%)", fontweight="bold")
    ax.set_ylabel("Subject ID", fontweight="bold")
    ax.set_title(
        f"{model_name} — LOSO Subject Ranking by Macro F1",
        fontweight="bold", pad=12,
    )
    ax.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_subject_ranking_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Box and violin plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_distributions(
    fold_metrics: List[Dict[str, Any]],
    figures_dir: Path,
    model_name: str,
) -> None:
    """Combined box + violin plots for key metrics."""
    plt.rcParams.update(_STYLE)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics)
    metric_cols = ["Accuracy", "Balanced Accuracy", "Macro F1",
                   "Weighted F1", "MCC", "Cohen Kappa"]
    df_long = df[metric_cols].melt(var_name="Metric", value_name="Score")

    # ── Boxplot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_long, x="Metric", y="Score", hue="Metric", ax=ax,
                palette="Set2", width=0.5, fliersize=4, legend=False)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(
        f"{model_name} — LOSO Metric Distributions (Box Plot)",
        fontweight="bold", pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_boxplot_{model_name.lower()}")

    # ── Violin ──
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df_long, x="Metric", y="Score", hue="Metric", ax=ax,
                   palette="Set2", inner="quart", cut=0, legend=False)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(
        f"{model_name} — LOSO Metric Distributions (Violin Plot)",
        fontweight="bold", pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_violin_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Aggregated confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggregated_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
    model_name: str,
    n_classes: int = 50,
) -> None:
    """Normalised aggregated confusion matrix heatmap."""
    plt.rcParams.update(_STYLE)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Recall (Row-Normalised)", "shrink": 0.6},
        ax=ax,
    )
    ax.set_xlabel("Predicted Gesture Class", fontweight="bold")
    ax.set_ylabel("True Gesture Class", fontweight="bold")
    ax.set_title(
        f"{model_name} — Aggregated LOSO Confusion Matrix (Row-Normalised)",
        fontweight="bold", pad=12,
    )
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_confusion_matrix_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Best / worst subject comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_worst_subjects(
    fold_metrics: List[Dict[str, Any]],
    figures_dir: Path,
    model_name: str,
    n: int = 5,
) -> None:
    """Side-by-side bars comparing the top-N best and bottom-N worst subjects."""
    plt.rcParams.update(_STYLE)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics).sort_values("Macro F1", ascending=False)
    best = df.head(n)
    worst = df.tail(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].barh(
        best["Subject ID"].astype(str),
        best["Macro F1"] * 100,
        color="#84BA5B", edgecolor="black", linewidth=0.8,
    )
    axes[0].set_xlabel("Macro F1 (%)", fontweight="bold")
    axes[0].set_title(f"Top {n} Best Subjects", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", ls="--", alpha=0.4)

    axes[1].barh(
        worst["Subject ID"].astype(str),
        worst["Macro F1"] * 100,
        color="#D35E60", edgecolor="black", linewidth=0.8,
    )
    axes[1].set_xlabel("Macro F1 (%)", fontweight="bold")
    axes[1].set_title(f"Bottom {n} Worst Subjects", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", ls="--", alpha=0.4)

    fig.suptitle(
        f"{model_name} — LOSO Best vs Worst Subjects",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, figures_dir, f"loso_best_worst_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_loso_plots(
    loso_results: Dict[str, Any],
    figures_dir: Path,
) -> None:
    """Generate every publication figure for the given LOSO run."""
    model_name = loso_results["model_name"]
    fold_metrics = loso_results["fold_metrics"]
    y_true = loso_results["all_y_true"]
    y_pred = loso_results["all_y_pred"]

    plot_fold_metrics(fold_metrics, figures_dir, model_name)
    plot_subject_ranking(fold_metrics, figures_dir, model_name)
    plot_metric_distributions(fold_metrics, figures_dir, model_name)
    plot_aggregated_confusion_matrix(y_true, y_pred, figures_dir, model_name)
    plot_best_worst_subjects(fold_metrics, figures_dir, model_name)

    logger.info(f"All LOSO publication figures saved to {figures_dir}")
