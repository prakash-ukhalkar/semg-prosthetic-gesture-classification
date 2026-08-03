"""
sEMG Prosthetic Gesture Classification
Module: ml.visualization

Generates publication-quality figures (comparing performance, times, heatmaps, and matrices)
in PNG, SVG, and PDF formats at 300-600 DPI.
"""

import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("semg_prosthetic_classification")

def set_publication_style():
    """Set publication-style matplotlib parameters."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })

def save_publication_figure(fig: plt.Figure, filepath_no_ext: Path, dpi: int = 300):
    """
    Save a matplotlib figure in PNG, SVG, and PDF formats.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib Figure object.
    filepath_no_ext : Path
        Target filepath without extension.
    dpi : int, default=300
        DPI for raster (PNG) export.
    """
    filepath_no_ext = Path(filepath_no_ext)
    filepath_no_ext.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    fig.savefig(filepath_no_ext.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    # Save SVG
    fig.savefig(filepath_no_ext.with_suffix(".svg"), bbox_inches="tight")
    # Save PDF
    fig.savefig(filepath_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    
    logger.info(f"Saved figure in PNG, SVG, PDF at: {filepath_no_ext}.*")

def plot_metric_comparison(ranking_df: pd.DataFrame, metric_name: str, y_label: str, title: str, save_path: Path, dpi: int = 300):
    """
    Generate side-by-side bar chart comparing a metric (e.g. Accuracy, F1 macro)
    across all models for Top 25 and Top 50 features.
    """
    set_publication_style()
    
    # Make sure we map column names correctly
    col_map = {
        "accuracy": "test_accuracy",
        "balanced_accuracy": "test_balanced_accuracy",
        "f1_macro": "test_f1_macro",
        "mcc": "test_mcc"
    }
    col_in_df = col_map.get(metric_name.lower(), metric_name)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Clean model names for plotting
    plot_df = ranking_df.copy()
    plot_df["model_name"] = plot_df["model_name"].str.replace("_", " ").str.title()
    # Replace LDA, QDA, KNN, SVM with uppercase
    acronyms = {"Lda": "LDA", "Qda": "QDA", "Knn": "KNN", "Svm": "SVM", "Gnb": "GNB", "Rbf Svm": "RBF SVM", "Linear Svm": "Linear SVM", "Xgboost": "XGBoost", "Lightgbm": "LightGBM", "Catboost": "CatBoost"}
    plot_df["model_name"] = plot_df["model_name"].apply(lambda x: acronyms.get(x, x))
    
    # Re-order models by average metric score to keep the chart clean
    avg_scores = plot_df.groupby("model_name")[col_in_df].mean().sort_values(ascending=False)
    order = avg_scores.index
    
    sns.barplot(
        data=plot_df,
        x="model_name",
        y=col_in_df,
        hue="feature_set",
        order=order,
        palette="viridis",
        ax=ax
    )
    
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Machine Learning Classifier", fontweight="bold", labelpad=10)
    ax.set_ylabel(y_label, fontweight="bold", labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Feature Configuration", loc="upper right")
    
    plt.tight_layout()
    save_publication_figure(fig, save_path, dpi=dpi)
    plt.close(fig)

def plot_time_comparison(ranking_df: pd.DataFrame, save_path: Path, dpi: int = 300):
    """
    Generate training and inference time comparisons on log scales.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clean model names
    plot_df = ranking_df.copy()
    plot_df["model_name"] = plot_df["model_name"].str.replace("_", " ").str.title()
    acronyms = {"Lda": "LDA", "Qda": "QDA", "Knn": "KNN", "Rbf Svm": "RBF SVM", "Linear Svm": "Linear SVM", "Xgboost": "XGBoost", "Lightgbm": "LightGBM", "Catboost": "CatBoost"}
    plot_df["model_name"] = plot_df["model_name"].apply(lambda x: acronyms.get(x, x))
    
    # Sort order by training time on Top 50 features
    t50_df = plot_df[plot_df["feature_set"] == "top50"].sort_values("train_time_sec", ascending=False)
    order = t50_df["model_name"].tolist()
    
    # Panel A: Training Time
    sns.barplot(
        data=plot_df,
        x="model_name",
        y="train_time_sec",
        hue="feature_set",
        order=order,
        palette="magma",
        ax=axes[0]
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training Execution Time", fontweight="bold", pad=10)
    axes[0].set_xlabel("Machine Learning Classifier", fontweight="bold")
    axes[0].set_ylabel("Training Time (seconds) - Log Scale", fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].legend(title="Features")
    
    # Panel B: Inference Throughput
    sns.barplot(
        data=plot_df,
        x="model_name",
        y="test_throughput",
        hue="feature_set",
        order=order,
        palette="mako",
        ax=axes[1]
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Prediction Throughput", fontweight="bold", pad=10)
    axes[1].set_xlabel("Machine Learning Classifier", fontweight="bold")
    axes[1].set_ylabel("Inference Speed (samples/second) - Log Scale", fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].legend(title="Features")
    
    plt.tight_layout()
    save_publication_figure(fig, save_path, dpi=dpi)
    plt.close(fig)

def plot_metrics_heatmap(ranking_df: pd.DataFrame, feature_set: str, save_path: Path, dpi: int = 300):
    """
    Generate metric heatmap for all classifiers on a specific feature set.
    """
    set_publication_style()
    
    fs_df = ranking_df[ranking_df["feature_set"] == feature_set].copy()
    fs_df["model_name"] = fs_df["model_name"].str.replace("_", " ").str.title()
    acronyms = {"Lda": "LDA", "Qda": "QDA", "Knn": "KNN", "Rbf Svm": "RBF SVM", "Linear Svm": "Linear SVM", "Xgboost": "XGBoost", "Lightgbm": "LightGBM", "Catboost": "CatBoost"}
    fs_df["model_name"] = fs_df["model_name"].apply(lambda x: acronyms.get(x, x))
    
    # Set index for heatmap
    fs_df = fs_df.set_index("model_name")
    
    # Metrics to display
    metrics_cols = [
        "test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_mcc", "test_cohen_kappa"
    ]
    heatmap_df = fs_df[metrics_cols]
    
    # Rename columns for presentation
    heatmap_df.columns = ["Accuracy", "Balanced Accuracy", "Macro F1", "MCC", "Cohen Kappa"]
    
    # Sort by Macro F1
    heatmap_df = heatmap_df.sort_values("Macro F1", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar_kws={"label": "Metric Score"},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(f"Classifier Performance Matrix ({feature_set.upper()} Features)", fontweight="bold", pad=15)
    ax.set_ylabel("Machine Learning Classifier", fontweight="bold", labelpad=10)
    ax.set_xlabel("Evaluation Metrics", fontweight="bold", labelpad=10)
    
    plt.tight_layout()
    save_publication_figure(fig, save_path, dpi=dpi)
    plt.close(fig)

def plot_radar_chart(ranking_df: pd.DataFrame, feature_set: str, top_n_models: List[str], save_path: Path, dpi: int = 300):
    """
    Generate a radar/spider chart comparing the top performing classifiers
    across multiple normalized dimensions.
    """
    set_publication_style()
    
    # Filter for feature set and top models
    plot_df = ranking_df[(ranking_df["feature_set"] == feature_set) & (ranking_df["model_name"].isin(top_n_models))].copy()
    plot_df["model_name"] = plot_df["model_name"].str.replace("_", " ").str.title()
    acronyms = {"Lda": "LDA", "Qda": "QDA", "Knn": "KNN", "Rbf Svm": "RBF SVM", "Linear Svm": "Linear SVM", "Xgboost": "XGBoost", "Lightgbm": "LightGBM", "Catboost": "CatBoost"}
    plot_df["model_name"] = plot_df["model_name"].apply(lambda x: acronyms.get(x, x))
    
    metrics = ["test_accuracy", "test_balanced_accuracy", "test_f1_macro", "test_mcc", "test_cohen_kappa"]
    metric_labels = ["Accuracy", "Balanced Acc", "Macro F1", "MCC", "Cohen Kappa"]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metric_labels, color="grey", size=10, fontweight="bold")
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    colors = plt.colormaps.get_cmap("tab10")
    
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values += values[:1]  # complete the loop
        
        lbl = row["model_name"]
        color = colors(idx % 10)
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=lbl, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title(f"Multi-Metric Profiling of Top Classifiers ({feature_set.upper()})", fontweight="bold", pad=20)
    
    plt.tight_layout()
    save_publication_figure(fig, save_path, dpi=dpi)
    plt.close(fig)

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name: str, feature_set: str, save_path: Path, dpi: int = 300):
    """
    Generate confusion matrix heatmap for a specific classifier.
    Since we have 50 gestures, a full 50x50 confusion matrix can be large,
    so we configure a high-resolution, readable heatmap representation.
    """
    set_publication_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Handle division by zero
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        cbar=True,
        xticklabels=5, # show label every 5
        yticklabels=5,
        ax=ax
    )
    
    # Clean model title
    model_title = model_name.replace("_", " ").title()
    acronyms = {"Lda": "LDA", "Qda": "QDA", "Knn": "KNN", "Rbf Svm": "RBF SVM", "Linear Svm": "Linear SVM", "Xgboost": "XGBoost", "Lightgbm": "LightGBM", "Catboost": "CatBoost"}
    model_title = acronyms.get(model_title, model_title)
    
    ax.set_title(f"Normalized Confusion Matrix: {model_title} ({feature_set.upper()} Features)", fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Gesture ID", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Gesture ID", fontweight="bold", labelpad=10)
    
    plt.tight_layout()
    save_publication_figure(fig, save_path, dpi=dpi)
    plt.close(fig)
