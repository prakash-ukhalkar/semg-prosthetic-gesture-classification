"""
sEMG Prosthetic Gesture Classification
Module: ml.evaluation_pipeline

Handles dynamic detection of optimized models, multi-metric test-set evaluations,
multiclass ROC/PR curves, probability calibration diagnostics (ECE, Brier),
paired McNemar statistical tests, class confusion error analyses,
and saves publication-quality figures and tables in CSV, Markdown, and LaTeX.
"""

import os
import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

logger = logging.getLogger("semg_prosthetic_classification")

# ==============================================================================
# 1. CORE PERFORMANCE & CALIBRATION METRICS HELPERS
# ==============================================================================

def compute_top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 2) -> float:
    """Compute Top-k classification accuracy."""
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    is_correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return float(np.mean(is_correct))

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class classification."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    correct = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (confidences == bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def compute_multiclass_brier(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Compute Brier Score for multi-class probability forecasts."""
    n_classes = y_probs.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1)))

# ==============================================================================
# 2. STATISTICAL PAIRED TEST
# ==============================================================================

def compute_mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
    """Perform McNemar's test for paired classification predictions."""
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    a = np.sum(correct1 & correct2)
    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 incorrect
    c = np.sum(~correct1 & correct2)  # Model 1 incorrect, Model 2 correct
    d = np.sum(~correct1 & ~correct2)
    
    n_discordant = b + c
    if n_discordant < 25:
        p_val = float(stats.binom.cdf(min(b, c), n_discordant, 0.5) * 2.0)
        p_val = min(1.0, p_val)
        stat = float(b)
        test_type = "exact_binomial"
    else:
        stat = float(((abs(b - c) - 1.0) ** 2) / n_discordant)
        p_val = float(stats.chi2.sf(stat, df=1))
        test_type = "chi2_continuity_corrected"
        
    return {
        "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
        "statistic": stat,
        "p_value": p_val,
        "test_type": test_type,
        "significant": p_val < 0.05
    }

# ==============================================================================
# 3. COMPILATION & VISUALIZATION PIPELINE CLASS
# ==============================================================================

class FinalModelEvaluator:
    def __init__(self, workspace_path: str):
        self.workspace_dir = Path(workspace_path)
        self.outputs_dir = self.workspace_dir / "outputs"
        self.tables_dir = self.outputs_dir / "tables"
        self.figures_dir = self.outputs_dir / "figures"
        self.reports_dir = self.outputs_dir / "reports"
        self.models_dir = self.workspace_dir / "models/optimized"
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot style configuration
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'Inter', 'DejaVu Sans'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
    def detect_optimized_models(self) -> List[Tuple[str, Path]]:
        """Automatically scan and return model names and pickle paths."""
        pkl_files = list(self.models_dir.glob("*.pkl"))
        detected = []
        for p in pkl_files:
            model_name = p.stem.upper()
            detected.append((model_name, p))
        logger.info(f"Detected optimized models: {[n for n, _ in detected]}")
        return detected
        
    def load_dataset(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Load selected features and filter train/val/test splits using split metadata."""
        data_path = self.workspace_dir / "data/final/selected_features_top50.parquet"
        split_meta_path = self.reports_dir / "split_metadata_top50.json"
        
        logger.info("Loading Top 50 features dataset...")
        df = pd.read_parquet(data_path)
        
        with open(split_meta_path, "r", encoding="utf-8") as f:
            split_metadata = json.load(f)
            
        test_subjects = split_metadata["test_subjects"]
        val_subjects = split_metadata["val_subjects"]
        
        df_test = df[df["subject_id"].isin(test_subjects)].copy()
        df_val = df[df["subject_id"].isin(val_subjects)].copy()
        
        meta_cols = [
            "subject_id", "exercise_id", "gesture_id", "window_id", "repetition_id",
            "start_sample", "end_sample", "window_size_samples", "sampling_frequency_hz"
        ]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        X_test = df_test[feature_cols]
        y_test = df_test["gesture_id"].values
        
        # Keep subject_id for subject-specific evaluations
        X_test_with_subj = df_test[feature_cols + ["subject_id"]]
        
        return X_test, y_test, X_test_with_subj, df_val[feature_cols], df_val["gesture_id"].values

    def run_evaluations(self) -> Dict[str, Any]:
        X_test, y_test, X_test_with_subj, _, _ = self.load_dataset()
        detected_models = self.detect_optimized_models()
        
        # Load baseline vs optimized raw to get Notebook 09 training times
        comp_csv = self.tables_dir / "baseline_vs_optimized_comparison.csv"
        df_comp_raw = pd.read_csv(comp_csv) if comp_csv.exists() else None
        
        evaluation_results = {}
        predictions_dict = {}
        probabilities_dict = {}
        
        for name, pkl_path in detected_models:
            logger.info(f"Loading pipeline '{name}' from {pkl_path}...")
            with open(pkl_path, "rb") as f:
                pipeline = pickle.load(f)
                
            # Perform inference timing
            logger.info(f"Running inference on {len(X_test)} samples for '{name}'...")
            t_start = time.perf_counter()
            y_probs = pipeline.predict_proba(X_test)
            inference_time = time.perf_counter() - t_start
            
            y_pred = np.argmax(y_probs, axis=1)
            
            # Save predictions and probabilities for later export
            predictions_dict[name] = y_pred
            probabilities_dict[name] = y_probs
            
            # Calculate overall metrics
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            
            loss = log_loss(y_test, y_probs)
            top2_acc = compute_top_k_accuracy(y_test, y_probs, k=2)
            top3_acc = compute_top_k_accuracy(y_test, y_probs, k=3)
            
            # Calculate ECE and Brier
            ece = compute_ece(y_test, y_probs)
            brier = compute_ece(y_test, y_probs) # wait, let's call brier helper
            brier = compute_multiclass_brier(y_test, y_probs)
            
            # Throughput
            throughput = len(X_test) / inference_time
            latency = (inference_time / len(X_test)) * 1000.0
            
            # Model size
            model_size_mb = pkl_path.stat().st_size / (1024 * 1024)
            
            # Get training time from Notebook 09 outputs
            train_time = 0.0
            if df_comp_raw is not None:
                match = df_comp_raw[(df_comp_raw["Model"] == name) & (df_comp_raw["Metric"].str.startswith("Training Time"))]
                if not match.empty:
                    train_time = float(match["Optimized"].values[0])
            
            # Per-class metrics
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Subject-specific metrics
            subject_f1s = {}
            for subj in np.unique(X_test_with_subj["subject_id"]):
                subj_mask = X_test_with_subj["subject_id"] == subj
                y_true_s = y_test[subj_mask]
                y_pred_s = y_pred[subj_mask]
                subject_f1s[int(subj)] = f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
                
            evaluation_results[name] = {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_precision": prec,
                "macro_recall": rec,
                "macro_f1": f1,
                "weighted_f1": weighted_f1,
                "mcc": mcc,
                "cohen_kappa": kappa,
                "log_loss": loss,
                "top2_accuracy": top2_acc,
                "top3_accuracy": top3_acc,
                "ece": ece,
                "brier_score": brier,
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "latency_ms": latency,
                "throughput_samples_per_sec": throughput,
                "model_size_mb": model_size_mb,
                "class_reports": report_dict,
                "subject_f1s": subject_f1s
            }
            
        # ==============================================================================
        # STATISTICAL ANALYSIS (MCNEMAR PAIRED TESTS)
        # ==============================================================================
        mcnemar_results = []
        for i in range(len(detected_models)):
            for j in range(i + 1, len(detected_models)):
                name1, _ = detected_models[i]
                name2, _ = detected_models[j]
                
                m_test = compute_mcnemar_test(y_test, predictions_dict[name1], predictions_dict[name2])
                mcnemar_results.append({
                    "Model A": name1,
                    "Model B": name2,
                    "Statistic": m_test["statistic"],
                    "p-value": m_test["p_value"],
                    "Significant (p < 0.05)": m_test["significant"],
                    "Test Type": m_test["test_type"]
                })
                
        df_mcnemar = pd.DataFrame(mcnemar_results)
        df_mcnemar.to_csv(self.tables_dir / "mcnemar_paired_tests.csv", index=False)
        df_mcnemar.to_markdown(self.tables_dir / "mcnemar_paired_tests.md", index=False)
        df_mcnemar.to_latex(self.tables_dir / "mcnemar_paired_tests.tex", index=False, float_format="%.4e")
        
        # Save Predictions and Probability Scores
        logger.info("Exporting test predictions and probabilities...")
        export_df = pd.DataFrame({"y_true": y_test})
        for name, y_pred in predictions_dict.items():
            export_df[f"y_pred_{name}"] = y_pred
            
        export_df.to_parquet(self.outputs_dir / "test_predictions.parquet", compression="snappy")
        
        # Export probabilities as dictionary npz
        np.savez_compressed(
            self.outputs_dir / "test_probability_scores.npz",
            **{f"y_probs_{n}": p for n, p in probabilities_dict.items()}
        )
        
        return {
            "eval_results": evaluation_results,
            "predictions": predictions_dict,
            "probabilities": probabilities_dict,
            "y_test": y_test,
            "X_test_with_subj": X_test_with_subj,
            "mcnemar": df_mcnemar
        }
        
    def generate_all_tables(self, eval_data: Dict[str, Any]):
        eval_results = eval_data["eval_results"]
        y_test = eval_data["y_test"]
        
        # 1. Overall Results Table
        overall_rows = []
        for name, metrics in eval_results.items():
            overall_rows.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "MCC": metrics["mcc"],
                "Cohen Kappa": metrics["cohen_kappa"],
                "Log Loss": metrics["log_loss"],
                "Top-2 Accuracy": metrics["top2_accuracy"],
                "Top-3 Accuracy": metrics["top3_accuracy"]
            })
        df_overall = pd.DataFrame(overall_rows)
        df_overall.to_csv(self.tables_dir / "overall_results.csv", index=False)
        df_overall.to_markdown(self.tables_dir / "overall_results.md", index=False)
        df_overall.to_latex(self.tables_dir / "overall_results.tex", index=False, float_format="%.4f")
        
        # 2. Inference Statistics Table
        inf_rows = []
        for name, metrics in eval_results.items():
            inf_rows.append({
                "Model": name,
                "Model Size (MB)": metrics["model_size_mb"],
                "Training Time (s)": metrics["train_time_sec"],
                "Test Inference Time (s)": metrics["inference_time_sec"],
                "Latency per Sample (ms)": metrics["latency_ms"],
                "Prediction Throughput (sps)": metrics["throughput_samples_per_sec"]
            })
        df_inf = pd.DataFrame(inf_rows)
        df_inf.to_csv(self.tables_dir / "inference_statistics.csv", index=False)
        df_inf.to_markdown(self.tables_dir / "inference_statistics.md", index=False)
        df_inf.to_latex(self.tables_dir / "inference_statistics.tex", index=False, float_format="%.4f")
        
        # 3. Calibration Statistics Table
        cal_rows = []
        for name, metrics in eval_results.items():
            cal_rows.append({
                "Model": name,
                "Expected Calibration Error (ECE)": metrics["ece"],
                "Brier Score": metrics["brier_score"]
            })
        df_cal = pd.DataFrame(cal_rows)
        df_cal.to_csv(self.tables_dir / "calibration_statistics.csv", index=False)
        df_cal.to_markdown(self.tables_dir / "calibration_statistics.md", index=False)
        df_cal.to_latex(self.tables_dir / "calibration_statistics.tex", index=False, float_format="%.4f")
        
        # 4. Model Comparison Leaderboard (Ranked by Macro F1)
        df_leaderboard = df_overall.sort_values(by=["Macro F1", "Balanced Accuracy", "MCC"], ascending=False).reset_index(drop=True)
        df_leaderboard.index += 1
        df_leaderboard.to_csv(self.tables_dir / "model_comparison.csv")
        df_leaderboard.to_markdown(self.tables_dir / "model_comparison.md")
        df_leaderboard.to_latex(self.tables_dir / "model_comparison.tex", float_format="%.4f")
        
        # 5. Per-class Results Table
        # Extract per-class F1, Recall, Precision, Support for all models
        class_rows = []
        for name, metrics in eval_results.items():
            reports = metrics["class_reports"]
            for class_id in range(50):
                cls_str = str(class_id)
                if cls_str in reports:
                    class_rows.append({
                        "Model": name,
                        "Gesture Class": class_id,
                        "Precision": reports[cls_str]["precision"],
                        "Recall": reports[cls_str]["recall"],
                        "F1-Score": reports[cls_str]["f1-score"],
                        "Support": int(reports[cls_str]["support"])
                    })
        df_class_results = pd.DataFrame(class_rows)
        df_class_results.to_csv(self.tables_dir / "per_class_results.csv", index=False)
        
        # Make a Markdown pivot version for representation of top/bottom gestures
        # Let's save a summary of hardest and easiest gestures
        # hardest: lowest average F1 across all models
        avg_f1 = df_class_results.groupby("Gesture Class")["F1-Score"].mean().sort_values()
        hardest_10 = avg_f1.head(10).index.tolist()
        easiest_10 = avg_f1.tail(10).index.tolist()
        
        # Save a summary table of the top confused gesture pairs
        error_pairs = []
        for name, preds in eval_data["predictions"].items():
            cm = confusion_matrix(y_test, preds)
            # Remove diagonal (correct predictions)
            np.fill_diagonal(cm, 0)
            # Get flat indices of top 5 error entries
            flat_indices = np.argsort(cm.ravel())[-5:]
            for idx in flat_indices:
                t_class, p_class = np.unravel_index(idx, cm.shape)
                count = cm[t_class, p_class]
                if count > 0:
                    error_pairs.append({
                        "Model": name,
                        "True Gesture Class": t_class,
                        "Predicted Class": p_class,
                        "Misclassification Count": int(count),
                        "Rate (%)": float((count / np.sum(y_test == t_class)) * 100.0)
                    })
        df_errors = pd.DataFrame(error_pairs).sort_values("Misclassification Count", ascending=False)
        df_errors.to_csv(self.tables_dir / "error_analysis_summary.csv", index=False)
        df_errors.to_markdown(self.tables_dir / "error_analysis_summary.md", index=False)
        df_errors.to_latex(self.tables_dir / "error_analysis_summary.tex", index=False, float_format="%.2f")
        
        # Select best model and save outputs/final_best_model.json
        best_model_name = df_leaderboard.iloc[0]["Model"]
        best_model_metrics = eval_results[best_model_name]
        best_model_meta = {
            "best_model_name": best_model_name,
            "metrics": {
                "macro_f1": best_model_metrics["macro_f1"],
                "accuracy": best_model_metrics["accuracy"],
                "balanced_accuracy": best_model_metrics["balanced_accuracy"],
                "mcc": best_model_metrics["mcc"],
                "inference_time_sec": best_model_metrics["inference_time_sec"],
                "latency_ms": best_model_metrics["latency_ms"],
                "throughput_samples_per_sec": best_model_metrics["throughput_samples_per_sec"],
                "model_size_mb": best_model_metrics["model_size_mb"]
            },
            "justification": (
                f"The overall best-performing optimized model is {best_model_name}, which achieved a Macro F1 score of "
                f"{best_model_metrics['macro_f1'] * 100.0:.2f}% and a test classification accuracy of "
                f"{best_model_metrics['accuracy'] * 100.0:.2f}% on the disjoint subject test split. It provides a highly "
                f"compatible footprint of {best_model_metrics['model_size_mb']:.2f} MB and latency of "
                f"{best_model_metrics['latency_ms']:.4f} ms per sample, which is far below the real-time prosthetic control loop threshold (<50 ms)."
            )
        }
        with open(self.outputs_dir / "final_best_model.json", "w", encoding="utf-8") as f:
            json.dump(best_model_meta, f, indent=4)
            
        print("Successfully generated and saved all tables and final_best_model.json.")

    def generate_all_plots(self, eval_data: Dict[str, Any]):
        eval_results = eval_data["eval_results"]
        predictions = eval_data["predictions"]
        probabilities = eval_data["probabilities"]
        y_test = eval_data["y_test"]
        X_test_with_subj = eval_data["X_test_with_subj"]
        
        n_classes = 50
        y_true_onehot = np.eye(n_classes)[y_test]
        
        # Colors map
        colors = {"CATBOOST": "#D35E60", "XGBOOST": "#7293CB", "LIGHTGBM": "#84BA5B"}
        
        # 1. confusion matrices (Normalized and Absolute) for all models
        for name, pred in predictions.items():
            cm = confusion_matrix(y_test, pred)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            
            # Plot Normalized
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_norm, cmap="Blues", cbar=True, xticklabels=5, yticklabels=5, ax=ax)
            ax.set_title(f"Normalized Confusion Matrix: {name} (Test Split)", fontweight="bold")
            ax.set_xlabel("Predicted Gesture ID")
            ax.set_ylabel("True Gesture ID")
            plt.tight_layout()
            fig.savefig(self.figures_dir / f"confusion_matrix_norm_{name.lower()}.png", dpi=300)
            fig.savefig(self.figures_dir / f"confusion_matrix_norm_{name.lower()}.svg")
            fig.savefig(self.figures_dir / f"confusion_matrix_norm_{name.lower()}.pdf")
            plt.close(fig)
            
            # Plot Absolute
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, cmap="Greens", cbar=True, xticklabels=5, yticklabels=5, ax=ax)
            ax.set_title(f"Absolute Confusion Matrix: {name} (Test Split)", fontweight="bold")
            ax.set_xlabel("Predicted Gesture ID")
            ax.set_ylabel("True Gesture ID")
            plt.tight_layout()
            fig.savefig(self.figures_dir / f"confusion_matrix_abs_{name.lower()}.png", dpi=300)
            fig.savefig(self.figures_dir / f"confusion_matrix_abs_{name.lower()}.svg")
            fig.savefig(self.figures_dir / f"confusion_matrix_abs_{name.lower()}.pdf")
            plt.close(fig)
            
        # 2. ROC Curves (per-class in background, macro/micro in foreground)
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, probs in probabilities.items():
            # Compute micro ROC
            fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), probs.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            
            # Compute macro ROC
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            tpr_sum = np.zeros_like(fpr_grid)
            for c in range(n_classes):
                fpr_c, tpr_c, _ = roc_curve(y_true_onehot[:, c], probs[:, c])
                tpr_sum += np.interp(fpr_grid, fpr_c, tpr_c)
            tpr_macro = tpr_sum / n_classes
            roc_auc_macro = auc(fpr_grid, tpr_macro)
            
            ax.plot(fpr_grid, tpr_macro, label=f"{name} Macro (AUC = {roc_auc_macro:.4f})", color=colors[name], linewidth=2)
            ax.plot(fpr_micro, tpr_micro, label=f"{name} Micro (AUC = {roc_auc_micro:.4f})", color=colors[name], linestyle="--", linewidth=1.5)
            
        ax.plot([0, 1], [0, 1], 'k--', color='grey', label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multiclass One-vs-Rest ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "multiclass_roc_curves.png", dpi=300)
        fig.savefig(self.figures_dir / "multiclass_roc_curves.svg")
        fig.savefig(self.figures_dir / "multiclass_roc_curves.pdf")
        plt.close(fig)
        
        # 3. Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, probs in probabilities.items():
            # Micro-average PR
            precision_micro, recall_micro, _ = precision_recall_curve(y_true_onehot.ravel(), probs.ravel())
            ap_micro = average_precision_score(y_true_onehot, probs, average="micro")
            
            # Macro average AP
            ap_macro = average_precision_score(y_true_onehot, probs, average="macro")
            
            # We can interpolate class-wise PR curves to get a clean Macro PR curve
            recall_grid = np.linspace(0.0, 1.0, 1000)
            precision_sum = np.zeros_like(recall_grid)
            for c in range(n_classes):
                prec_c, rec_c, _ = precision_recall_curve(y_true_onehot[:, c], probs[:, c])
                # We reverse rec_c and prec_c to make interpolation monotonic increasing
                precision_sum += np.interp(recall_grid, rec_c[::-1], prec_c[::-1])
            precision_macro = precision_sum / n_classes
            
            ax.plot(recall_grid, precision_macro, label=f"{name} Macro (AP = {ap_macro:.4f})", color=colors[name], linewidth=2)
            ax.plot(recall_micro, precision_micro, label=f"{name} Micro (AP = {ap_micro:.4f})", color=colors[name], linestyle="--", linewidth=1.5)
            
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Multiclass Precision-Recall Curves')
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "precision_recall_curves.png", dpi=300)
        fig.savefig(self.figures_dir / "precision_recall_curves.svg")
        fig.savefig(self.figures_dir / "precision_recall_curves.pdf")
        plt.close(fig)
        
        # 4. Calibration Curves (Reliability Diagrams)
        fig, ax = plt.subplots(figsize=(8, 6))
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for name, probs in probabilities.items():
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            correct = (predictions == y_test)
            
            bin_accuracies = []
            bin_confidences = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                if i == n_bins - 1:
                    in_bin = in_bin | (confidences == bin_upper)
                    
                if np.sum(in_bin) > 0:
                    bin_accuracies.append(np.mean(correct[in_bin]))
                    bin_confidences.append(np.mean(confidences[in_bin]))
                else:
                    # If empty bin, use boundary midpoint
                    bin_accuracies.append(np.nan)
                    bin_confidences.append((bin_lower + bin_upper) / 2.0)
                    
            ax.plot(bin_confidences, bin_accuracies, marker='o', linewidth=2, label=f"{name} (ECE = {eval_results[name]['ece']:.4f})", color=colors[name])
            
        ax.plot([0, 1], [0, 1], 'k--', color='grey', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Accuracy in Bin')
        ax.set_title('Reliability Diagram (Probability Calibration)')
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "probability_calibration.png", dpi=300)
        fig.savefig(self.figures_dir / "probability_calibration.svg")
        fig.savefig(self.figures_dir / "probability_calibration.pdf")
        plt.close(fig)
        
        # 5. Per-class F-1 and Recall Comparisons
        # Load per-class results CSV
        df_cls = pd.read_csv(self.tables_dir / "per_class_results.csv")
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # We will plot F1 and Recall side-by-side for each class as lines or points
        for name in df_cls["Model"].unique():
            df_m = df_cls[df_cls["Model"] == name].sort_values("Gesture Class")
            axes[0].plot(df_m["Gesture Class"], df_m["F1-Score"], label=name, color=colors[name], marker='x', alpha=0.7)
            axes[1].plot(df_m["Gesture Class"], df_m["Recall"], label=name, color=colors[name], marker='o', alpha=0.7)
            
        axes[0].set_ylabel("F1-Score")
        axes[0].set_title("Per-class F1-Score Performance Comparison")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)
        
        axes[1].set_ylabel("Recall")
        axes[1].set_xlabel("Gesture Class ID")
        axes[1].set_title("Per-class Recall Performance Comparison")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / "per_class_performance_comparison.png", dpi=300)
        fig.savefig(self.figures_dir / "per_class_performance_comparison.svg")
        fig.savefig(self.figures_dir / "per_class_performance_comparison.pdf")
        plt.close(fig)
        
        # 6. Latency, Throughput, and Memory Usage Comparisons
        fig, ax = plt.subplots(figsize=(8, 5))
        model_names = []
        throughputs = []
        sizes = []
        
        for name, metrics in eval_results.items():
            model_names.append(name)
            throughputs.append(metrics["throughput_samples_per_sec"])
            sizes.append(metrics["model_size_mb"] * 50)
            
        scatter = ax.scatter(throughputs, [eval_results[n]["macro_f1"]*100 for n in model_names], s=sizes, color=['#D35E60', '#7293CB', '#84BA5B'], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.set_xlabel("Prediction Throughput (samples/second)")
        ax.set_ylabel("Test Macro F1 (%)")
        ax.set_title("Model Throughput vs. Macro F1 Score (Bubble Size represents Model Size)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Annotate
        for i, name in enumerate(model_names):
            ax.annotate(f"{name}\n({sizes[i]/50:.1f} MB)", xy=(throughputs[i], eval_results[name]["macro_f1"]*100), xytext=(12, 0), textcoords="offset points", ha='left', va='center', fontsize=9, weight='bold')
            
        plt.tight_layout()
        fig.savefig(self.figures_dir / "computational_efficiency_comparison.png", dpi=300)
        fig.savefig(self.figures_dir / "computational_efficiency_comparison.svg")
        fig.savefig(self.figures_dir / "computational_efficiency_comparison.pdf")
        plt.close(fig)
        
        # 7. Subject-Specific macro F1 comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        subject_rows = []
        for name, metrics in eval_results.items():
            for subj, f1 in metrics["subject_f1s"].items():
                subject_rows.append({
                    "Model": name,
                    "Subject ID": f"Subj {subj}",
                    "Macro F1": f1
                })
        df_subj = pd.DataFrame(subject_rows)
        sns.barplot(data=df_subj, x="Subject ID", y="Macro F1", hue="Model", palette="Set2", ax=ax)
        ax.set_title("Cross-Subject Performance Variation on Test Split (Held-out Subjects)")
        ax.set_ylabel("Macro F1-Score")
        ax.set_xlabel("Held-out Subject ID")
        ax.grid(True, linestyle="--", alpha=0.4, axis='y')
        plt.tight_layout()
        fig.savefig(self.figures_dir / "cross_subject_performance_variance.png", dpi=300)
        fig.savefig(self.figures_dir / "cross_subject_performance_variance.svg")
        fig.savefig(self.figures_dir / "cross_subject_performance_variance.pdf")
        plt.close(fig)
        
        print("Successfully generated and saved all publication figures.")
