"""
sEMG Prosthetic Gesture Classification
Sub-package: ml

Reusable components for machine learning benchmarking, splits, models, pipelines,
training, evaluation, and visualizations.
"""

from src.ml.splits import verify_dataset_integrity, get_subject_splits
from src.ml.models import initialize_model, get_model_catalog
from src.ml.pipelines import create_pipeline
from src.ml.training import train_classifier
from src.ml.evaluation import evaluate_classifier
from src.ml.benchmark import run_benchmark

# Optimization modules
from src.ml.search_space import get_search_space
from src.ml.early_stopping import get_pruner, get_sampler, get_early_stopping_params
from src.ml.objectives import create_objective
from src.ml.trials import save_trial_history
from src.ml.optimization import run_optuna_optimization, finalize_from_existing_study
from src.ml.evaluation_pipeline import FinalModelEvaluator

# LOSO cross-subject validation modules
from src.ml.fold_metrics import compute_fold_metrics, save_fold_checkpoint, load_fold_checkpoint
from src.ml.aggregate_results import aggregate_fold_metrics, compute_95_ci
from src.ml.subject_analysis import analyze_subject_performances, analyze_confusion_matrices
from src.ml.loso import run_loso_validation, load_optimized_pipeline
from src.ml.cross_subject import generate_all_loso_plots
